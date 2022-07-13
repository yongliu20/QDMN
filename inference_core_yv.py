"""
This file specifies an advanced version of inference_core.py
specific for YouTubeVOS evaluation (which is not trivial and one has to be very careful!)

In a very high level, we perform propagation independently for each object
and we start memorization for each object only after their first appearance
which is also when YouTubeVOS gives the "first frame"

The "first frame" for each object is different
"""

from collections import defaultdict
import torch
import numpy as np
import cv2

from model.eval_network import PropagationNetwork
from model.aggregate import aggregate_wbg

from util.tensor_util import pad_divide_by
import torch.nn.functional as F



class InferenceCore:
    def __init__(self, prop_net:PropagationNetwork, images, num_objects, mem_freq=5):
        self.prop_net = prop_net
        self.mem_freq = mem_freq

        # True dimensions
        t = images.shape[1]
        h, w = images.shape[-2:]

        # Pad each side to multiple of 16
        images, self.pad = pad_divide_by(images, 16)
        # Padded dimensions
        nh, nw = images.shape[-2:]

        self.images = images
        self.device = 'cuda'

        self.k = num_objects
        self.masks = torch.zeros((t, 1, nh, nw), dtype=torch.uint8, device=self.device)
        self.out_masks = np.zeros((t, h, w), dtype=np.uint8)

        # Background included, not always consistent (i.e. sum up to 1)
        self.prob = torch.zeros((self.k+1, t, 1, nh, nw), dtype=torch.float32, device=self.device)
        self.prob[0] = 1e-7

        self.t, self.h, self.w = t, h, w
        self.nh, self.nw = nh, nw
        self.kh = self.nh//16
        self.kw = self.nw//16
        self.memory_thr = 10

        # The keys/values are always presevered in YouTube testing
        # the reason is that we still consider it as a single propagation pass
        # just that some objects are arriving later than usual
        self.keys = dict()
        self.values = dict()

        # list of objects with usable memory
        self.enabled_obj = []

    def get_query_kv_buffered(self, idx, first, score=None):
        # not actually buffered
        f16, f8, f4 = self.prop_net.get_query_values(self.images[:,idx].cuda())
        _, _, h, w = f16.size()
        if score is None:
            previous_mask = F.softmax(self.prob[:, idx - 1], dim=0)
            previous_mask, _ = torch.max(previous_mask[1:], dim=0)
        else:
            if score < 0.0:
                previous_mask = F.softmax(self.prob[:, first], dim=0)
                previous_mask, _ = torch.max(previous_mask[1:], dim=0)
            else:
                previous_mask = F.softmax(self.prob[:, idx - 1], dim=0)
                previous_mask, _ = torch.max(previous_mask[1:], dim=0)

        previous_mask = previous_mask.unsqueeze(0)
        previous_mask = F.interpolate(previous_mask, size=[h, w], mode='bilinear')
        concat_f16 = torch.cat((f16, previous_mask), dim=1)
        concat_f16 = self.prop_net.concat_conv(concat_f16)
        concat_f16 = torch.sigmoid(concat_f16)
        f16_final = f16 * concat_f16
        k16, v16 = self.prop_net.kv_q_f16(f16_final)
        result = (f16_final, f8, f4, k16, v16)
        return result

    def do_pass(self, key_k, key_v, idx, end_idx):
        """
        key_k, key_v - Memory feature of the starting frame
        idx - Frame index of the starting frame
        end_idx - Frame index at which we stop the propagation
        """
        closest_ti = end_idx
        memory = []
        sa = {}

        K, CK, _, H, W = key_k.shape
        _, CV, _, _, _ = key_v.shape

        for i, oi in enumerate(self.enabled_obj):       #先把第一帧的key和value存进去
            if oi not in self.keys:                                 #如果一个物体第一次出现，那么就将它的key和value添加到dict key和dict value中
                self.keys[oi] = key_k[i:i+1]
                self.values[oi] = key_v[i:i+1]
            else:                                                   #如果一个物体出现过，那么就将它的key和value按T维度和此前的进行concat
                self.keys[oi] = torch.cat([self.keys[oi], key_k[i:i+1]], 2)
                self.values[oi] = torch.cat([self.values[oi], key_v[i:i+1]], 2)

        prev_in_mem = True
        prev_key = {}
        prev_value = {}
        last_ti = idx       #last_ti是memory中除第一帧外最早的帧的索引，idx是第一帧的索引
        index = 0

        # Note that we never reach closest_ti, just the frame before it
        this_range = range(idx+1, closest_ti)
        step = +1
        end = closest_ti - 1



        for ti in this_range:       #ti下一帧到最后一帧的索引
            if prev_in_mem:
                # if the previous frame has already been added to the memory bank
                this_k = self.keys
                this_v = self.values
            else:
                # append it to a temporary memory bank otherwise
                # everything has to be done independently for each object
                this_k = {}
                this_v = {}
                for i, oi in enumerate(self.enabled_obj):
                    this_k[oi] = torch.cat([self.keys[oi], prev_key[i:i+1]], 2)
                    this_v[oi] = torch.cat([self.values[oi], prev_value[i:i+1]], 2)

            if ti == idx+1:     #第二帧没有score
                query = self.get_query_kv_buffered(ti, idx)      #提取当前帧的key、value以及图片特征
            else:
                query = self.get_query_kv_buffered(ti, idx, scores)

            out_mask = torch.cat([
                self.prop_net.segment_with_query(this_k[oi], this_v[oi], *query)
            for oi in self.enabled_obj], 0)     #在不同物体的channel（dim=0）上进行预测      N,1,H,W



            out_mask = aggregate_wbg(out_mask, keep_bg=True)    #soft aggregation操作
            self.prob[0,ti] = out_mask[0]
            # output mapping to the full object id space
            for i, oi in enumerate(self.enabled_obj):
                self.prob[oi,ti] = out_mask[i+1]            #N,1,H,W

            if ti != end:
                # memorize this frame
                prev_key, prev_value, score = self.prop_net.memorize(self.images[:,ti].cuda(), out_mask[1:])   #得到刚刚处理完的这一帧的key和value

                score = score.cpu().numpy()
                score = list(filter(lambda a: a != 0, score))
                scores = np.sum(score)
                scores = scores / len(score)


                if ti == idx+1:
                    max_score = scores
                scores = scores / max_score


                if (abs(ti-last_ti) >= self.mem_freq) and (scores > 0.75):
                    index += 1
                    if len(memory) > self.memory_thr:
                        sr = {}
                        for m in memory:
                            score_c = np.exp(-abs(index-m))
                            sr[m] = score_c + sa[m]
                            result_min = min(sr, key=lambda x: sr[x])
                            pos = memory.index(result_min)
                            memory.remove(result_min)
                            del sa[result_min]
                            memory.append(index)
                            sa[index] = score
                            for i, oi in enumerate(self.enabled_obj):
                                self.keys[oi] = torch.cat([self.keys[oi][:, :, :pos], self.keys[oi][:, :, pos+1:]], dim=2)
                                self.keys[oi] = torch.cat([self.keys[oi], prev_key[i:i+1]], 2)
                                self.values[oi] = torch.cat([self.values[oi][:, :, :pos], self.values[oi][:, :, pos+1:]], dim=2)
                                self.values[oi] = torch.cat([self.values[oi], prev_value[i:i+1]], 2)
                    else:
                        memory.append(index)
                        sa[index] = score
                        for i, oi in enumerate(self.enabled_obj):
                            self.keys[oi] = torch.cat([self.keys[oi], prev_key[i:i+1]], 2)
                            self.values[oi] = torch.cat([self.values[oi], prev_value[i:i+1]], 2)


                    last_ti = ti
                    prev_in_mem = True
                else:
                    prev_in_mem = False


        return closest_ti

    def interact(self, mask, frame_idx, end_idx, obj_idx):
        """
        mask - Input one-hot encoded mask WITHOUT the background class
        frame_idx, end_idx - Start and end idx of propagation
        obj_idx - list of object IDs that first appear on this frame
        """

        # In youtube mode, we interact with a subset of object id at a time
        mask, _ = pad_divide_by(mask.cuda(), 16)

        # update objects that have been labeled
        self.enabled_obj.extend(obj_idx)

        # Set other prob of mask regions to zero
        mask_regions = (mask[1:].sum(0) > 0.5)
        self.prob[:, frame_idx, mask_regions] = 0
        self.prob[obj_idx, frame_idx] = mask[obj_idx]

        self.prob[:, frame_idx] = aggregate_wbg(self.prob[1:, frame_idx], keep_bg=True)

        # KV pair for the interacting frame
        key_k, key_v, _ = self.prop_net.memorize(self.images[:,frame_idx].cuda(), self.prob[self.enabled_obj,frame_idx].cuda())

        # Propagate
        self.do_pass(key_k, key_v, frame_idx, end_idx)
