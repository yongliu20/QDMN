"""
This file can handle DAVIS 2016/2017 evaluation.
"""

import torch
import numpy as np
import cv2
import torch.nn.functional as F
from model.eval_network import PropagationNetwork
from model.aggregate import aggregate_wbg
from util.tensor_util import pad_divide_by


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
        self.memory_thr = 20

    def get_query_kv_buffered(self, idx):
        # not actually buffered
        f16, f8, f4 = self.prop_net.get_query_values(self.images[:,idx].cuda())
        _, _, h, w = f16.size()
        pre_mask, _ = torch.max(self.prob[1:, idx-1], dim=0)
        pre_mask = pre_mask.unsqueeze(0)
        pre_mask = F.interpolate(pre_mask, size=[h, w], mode='bilinear')
        concat_f16 = torch.cat([f16, pre_mask], dim=1)
        concat_f16 = self.prop_net.concat_conv(concat_f16)
        concat_f16 = torch.sigmoid(concat_f16)
        concat_f16 = f16 * concat_f16
        k16, v16 = self.prop_net.kv_q_f16(concat_f16)
        result = (concat_f16, f8, f4, k16, v16)
        return result

    def do_pass(self, key_k, key_v, idx, end_idx, scores_fir):
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

        keys = key_k
        values = key_v

        prev_in_mem = True
        prev_key = prev_value = None
        last_ti = idx
        index = 0

        # Note that we never reach closest_ti, just the frame before it
        this_range = range(idx+1, closest_ti)
        end = closest_ti - 1
        
        scores_fir = list(filter(lambda a: a != 0, scores_fir))
        score_fir = np.sum(scores_fir)
        score_fir = score_fir / len(scores_fir)

        for ti in this_range:
            if prev_in_mem:
                # if the previous frame has already been added to the memory bank
                this_k = keys
                this_v = values
            else:
                # append it to a temporary memory bank otherwise
                this_k = torch.cat([keys, prev_key], 2)
                this_v = torch.cat([values, prev_value], 2)
            query = self.get_query_kv_buffered(ti)
            out_mask = self.prop_net.segment_with_query(this_k, this_v, *query)
            out_mask = aggregate_wbg(out_mask, keep_bg=True)
            self.prob[:,ti] = out_mask

            if ti != end:
                # Memorize this frame
                prev_key, prev_value, scores = self.prop_net.memorize(self.images[:,ti].cuda(), out_mask[1:])
                scores = scores.cpu().numpy()
                scores = list(filter(lambda a: a != 0, scores))
                score = np.sum(scores)
                score = score / len(scores)
                score = score / score_fir   
                if (abs(ti-last_ti) >= self.mem_freq) and (score > 0.9):
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
                        keys = torch.cat([keys[:, :, :pos], keys[:, :, pos+1:]], dim=2)
                        keys = torch.cat([keys, prev_key], dim=2)
                        values = torch.cat([values[:, :, :pos], values[:, :, pos+1:]], dim=2)
                        values = torch.cat([values, prev_value], 2)
                    else:
                        memory.append(index)
                        sa[index] = score
                        keys = torch.cat([keys, prev_key], 2)
                        values = torch.cat([values, prev_value], 2)
                    last_ti = ti
                    prev_in_mem = True
                else:
                    prev_in_mem = False

        return closest_ti

    def interact(self, mask, frame_idx, end_idx):
        """
        mask - Input one-hot encoded mask WITHOUT the background class
        frame_idx, end_idx - Start and end idx of propagation
        """
        mask, _ = pad_divide_by(mask.cuda(), 16)

        self.prob[:, frame_idx] = aggregate_wbg(mask, keep_bg=True)

        # KV pair for the interacting frame
        key_k, key_v, scores_fir = self.prop_net.memorize(self.images[:,frame_idx].cuda(), self.prob[1:,frame_idx].cuda())

        # Propagate
        self.do_pass(key_k, key_v, frame_idx, end_idx, scores_fir)
