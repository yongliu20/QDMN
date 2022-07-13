import torch
import torch.nn.functional as F

def compute_tensor_iu(seg, gt):
    intersection = (seg & gt).float().sum()
    union = (seg | gt).float().sum()

    return intersection, union

def compute_tensor_iou(seg, gt):
    intersection, union = compute_tensor_iu(seg, gt)
    iou = (intersection + 1e-6) / (union + 1e-6)
    
    return iou 

def pad_divide_by(in_img, d):
    h, w = in_img.shape[-2:]

    if h % d > 0:
        new_h = h + d - h % d
    else:
        new_h = h
    if w % d > 0:
        new_w = w + d - w % d
    else:
        new_w = w
    lh, uh = int((new_h-h) / 2), int(new_h-h) - int((new_h-h) / 2)
    lw, uw = int((new_w-w) / 2), int(new_w-w) - int((new_w-w) / 2)
    pad_array = (int(lw), int(uw), int(lh), int(uh))
    out = F.pad(in_img, pad_array)
    return out, pad_array

def unpad(img, pad):
    if pad[2]+pad[3] > 0:
        img = img[:,:,pad[2]:-pad[3],:]
    if pad[0]+pad[1] > 0:
        img = img[:,:,:,pad[0]:-pad[1]]
    return img

def maskiou(mask1, mask2):
    b, c, h, w = mask1.size()
    mask1 = mask1.view(b, -1)
    mask2 = mask2.view(b, -1)
    area1 = mask1.sum(dim=1, keepdim=True)
    area2 = mask2.sum(dim=1, keepdim=True)
    inter = ((mask1 + mask2) == 2).sum(dim=1, keepdim=True)
    union = (area1 + area2 - inter)
    for a in range(b):
        if union[a][0] == torch.tensor(0):
            union[a][0] = torch.tensor(1)
    maskiou = inter / union
    return maskiou