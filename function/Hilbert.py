# @Time: 2023/8/23 11:06
from scipy.signal import hilbert
import torch


def myhilbert(img_batch, axis=-1):
    """
    img shape is (b, Wnet(SEP2Embedwater)+SEDecoder_COCO, h, w)
    """
    img_list = []
    for img in img_batch:
        if axis == 10:
            h = img.shape[-1]
            img = torch.reshape(img, (1, -1))
            image = hilbert(img, axis=-1)
            image = image.reshape(1, h, h)
        else:
            image = hilbert(img, axis=axis)
        img_list.append(torch.tensor(image))
    return torch.stack(img_list)


# u = torch.zeros((Wnet(SEP2Embedwater)+SEDecoder_COCO, Wnet(SEP2Embedwater)+SEDecoder_COCO, 256, 256))
# s = myhilbert(u, axis=10)
# print(s.shape)










