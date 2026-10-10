# @Time: 2023/8/12 15:15
import torch


def arnold(img_batch, times):
    """
    img: shape is (h, w)
    """
    b, c, h, w = img_batch.shape
    p = img_batch.clone()
    a = 1
    b = 1
    for batch in range(b):
        for channel in range(c):
            for _ in range(times):
                for i in range(h):
                    for j in range(w):
                        x = (i + b * j) % h
                        y = (a * i + (a * b + 1) * j) % w
                        p[batch, channel, x, y] = img_batch[batch, channel, i, j]
    return p


def dearnold(img, times):
    """
    img: shape is (h, w)
    """
    r, c = img.shape
    p = img.clone()
    a = 1
    b = 1
    for _ in range(times):
        for i in range(r):
            for j in range(c):
                x = ((a * b + 1) * i - b * j) % r
                y = (-a * i + j) % c
                p[x, y] = img[i, j]
    return p



















