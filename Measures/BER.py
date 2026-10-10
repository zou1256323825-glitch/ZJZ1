import torch
import numpy as np

def BER(image1, image2):
    round_image1 = np.round(image1.cpu(), 2)
    round_image1_down = round_image1 - 0.2
    round_image1_up = round_image1 + 0.2
    round_image2 = np.round(image2.cpu(), 2)
    true_pixel = torch.sum(np.logical_and(round_image1_down < round_image2, round_image2 < round_image1_up))

    total_pixel = image1.numel()
    ber = 1 - true_pixel / total_pixel
    return ber


if __name__ == '__main__':
    tensor1 = torch.rand(16, 1, 32, 32).cuda()
    tensor2 = torch.rand(16, 1, 32, 32).cuda()
    print(BER(tensor1, tensor2))








