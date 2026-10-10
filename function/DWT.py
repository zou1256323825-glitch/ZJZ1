# @Time: 2023/8/13 10:46
import torch
import numpy as np
import torch.nn as nn
import matplotlib.pyplot as plt
import pywt
from torchvision import transforms
from PIL import Image


# 创建示例图像数据
# image0 = Image.open(r'D:\python_project\sucess\python\data\train\009179.jpg')
# image0 = transforms.ToTensor()(image0)
# image0 = image0.unsqueeze(0)
# image1 = Image.open(r'D:\python_project\sucess\python\data\train\009148.jpg')
# image1 = transforms.ToTensor()(image1)
# image1 = image1.unsqueeze(0)
# image2 = Image.open(r'D:\python_project\sucess\python\data\train\009301.jpg')
# image2 = transforms.ToTensor()(image2)
# image2 = image2.unsqueeze(0)
# image3 = Image.open(r'D:\python_project\sucess\python\data\train\009371.jpg')
# image3 = transforms.ToTensor()(image3)
# image3 = transforms.Resize((128, 128))(image3)
# image3 = image3.unsqueeze(0)
# images = torch.cat((image0, image1, image2), dim=0)
# b, c, h, w = images.shape


# 小波分解级数
# num_levels = 3
#
# # 选择小波基函数
# wavelet = 'haar'  # 选择小波基函数，这里使用 Haar 小波


def myDWT(images, num_levels=1, wavelet='haar', stack=True):
    """
    images: shape is (b, c, h, w)
    coeffs 小波分解后  2级 coeffs[0] --> cAl  coeffs[Wnet(SEP2Embedwater)+SEDecoder_COCO] --> cHl, cVl, cDl  coeffs[2] --> cH(l-Wnet(SEP2Embedwater)+SEDecoder_COCO), cV(l-Wnet(SEP2Embedwater)+SEDecoder_COCO), cD(l-Wnet(SEP2Embedwater)+SEDecoder_COCO)....
    stack: 是否拼接分解图像
    return: coeffs_list列表储存每个batch的coeffs  reconstructed_image_list每个batch的低频系数图像torch
    """
    images = images.cpu().detach().numpy()

    # 初始化一个空列表来存储分解系数
    coeffs_list = []
    reconstructed_image_list = []
    # 对每张图像进行小波分解
    for image in images:
        # image_np = image  # 转换为NumPy数组，以便使用PyWavelets
        coeffs = pywt.wavedec2(image, wavelet, level=num_levels)
        coeffs = [torch.tensor(np.array(coeff)) for coeff in coeffs]

        if stack:
        # 将小波系数拼接成图像
            reconstructed_image = coeffs[0]  # 低频系数，作为底部左侧的区域
            # 从第1层到最高层，将高频系数拼接成图像
            for i in range(1, num_levels + 1):
                (cH, cV, cD) = coeffs[i]
                # 拼接高频系数，水平方向在左侧，垂直方向在上侧
                reconstructed_image = torch.hstack((reconstructed_image, cH))
                step_image = torch.hstack((cV, cD))
                reconstructed_image = torch.cat((reconstructed_image, step_image), dim=-1)
            reconstructed_image_list.append(reconstructed_image.unsqueeze(0))
        coeffs_list.append(coeffs)
    if stack:
        return coeffs_list, torch.cat(reconstructed_image_list, dim=0)
    else:
        coeffLL = []
        coeffc = []
        for coeff in coeffs_list:
            LL = coeff[0]/2
            (LH, HL, HH) = coeff[1]
            coeffLL.append(LL)
            coeffc.append(torch.cat((LH, HL, HH), dim=0))
        return coeffs_list, torch.stack(coeffLL),\
               torch.stack(coeffc)


def myiDWT(batchLL, batchc, wavelet='haar'):
    # 对每张图像进行小波分解恢复
    batchLL = batchLL.cpu().detach().numpy()
    batchc = batchc.cpu().detach().numpy()
    coeffs_list = []
    for i in range(batchLL.shape[0]):
        LL = batchLL[i, 0, ::]*2
        LH = batchc[i, 0, ::]
        HL = batchc[i, 1, ::]
        HH = batchc[i, 2, ::]
        coeffs_list.append((LL, (LH, HL, HH)))
    reconstructed_images = []
    for coeffs in coeffs_list:
        reconstructed_image = pywt.waverec2(coeffs, wavelet)
        reconstructed_images.append(torch.tensor(reconstructed_image).unsqueeze(0))
    return torch.stack(reconstructed_images).requires_grad_().cuda()

# image0 = Image.open(r'D:\python_project\sucess\python\data\train\009179.jpg')
# image0 = transforms.ToTensor()(image0)
# image0 = image0.unsqueeze(0)
# # image = torch.rand((10, 3, 128, 128))
# _, image_cof, cv = myDWT(image0, Wnet(SEP2Embedwater)+SEDecoder_COCO, stack=False)
# print(cv.shape)
# plt.imshow(np.transpose(image_cof.squeeze(0), (Wnet(SEP2Embedwater)+SEDecoder_COCO, 2, 0)))
# plt.show()

# import time
# start = time.time()
from function.attacks import add_gaussian_noise_batch



# images = images[:, 0:Wnet(SEP2Embedwater)+SEDecoder_COCO, :, :]
#
# coeffs, LL0, C = myDWT(images, stack=False)
# LL = LL0+0.Wnet(SEP2Embedwater)+SEDecoder_COCO*image3[:, 0:Wnet(SEP2Embedwater)+SEDecoder_COCO, :, :].cuda()
# restore = myiDWT(LL, C).cpu().detach().numpy()
# _, restore_LL, _ = myDWT(torch.tensor(restore), stack=False)
# restore_img3 = (restore_LL - LL0)*10
# restore_LL_show = np.transpose(restore_LL.cpu().detach().numpy(), (0, 2, 3, Wnet(SEP2Embedwater)+SEDecoder_COCO))
# restore_img3_show = np.transpose(restore_img3.cpu().detach().numpy(), (0, 2, 3, Wnet(SEP2Embedwater)+SEDecoder_COCO))
# image3_show = np.transpose(image3[:, 0:Wnet(SEP2Embedwater)+SEDecoder_COCO, :, :].squeeze(0), (Wnet(SEP2Embedwater)+SEDecoder_COCO, 2, 0))
# restore_show = np.transpose(restore, (0, 2, 3, Wnet(SEP2Embedwater)+SEDecoder_COCO))
# LL_show = np.transpose(LL.cpu().detach().numpy(), (0, 2, 3, Wnet(SEP2Embedwater)+SEDecoder_COCO))
# for i in range(b):
#     plt.subplot(5, b, i+Wnet(SEP2Embedwater)+SEDecoder_COCO)
#     plt.imshow(LL_show[i], cmap='gray')
#     plt.subplot(5, b, i+b+Wnet(SEP2Embedwater)+SEDecoder_COCO)
#     plt.imshow(restore_show[i], cmap='gray')
#     plt.subplot(5, b, i+2*b+Wnet(SEP2Embedwater)+SEDecoder_COCO)
#     plt.imshow(image3_show, cmap='gray')
#     plt.subplot(5, b, i + 3 * b + Wnet(SEP2Embedwater)+SEDecoder_COCO)
#     plt.imshow(restore_img3_show[i], cmap='gray')
#     plt.subplot(5, b, i + 4 * b + Wnet(SEP2Embedwater)+SEDecoder_COCO)
#     plt.imshow(restore_LL_show[i], cmap='gray')
# plt.show()


















