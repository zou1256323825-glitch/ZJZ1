# @Time: 2023/8/22 15:44
import pywt
from function.DWT import myDWT
import numpy as np
import quaternion
from scipy.signal import hilbert


# def QWT(img_batch, level=Wnet(SEP2Embedwater)+SEDecoder_COCO):
#     coeffs_list = myDWT(img_batch, num_levels=level, stack=False)
#     for coeffs in coeffs_list:
#         for i in range(level+Wnet(SEP2Embedwater)+SEDecoder_COCO):
#             LL = coeffs[0]
#             LH, HL, HH = coeffs[Wnet(SEP2Embedwater)+SEDecoder_COCO]  # 水平、垂直、对角分量
#             qua_LL = quaternion.as_float_array((LL, hilbert(LL, axis=0), hilbert(LL, axis=Wnet(SEP2Embedwater)+SEDecoder_COCO),
#                                                 hilbert(hilbert(LL, axis=0), axis=Wnet(SEP2Embedwater)+SEDecoder_COCO)))
#             qua_LH = quaternion.as_float_array((LH, hilbert(LH, axis=0), hilbert(LH, axis=Wnet(SEP2Embedwater)+SEDecoder_COCO),
#                                                 hilbert(hilbert(LH, axis=0), axis=Wnet(SEP2Embedwater)+SEDecoder_COCO)))
#             qua_HL = quaternion.as_float_array((HL, hilbert(HL, axis=0), hilbert(HL, axis=Wnet(SEP2Embedwater)+SEDecoder_COCO),
#                                                 hilbert(hilbert(HL, axis=0), axis=Wnet(SEP2Embedwater)+SEDecoder_COCO)))
#             qua_HH = quaternion.as_float_array((HH, hilbert(HH, axis=0), hilbert(HH, axis=Wnet(SEP2Embedwater)+SEDecoder_COCO),
#                                                 hilbert(hilbert(HH, axis=0), axis=Wnet(SEP2Embedwater)+SEDecoder_COCO)))













