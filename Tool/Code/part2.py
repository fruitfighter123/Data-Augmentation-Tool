from __future__ import division
import cv2
import matplotlib.pyplot as plt
import numpy as np
import copy
import random


def Image_fution(im, obj, angle, vection, iou, gap=[0, 0]):
    # im背景，obj是前景；angle是obj的旋转角度；vection是移动方向；iou是重叠率；gap是最后间隔
    height, width, channels = im.shape
    h_p = (9000 - height) // 2
    w_p = (9000 - width) // 2
    im = cv2.copyMakeBorder(im, h_p, h_p, w_p, w_p, cv2.BORDER_CONSTANT, value=[255, 255, 255])
    gray1 = cv2.cvtColor(obj, cv2.COLOR_BGR2GRAY)
    maks1, mask1 = cv2.threshold(255 - gray1, 10, 255, cv2.THRESH_BINARY)
    my_im_r, my_im_c = np.where(mask1 > 1)
    r_max = max(my_im_r)
    r_min = min(my_im_r)
    c_max = max(my_im_c)
    c_min = min(my_im_c)
    obj = obj[r_min:r_max, c_min:c_max, :]
    plt.close('all')
    height, width, channels = obj.shape
    degree = angle
    # 旋转后的尺寸
    heightNew = int(width * np.fabs(np.sin(np.radians(degree))) + height * np.fabs(np.cos(np.radians(degree))))
    widthNew = int(height * np.fabs(np.sin(np.radians(degree))) + width * np.fabs(np.cos(np.radians(degree))))

    matRotation = cv2.getRotationMatrix2D((width / 2, height / 2), degree, 1)

    matRotation[0, 2] += (widthNew - width) / 2  # 重点在这步
    matRotation[1, 2] += (heightNew - height) / 2  # 重点在这步

    obj_imgRotation = cv2.warpAffine(obj, matRotation, (widthNew, heightNew), borderValue=(255, 255, 255))
    gray1 = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    maks1, mask1 = cv2.threshold(255 - gray1, 10, 255, cv2.THRESH_BINARY)
    my_im_r, my_im_c = np.where(mask1 > 1)

    gray2 = cv2.cvtColor(obj_imgRotation, cv2.COLOR_BGR2GRAY)
    maks2, mask2 = cv2.threshold(255 - gray2, 10, 255, cv2.THRESH_BINARY)
    my_obj_r, my_obj_c = np.where(mask2 > 1)

    im_r_mean = int(np.mean(my_im_r))
    im_c_mean = int(np.mean(my_im_c))
    obj_r_mean = int(np.mean(my_obj_r))
    obj_c_mean = int(np.mean(my_obj_c))

    width1, height1, channels1 = im.shape
    width2, height2, channels21 = obj_imgRotation.shape
    my_iou = iou
    im_i = 0
    im_j = 0
    im2 = copy.deepcopy(im)
    im3 = copy.deepcopy(im2)
    im3[my_obj_r - obj_r_mean + im_r_mean + im_i, my_obj_c - obj_c_mean + im_c_mean + im_j, :] = obj_imgRotation[
                                                                                                 my_obj_r, my_obj_c, :]
    im4 = copy.deepcopy(im3)
    im4[:, :, :] = 255;
    im4[my_obj_r - obj_r_mean + im_r_mean + im_i, my_obj_c - obj_c_mean + im_c_mean + im_j, :] = obj_imgRotation[
                                                                                                 my_obj_r, my_obj_c, :]

    gray3 = cv2.cvtColor(im3, cv2.COLOR_BGR2GRAY)
    maks3, mask3 = cv2.threshold(255 - gray3, 10, 255, cv2.THRESH_BINARY)

    gray4 = cv2.cvtColor(im4, cv2.COLOR_BGR2GRAY)
    maks4, mask4 = cv2.threshold(255 - gray4, 10, 255, cv2.THRESH_BINARY)

    im_obj_iou = np.sum(((mask1 > 1) & (mask4 > 1))) / np.sum(mask3 > 1)

    while im_obj_iou > my_iou:
        im3 = copy.deepcopy(im2)
        im3[my_obj_r - obj_r_mean + im_r_mean + im_i, my_obj_c - obj_c_mean + im_c_mean + im_j, :] = obj_imgRotation[
                                                                                                     my_obj_r, my_obj_c,
                                                                                                     :]

        im4 = copy.deepcopy(im3)
        im4[:, :, :] = 255;
        im4[my_obj_r - obj_r_mean + im_r_mean + im_i, my_obj_c - obj_c_mean + im_c_mean + im_j, :] = obj_imgRotation[
                                                                                                     my_obj_r, my_obj_c,
                                                                                                     :]

        gray3 = cv2.cvtColor(im3, cv2.COLOR_BGR2GRAY)
        maks3, mask3 = cv2.threshold(255 - gray3, 10, 255, cv2.THRESH_BINARY)

        gray4 = cv2.cvtColor(im4, cv2.COLOR_BGR2GRAY)
        maks4, mask4 = cv2.threshold(255 - gray4, 10, 255, cv2.THRESH_BINARY)

        im_obj_iou = np.sum(((mask1 > 1) & (mask4 > 1))) / np.sum(mask3 > 1)
        # print('im_i=%d im_j=%d im_obj_iou=%f\n' % (im_i, im_j, im_obj_iou))
        im_i = im_i + vection[0]
        im_j = im_j + vection[1]
    im3 = copy.deepcopy(im2)
    im3[my_obj_r - obj_r_mean + im_r_mean + im_i - vection[0] + gap[0],
    my_obj_c - obj_c_mean + im_c_mean + im_j - vection[1] + gap[1], :] = obj_imgRotation[my_obj_r, my_obj_c, :]

    return im3


def Image_synthesis(im1_name, im2_name, obj1_name, obj2_name, im1_iou, im2_iou):
    # 读取四个图像，其中im1和im2为背景，obj1和obj2是前景；im1_obj1_iou是im1与obj1融合时的重叠率；im2_obj2_iou是im2与obj2融合时的重叠率
    im1 = cv2.imread(im1_name)
    im2 = cv2.imread(im2_name)
    obj1 = cv2.imread(obj1_name)
    obj2 = cv2.imread(obj2_name)
    print('进行中......')
    im3 = Image_fution(im1, obj1, random.randint(-90, 90), [random.randint(-3, 3) * 10, random.randint(-3, 3) * 10],
                       im1_iou)
    print('进行中......')
    im4 = Image_fution(im2, obj2, random.randint(-90, 90), [random.randint(-3, 3) * 10, random.randint(-3, 3) * 10],
                       im2_iou)
    print('进行中......')
    im5 = Image_fution(im3, im4, random.randint(-90, 90),
                       [random.randint(-3, 3) * 10, random.randint(-3, 3) * 10], 0.0,
                       gap=[50 * random.randint(2, 4), 50 * random.randint(4, 4)])
    print('已完成！！')
    return im5


if __name__ == '__main__':
    im1_name = input("输入第一个图像的名称:")
    im2_name = input("输入第一个图像的名称:")
    obj1_name = input("输入第一个图像的名称:")
    obj2_name = input("输入第一个图像的名称:")
    im1_obj1_iou = float(input("前两个图像的IoU:"))
    im2_obj2_iou = float(input("后两个图像的IoU:"))
    im3 = Image_synthesis(im1_name, im2_name, obj1_name, obj2_name, im1_obj1_iou, im2_obj2_iou)
    cv2.imwrite('SynthesisImage.jpg', im3)
    im1 = cv2.imread(im1_name)
    im2 = cv2.imread(im2_name)
    obj1 = cv2.imread(obj1_name)
    obj2 = cv2.imread(obj2_name)
    plt.figure(1)
    plt.subplot(2, 4, 1)
    plt.imshow(im1)
    plt.subplot(2, 4, 2)
    plt.imshow(obj1)
    plt.subplot(2, 4, 3)
    plt.imshow(im2)
    plt.subplot(2, 4, 4)
    plt.imshow(obj2)
    plt.subplot(2, 4, 5)
    plt.imshow(im3)
    plt.show()