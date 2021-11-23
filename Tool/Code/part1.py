from __future__ import division
import glob
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
def PreciseMaskExtraction():
    Images = []
    path = glob.glob('photo/*.jpg')
    for x in path:
        img = Image.open(x)
        Images.append(np.float32(img))
    average = np.mean(Images, axis=0)
    std = np.std(Images, axis=0)
    threshold = 60
    average[np.any(std > threshold, axis=2)] = [255, 255, 255]
    mask = average
    res = Image.fromarray(mask.astype(np.uint8))
    res.save("Mask/mask.jpg")
if __name__ == '__main__':
    PreciseMaskExtraction()
    print("操作完成！")
