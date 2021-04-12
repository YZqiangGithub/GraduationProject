#encoding:utf-8
import pandas as pd
import numpy as np
from collections import *
from PIL import Image

def getMatrixFromHex(filename):
    hexarr = []
    savename = datapath + filename
    filename = basepath + filename + '.bytes'
    with open(filename, 'rb') as f:
        for line in f:
            hexarr.extend(int(el, 16) for el in line.split()[1:] if el != b"??")
    fh = np.array(hexarr)
    fh = fh.astype('uint8')
    txt = fh.reshape(-1,1)
    np.savetxt(savename, txt,fmt='%d')
    return fh

basepath = '/media/yzq/新加卷/subtrain/'
datapath = './data/trainData/'
subtrianLabels = pd.read_csv('/media/yzq/新加卷/subtrainLabels.csv')
count = 1
maxImgLen = 0
minImgLen = float('inf')
for sid in subtrianLabels.Id:
    print("dealing with file {0}...".format(str(count)))
    count += 1
    img = getMatrixFromHex(sid)
    imglen = len(img)
    maxImgLen = max(maxImgLen, imglen)
    minImgLen = min(minImgLen, imglen)

print("max len:", maxImgLen) #3903136
print("min len:", minImgLen) #6476