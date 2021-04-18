#encoding:utf-8
import pandas as pd
import numpy as np
from tqdm import tqdm
from PIL import Image
#
# #
# def getMatrixFromHex(filename):
#     hexarr = []
#     savename = datapath + filename
#     filename = basepath + filename + '.bytes'
#     with open(filename, 'rb') as f:
#         for line in f:
#             hexarr.extend(int(el, 16) for el in line.split()[1:] if el != b"??")
#     fh = np.array(hexarr)
#     fh = fh.astype('uint8')
#     txt = fh.reshape(-1,1)
#     np.savetxt(savename, txt,fmt='%d')
#     return fh

def getMatrixFromHex(filename, tro_class):
    hexarr = []
    savename = pngdatapath + str(tro_class) + '/' + filename + '.png'
    filename = basepath + filename + '.bytes'
    with open(filename, 'rb') as f:
        for line in f:
            hexarr.extend(int(el, 16) for el in line.split()[1:] if el != b"??")
    txt = np.resize(hexarr,(256,256))
    txt = np.uint8(txt)
    im = Image.fromarray(txt)
    im = im.convert("L")
    im.save(savename)

basepath = '/media/yzq/新加卷/subtrain/'
jpgdatapath = './data/bytes_jpg/'
pngdatapath = './data/bytes_png/'
subtrianLabels = pd.read_csv('./subtrainLabels.csv')
maxImgLen = 0
minImgLen = float('inf')
for i in tqdm(range(subtrianLabels.shape[0])):
    getMatrixFromHex(subtrianLabels['Id'][i], subtrianLabels['Class'][i])
#     img = getMatrixFromHex(sid)
#     imglen = len(img)
#     maxImgLen = max(maxImgLen, imglen)
#     minImgLen = min(minImgLen, imglen)
#
# print("max len:", maxImgLen) #3903136
# print("min len:", minImgLen) #6476