#encoding:utf-8
import re
import pandas as pd
import binascii
import numpy as np
from tqdm import tqdm
from PIL import Image

def getMatrixFromAsm(filename, tro_class, startindex = 0, pix_num = 5000):
    savename = pngdatapath + str(tro_class) + '/' + filename + '.png'
    filename = basepath + filename + '.asm'
    with open(filename, 'rb') as f:
        f.seek(startindex,0)
        content = f.read(pix_num)
    hexst = binascii.hexlify(content)
    fh = np.array([int(hexst[i:i+2],16) for i in range(0, len(hexst), 2)])
    cxt = np.resize(fh, (32,32))
    cxt = np.uint8(cxt)
    im = Image.fromarray(cxt)
    im = im.convert("L")
    im.save(savename)

basepath = '/media/yzq/新加卷/subtrain/'
jpgdatapath = './data/asm_jpg/'
pngdatapath = './data/asm_png/'
subtrianLabels = pd.read_csv('./subtrainLabels.csv')
maxImgLen = 0
minImgLen = float('inf')
for i in tqdm(range(subtrianLabels.shape[0])):
    getMatrixFromAsm(subtrianLabels['Id'][i], subtrianLabels['Class'][i], pix_num=1024)