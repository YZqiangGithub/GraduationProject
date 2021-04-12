#encoding:utf-8
import re
import pandas as pd
import numpy
from collections import *
import binascii

def getMatrixFromBin(filename, width = 512, one_row = False):
    with open(filename, 'rb') as f:
        content = f.read()
    hexst = binascii.hexlify(content)
    fh = numpy.array([int(hexst[i:i+2],16) for i in range(0, len(hexst), 2)])
    if one_row is False:
        rn = len(fh)/width
        fh = numpy.reshape(fh[:rn*width], (-1,width))
    fh = numpy.uint8(fh)
    return fh

def getMatrixFromAsm(filename, startindex = 0, pix_num = 5000):
    with open(filename, 'rb') as f:
        f.seek(startindex,0)
        content = f.read(pix_num)
    hexst = binascii.hexlify(content)
    fh = numpy.array([int(hexst[i:i+2],16) for i in range(0, len(hexst), 2)])
    fh = numpy.uint8(fh)
    return fh

def getMatrixFromHex(filename, width):
    hexar = []
    with open(filename, 'rb') as f:
        for line in f:
            hexar.extend(int(el, 16) for el in line.split()[1:] if el != "??")
    rn = len(hexar) / width
    fh = numpy.reshape(hexar[:rn*width], (-1, width))
    fh = numpy.uint8(fh)
    return fh

def read_hexbytes(filename):
    hexar = []
    with open(filename, 'rb') as f:
        for line in f:
            hexar.extend(int(el,16) for el in line.split()[1:] if el != "??")
    rn = len(hexar) / 256
    fh = numpy.reshape(hexar[:rn * 256], (-1,256))
    fh = numpy.uint8(fh)
    return fh

basepath = '/media/yzq/新加卷/subtrain/'
mapimg = defaultdict(list)
subtrian = pd.read_csv('/media/yzq/新加卷/subtrainLabels.csv')
count = 1

for sid in subtrian.Id:
    print("dealing with file {0}...".format(str(count)))
    count += 1
    file_path = basepath + sid + '.asm'
    img = getMatrixFromAsm(file_path,startindex=0, pix_num=1600)
    mapimg[sid] = img

dataFrameList = []
for fid, imf in mapimg.items():
    standard = {}
    standard['Id'] = fid
    for k, v in enumerate(imf):
        col_name = "pix{0}".format(str(k))
        standard[col_name] = v
    dataFrameList.append(standard)

df = pd.DataFrame(dataFrameList)
df.to_csv('./data/train/asm_imgfeature.csv', index = False)
