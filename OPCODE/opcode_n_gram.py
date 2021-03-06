#encoding:utf-8
import re
from collections import *
import os
import pandas as pd

def getOpcodeSequence(filename):
    opcode_seq = []
    p = re.compile(r'\s([a-fA-F0-9]{2}\s)+\s*([a-z]+)')
    with open(filename,'rb') as f:
        for biline in f:
            line = biline.decode('utf-8','ignore')
            if line.startswith(".text"):
                m = re.findall(p,line)
                if m:
                    opc = m[0][1]
                    if opc != "align":
                        opcode_seq.append(opc)
    return opcode_seq

def train_opcode_lm(ops, order=4):
    lm = defaultdict(Counter)
    prefix = ["~"] * order
    prefix.extend(ops)
    data = prefix
    for i in xrange(len(data)-order):
        history, char = tuple(data[i:i+order]), data[i+order]
        lm[history][char]+=1
    def normalize(counter):
        s = float(sum(counter.values()))
        return [(c,cnt/s) for c,cnt in counter.iteritems()]
    outlm = {hist:chars for hist, chars in lm.iteritems()}
    return outlm

def getOpcodeNgram(ops, n=3):
    opngramlist = [tuple(ops[i:i+n]) for i in range(len(ops)-n)]
    opngram = Counter(opngramlist)
    return opngram

basepath = "/media/yzq/新加卷/subtrain/"
map3gram = defaultdict(Counter)
subtrain = pd.read_csv('./subtrainLabels.csv')
count = 1
for sid in subtrain.Id:
    print( "counting the 3-gram of the {0} file...".format(str(count)))
    count += 1
    filename = basepath + sid + ".asm"
    ops = getOpcodeSequence(filename)
    op3gram = getOpcodeNgram(ops)
    map3gram[sid] = op3gram

cc = Counter([])
for d in map3gram.values():
    cc += d
selectedfeatures = {}
# tc = 0
# for k,v in cc.items():
#     if v >= 500:
#         selectedfeatures[k] = v
#         print(k,v)
#         tc += 10

cc = sorted()

dataframelist = []
for fid,op3gram in map3gram.items():
    standard = {}
    standard["Id"] = fid
    for feature in selectedfeatures:
        if feature in op3gram:
            standard[feature] = op3gram[feature]
        else:
            standard[feature] = 0
    dataframelist.append(standard)
df = pd.DataFrame(dataframelist)
df.to_csv("3gramfeature.csv",index=False)
