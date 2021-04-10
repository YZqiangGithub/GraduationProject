#encoding:utf-8
import pandas as pd
import re


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

basepath = "/media/yzq/新加卷/subtrain/"
dataframelist = []
subtrain = pd.read_csv('./subtrainLabels.csv')
count = 1
for sid in subtrain.Id:
    standard = {}
    standard['Id'] = sid
    print( "store opcode sequence of the {0} file...".format(str(count)))
    count += 1
    filename = basepath + sid + ".asm"
    op_s = getOpcodeSequence(filename)
    ops = ' '.join(op_s)
    standard['ops'] = ops
    dataframelist.append(standard)

df = pd.DataFrame(dataframelist)
df.to_csv('op_seq.csv',index=False)