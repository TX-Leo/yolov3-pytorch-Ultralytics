
import os
import random

'''create path'''
if not os.path.exists('ImageSets/Main'):  # 改成自己建立的myData
    os.makedirs('ImageSets/Main')

'''create and open file'''
ftrainval = open('ImageSets/Main/trainval.txt', 'w')
ftrain = open('ImageSets/Main/train.txt', 'w')
fval = open('ImageSets/Main/val.txt', 'w')

'''train/trainval'''
xmlfilepath = 'Annotations/train'
total_xml = os.listdir(xmlfilepath)
num = len(total_xml)
list = range(num)
for i in list:
    name = total_xml[i][:-4] + '\n'
    ftrain.write(name)
    ftrainval.write(name)

'''val/trainval'''
xmlfilepath = 'Annotations/val'
total_xml = os.listdir(xmlfilepath)
num = len(total_xml)
list = range(num)
for i in list:
    name = total_xml[i][:-4] + '\n'
    fval.write(name)
    ftrainval.write(name)

'''close file'''
ftrainval.close()
ftrain.close()
fval.close()

