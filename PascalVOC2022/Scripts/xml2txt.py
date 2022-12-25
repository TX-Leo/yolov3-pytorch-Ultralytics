
import xml.etree.ElementTree as ET
import pickle
import os
from os import listdir, getcwd
from os.path import join
train_sets = [('PascalVOC2022', 'train')]
val_sets = [('PascalVOC2022', 'val')]
classes = ['RBC']  # 改成自己的类别

if not os.path.exists('labels/train'):
    os.makedirs('labels/train')
if not os.path.exists('labels/val'):
    os.makedirs('labels/val')
if not os.path.exists('cfg/'):
    os.makedirs('cfg/')

def convert(size, box):
    dw = 1. / (size[0])
    dh = 1. / (size[1])
    x = (box[0] + box[1]) / 2.0 - 1
    y = (box[2] + box[3]) / 2.0 - 1
    w = box[1] - box[0]
    h = box[3] - box[2]
    x = x * dw
    w = w * dw
    y = y * dh
    h = h * dh
    return (x, y, w, h)


def convert_annotation(year, image_set,image_id):
    # 打开Annotations/train/1.xml(或者val)
    in_file = open('Annotations/%s/%s.xml' % (image_set,image_id))
    # 创建并打开labels/train/1.txt(或者val)
    out_file = open('labels/%s/%s.txt' % (image_set,image_id), 'w')
    tree = ET.parse(in_file)
    root = tree.getroot()
    size = root.find('size')
    w = int(size.find('width').text)
    h = int(size.find('height').text)

    for obj in root.iter('object'):
        difficult = obj.find('difficult').text
        cls = obj.find('name').text
        if cls not in classes or int(difficult) == 1:
            continue
        cls_id = classes.index(cls)
        xmlbox = obj.find('bndbox')
        b = (float(xmlbox.find('xmin').text), float(xmlbox.find('xmax').text), float(xmlbox.find('ymin').text),
             float(xmlbox.find('ymax').text))
        bb = convert((w, h), b)
        out_file.write(str(cls_id) + ' ' + ' '.join([str(a) for a in bb]) + '\n')

wd = getcwd()

# 创建labels/trian以及PascalVOC2022_train.txt
# year = PascalVOC2022;image_set=train
for year, image_set in train_sets:
    #打开并读取train.txt
    image_ids = open('ImageSets/Main/%s.txt' % (image_set)).read().strip().split()
    #创建PascalVOC2022_trian.txt
    list_file = open('cfg/%s_%s.txt' % (year, image_set), 'w')
    for image_id in image_ids:
        list_file.write('PascalVOC2022/JPEGImages/%s/%s.jpg\n' % (image_set,image_id))  #写入相对路径
        #list_file.write('%s/PascalVOC2022/JPEGImages/%s/%s.jpg\n' % (wd, image_set, image_id)) #写入绝对路径
        convert_annotation(year, image_set,image_id)
    list_file.close()

# 创建labels/val以及PascalVOC2022_val.txt
# year = PascalVOC2022;image_set=val
for year, image_set in val_sets:
    # 打开并读取val.txt
    image_ids = open('ImageSets/Main/%s.txt' % (image_set)).read().strip().split()
    # 创建PascalVOC2022_val.txt
    list_file = open('cfg/%s_%s.txt' % (year, image_set), 'w')
    for image_id in image_ids:
        list_file.write('PascalVOC2022/JPEGImages/%s/%s.jpg\n' % (image_set, image_id))  # 写入相对路径
        # list_file.write('%s/PascalVOC2022/JPEGImages/%s/%s.jpg\n' % (wd, image_set, image_id)) #写入绝对路径
        convert_annotation(year, image_set, image_id)
    list_file.close()

