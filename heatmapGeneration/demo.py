import os 
import pickle
import numpy as np
import copy
import json
import argparse
from   pycocotools.coco     import COCO 
from   pycocotools.cocoeval import COCOeval
from   PIL import Image

'''
    coarse heatmap generation on HICO-DET dataset
    e.g.:
        0 0 0 0 0 0 0 0 0 0 
        0 0 0 1 1 1 1 1 0 0
        0 0 0 1 1 1 1 1 0 0
        0 0 0 1 1 1 1 1 0 0
        0 0 0 1 1 1 1 1 0 0
        0 0 0 1 1 1 1 1 0 0
        0 0 0 0 0 0 0 0 0 0 
'''

def getUnionBox(a, b):
    x = min(a[0], b[0])
    y = min(a[1], b[1])
    w = max(a[0]+a[2], b[0]+b[2]) - x
    h = max(a[1]+a[3], b[1]+b[3]) - y
    return (x, y, w, h)

if __name__ == "__main__":
    # loading data
    datasetPart      = 'test' # train or test
    hicoGroundTruth  = "/data01/zzk/data/hico_20160224_det/hico_annotations_{}2015.json".format(datasetPart)
    DATA_DIR         = "/data01/zzk/data/hico_20160224_det/images/{}2015/".format(datasetPart)
    cocoGt           = COCO(hicoGroundTruth)
    GTBoxes          = dict()

    '''
    key:    2494 
    anno:   {'image_id': 902, 'category_id': 4, 
             'bbox': [262.0, 153.0, 333.0, 207.0], 'iscrowd': 0, 'id': 2494}
    '''
    for key, anno in cocoGt.anns.items():
        image_id = anno['image_id']
        if image_id not in GTBoxes.keys():
            tmp_dict = dict()
            tmp_dict['humans']  = []
            tmp_dict['objects'] = []
            GTBoxes[image_id]   = tmp_dict

        if anno['category_id'] == 1:
            GTBoxes[image_id]['humans'].append(anno['bbox'])
        else:
            GTBoxes[image_id]['objects'].append(anno['bbox'])

    print(len(GTBoxes.keys()))
    print("==> finish preparing data")

    cnt = 1
    for image_id, bboxes in GTBoxes.items():
        imgPath = DATA_DIR + 'HICO_{}2015_'.format(datasetPart) + (str(image_id)).zfill(8) + '.jpg'
        oriImg  = Image.open(imgPath)
        oriImg.save('./heatmapGeneration/outputs/demo_{}_ori.jpg'.format(cnt))
        # print(oriImg.size)

        npImg = np.zeros((oriImg.size[1], oriImg.size[0], 3))
        for H in bboxes['humans']:
            for O in bboxes['objects']:
                [x,y,w,h] = getUnionBox(H,O)
                x1,y1,x2,y2 = int(x),int(y),int(x+w),int(y+h)
                npImg[y1:y2, x1:x2] = [255, 255, 255]

        im = Image.fromarray(np.uint8(npImg))
        im.save('./heatmapGeneration/outputs/demo_{}_heatmap.jpg'.format(cnt))
        

        # exit()
        cnt += 1
        if cnt > 10:
            exit()