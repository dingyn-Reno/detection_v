import json
import os
import pdb

with open('/share_ssd/dongbingcheng/lvis_v1_val/configs/val/val/inference_lvis_v1_val/lvis_instances_results.json','rb') as f:
    data=json.load(f)


dir='/share_ssd/dongbingcheng/LVIS'

with open('saved_anno.json','rb') as f:
    data_origin=json.load(f)

def compute_iou(rec1, rec2):
    """
    computing IoU
    :param rec1: (y0, x0, y1, x1), which reflects
            (top, left, bottom, right)
    :param rec2: (y0, x0, y1, x1)
    :return: scala value of IoU
    """
    # computing area of each rectangles
    S_rec1 = (rec1[2] - rec1[0]) * (rec1[3] - rec1[1])
    S_rec2 = (rec2[2] - rec2[0]) * (rec2[3] - rec2[1])

    # computing the sum_area
    sum_area = S_rec1 + S_rec2

    # find the each edge of intersect rectangle
    left_line = max(rec1[1], rec2[1])
    right_line = min(rec1[3], rec2[3])
    top_line = max(rec1[0], rec2[0])
    bottom_line = min(rec1[2], rec2[2])

    # judge if there is an intersect
    if left_line >= right_line or top_line >= bottom_line:
        return 0
    else:
        intersect = (right_line - left_line) * (bottom_line - top_line)
        return (intersect / (sum_area - intersect)) * 1.0

iou_c=0.5
scores_c=0.5
gts=0
gts_per_class=[0 for i in range(0,len(data_origin['cates'])+1)]
preds=0
preds_per_class=[0 for i in range(0,len(data_origin['cates'])+1)]
acc_count=0
acc_count_per_class=[0 for i in range(0,len(data_origin['cates'])+1)]
# pdb.set_trace()
for i in range(0,len(data)):

    image_id=data[i]['image_id']
    category_id=data[i]['category_id']
    bbox=data[i]['bbox']
    bbox[2]=bbox[0]+bbox[2]
    bbox[3]=bbox[1]+bbox[3]
    score=data[i]['score']
    if score < scores_c:
        continue
    preds+=1
    preds_per_class[category_id]+=1
    image_id=str(image_id)
    # pdb.set_trace()
    for j in range(0,len(data_origin[image_id]['cates'])):
        if category_id==data_origin[image_id]['cates'][j]:
            bbox_origin=data_origin[image_id]['annos'][j]
            bbox_origin[2]=bbox_origin[0]+bbox_origin[2]
            bbox_origin[3]=bbox_origin[1]+bbox_origin[3]
            if compute_iou(bbox,bbox_origin)>=iou_c:
                acc_count+=1
                acc_count_per_class[category_id]+=1
                break

for image_id in data_origin.keys():
    if image_id!='cates':
        gts=gts+len(data_origin[image_id]['cates'])
        for cate in data_origin[image_id]['cates']:
            gts_per_class[cate]+=1
print('gts:',gts)
print('preds:',preds)
print('acc_count:',acc_count)
acc=acc_count/preds
print('acc:',acc)
rec=acc_count/gts
print('rec:',rec)














a=0