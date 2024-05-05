import json
import os
import pdb

with open('/share_ssd/dongbingcheng/lvis_v1_val/configs/val/val/inference_lvis_v1_val/lvis_instances_results.json','rb') as f:
    data=json.load(f)

dataset_path='/share_ssd/dongbingcheng/LVIS/lvis_v1_val.json'

with open(dataset_path,'rb') as f:
    data_origin=json.load(f)


dir='/share_ssd/dongbingcheng/LVIS'

def rebuild_image_query(js):
    dic_bbox = {}
    images=js['images']
    annos=js['annotations']
    cates=js['categories']
    dic_bbox['cates']=cates
    for img in images:
        img_id=img['id']
        url=img['coco_url'].split('http://images.cocodataset.org/')[-1]
        dic_bbox[img_id]={}
        dic_bbox[img_id]['path']=url
        dic_bbox[img_id]['annos']=[]
        dic_bbox[img_id]['cates']=[]
        for anno in annos:
            if anno['image_id']==img_id:
                dic_bbox[img_id]['annos'].append(anno['bbox'])
                dic_bbox[img_id]['cates'].append(anno['category_id'])
    return dic_bbox
# pdb.set_trace()
dic_bbox=rebuild_image_query(data_origin)

with open('saved_anno.json','w') as f:
    json.dump(dic_bbox,f)

a=0