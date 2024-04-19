'''

用于可视化展示数据集，修改参数：
first_json ：json路径；
pa：图片根路径;
maxsize:保存图片的上限数量，-1为保存全部图片

程序执行后显示在：'./output_img2/test_case/{}/'

'''
import os, cv2
import json
import random
first_json = ""

# first_json = "/home/vis/shenzhiyong/code/00_BM/GLIP-OWLv2/MQ-Det/DATASET/chengguan/all_train_data_gb_new_13cls.json"

maxsize=10

with open(first_json, "r") as f:
    data = json.load(f)

def exportJSON(jsonname, record_dict):
    json_data = json.dumps(record_dict)
    with open(jsonname, 'w+') as fp:
        fp.write(json_data)

analy = [6] # [1, 3, 6, 7, 16, 17, 24, 34, 37, 38, 50, 52] # 45, 2, 12

cate = data["categories"]
catgs = dict()
for cat in cate:
    catgs[cat["id"]] = cat["name"]

file_d = {}

imageso = data["images"]
print(len(imageso))
for im in imageso:
    file_d[im["id"]] = im["file_name"]

anno = data["annotations"]
record = dict()
filelists = []
pa = ""
# pa = "/home/vis/dingyuning03/baidu/personal-code/dingyuning_APE/APE/datasets/13cls_weizhang_det/"
img_list=[]
idx=0

if not os.path.exists('./output_img2/test_case/{}/'.format(analy[0])):
    os.makedirs('./output_img2/test_case/{}/'.format(analy[0]))

for ann in anno:
    if ann["category_id"] in analy:
        img=cv2.imread(pa+file_d[ann["image_id"]])
        X, Y, XZ, YZ=ann['bbox'][0],ann['bbox'][1],ann['bbox'][0]+ann['bbox'][2], ann['bbox'][1]+ann['bbox'][3]
        file_name=str(ann['image_id'])+".jpg"
        img = cv2.rectangle(img, (int(X), int(Y)), \
                    (int(XZ), int(YZ)), \
                    [255,0,0], 2)
        img_list.append(ann["image_id"])
        cv2.imwrite('./output_img2/test_case/{}/'.format(analy[0])+file_name, img)
        idx+=1
    if idx==maxsize:
        break

print(img_list)




