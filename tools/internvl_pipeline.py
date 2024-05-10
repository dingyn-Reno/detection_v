from transformers import AutoTokenizer, AutoModel
import torch
import torchvision.transforms as T
from PIL import Image

from torchvision.transforms.functional import InterpolationMode

import json
import cv2
import os
import pdb
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def build_transform(input_size):
    MEAN, STD = IMAGENET_MEAN, IMAGENET_STD
    transform = T.Compose([
        T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
        T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=MEAN, std=STD)
    ])
    return transform


def find_closest_aspect_ratio(aspect_ratio, target_ratios, width, height, image_size):
    best_ratio_diff = float('inf')
    best_ratio = (1, 1)
    area = width * height
    for ratio in target_ratios:
        target_aspect_ratio = ratio[0] / ratio[1]
        ratio_diff = abs(aspect_ratio - target_aspect_ratio)
        if ratio_diff < best_ratio_diff:
            best_ratio_diff = ratio_diff
            best_ratio = ratio
        elif ratio_diff == best_ratio_diff:
            if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                best_ratio = ratio
    return best_ratio


def dynamic_preprocess(image, min_num=1, max_num=6, image_size=448, use_thumbnail=False):
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height

    # calculate the existing image aspect ratio
    target_ratios = set(
        (i, j) for n in range(min_num, max_num + 1) for i in range(1, n + 1) for j in range(1, n + 1) if
        i * j <= max_num and i * j >= min_num)
    target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])

    # find the closest aspect ratio to the target
    target_aspect_ratio = find_closest_aspect_ratio(
        aspect_ratio, target_ratios, orig_width, orig_height, image_size)

    # calculate the target width and height
    target_width = image_size * target_aspect_ratio[0]
    target_height = image_size * target_aspect_ratio[1]
    blocks = target_aspect_ratio[0] * target_aspect_ratio[1]

    # resize the image
    resized_img = image.resize((target_width, target_height))
    processed_images = []
    for i in range(blocks):
        box = (
            (i % (target_width // image_size)) * image_size,
            (i // (target_width // image_size)) * image_size,
            ((i % (target_width // image_size)) + 1) * image_size,
            ((i // (target_width // image_size)) + 1) * image_size
        )
        # split the image
        split_img = resized_img.crop(box)
        processed_images.append(split_img)
    assert len(processed_images) == blocks
    if use_thumbnail and len(processed_images) != 1:
        thumbnail_img = image.resize((image_size, image_size))
        processed_images.append(thumbnail_img)
    return processed_images


def load_image(image_file, input_size=448, max_num=6):
    image = Image.open(image_file).convert('RGB')
    transform = build_transform(input_size=input_size)
    images = dynamic_preprocess(image, image_size=input_size, use_thumbnail=True, max_num=max_num)
    pixel_values = [transform(image) for image in images]
    pixel_values = torch.stack(pixel_values)
    return pixel_values


path = "models/InternVL"
# If you have an 80G A100 GPU, you can put the entire model on a single GPU.
# model = AutoModel.from_pretrained(
#     path,
#     torch_dtype=torch.bfloat16,
#     low_cpu_mem_usage=True,
#     trust_remote_code=True).eval().cuda()
# Otherwise, you need to set device_map='auto' to use multiple GPUs for inference.




with open('/home/vis/dingyuning03/baidu/personal-code/dingyuning_APE/APE/output/APE/configs/city/city_13/city_58_debug/inference_chengguan_58_test/coco_instances_results.json','r') as f:
    results = json.load(f)

label_json_path = 'datasets/add_6cls_26/chengguan_20240412_58cls_add_xiaoguanggao_test.json'

with open(label_json_path,'r') as f:
    labels=json.load(f)

categories=[{'id': 1, 'name': 'trading at the door', 'ori_name': '01_kuamen'},
 {'id': 2, 'name': 'vendor' , 'ori_name': '02_youshang'},
 {'id': 3, 'name': 'piled material', 'ori_name': '03_luanduiwu'},
 {'id': 4, 'name': 'hanging clothes', 'ori_name': '04_yanjieliangshai'},
 {'id': 5, 'name': 'exposed trash', 'ori_name': '05_baolulaji'},
 {'id': 6, 'name': 'muck pile', 'ori_name': '06_jicunlajizhatu'},
 {'id': 7, 'name': 'overflowing bin', 'ori_name': '07_lajimanyi'},
 {'id': 8, 'name': 'packed garbage', 'ori_name': '08_dabaolaji'},
 {'id': 9, 'name': 'household items', 'ori_name': '09_shenghuozawu'},
 {'id': 10, 'name': 'umbrella', 'ori_name': '10_zhandaochengsan'},
 {'id': 11, 'name': 'table and chairs', 'ori_name': '11_lutiancanyin'},
 {'id': 12, 'name': 'stall on the road', 'ori_name': '12_zhandaojingying'},
 {'id': 13, 'name': 'advertisement', 'ori_name': '13_huwaiguanggao'},
 {'id': 14, 'name': 'hazardous chemical vehicle', 'ori_name': 'weihuapinche'},
 {'id': 15, 'name': 'courier truck', 'ori_name': 'kuaidiche'},
 {'id': 16, 'name': 'takeaway truck', 'ori_name': 'waimaiche'},
 {'id': 17, 'name': 'dump truck', 'ori_name': 'zhatuche'},
 {'id': 18, 'name': 'boats', 'ori_name': 'chuanzhi'},
 {'id': 19, 'name': 'Non-motorized Vehicle Parking', 'ori_name': 'feijidongcheweiting'},
 {'id': 20, 'name': 'trash floating on the river', 'ori_name': 'hedaopiaofuwu'},
 {'id': 21, 'name': 'junked vehicle', 'ori_name': 'feiqicheliang_jidongche'},
 {'id': 22, 'name': 'junked non-motorized vehicle', 'ori_name': 'feiqicheliang_feijidongche'},
{'id': 23, 'name': 'blue corrugated metal', 'ori_name': 'luandaluanjian'},
{'id': 24, 'name': 'slope extension on the road', 'ori_name': 'weizhangjiepo'},
{'id': 25, 'name': 'building unclean or building damaged', 'ori_name': 'jianzhuwuwailimianbujie'},
{'id': 26, 'name': 'trash in road or pavement', 'ori_name': 'daolubujie'},
{'id': 27, 'name': 'broken road', 'ori_name': 'daoluposun'},
{'id': 28, 'name': 'sand or gravel in road', 'ori_name': 'daoluyisa'},
{'id': 29, 'name': 'trash in park area', 'ori_name': 'lvdizangluan'},
{'id': 30, 'name': 'Non-decorative tree hangings', 'ori_name': 'feizhuangshixingshugua'},
{'id': 31, 'name': 'dirty around trash can or overflowing trash can', 'ori_name': 'guopixiangzangwu'},
{'id':32,'name':'dirty around dumpster or overflowing dumpster','ori_name':'lazitongzangwu'},
{'id':33,'name':'burning trash and leaves','ori_name':'fenshaolajishuye'},
{'id':34,'name':'person swimming in the river','ori_name':'weiguixiashuiyouyong'},
{'id':35,'name':'human fishing at the water','ori_name':'weiguibuyu'},
{'id':36,'name':'trash beside the river','ori_name':'heanlaji'},
{'id':37,'name':'rubbish on the river','ori_name':'hedaowuran'},
{'id':38,'name':'discarded furniture','ori_name':'feiqijiaju'},
{'id':39,'name':'dead animal','ori_name':'dongwushiti'},
{'id':40,'name':'poultry or livestock','ori_name':'siyangjiajinjiachu'},
{'id':41,'name':'slogan or publicity material','ori_name':'weiguibiaoyu'},
{'id':42,'name':'road obstructed by construction','ori_name':'shigongzhandao'},
{'id':43,'name':'digging up the road','ori_name':'wuzhengwalu'},
{'id':44,'name':'begging or panhandling on the streets','ori_name':'liulangqitao'},
{'id':45,'name':'scrap collection vehicle','ori_name':'zhandaofeipinshougou'},
{'id':46,'name':'crumbling roadway','ori_name':'lumiantaxian'},
{'id':47,'name':'sagging power lines','ori_name':'jiakongxianlantuoluo'},
{'id':48,'name':'fallen trees','ori_name':'shumudaofu'},
{'id':49,'name':'missing or broken manhole cover','ori_name':'jinggaiposun'},
{'id':50,'name':'missing or broken grate','ori_name':'yushuiposun'},
{'id':51,'name':'vehicles blocking the tactile paths','ori_name':'jidongchezhanyamangdao'},
 {'id': 52, 'name': 'motorized Vehicle Parking', 'ori_name': 'jidongcheweiting'},
 {'id': 55, 'name': 'snow or ice on the road', 'ori_name': 'daolujixue'},
 {'id': 54, 'name': 'water on the road', 'ori_name': 'daolujishui'},
  {'id': 53, 'name': 'damaged or tilted large billboards', 'ori_name': 'guanggaopaiposunqingxie'},
    {'id':56, 'name': 'sleeping human', 'ori_name': 'shuigang'},
       {'id':57, 'name': 'smoking human', 'ori_name': 'chouyan'},
    {'id':58, 'name': 'illegally posted or sprayed advertisements', 'ori_name': 'feifaxiaoguanggao'},
 ]

from NAME_TO_TASK import NAME_TO_TASK

name_to_task=NAME_TO_TASK()

for cate in categories:
    cate['name']=name_to_task.caption_name_mapping[cate['name']]
    print(cate['name'])
    # pdb.set_trace()

print(categories)

model = AutoModel.from_pretrained(
    path,
    torch_dtype=torch.bfloat16,
    low_cpu_mem_usage=True,
    trust_remote_code=True,
    device_map='auto').eval()

tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True)
# set the max number of tiles in `max_num`

skip_cates=[]

generation_config = dict(
    num_beams=1,
    max_new_tokens=512,
    do_sample=False,
)

base_path = 'datasets/13cls_weizhang_det/'
debug=False
new_results=[]
print(len(results))
score_thrs=0.1
count=0
update_results=[]
for res in results:
    if res['category_id'] in skip_cates:
        continue
    if res['category_id']!=25:
        continue
    if res['score']>score_thrs:
        update_results.append(res)
    else:
        continue

import time
for res in update_results:
    start=time.time()
    count+=1
    score=res['score']

    image_id=res['image_id']
    category_id=res['category_id']
    bbox=res['bbox']
    bbox[2]=bbox[0]+bbox[2]
    bbox[3]=bbox[1]+bbox[3]
    print(count,len(update_results))
    for img in labels['images']:
        if img['id']==image_id:
            img_path=base_path+img['file_name']
            break
    for cat in categories:
        if cat['id']==category_id:
            cat_name=cat['name']
            break
    assert img_path is not None
    assert cat_name is not None
    img=cv2.imread(img_path)
    pixel_name='tmp.png'
    X, Y, XZ, YZ=bbox[0],bbox[1],bbox[2],bbox[3]
    img = cv2.rectangle(img, (int(X), int(Y)), \
                        (int(XZ), int(YZ)), \
                        [255,0,0], 2)
    cv2.imwrite(pixel_name, img)
    pixel_values = load_image(pixel_name, max_num=6).to(torch.bfloat16).cuda()
    ques="想象你是一个城管，请判断图中蓝色内的内容是否为\'{}\'场景，回答问题时输出\'是\'或\'否\'。".format(cat_name)
    response = model.chat(tokenizer, pixel_values, ques, generation_config)
    print(ques,response)
    if '是' in response:
        new_results.append(res)
    else:
        pass
    os.remove(pixel_name)
    each_time=time.time()-start
    print('用时:',each_time)
    if debug:
        exit(0)

print(len(results),len(new_results))

with open('/home/vis/dingyuning03/baidu/personal-code/dingyuning_APE/APE/output/APE/configs/city/city_13/city_58_debug/internvl/coco_instances_results.json','wb') as f:
    json.dump(new_results,f)



