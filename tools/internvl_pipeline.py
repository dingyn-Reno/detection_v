from transformers import AutoTokenizer, AutoModel
import torch
import torchvision.transforms as T
from PIL import Image
import time
from torchvision.transforms.functional import InterpolationMode
import argparse
import logging
import os
# 配置日志基本设置

print(os.getenv('CUDA_VISIBLE_DEVICES'))

devices_list=os.getenv('CUDA_VISIBLE_DEVICES').capitalize().split(',')
batch_size=len(devices_list)

# 写入不同级别的日志
# logging.debug('这是一条debug日志')
# logging.info('这是一条info日志')
# logging.warning('这是一条warning日志')
# logging.error('这是一条error日志')
# logging.critical('这是一条critical日志')

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



# If you have an 80G A100 GPU, you can put the entire model on a single GPU.
# model = AutoModel.from_pretrained(
#     path,
#     torch_dtype=torch.bfloat16,
#     low_cpu_mem_usage=True,
#     trust_remote_code=True).eval().cuda()
# Otherwise, you need to set device_map='auto' to use multiple GPUs for inference.

with open('evals/res_84/coco_instances_results.json','r') as f:
    results = json.load(f)

label_json_path = '/ssd3/vis_data/chengguan_data/add_traffic_data/chengguan_20240511_84cls_add_traffic_test_check_wh.json'

with open(label_json_path,'r') as f:
    labels=json.load(f)

categories=[{"id": 1, "name": "01_跨店经营", "supercategory": ""}, {"id": 2, "name": "02_无照经营游商", "supercategory": ""}, {"id": 3, "name": "03_乱堆物堆料", "supercategory": ""}, {"id": 4, "name": "04_沿街晾晒", "supercategory": ""}, {"id": 5, "name": "05_暴露垃圾", "supercategory": ""}, {"id": 6, "name": "06_积存垃圾渣土", "supercategory": ""}, {"id": 7, "name": "07_打包垃圾", "supercategory": ""}, {"id": 8, "name": "08_占道撑伞", "supercategory": ""}, {"id": 9, "name": "09_露天餐饮", "supercategory": ""}, {"id": 10, "name": "10_占道经营", "supercategory": ""}, {"id": 11, "name": "11_户外广告", "supercategory": ""}, {"id": 12, "name": "12_危化品车", "supercategory": ""}, {"id": 13, "name": "13_快递车", "supercategory": ""}, {"id": 14, "name": "14_外卖车", "supercategory": ""}, {"id": 15, "name": "15_渣土车", "supercategory": ""}, {"id": 16, "name": "16_船只", "supercategory": ""}, {"id": 17, "name": "17_非机动车违停", "supercategory": ""}, {"id": 18, "name": "18_河道漂浮物", "supercategory": ""}, {"id": 19, "name": "19_废弃车辆-机动车（僵尸车）", "supercategory": ""}, {"id": 20, "name": "20_废弃车辆-非机动车", "supercategory": ""}, {"id": 21, "name": "21_乱搭乱建", "supercategory": ""}, {"id": 22, "name": "22_违章接坡", "supercategory": ""}, {"id": 23, "name": "23_建筑物外立面不洁", "supercategory": ""}, {"id": 24, "name": "24_道路不洁", "supercategory": ""}, {"id": 25, "name": "25_道路破损", "supercategory": ""}, {"id": 26, "name": "26_道路遗撒", "supercategory": ""}, {"id": 27, "name": "27_绿地脏乱", "supercategory": ""}, {"id": 28, "name": "28_非装饰性树挂", "supercategory": ""}, {"id": 29, "name": "29_果皮箱脏污、满冒、周边不洁", "supercategory": ""}, {"id": 30, "name": "30_垃圾桶脏污、满冒", "supercategory": ""}, {"id": 31, "name": "31_焚烧垃圾树叶", "supercategory": ""}, {"id": 32, "name": "32_水域秩序问题-违规下水游泳", "supercategory": ""}, {"id": 33, "name": "33_水域秩序问题-违规捕鱼", "supercategory": ""}, {"id": 34, "name": "34_河岸垃圾", "supercategory": ""}, {"id": 35, "name": "35_河道污染", "supercategory": ""}, {"id": 36, "name": "36_废弃家具设备", "supercategory": ""}, {"id": 37, "name": "37_动物尸体", "supercategory": ""}, {"id": 38, "name": "38_擅自饲养家禽家畜", "supercategory": ""}, {"id": 39, "name": "39_违规标语宣传品", "supercategory": ""}, {"id": 40, "name": "40_施工占道", "supercategory": ""}, {"id": 41, "name": "41_无证掘路（道路开挖）", "supercategory": ""}, {"id": 42, "name": "42_流浪乞讨", "supercategory": ""}, {"id": 43, "name": "43_占道废品收购", "supercategory": ""}, {"id": 44, "name": "44_路面塌陷", "supercategory": ""}, {"id": 45, "name": "45_架空线缆脱落", "supercategory": ""}, {"id": 46, "name": "46_树木倒伏", "supercategory": ""}, {"id": 47, "name": "47_井盖破损", "supercategory": ""}, {"id": 48, "name": "48_雨水篦子破损/缺失", "supercategory": ""}, {"id": 49, "name": "49_机动车占压盲道", "supercategory": ""}, {"id": 50, "name": "50_机动车乱停放", "supercategory": ""}, {"id": 51, "name": "51_广告牌破损倾斜", "supercategory": ""}, {"id": 52, "name": "52_道路积水", "supercategory": ""}, {"id": 53, "name": "53_道路积雪", "supercategory": ""}, {"id": 54, "name": "54_睡岗", "supercategory": ""}, {"id": 55, "name": "55_抽烟", "supercategory": ""}, {"id": 56, "name": "56_非法小广告", "supercategory": ""}, {"id": 57, "name": "57_建筑外立面乱拉乱挂（飞线充电）", "supercategory": ""}, {"id": 58, "name": "58_交通标线不清晰", "supercategory": ""}, {"id": 59, "name": "59_非法伐树", "supercategory": ""}, {"id": 60, "name": "60_河堤破损", "supercategory": ""}, {"id": 61, "name": "61_水域秩序问题-河岸烧烤露营餐饮", "supercategory": ""}, {"id": 62, "name": "62_露天烧烤", "supercategory": ""}, {"id": 63, "name": "63_乱倒乱排污水、废水", "supercategory": ""}, {"id": 64, "name": "64_街头散发广告", "supercategory": ""}, {"id": 65, "name": "65_户外广告设置位置不合理", "supercategory": ""}, {"id": 66, "name": "66_违规牌匾标识", "supercategory": ""}, {"id": 67, "name": "67_工地扬尘", "supercategory": ""}, {"id": 68, "name": "68_施工废弃料（建筑垃圾）", "supercategory": ""}, {"id": 69, "name": "69_施工工地道路未硬化", "supercategory": ""}, {"id": 70, "name": "70_快递分拣占道", "supercategory": ""}, {"id": 71, "name": "71_绿地破损（黄土裸露）", "supercategory": ""}, {"id": 72, "name": "72_非道路移动机械乱停放", "supercategory": ""}, {"id": 73, "name": "73_盲道破损", "supercategory": ""}, {"id": 74, "name": "74_桥下空间私搭乱建", "supercategory": ""}, {"id": 75, "name": "75_台阶破损", "supercategory": ""}, {"id": 76, "name": "76_防撞桶破损", "supercategory": ""}, {"id": 77, "name": "77_护栏应撤未撤", "supercategory": ""}, {"id": 78, "name": "78_施工围挡破损", "supercategory": ""}, {"id": 79, "name": "79_擅自搭建气拱门", "supercategory": ""}, {"id": 80, "name": "80_垃圾桶周边不洁", "supercategory": ""}, {"id": 81, "name": "81_施工工地出入口道路破损", "supercategory": ""}, {"id": 82, "name": "82_渣土车运输车辆未安装密闭装置（渣土车未苫盖）", "supercategory": ""}, {"id": 83, "name": "83_施工工地车轮夹带", "supercategory": ""}, {"id": 84, "name": "84_共享单车乱停放", "supercategory": ""}]


model_path = '/ssd2/dingyuning/internvl/evals/InternVL/'
# If you have an 80G A100 GPU, you can put the entire model on a single GPU.
model = AutoModel.from_pretrained(
    model_path,
    torch_dtype=torch.bfloat16,
    low_cpu_mem_usage=True,
    trust_remote_code=True).eval().cuda()
# Otherwise, you need to set device_map='auto' to use multiple GPUs for inference.

tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
# set the max number of tiles in `max_num`

generation_config = dict(
    num_beams=1,
    max_new_tokens=512,
    do_sample=False,
)

skip_cates=[]
base_path = '/ssd3/vis_data/'
new_dir='prompt_res_84'
debug=False
# new_results=[]
print(len(results))
score_thrs=0.1
skip_score_thrs=0.4
count=0

explanation_dicts={
    1:'超出正常门店范围内的经营物料摆放的现象',
    2:'无营业执照，未经许可在城市道路、公共场所从事流动性经营的现象',
    3:'未经许可在公共场所堆放物料',
    4:'在主要道路及公共场所的树木和护栏、路牌、电线、电杆等设施上吊挂、晾晒物品的现象',
    5:'功能场所未导入垃圾容器的生活垃圾',
    6:'未按照规定及时清理的非生活垃圾、渣土',
    7:'成堆堆放的打包好的垃圾',
    8:'大型遮阳伞，伞下有多个人或者经营行为',
    9:'露天摆放的经营餐饮物料和露天餐饮就餐行为',
    10:'违法占用道路、桥梁、广场等公共场所进行买卖商品或服务的行为',
    11:'违规设置以灯箱、霓虹灯、电子显示装置、展示牌等为载体形式的户外广告设施',
    12:'带有明显危化品标识的车（洒水车、混凝土搅拌车不算）',
    17:'在未经许可、未合法设置停车泊位的地点停放非机动车辆',
    19:'长期占道无人使用的汽车',
    20:'长期占道无人使用的非机动车（二轮、三轮）',
    21:'城市通行道路旁、绿地旁搭建的危房、棚户（水泥房或者彩钢瓦房）',
    22:'马路牙子水泥接坡、水泥板、砖石接坡',
    23:'外立（墙）面明显脏污、破损或脱落',
    24:'通行道路上的垃圾（垃圾袋、纸团），包含生活垃圾、树叶等',
    25:'道路损坏、塌陷、坑洼等影响通行的情况',
    26:'交通主干道上的泥土，材料等遗撒',
    27:'草地/绿地上的垃圾（落叶、积雪不算）',
    28:'树木、灌木上的挂的塑料袋、破旧布条等',
    29:'公园以及城市主干道旁边的小型垃圾箱（不是翻盖的那种）脏污、满冒、周边存在油污以及遗落垃圾',
    30:'大型垃圾桶的脏污、满冒',
    31:'堆放树叶、垃圾进行焚烧的现象',
    32:'人员违规进入河道、违规下水游泳、违规划船',
    33:'人员违规捕鱼、垂钓行为',
    34:'河沟，河岸上存在堆放垃圾（能够抓拍水面以及水岸）',
    35:'充斥大量漂浮物、水体发黑有凝结物、水体存在大量白色泡沫、水体颜色异常（深绿色、棕黑色、红色）',
    36:'沿街摆放的破旧生活家具，非餐饮用的椅子，沙发等',
    39:'横幅、直幅',
    40:'施工车辆、施工标识（标识牌、锥形桶等）占用道路的现象',
    42:'在公共场所从事卖艺、乞讨、露宿等行为',
    43:'未经许可在公共场所从事收购废品的占道行为',
    44:'主干道破损下沉出现深坑',
    45:'在道路、小区和其他室外公共空间架空的线缆破损下坠',
    46:'树木倒在地面或者倾斜角度较大',
    47:'下水道井盖破损或者缺失',
    48:'雨水篦子破损或者缺失',
    49:'机动车违规停放在盲道上',
    50:'机动车停留在通行道路上',
    51:'广告牌破损或者倾斜',
    52:'道路大面积积水的现象',
    54:'人在工位上睡觉',
    56:'地面或者墙上的手写广告，张贴的纸质广告',
    57:'居民建筑存在私自扯线或者悬挂移动插排进行充电的现象',
    58:'交通标识线破损缺失',
    59:'园林树木乱砍乱伐现象',
    60:'河堤及其附属设施损坏、塌陷的现象',
    61:'在公共水域露营、野炊等活动',
    62:'在露天公共场所内烧烤食物的行为',
    63:'在公共场所乱倒乱排污水、废水现象',
    64:'在公共场所散发广告的现象',
    65:'广告牌占道、广告牌遮挡交通标识牌的现象',
    66:'违规设置以灯箱、霓虹灯、展示牌等形式显示单位名称、字号和标志的设施',
    67:'施工工地中的扬尘现象',
    68:'在公共场所堆放建筑垃圾、工程渣土的现象',
    69:'施工现场，通行道路为土路，存在清晰车辙印或者路面被碾压痕迹',
    70:'快递车停放在路边以及路边存在摆放的快递',
    71:'草地出现明显斑秃',
    72:'工程车辆违规停放在道路上，例如铲车、压路机、叉车',
    73:'盲道缺失或破损',
    74:'过街天桥下搭建临时建筑或者居住点',
    75:'台阶破损或缺失',
    76:'道路防撞桶破损',
    77:'道路上的护栏',
    79:'彩虹门',
    80:'大型垃圾桶周边存在遗撒垃圾',
    82:'满载的渣土车没有苫盖',
    83:'施工车辆车轮夹带泥土行驶时造成路面污染',
    84:'占压盲道、阻碍行人车辆通行、淤积堆叠、停在草坪的共享单车'
}

if __name__=='__main__':
    # update_results=results
    parser = argparse.ArgumentParser(description="Process some integers.")
    parser.add_argument("log", type=str,default='a1.log', help="日志记录")
    parser.add_argument("start",default=1, type=str, help="开始的类别序号")
    parser.add_argument("end",default=59, type=str, help="结束的类别序号")
    args = parser.parse_args()
    log_path=args.log
    logging.basicConfig(filename=log_path, level=logging.INFO,
                    format='%(asctime)s:%(levelname)s:%(message)s')
    start=int(args.start)
    end=int(args.end)
    cates_list=[i for i in range(start,end)]
    for cate in cates_list:
        logging.info('类别{}开始'.format(cate))
        update_results=[]
        new_results=[]
        count=0
        for res in results:
            if res['score']<score_thrs:
                continue
            if res['category_id'] in skip_cates:
                new_results.append(res)
                continue
            if res['category_id']==cate:
                if res['score']> skip_score_thrs:
                    new_results.append(res)
                    continue
                else:
                    update_results.append(res)
        for res in update_results:
            count+=1
            start=time.time()
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
            try:
                img=cv2.imread(img_path)
            except:
                logging.error(img_path)
                new_results.append(res)
            pixel_name='{}.png'.format(log_path.split('.')[0])
            X, Y, XZ, YZ=bbox[0],bbox[1],bbox[2],bbox[3]
            img = cv2.rectangle(img, (int(X), int(Y)), \
                                (int(XZ), int(YZ)), \
                                [255,0,0], 2)
            cv2.imwrite(pixel_name, img)
            pixel_values = load_image(pixel_name, max_num=6).to(torch.bfloat16).cuda()
            if cate in explanation_dicts.keys():
                text_sample=explanation_dicts[cate]
                ques="我们将{}定义为\'{}\'。从现在开始，想象你是一名城管，请判断图中蓝色内的内容是否为\'{}\'场景，回答问题时输出\'是\'或\'否\'。".format(text_sample,cat_name,cat_name)
            else:
                ques="想象你是一个城管，请判断图中蓝色内的内容是否为\'{}\'场景，回答问题时输出\'是\'或\'否\'。".format(cat_name)
            response = model.chat(tokenizer, pixel_values, ques, generation_config)
            logging.info('images:{}/{}'.format(count,len(update_results)))
            logging.info(response)
            print(ques,response)
            if '否' in response:
                logging.info('drop')
            elif 'No' in response:
                logging.info('drop')
            else:
                new_results.append(res)
            os.remove(pixel_name)
            each_time=time.time()-start
            print('用时:',each_time)
            if debug and count>1000:
                break
        logging.info('类别{}完成'.format(cate))
        with open('evals/{}/coco_instances_results_class_{}.json'.format(new_dir,cate),'w') as f:
            json.dump(new_results,f)


