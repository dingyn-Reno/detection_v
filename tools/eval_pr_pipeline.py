import numpy as np
from multiprocessing import Pool
from pycocotools.coco import COCO
import json
import cv2
import random
import os
from PIL import Image, ImageDraw, ImageFont
import time


def load_json(file):
    with open(file, 'r') as fp:
        json_obj = json.load(fp)
    return json_obj


def filter_according_score(preds, score_thr):
    new_preds = []
    for pred in preds:
        if pred['score'] >= score_thr:
            pred['bbox'].append(pred['score'])
            new_preds.append(pred['bbox'])
    return np.array(new_preds)


def nms(prediction, threshold=0.2):
    if prediction.shape[0] <= 0:
        return prediction
    # import pdb;pdb.set_trace()

    x1, y1, x2, y2, scores = prediction[:, 0], prediction[:,
                                                          1], prediction[:, 2], prediction[:, 3], prediction[:, 4]

    x1 = np.asarray(x1)
    y1 = np.asarray(y1)
    x2 = np.asarray(x2)
    y2 = np.asarray(y2)
    scores = np.asarray(scores)
    areas = (x2 - x1) * (y2 - y1)

    # 从大到小对应的的索引
    order = scores.argsort()[::-1]

    # 记录输出的bbox
    keep = []
    while order.size > 0:
        i = order[0]
        # 记录本轮最大的score对应的index
        keep.append(i)

        if order.size == 1:
            break

        # 计算当前bbox与剩余的bbox之间的IoU
        # 计算IoU需要两个bbox中最大左上角的坐标点和最小右下角的坐标点
        # 即重合区域的左上角坐标点和右下角坐标点
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        # 如果两个bbox之间没有重合, 那么有可能出现负值
        w = np.maximum(0.0, (xx2 - xx1))
        h = np.maximum(0.0, (yy2 - yy1))
        inter = w * h

        iou = inter / (areas[i] + areas[order[1:]] - inter)

        # 删除IoU大于指定阈值的bbox(重合度高), 保留小于指定阈值的bbox
        ids = np.where(iou <= threshold)[0]
        # 因为ids表示剩余的bbox的索引长度
        # +1恢复到order的长度
        order = order[ids + 1]
    # prediction["instances"] = prediction["instances"][keep]
    new_prediction = []
    for kk in keep:
        new_prediction.append(prediction[kk])
    # import pdb;pdb.set_trace()
    return np.asarray(new_prediction)


# categories=[{"id": 1, "name": "trading at the door", "ori_name": "01_跨店经营"}, {"id": 2, "name": "vendor", "ori_name": "02_无照经营游商"}, {"id": 3, "name": "piled material", "ori_name": "03_乱堆物堆料"}, {"id": 4, "name": "hanging clothes", "ori_name": "04_沿街晾晒"}, {"id": 5, "name": "exposed trash", "ori_name": "05_暴露垃圾"}, {"id": 6, "name": "muck pile", "ori_name": "06_积存垃圾渣土"}, {"id": 7, "name": "packed garbage", "ori_name": "07_打包垃圾"}, {"id": 8, "name": "umbrella", "ori_name": "08_占道撑伞"}, {"id": 9, "name": "table and chairs", "ori_name": "09_露天餐饮"}, {"id": 10, "name": "stall on the road", "ori_name": "10_占道经营"}, {"id": 11, "name": "advertisement", "ori_name": "11_户外广告"}, {"id": 12, "name": "hazardous chemical vehicle", "ori_name": "12_危化品车"}, {"id": 13, "name": "courier truck", "ori_name": "13_快递车"}, {"id": 14, "name": "takeaway truck", "ori_name": "14_外卖车"}, {"id": 15, "name": "dump truck", "ori_name": "15_渣土车"}, {"id": 16, "name": "boats", "ori_name": "16_船只"}, {"id": 17, "name": "Non-motorized Vehicle Parking", "ori_name": "17_非机动车违停"}, {"id": 18, "name": "trash floating on the river", "ori_name": "18_河道漂浮物"}, {"id": 19, "name": "junked vehicle", "ori_name": "19_废弃车辆-机动车（僵尸车）"}, {"id": 20, "name": "junked non-motorized vehicle", "ori_name": "20_废弃车辆-非机动车"}, {"id": 21, "name": "blue corrugated metal", "ori_name": "21_乱搭乱建"}, {"id": 22, "name": "slope extension on the road", "ori_name": "22_违章接坡"}, {"id": 23, "name": "building unclean or building damaged", "ori_name": "23_建筑物外立面不洁"}, {"id": 24, "name": "trash in road or pavement", "ori_name": "24_道路不洁"}, {"id": 25, "name": "broken road", "ori_name": "25_道路破损"}, {"id": 26, "name": "sand or gravel in road", "ori_name": "26_道路遗撒"}, {"id": 27, "name": "trash in park area", "ori_name": "27_绿地脏乱"}, {"id": 28, "name": "Non-decorative tree hangings", "ori_name": "28_非装饰性树挂"}, {"id": 29, "name": "dirty around trash can or overflowing trash can", "ori_name": "29_果皮箱脏污、满冒、周边不洁"}, {"id": 30, "name": "dirty around dumpster or overflowing dumpster", "ori_name": "30_垃圾桶脏污、满冒"}, {"id": 31, "name": "burning trash and leaves", "ori_name": "31_焚烧垃圾树叶"}, {"id": 32, "name": "person swimming in the river", "ori_name": "32_水域秩序问题-违规下水游泳"}, {"id": 33, "name": "human fishing at the water", "ori_name": "33_水域秩序问题-违规捕鱼"}, {"id": 34, "name": "trash beside the river", "ori_name": "34_河岸垃圾"}, {"id": 35, "name": "rubbish on the river", "ori_name": "35_河道污染"}, {"id": 36, "name": "discarded furniture", "ori_name": "36_废弃家具设备"}, {"id": 37, "name": "dead animal", "ori_name": "37_动物尸体"}, {"id": 38, "name": "poultry or livestock", "ori_name": "38_擅自饲养家禽家畜"}, {"id": 39, "name": "slogan or publicity material", "ori_name": "39_违规标语宣传品"}, {"id": 40, "name": "road obstructed by construction", "ori_name": "40_施工占道"}, {"id": 41, "name": "digging up the road", "ori_name": "41_无证掘路（道路开挖）"}, {"id": 42, "name": "begging or panhandling on the streets", "ori_name": "42_流浪乞讨"}, {"id": 43, "name": "scrap collection vehicle", "ori_name": "43_占道废品收购"}, {"id": 44, "name": "crumbling roadway", "ori_name": "44_路面塌陷"}, {"id": 45, "name": "sagging power lines", "ori_name": "45_架空线缆脱落"}, {"id": 46, "name": "fallen trees", "ori_name": "46_树木倒伏"}, {"id": 47, "name": "missing or broken manhole cover", "ori_name": "47_井盖破损"}, {"id": 48, "name": "missing or broken grate", "ori_name": "48_雨水篦子破损/缺失"}, {"id": 49, "name": "vehicles blocking the tactile paths", "ori_name": "49_机动车占压盲道"}, {"id": 50, "name": "motorized Vehicle Parking", "ori_name": "50_机动车乱停放"}, {"id": 51, "name": "damaged or tilted large billboards", "ori_name": "51_广告牌破损倾斜"}, {"id": 52, "name": "water on the road", "ori_name": "52_道路积水"}, {"id": 53, "name": "snow or ice on the road", "ori_name": "53_道路积雪"}, {"id": 54, "name": "sleeping human", "ori_name": "54_睡岗"}, {"id": 55, "name": "smoking human", "ori_name": "55_抽烟"}, {"id": 56, "name": "illegally posted or sprayed advertisements", "ori_name": "56_非法小广告"}, {"id": 57, "name": "vertical power lines on building surfaces", "ori_name": "57_建筑外立面乱拉乱挂（飞线充电）"}, {"id": 58, "name": "broken or missing traffic markings", "ori_name": "58_交通标线不清晰"}, {"id": 59, "name": "The workers are cutting down trees", "ori_name": "59_非法伐树"}, {"id": 60, "name": "damaged river bank", "ori_name": "60_河堤破损"}, {"id": 61, "name": "barbecue or camp by the riverbank", "ori_name": "61_水域秩序问题-河岸烧烤露营餐饮"}, {"id": 62, "name": "open-air barbecue", "ori_name": "62_露天烧烤"}, {"id": 63, "name": "dumping sewage", "ori_name": "63_乱倒乱排污水、废水"}, {"id": 64, "name": "distribute advertisements on the streets", "ori_name": "64_街头散发广告"}, {"id": 65, "name": "skyscraper Ad in the green belt or long vertical advertisement blocking windows", "ori_name": "65_户外广告设置位置不合理"}, {"id": 66, "name": "billboard disguised as sign", "ori_name": "66_违规牌匾标识"}, {"id": 67, "name": "construction site dust", "ori_name": "67_工地扬尘"}, {"id": 68, "name": "scattered construction waste", "ori_name": "68_施工废弃料（建筑垃圾）"}, {"id": 69, "name": "dirt road at construction site", "ori_name": "69_施工工地道路未硬化"}, {"id": 70, "name": "sorting the delivery occupies the road", "ori_name": "70_快递分拣占道"}, {"id": 71, "name": "The green space was destroyed and loess was exposed", "ori_name": "71_绿地破损（黄土裸露）"}, {"id": 72, "name": "non-road mobile machinery parked on the road", "ori_name": "72_非道路移动机械乱停放"}, {"id": 73, "name": "damaged blind sidewalk", "ori_name": "73_盲道破损"}, {"id": 74, "name": "build temporary settlement under the bridge", "ori_name": "74_桥下空间私搭乱建"}, {"id": 75, "name": "damaged steps", "ori_name": "75_台阶破损"}, {"id": 76, "name": "damaged road safety barrel", "ori_name": "76_防撞桶破损"}, {"id": 77, "name": "road side guardrail", "ori_name": "77_护栏应撤未撤"}, {"id": 78, "name": "damaged construction hoarding", "ori_name": "78_施工围挡破损"}, {"id": 79, "name": "inflatable Arch", "ori_name": "79_擅自搭建气拱门"}, {"id": 80, "name": "There is garbage around the trash can", "ori_name": "80_垃圾桶周边不洁"}, {"id": 81, "name": "damaged road at construction site entrance", "ori_name": "81_施工工地出入口道路破损"}, {"id": 82, "name": "The dump truck loading muck has no cover", "ori_name": "82_渣土车运输车辆未安装密闭装置（渣土车未苫盖）"}, {"id": 83, "name": "construction vehicle wheels carry dirt", "ori_name": "83_施工工地车轮夹带"}, {"id": 84, "name": "The shared bikes are parked indiscriminately", "ori_name": "84_共享单车乱停放"}]

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

def cv2ImgAddText(img_back, text, left, top):

    # import pdb;pdb.set_trace()

    if (isinstance(img_back, np.ndarray)):  # 判断是否OpenCV图片类型
        img_back = Image.fromarray(cv2.cvtColor(
            img_back.astype(np.uint8), cv2.COLOR_BGR2RGB))
    # 创建一个可以在给定图像上绘图的对象

    draw = ImageDraw.Draw(img_back)
    fontStyle = ImageFont.truetype(
        "/root/paddlejob/workspace/env_run/zhengwu/APE/tools/chinese_stsong.ttf", 15, encoding="utf-8")
    # # 绘制文本
    draw.text((left, top), text, (0, 255, 0), font=fontStyle)
    # 转换回OpenCV格式
    return cv2.cvtColor(np.asarray(img_back), cv2.COLOR_RGB2BGR)


def eval(anno, pred, predict_score_thr, img_base_path, vis_dir, iou_thr=0.2, nproc=4, have_vis=True):
    ids = list(anno.imgs.keys())
    tp_num, all_pred_num, all_gt_num = 0, 0, 0
    eval_result, class_eval_result = {}, {}
    cate_map={}
    for cate in categories:
        cate_map[cate["name"]]=cate['ori_name']

    num_classes = len(anno.getCatIds())
    print('num_classes:', num_classes)
    for class_id in anno.getCatIds():
    # for class_id in [77]:
        cat_info = anno.loadCats(class_id)
        if cat_info[0]['name'] in cate_map:
            # print(cat_info[0]['name'])
            class_tp_num, class_all_pred_num, class_all_gt_num = 0, 0, 0
            for img_id in ids:
                vis_current_img = False
                tp_in_dt_index, tp_in_gt_index = [], []

                ann_ids = anno.getAnnIds(imgIds=img_id, catIds=class_id)
                gts = anno.loadAnns(ann_ids)

                cls_gts = []
                for ann in gts:
                    cls_gts.append(ann['bbox'])
                cls_gts = np.array(cls_gts)
                if cls_gts.shape[0] > 0:
                    cls_gts[:, 2] = cls_gts[:, 2]+cls_gts[:, 0]
                    cls_gts[:, 3] = cls_gts[:, 3]+cls_gts[:, 1]

                pred_ids = pred.getAnnIds(imgIds=img_id, catIds=class_id)
                preds = pred.loadAnns(pred_ids)
                cls_dets = filter_according_score(preds, predict_score_thr)
                if cls_dets.shape[0] > 0:
                    cls_dets[:, 2] = cls_dets[:, 2]+cls_dets[:, 0]
                    cls_dets[:, 3] = cls_dets[:, 3]+cls_dets[:, 1]

                cls_dets = nms(cls_dets)
                if have_vis:
                    if cls_gts.shape[0] > 0 or cls_dets.shape[0] > 0:
                        img_path = anno.loadImgs(img_id)[0]['file_name']
                        img = cv2.imread(os.path.join(img_base_path, img_path))

                gt_is_matched = np.zeros(cls_gts.shape[0], dtype=bool)
                if cls_gts.shape[0] > 0 and cls_dets.shape[0] > 0:
                    ious = bbox_overlaps(cls_dets[:, :-1], cls_gts)

                    ious_max = ious.max(axis=1)  # 找出每个预测框有最大IoU值的真实框
                    ious_argmax = ious.argmax(axis=1)
                    sort_inds = np.argsort(-cls_dets[:, -1])
                    for i in sort_inds:
                        if ious_max[i] >= iou_thr:
                            matched_gt_index = ious_argmax[i]  # 匹配对应的真实框
                            if not gt_is_matched[matched_gt_index]:  # 若真实框没有被匹配，则匹配之
                                gt_is_matched[matched_gt_index] = True
                                tp_num += 1
                                class_tp_num += 1
                                tp_in_dt_index.append(i)
                                tp_in_gt_index.append(matched_gt_index)

                                if have_vis:
                                    vis_current_img = True
                                    random_color = (random.randint(5, 200), random.randint(
                                        5, 200), random.randint(5, 200))
                                    x0, y0, x1, y1 = cls_gts[matched_gt_index]
                                    x0, y0, x1, y1 = map(
                                        int, (x0, y0, x1, y1))
                                    cv2.rectangle(
                                        img, (x0, y0), (x1, y1), random_color, 2)
                                    cv2.putText(img, 'gt', (int(x0), int(
                                        y1+13)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

                                    x0, y0, x1, y1 = cls_dets[i][:-1]
                                    x0, y0, x1, y1 = map(
                                        int, (x0, y0, x1, y1))
                                    cv2.rectangle(
                                        img, (x0, y0), (x1, y1), random_color, 2)
                                    cv2.putText(img, 'dt', (int(x0), int(
                                        y1+5)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

                if have_vis and len(tp_in_dt_index) < len(cls_dets):
                    vis_current_img = True
                    for i in range(len(cls_dets)):
                        if i not in tp_in_dt_index:
                            x0, y0, x1, y1 = cls_dets[i][:-1]
                            x0, y0, x1, y1 = map(int, (x0, y0, x1, y1))
                            cv2.putText(img, 'fp', (int(x0), int(
                                y0-5)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
                            cv2.rectangle(
                                img, (x0, y0), (x1, y1), (255, 0, 0), 2)

                if have_vis and len(tp_in_gt_index) < len(cls_gts):
                    vis_current_img = True
                    for i in range(len(cls_gts)):
                        if i not in tp_in_gt_index:
                            x0, y0, x1, y1 = cls_gts[i]
                            x0, y0, x1, y1 = map(int, (x0, y0, x1, y1))
                            cv2.putText(img, 'fn', (int(x0), int(
                                y0-5)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                            cv2.rectangle(
                                img, (x0, y0), (x1, y1), (0, 0, 255), 2)

                if have_vis and vis_current_img:
                    # if have_vis:
                    os.makedirs(vis_dir+str(predict_score_thr)+'/' +
                                cate_map[cat_info[0]['name']], exist_ok=True)
                    # import pdb;pdb.set_trace()
                    h, w, c = img.shape
                    img_back = np.ones((h+50, w, c))*255
                    img_back[0:h, 0:w, :] = img
                    current_img_ann_ids = anno.getAnnIds(imgIds=img_id)
                    current_img_gts = anno.loadAnns(current_img_ann_ids)
                    for ann in current_img_gts:
                        ann_cate = cate_map[anno.loadCats(
                            ann['category_id'])[0]['name']]
                        x0, y0, x1, y1 = ann['bbox'][0], ann['bbox'][1], ann['bbox'][0] + \
                            ann['bbox'][2], ann['bbox'][1]+ann['bbox'][3]
                        x0, y0, x1, y1 = map(int, (x0, y0, x1, y1))
                        # import pdb;pdb.set_trace()
                        img_back = cv2ImgAddText(
                            img_back, 'gt : '+ann_cate, int(x0), y0-int(20))
                        # cv2.putText(img, 'gt:', (int(x0), int(
                        #         y0-5)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                        cv2.rectangle(img_back, (x0, y0),
                                      (x1, y1), (0, 255, 0), 2)
                    # cv2.putText(img_back, 'all_p :'+str(current_frame_all_p+current_frame_fp)+' tp:'+str(current_frame_tp)+' fn:'+str(current_frame_fn)+' fp:' +
                    #     str(current_frame_fp) + ' predict_wrong:'+str(current_frame_predict_wrong), (int(10), int(h+24)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
                    cv2.imwrite(vis_dir+str(predict_score_thr)+'/' +
                                cate_map[cat_info[0]['name']]+'/'+img_path.split('/')[-1], img_back)

                class_all_pred_num += len(cls_dets)
                all_pred_num += len(cls_dets)
                class_all_gt_num += len(cls_gts)
                all_gt_num += len(cls_gts)

            if class_all_gt_num > 0 and class_all_pred_num > 0:
                class_eval_result[cate_map[cat_info[0]['name']]] = {'recall': np.round(class_tp_num/class_all_gt_num, 4), 'precision': np.round(
                    class_tp_num/class_all_pred_num, 4), 'class_tp_num': class_tp_num, 'class_all_gt_num': class_all_gt_num, 'class_all_pred_num': class_all_pred_num}
        else:
            print(cat_info[0]['name'])
            import pdb
            # pdb.set_trace()
            # continue

    if all_gt_num > 0 and all_pred_num > 0:
        eval_result = {'recall': np.round(tp_num/all_gt_num, 4), 'precision': np.round(
            tp_num/all_pred_num, 4), 'tp_num': tp_num, 'all_gt_num': all_gt_num, 'all_pred_num': all_pred_num}
        return class_eval_result, eval_result
    else:
        return None


def bbox_overlaps(bboxes1, bboxes2, eps=1e-6):
    """Calculate the ious between each bbox of bboxes1 and bboxes2.
    Args:
        bboxes1(ndarray): shape (n, 4)
        bboxes2(ndarray): shape (k, 4)
    Returns:
        ious(ndarray): shape (n, k)
    """
    bboxes1 = bboxes1.astype(np.float32)
    bboxes2 = bboxes2.astype(np.float32)
    rows = bboxes1.shape[0]
    cols = bboxes2.shape[0]
    ious = np.zeros((rows, cols), dtype=np.float32)
    if rows * cols == 0:
        return ious
    exchange = False
    if bboxes1.shape[0] > bboxes2.shape[0]:
        bboxes1, bboxes2 = bboxes2, bboxes1
        ious = np.zeros((cols, rows), dtype=np.float32)
        exchange = True
    area1 = (bboxes1[:, 2] - bboxes1[:, 0]) * (bboxes1[:, 3] - bboxes1[:, 1])
    area2 = (bboxes2[:, 2] - bboxes2[:, 0]) * (bboxes2[:, 3] - bboxes2[:, 1])
    for i in range(bboxes1.shape[0]):
        x_start = np.maximum(bboxes1[i, 0], bboxes2[:, 0])
        y_start = np.maximum(bboxes1[i, 1], bboxes2[:, 1])
        x_end = np.minimum(bboxes1[i, 2], bboxes2[:, 2])
        y_end = np.minimum(bboxes1[i, 3], bboxes2[:, 3])
        overlap = np.maximum(x_end - x_start, 0) * np.maximum(
            y_end - y_start, 0)
        union = area1[i] + area2 - overlap
        union = np.maximum(union, eps)
        ious[i, :] = overlap / union
    if exchange:
        ious = ious.T
    return ious


def sample_test_json():
    label_json_path = '/root/paddlejob/workspace/env_run/zhengwu/DATA/cheng_guan/chengguan_20240329_57cls_fix_hedao_test.json'
    new_json_path = '/root/paddlejob/workspace/env_run/zhengwu/DATA/cheng_guan/chengguan_test_sample_for_debug.json'
    json_data = load_json(label_json_path)
    new_json_data = {
        "annotations": [],
        "images": [],
        "categories": json_data["categories"]
    }
    img_ids = []
    num_save = 0

    for anno_info in json_data['annotations']:
        if num_save < 100:
            num_save += 1
            if anno_info['image_id'] not in img_ids:
                img_ids.append(anno_info['image_id'])
            new_json_data['annotations'].append(anno_info)
    for img_info in json_data['images']:
        if img_info['id'] in img_ids:
            new_json_data['images'].append(img_info)
    with open(new_json_path, 'w') as fp:
        # fp.write(json.dumps(json_data, indent=4))
        json.dump(new_json_data, fp)


if __name__ == "__main__":
    label_json_path = 'datasets/add_6cls_26/chengguan_20240412_58cls_add_xiaoguanggao_test.json'
    base_path = '/home/vis/dingyuning03/baidu/personal-code/dingyuning_APE/APE/output/APE/configs/city/city_13/city_58_debug/inference_chengguan_58_test/'
    predict_result_path = base_path+'coco_instances_results.json'

    #需要可视化时，have_vis=True
    have_vis=False
    img_base_path = 'datasets/chengguan_data'
    vis_dir = 'vis_result/'

    predict_score_thrs = np.linspace(.2, 0.6, int(
        np.round((0.6 - .2) / .1)) + 1, endpoint=True)
    # predict_score_thrs=[0.3]
    print(predict_score_thrs)
    eval_result_1, eval_result_2,class_eval_result = {}, {},{}
    anno = COCO(label_json_path)
    pred = anno.loadRes(load_json(predict_result_path))
    for predict_score_thr in predict_score_thrs:
        predict_score_thr = np.round(predict_score_thr, 2)
        print('\n\n\n\nscore:', predict_score_thr)

        class_eval_result['score_'+str(predict_score_thr)], eval_result_2['score_'+str(predict_score_thr)] = eval(anno, pred, predict_score_thr, img_base_path, vis_dir, have_vis=have_vis)

        precision,recall=0.0,0.0
        cate_num=0
        for cate in class_eval_result['score_'+str(predict_score_thr)]:
            precision+=class_eval_result['score_'+str(predict_score_thr)][cate]['precision']
            recall+=class_eval_result['score_'+str(predict_score_thr)][cate]['recall']
            cate_num+=1
        print(cate_num)
        eval_result_1['score_'+str(predict_score_thr)]={'recall': np.round(recall/84, 4), 'precision': np.round(precision/84,4)}
    print('各类别PR的平均得到的PR,macro')
    print(eval_result_1)

    print('总和计算的PR,micro')
    print(eval_result_2)

    # PR保存
    with open(base_path+'PR_macro.json', 'w') as fp:
        json.dump(eval_result_1, fp, indent=4, ensure_ascii=False)
    with open(base_path+'PR_micro.json', 'w') as fp:
        json.dump(eval_result_2, fp, indent=4, ensure_ascii=False)
    with open(base_path+'PR_classes.json', 'w') as fp:
        json.dump(class_eval_result, fp, indent=4, ensure_ascii=False)
