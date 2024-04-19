```python
import os

from detectron2.data.datasets.register_coco import register_coco_instances

OBJECTS365_CATEGORIES_FIXNAME =   [{'id': 1, 'name': 'trading at the door', 'ori_name': '01_kuamen'},
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
OBJECTS365_CATEGORIES =    [{'id': 1, 'name': 'trading at the door', 'ori_name': '01_kuamen'},
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
   {'id': 53, 'name': 'damaged or tilted large billboards', 'ori_name': 'guanggaopaiposunqingxie'},
 {'id': 54, 'name': 'water on the road', 'ori_name': 'daolujishui'},
 {'id': 55, 'name': 'snow or ice on the road', 'ori_name': 'daolujixue'},
    {'id':56, 'name': 'sleeping human', 'ori_name': 'shuigang'},
       {'id':57, 'name': 'smoking human', 'ori_name': 'chouyan'},
    {'id':58, 'name': 'illegally posted or sprayed advertisements', 'ori_name': 'feifaxiaoguanggao'},
 ]
def _get_builtin_metadata(key):
    # return {}
    if "fixname" in key:
        id_to_name = {x["id"]: x["name"] for x in OBJECTS365_CATEGORIES_FIXNAME}
        thing_dataset_id_to_contiguous_id = {
            i + 1: i for i in range(len(OBJECTS365_CATEGORIES_FIXNAME))
        }
    else:
        id_to_name = {x["id"]: x["name"] for x in OBJECTS365_CATEGORIES}
        thing_dataset_id_to_contiguous_id = {i + 1: i for i in range(len(OBJECTS365_CATEGORIES))}
    thing_classes = [id_to_name[k] for k in sorted(id_to_name)]
    import json  
    import pdb
    import os 

    # 打开你的json文件  
    with open('datasets/add_6cls_26/chengguan_20240412_58cls_add_xiaoguanggao_train.json', 'r') as f:  
        data = json.load(f)  
    # 现在，data就是一个Python字典，你可以通过键来获取数据  
    # printdata)
    dics={}
    dics['info']=data['info']
    dics['licenses']=data['licenses']
    dics['categories']=  [{"id": 1, "name": "trading at the door", "ori_name": "01_kuamen"}, {"id": 2, "name": "vendor", "ori_name": "02_youshang"}, {"id": 3, "name": "piled material", "ori_name": "03_luanduiwu"}, {"id": 4, "name": "hanging clothes", "ori_name": "04_yanjieliangshai"}, {"id": 5, "name": "exposed trash", "ori_name": "05_baolulaji"}, {"id": 6, "name": "muck pile", "ori_name": "06_jicunlajizhatu"}, {"id": 7, "name": "overflowing bin", "ori_name": "07_lajimanyi"}, {"id": 8, "name": "packed garbage", "ori_name": "08_dabaolaji"}, {"id": 9, "name": "household items", "ori_name": "09_shenghuozawu"}, {"id": 10, "name": "umbrella", "ori_name": "10_zhandaochengsan"}, {"id": 11, "name": "table and chairs", "ori_name": "11_lutiancanyin"}, {"id": 12, "name": "stall on the road", "ori_name": "12_zhandaojingying"}, {"id": 13, "name": "advertisement", "ori_name": "13_huwaiguanggao"}, {"id": 14, "name": "hazardous chemical vehicle", "ori_name": "weihuapinche"}, {"id": 15, "name": "courier truck", "ori_name": "kuaidiche"}, {"id": 16, "name": "takeaway truck", "ori_name": "waimaiche"}, {"id": 17, "name": "dump truck", "ori_name": "zhatuche"}, {"id": 18, "name": "boats", "ori_name": "chuanzhi"}, {"id": 19, "name": "Non-motorized Vehicle Parking", "ori_name": "feijidongcheweiting"}, {"id": 20, "name": "trash floating on the river", "ori_name": "hedaopiaofuwu"}, {"id": 21, "name": "junked vehicle", "ori_name": "feiqicheliang_jidongche"}, {"id": 22, "name": "junked non-motorized vehicle", "ori_name": "feiqicheliang_feijidongche"}, {"id": 23, "name": "blue corrugated metal", "ori_name": "luandaluanjian"}, {"id": 24, "name": "slope extension on the road", "ori_name": "weizhangjiepo"}, {"id": 25, "name": "building unclean or building damaged", "ori_name": "jianzhuwuwailimianbujie"}, {"id": 26, "name": "trash in road or pavement", "ori_name": "daolubujie"}, {"id": 27, "name": "broken road", "ori_name": "daoluposun"}, {"id": 28, "name": "sand or gravel in road", "ori_name": "daoluyisa"}, {"id": 29, "name": "trash in park area", "ori_name": "lvdizangluan"}, {"id": 30, "name": "Non-decorative tree hangings", "ori_name": "feizhuangshixingshugua"}, {"id": 31, "name": "dirty around trash can or overflowing trash can", "ori_name": "guopixiangzangwu"}, {"id": 32, "name": "dirty around dumpster or overflowing dumpster", "ori_name": "lazitongzangwu"}, {"id": 33, "name": "burning trash and leaves", "ori_name": "fenshaolajishuye"}, {"id": 34, "name": "person swimming in the river", "ori_name": "weiguixiashuiyouyong"}, {"id": 35, "name": "human fishing at the water", "ori_name": "weiguibuyu"}, {"id": 36, "name": "trash beside the river", "ori_name": "heanlaji"}, {"id": 37, "name": "rubbish on the river", "ori_name": "hedaowuran"}, {"id": 38, "name": "discarded furniture", "ori_name": "feiqijiaju"}, {"id": 39, "name": "dead animal", "ori_name": "dongwushiti"}, {"id": 40, "name": "poultry or livestock", "ori_name": "siyangjiajinjiachu"}, {"id": 41, "name": "slogan or publicity material", "ori_name": "weiguibiaoyu"}, {"id": 42, "name": "road obstructed by construction", "ori_name": "shigongzhandao"}, {"id": 43, "name": "digging up the road", "ori_name": "wuzhengwalu"}, {"id": 44, "name": "begging or panhandling on the streets", "ori_name": "liulangqitao"}, {"id": 45, "name": "scrap collection vehicle", "ori_name": "zhandaofeipinshougou"}, {"id": 46, "name": "crumbling roadway", "ori_name": "lumiantaxian"}, {"id": 47, "name": "sagging power lines", "ori_name": "jiakongxianlantuoluo"}, {"id": 48, "name": "fallen trees", "ori_name": "shumudaofu"}, {"id": 49, "name": "missing or broken manhole cover", "ori_name": "jinggaiposun"}, {"id": 50, "name": "missing or broken grate", "ori_name": "yushuiposun"}, {"id": 51, "name": "vehicles blocking the tactile paths", "ori_name": "jidongchezhanyamangdao"}, {"id": 52, "name": "motorized Vehicle Parking", "ori_name": "jidongcheweiting"}, {"id": 55, "name": "snow or ice on the road", "ori_name": "daolujixue"}, {"id": 54, "name": "water on the road", "ori_name": "daolujishui"}, {"id": 53, "name": "damaged or tilted large billboards", "ori_name": "guanggaopaiposunqingxie"}, {"id": 56, "name": "sleeping human", "ori_name": "shuigang"}, {"id": 57, "name": "smoking human", "ori_name": "chouyan"}, {"id": 58, "name": "illegally posted or sprayed advertisements", "ori_name": "feifaxiaoguanggao"}]
    dics['images']=data['images']
    dics['annotations']=[]
    lss=[]
    
    anno_counts=[0 for i in range(59)]
    for anno in data['annotations']:
        anno_counts[anno['category_id']]+=1
    
    
    class_image_count=[]
    for i in range(1,59):
        # class_image_count[i]=anno_counts[i]/sum(anno_counts)
        class_image_count.append(
            {
                'id': i,
                "image_count": anno_counts[i],
            }
        )


        
    return {
        "thing_dataset_id_to_contiguous_id": thing_dataset_id_to_contiguous_id,
        "thing_classes": thing_classes,
        'class_image_count': class_image_count,
    }



_PREDEFINED_SPLITS_OBJECTS365 = {
       
    "chengguan_58_train": (
        "13cls_weizhang_det",
        "add_6cls_26/chengguan_20240412_58cls_add_xiaoguanggao_train.json",
    ),
    "chengguan_58_test": (
        "13cls_weizhang_det",
        "add_6cls_26/chengguan_20240412_58cls_add_xiaoguanggao_test.json",
    ),
}


def register_all_objects365(root):
    for key, (image_root, json_file) in _PREDEFINED_SPLITS_OBJECTS365.items():
        register_coco_instances(
            key,
            _get_builtin_metadata(key),
            os.path.join(root, json_file) if "://" not in json_file else json_file,
            os.path.join(root, image_root),
        )



if __name__.endswith(".city_58"):
    # Assume pre-defined datasets live in `./datasets`.
    _root = os.getenv("DETECTRON2_DATASETS", "datasets")
    register_all_objects365(_root)

```