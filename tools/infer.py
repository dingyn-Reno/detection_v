# Copyright (c) Facebook, Inc. and its affiliates.
import argparse
import glob
import json
import multiprocessing as mp
import os
import tempfile
import time
import warnings
from collections import abc
import cv2
import numpy as np
import tqdm
from detectron2.config import LazyConfig, get_cfg
from detectron2.data.detection_utils import read_image
from detectron2.evaluation.coco_evaluation import instances_to_coco_json
# from detectron2.projects.deeplab import add_deeplab_config
# from detectron2.projects.panoptic_deeplab import add_panoptic_deeplab_config
from detectron2.utils.logger import setup_logger
from predictor_lazy import VisualizationDemo
import matplotlib.pyplot as plt
import matplotlib.pylab as pylab
import os
import cv2
import json
import base64
import requests
import numpy as np
import time
from io import BytesIO
from PIL import Image
import torch
# constants
WINDOW_NAME = "APE"
import pdb
import jsonify

pylab.rcParams['figure.figsize'] = 20, 12



def load(url, if_remote=False):
    """
    Given an url of an image, downloads the image and
    returns a PIL image
    """
    if if_remote:
        response = requests.get(url)
        pil_image = Image.open(BytesIO(response.content)).convert("RGB")
    else:
        pil_image = Image.open(url).convert("RGB")
    # convert to BGR format
    image = np.array(pil_image)[:, :, [2, 1, 0]]
    return image

def imshow(img, caption):
    plt.imshow(img[:, :, [2, 1, 0]])
    plt.axis("off")
    plt.figtext(0.5, 0.09, caption, wrap=True, horizontalalignment='center', fontsize=20)

def base64_to_image(image):
    """
    base64转为opencv�~[��~I~G
    """
    img_data = base64.b64decode(image)
    img_array = np.frombuffer(img_data, np.uint8)
    img = cv2.imdecode(img_array, cv2.COLOR_RGB2BGR)
    return img

def cv2_to_base64(image):
    """
        cv2 image to base64 string
    """
    return base64.b64encode(image).decode('utf8')

def setup_cfg(args):
    # load config from file and command-line arguments
    cfg = LazyConfig.load(args.config_file)
    cfg = LazyConfig.apply_overrides(cfg, args.opts)

    if "output_dir" in cfg.model:
        cfg.model.output_dir = cfg.train.output_dir
    if "model_vision" in cfg.model and "output_dir" in cfg.model.model_vision:
        cfg.model.model_vision.output_dir = cfg.train.output_dir
    if "train" in cfg.dataloader:
        if isinstance(cfg.dataloader.train, abc.MutableSequence):
            for i in range(len(cfg.dataloader.train)):
                if "output_dir" in cfg.dataloader.train[i].mapper:
                    cfg.dataloader.train[i].mapper.output_dir = cfg.train.output_dir
        else:
            if "output_dir" in cfg.dataloader.train.mapper:
                cfg.dataloader.train.mapper.output_dir = cfg.train.output_dir

    if "model_vision" in cfg.model:
        cfg.model.model_vision.test_score_thresh = args.confidence_threshold
    else:
        cfg.model.test_score_thresh = args.confidence_threshold

    # default_setup(cfg, args)

    setup_logger(name="ape")
    setup_logger(name="timm")

    return cfg

def get_parser():
    parser = argparse.ArgumentParser(description="Detectron2 demo for builtin configs")
    parser.add_argument(
        "--config-file",
        default="configs/quick_schedules/mask_rcnn_R_50_FPN_inference_acc_test.yaml",
        metavar="FILE",
        help="path to config file",
    )
    parser.add_argument("--webcam", action="store_true", help="Take inputs from webcam.")
    parser.add_argument("--video-input", help="Path to video file.")
    parser.add_argument(
        "--input",
        nargs="+",
        help="A list of space separated input images; "
        "or a single glob pattern such as 'directory/*.jpg'",
    )
    parser.add_argument(
        "--output",
        help="A file or directory to save output visualizations. "
        "If not given, will show output in an OpenCV window.",
    )

    parser.add_argument(
        "--confidence-threshold",
        type=float,
        default=0.5,
        help="Minimum score for instance predictions to be shown",
    )
    parser.add_argument(
        "--opts",
        help="Modify config options using the command-line 'KEY VALUE' pairs",
        default=[],
        nargs=argparse.REMAINDER,
    )

    parser.add_argument("--text-prompt", default=None)

    parser.add_argument("--with-box", action="store_true", help="show box of instance")
    parser.add_argument("--with-mask", action="store_true", help="show mask of instance")
    parser.add_argument("--with-sseg", action="store_true", help="show mask of class")

    return parser

def updates(args):
    args.config_file='./configs/data_sample/ape_multi_v_infer_end_to_end.py'
    args.input=['./datasets/13cls_weizhang_det/chengguan_20240129_add7cls/weihuapinche_data/11010829001320029232022022051010180449114_01.Jpeg']
    args.output='output/dir'
    args.confidence_threshold=0.0
    args.text_prompt='snow or ice on the road'
    args.vision_prompt=[]
    return args

def infer(img,text_prompt,vision_prompt):
    args.input=img
    args.text_prompt=text_prompt
    if args.input:
        if len(args.input) == 1:
            args.input = glob.glob(os.path.expanduser(args.input[0]), recursive=True)
            assert args.input, "The input path(s) was not found"
        # pdb.set_trace()
        for path in tqdm.tqdm(args.input, disable=not args.output):
            # use PIL, to be consistent with evaluation
            try:
                img = read_image(path, format="BGR")
            except Exception as e:
                print("*" * 60)
                print("fail to open image: ", e)
                print("*" * 60)
                continue
            start_time = time.time()
            predictions, visualized_output, visualized_outputs, metadata = demo.run_on_image(
                img,
                text_prompt=args.text_prompt,
                with_box=True,
                with_mask=False,
                with_sseg=False,
                vision_prompt=vision_prompt
            )
            # pdb.set_trace()
            logger.info(
                "{}: {} in {:.2f}s".format(
                    path,
                    "detected {} instances".format(len(predictions["instances"]))
                    if "instances" in predictions
                    else "finished",
                    time.time() - start_time,
                )
            )
            if args.output:
                if "instances" in predictions:
                    results = instances_to_coco_json(
                        predictions["instances"].to(demo.cpu_device), path
                    )
                    for result in results:
                        result["category_name"] = metadata.thing_classes[result["category_id"]]
                        result["image_name"] = result["image_id"]
            return results

def rebuild_support_dict(support_dict_list):
    lens=len(support_dict_list)
    dic={}
    dic['image']={}
    dic['text']=[]
    for i in range(0,lens):
        dic['text'].append(support_dict_list[i]['text'][0])
        image_list=support_dict_list[i]['image'][1]
        image_feature=np.stack(image_list,axis=0)
        image_feature=torch.from_numpy(image_feature)
        dic['image'][i+1]=image_feature
    support_dict=dic
    return support_dict

mp.set_start_method("spawn", force=True)
args = get_parser().parse_args()
setup_logger(name="fvcore")
setup_logger(name="ape")
logger = setup_logger()
logger.info("Arguments: " + str(args))
print(args)
args = updates(args)
cfg = setup_cfg(args)
cfg.train.init_checkpoint='/share/dingyuning1/obj_model_final.pth'
cfg.model.model_language.cache_dir=""
cfg.model.model_vision.select_box_nums_for_evaluation=500
cfg.model.model_vision.text_feature_bank_reset=True
cfg.dataloader.test=cfg.dataloader.tests[0]
# pdb.set_trace()
if args.video_input:
    demo = VisualizationDemo(cfg, parallel=True, args=args)
else:
    demo = VisualizationDemo(cfg, args=args)

def poster(img_path=None,save_img='output.png'):
    result = 0
    if 0 == result:
        # pdb.set_trace()
        data={}
        support_dict_list=[]
        data['prompts']=""
        data["optimize_status"]=0
        data["task_name"]=""
        data["vision_prompt"]=["/home/dingyuning/APE_v/img/prompt/prompt1.png","/home/dingyuning/APE_v/img/prompt/prompt2.png",
                               "/home/dingyuning/APE_v/img/prompt/prompt3.png","/home/dingyuning/APE_v/img/prompt/prompt4.png",
                               "/home/dingyuning/APE_v/img/prompt/prompt5.png"]
        if data["optimize_status"]==1:
            import pickle
            task_names=data["task_name"].split(",")
            try:
                for task_name in task_names:
                    with open('/home/vis/dingyuning03/feature_docker/{}.pth'.format(task_name), 'rb') as f:
                        support_dict_list.append(pickle.load(f))
            except:
                return jsonify({"error": "one or some of the task_names is not exist"})
            support_dict=rebuild_support_dict(support_dict_list)
        else:
            support_dict=None

        # fn = str(time.time())
        # img_path="/home/vis/dingyuning03/baidu/personal-code/dingyuning_APE/APE/datasets/13cls_weizhang_det/chengguan_20240129_add7cls/weihuapinche_data/11010829001320029232022022051010180449114_01.Jpeg"
        # cv2.imwrite(img_path, base64_to_image(data["base64"]))
        img = [img_path]
        caption = data["prompts"]
        vision_prompt = data["vision_prompt"]
        print(img, caption)
        results = infer(img, caption, vision_prompt)
        # print(results)
        scores = []
        labels = []
        bboxes = []
        for result in results:
            scores.append(result["score"])
            labels.append(result["category_name"])
            bboxes.append(result["bbox"])
        lens = len(scores)
        outr = {"results": []}
        for idx in range(lens):
            _outr = {}
            _outr["scores"] = scores[idx]
            _outr["labels"] = labels[idx]
            boxes = bboxes[idx]
            print(scores[idx])
            X, Y, XZ, YZ = boxes[0], boxes[1], boxes[0] + boxes[2], boxes[1] + boxes[3]
            bboxes[idx] = [X, Y, XZ, YZ]
            _outr["boxes"] = bboxes[idx]
            outr["results"].append(_outr)
        # os.remove(img_path)
        img = cv2.imread(img_path)
        r = outr
        if len(r["results"]) > 0:
            for i in range(len(r["results"])):
                boxes = r["results"][i]["boxes"]
                score = r["results"][i]["scores"]
                if score < 0.1:continue
                X, Y, XZ, YZ = boxes[0], boxes[1], boxes[0] + boxes[2], boxes[1] + boxes[3]
                img = cv2.rectangle(img, (int(X), int(Y)), \
                                    (int(XZ), int(YZ)), \
                                    [255, 0, 0], 2)
                # img=cv2.putText(img, str(r["results"][i]["scores"]), (int(X), int(Y)), cv2.FONT_HERSHEY_COMPLEX, 1.0, (100, 200, 200), 1)
        cv2.imwrite(save_img, img)

    else:
        outr = {"results":"EASYPACK ERROR"}
    response = json.dumps(outr)

    return response

if __name__ == '__main__':

    response=poster('/home/dingyuning/APE_v/img/target_img.webp','./savez.png')
    pdb.set_trace()
    print(response)

# [4526, 4527, 4528, 4529, 4530, 4531, 4532, 4533, 4534, 4535]
# [5666, 5667, 5668, 5669, 5670, 5671, 5672, 6014, 6015, 6016]