import os
os.environ["NCCL_TIMEOUT"] = "360000"
import datetime
import logging
import random
import sys
import time
from collections import abc
from contextlib import nullcontext

import torch
from torch.nn.parallel import DataParallel, DistributedDataParallel
import pdb
import ape
from ape.checkpoint import DetectionCheckpointer
from ape.engine import SimpleTrainer
from ape.evaluation import inference_on_dataset
from detectron2.config import LazyConfig, instantiate
from detectron2.engine import default_argument_parser  # SimpleTrainer,
from detectron2.engine import default_setup, hooks, launch
from detectron2.engine.defaults import create_ddp_model
from detectron2.evaluation import print_csv_format
from detectron2.utils import comm
from detectron2.utils.events import (
    CommonMetricPrinter,
    JSONWriter,
    TensorboardXWriter,
    get_event_storage,
)
from detectron2.utils.file_io import PathManager
from detectron2.utils.logger import setup_logger
from detrex.modeling import ema
from detrex.utils import WandbWriter
import cv2
import pdb

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))

logger = logging.getLogger("ape")
def do_test(cfg, model, eval_only=False,checkpointer=None):
    logger = logging.getLogger("ape")
    if "evaluator" in cfg.dataloader:
        if isinstance(model, DistributedDataParallel):
            if hasattr(model.module, "set_eval_dataset"):
                model.module.set_eval_dataset(cfg.dataloader.test.dataset.names)
        else:
            if hasattr(model, "set_eval_dataset"):
                model.set_eval_dataset(cfg.dataloader.test.dataset.names)
        output_dir = os.path.join(
            cfg.train.output_dir, "inference_{}".format(cfg.dataloader.test.dataset.names)
        )
        if "cityscapes" in cfg.dataloader.test.dataset.names:
            pass
        else:
            if isinstance(cfg.dataloader.evaluator, abc.MutableSequence):
                for evaluator in cfg.dataloader.evaluator:
                    evaluator.output_dir = output_dir
            else:
                cfg.dataloader.evaluator.output_dir = output_dir

        ret = inference_on_dataset(
            model, instantiate(cfg.dataloader.test), instantiate(cfg.dataloader.evaluator)
        )
        logger.info(
            "Evaluation results for {} in csv format:".format(cfg.dataloader.test.dataset.names)
        )
        print_csv_format(ret)
        ret = {f"{k}_{cfg.dataloader.test.dataset.names}": v for k, v in ret.items()}
        ap50_result=ret[f'{k}_{cfg.dataloader.test.dataset.names}']['AP50']
                

    else:
        ret = {}

    if "evaluators" in cfg.dataloader:
        for test, evaluator in zip(cfg.dataloader.tests, cfg.dataloader.evaluators):
            if isinstance(model, DistributedDataParallel):
                model.module.set_eval_dataset(test.dataset.names)
            else:
                model.set_eval_dataset(test.dataset.names)
            output_dir = os.path.join(
                cfg.train.output_dir, "inference_{}".format(test.dataset.names)
            )
            if isinstance(evaluator, abc.MutableSequence):
                for eva in evaluator:
                    eva.output_dir = output_dir
            else:
                evaluator.output_dir = output_dir
            ret_ = inference_on_dataset(model, instantiate(test), instantiate(evaluator))
            logger.info("Evaluation results for {} in csv format:".format(test.dataset.names))
            print_csv_format(ret_)
            ret.update({f"{k}_{test.dataset.names}": v for k, v in ret_.items()})

    bbox_odinw_AP = {"AP": [], "AP50": [], "AP75": [], "APs": [], "APm": [], "APl": []}
    segm_seginw_AP = {"AP": [], "AP50": [], "AP75": [], "APs": [], "APm": [], "APl": []}
    bbox_rf100_AP = {"AP": [], "AP50": [], "AP75": [], "APs": [], "APm": [], "APl": []}
    for k, v in ret.items():
        for kk, vv in v.items():
            if k.startswith("bbox_odinw") and kk in bbox_odinw_AP and vv == vv:
                bbox_odinw_AP[kk].append(vv)
            if k.startswith("segm_seginw") and kk in segm_seginw_AP and vv == vv:
                segm_seginw_AP[kk].append(vv)
            if k.startswith("bbox_rf100") and kk in bbox_rf100_AP and vv == vv:
                bbox_rf100_AP[kk].append(vv)

    from statistics import median, mean

    logger.info("Evaluation results: {}".format(ret))
    for k, v in bbox_odinw_AP.items():
        if len(v) > 0:
            logger.info(
                "Evaluation results for odinw bbox {}: mean {} median {}".format(
                    k, mean(v), median(v)
                )
            )
    for k, v in segm_seginw_AP.items():
        if len(v) > 0:
            logger.info(
                "Evaluation results for seginw segm {}: mean {} median {}".format(
                    k, mean(v), median(v)
                )
            )
    for k, v in bbox_rf100_AP.items():
        if len(v) > 0:
            logger.info(
                "Evaluation results for rf100 bbox {}: mean {} median {}".format(
                    k, mean(v), median(v)
                )
            )

    return ret

def vis_picture(img, results , pre_score, classes):
    boxes = results[0]['instances'].pred_boxes.tensor
    scores = results[0]['instances'].scores
    pred_classes = results[0]['instances'].pred_classes
    for box, score, pred_class in zip(boxes, scores, pred_classes):
        if score.cpu().detach().numpy() < pre_score:
            continue
        box = box.cpu().detach().numpy()
        score = score.cpu().detach().numpy()
        pred_class = pred_class.cpu().numpy()
        X, Y, XZ, YZ = box[0], box[1],box[2], box[3]
        img = cv2.rectangle(img, (int(X), int(Y)), \
                (int(XZ), int(YZ)), \
                [255,0,0], 2)
        img=cv2.putText(img, str(f'{classes[int(pred_class)]}-{score}'), (int(X), int(Y)), cv2.FONT_HERSHEY_COMPLEX, 1.0, (100, 200, 200), 1)
    return img


if __name__ == "__main__":
    coco_classes = [
    'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light',
    'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
    'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
    'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard',
    'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
    'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
    'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
    'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear',
    'hair drier', 'toothbrush'
]

    args = default_argument_parser().parse_args()
    args.config_file = 'configs/data_sample/infer_single_img.py'
    img = 'test.jpg'
    cfg = LazyConfig.load(args.config_file)
    cfg = LazyConfig.apply_overrides(cfg, args.opts)
    if "output_dir" in cfg.model:
        cfg.model.output_dir = cfg.train.output_dir
    if "model_vision" in cfg.model and "output_dir" in cfg.model.model_vision:
        cfg.model.model_vision.output_dir = cfg.train.output_dir
    model = instantiate(cfg.model)
    model.to(cfg.train.device)
    DetectionCheckpointer(model, **ema.may_get_ema_checkpointer(cfg, model)).load(
            cfg.train.init_checkpoint
        )
    results = model.forward([img])
    print(results)
    pic = vis_picture(cv2.imread(img), results, 0.5, coco_classes)
    cv2.imwrite('result.jpg', pic)
    
