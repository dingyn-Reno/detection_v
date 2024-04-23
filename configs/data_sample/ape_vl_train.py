import torch.nn as nn

from detectron2.config import LazyCall as L
from detectron2.layers import ShapeSpec
from detectron2.solver import WarmupParamScheduler
from detrex.modeling.neck import ChannelMapper
from fvcore.common.param_scheduler import MultiStepParamScheduler

from ape.data.detection_utils import get_fed_loss_cls_weights
from ape.layers import VisionLanguageFusion
from ape.modeling.ape_deta import (
    DeformableDETRSegmVL,
    DeformableDetrTransformerDecoderVL,
    DeformableDetrTransformerEncoderVL,
    DeformableDetrTransformerVL,
    DeformableDETRSegmVV,
)
from ape.modeling.text import EVA02CLIP

from ...common.backbone.vitl_eva02_clip import backbone
from .cgdatasets import (
    dataloader,
)
from ...LVIS_InstanceSegmentation.ape_deta.ape_deta_vitl_eva02_lsj1024_cp_24ep import (
    model,
    optimizer,
    train,
)

import random

import detectron2.data.transforms as T
from detectron2.config import LazyCall as L
from detectron2.data import (
    DatasetMapper,
    MetadataCatalog,
    build_detection_test_loader,
    build_detection_train_loader,
    get_detection_dataset_dicts,
)
from detectron2.data.samplers import RepeatFactorTrainingSampler
from detectron2.evaluation import COCOEvaluator, LVISEvaluator, SemSegEvaluator
from omegaconf import OmegaConf
from ape.data import (
    DatasetMapper_detr_panoptic,
    DatasetMapper_detr_panoptic_copypaste,
    build_detection_train_loader_multi_dataset,
    build_detection_train_loader_multi_dataset_copypaste,
    get_detection_dataset_dicts_multi_dataset,
    get_detection_dataset_dicts_multi_dataset_copypaste,
)
from ape.evaluation import RefCOCOEvaluator
from ape.evaluation.oideval import OIDEvaluator
from detectron2.data.datasets import register_coco_instances

dataloader = OmegaConf.create()

image_size = 1024

dataloader.train = [
    L(build_detection_train_loader_multi_dataset_copypaste)(
        dataset=L(get_detection_dataset_dicts_multi_dataset_copypaste)(
            names=(dataset_name,),
            filter_emptys=[use_filter],
            copypastes=[use_cp],
            dataloader_id=dataloader_id,
            reduce_memory=True,
            reduce_memory_size=1e6,
        ),
        dataset_bg=L(get_detection_dataset_dicts)(
            names=(dataset_name,),
            filter_empty=use_filter,
        )
        if use_cp
        else [[]],
        mapper=L(DatasetMapper_detr_panoptic_copypaste)(
            is_train=True,
            augmentations=[
                L(T.RandomFlip)(horizontal=True),  # flip first
                L(T.ResizeScale)(
                    min_scale=0.1, max_scale=1.0, target_height=image_size, target_width=image_size
                ),
                L(T.FixedSizeCrop)(crop_size=(image_size, image_size), pad=False),
            ],
            augmentations_with_crop=[
                L(T.RandomFlip)(horizontal=True),  # flip first
                L(T.ResizeScale)(
                    min_scale=0.1, max_scale=2.0, target_height=image_size, target_width=image_size
                ),
                L(T.FixedSizeCrop)(crop_size=(image_size, image_size), pad=False),
            ],
            image_format="RGB",
            use_instance_mask=True,
            recompute_boxes=True,
            instance_mask_format="bitmask",
            ignore_label=MetadataCatalog.get(dataset_name).get("ignore_label", None),
            stuff_classes_offset=len(MetadataCatalog.get(dataset_name).get("thing_classes", [])),
            stuff_classes_decomposition=True,
            output_dir=None,
            vis_period=12800,
            dataset_names=(dataset_name,),
            max_num_phrase=128,
            nms_thresh_phrase=0.6,
        ),
        sampler=L(RepeatFactorTrainingSampler)(
            repeat_factors=L(RepeatFactorTrainingSampler.repeat_factors_from_category_frequency)(
                dataset_dicts="${...dataset}", repeat_thresh=0.001
            )
        )
        if use_rfs
        else None,
        sampler_bg=L(RepeatFactorTrainingSampler)(
            repeat_factors=L(RepeatFactorTrainingSampler.repeat_factors_from_category_frequency)(
                dataset_dicts="${...dataset}", repeat_thresh=0.001
            )
        )
        if use_rfs and use_cp
        else None,
        total_batch_size=16,
        total_batch_size_list=[16],
        aspect_ratio_grouping=True,
        num_workers=2,
        num_datasets=1,
    )
    for dataloader_id, use_rfs, use_cp, use_filter, dataset_name in [
        [0, True, False, True, "chengguan_58_train"], # chengguan_full_shot_train # objects365_train_fixname
    ]
]




dataloader.tests = [
    L(build_detection_test_loader)(
        dataset=L(get_detection_dataset_dicts)(names="chengguan_58_test", filter_empty=False),
        mapper=L(DatasetMapper)(
            is_train=False,
            augmentations=[
                L(T.ResizeShortestEdge)(short_edge_length=image_size, max_size=image_size),
            ],
            image_format="RGB",
        ),
        num_workers=2,
    ),
]

dataloader.evaluators = [
    L(COCOEvaluator)(
        dataset_name="chengguan_58_test", # chengguan_test # objects365_val_fixname
        tasks=("bbox",),
    ),
]



model.model_vision.backbone = backbone

# train.init_checkpoint = (
#     "models/QuanSun/EVA-CLIP/EVA02_CLIP_L_336_psz14to16_s6B.pt?matching_heuristics=True"
# )
train.init_checkpoint = (
    "/home/vis/dingyuning03/baidu/personal-code/dingyuning_APE/APE/output/APE/configs/city/city_13/city_58/best_model.pth"
)

# model.model_language = L(EVA02CLIP)(
#     clip_model="EVA02-CLIP-bigE-14-plus",
#     cache_dir="models/QuanSun/EVA-CLIP/EVA02_CLIP_E_psz14_plus_s9B.pt",
#     dtype="float16",
# )
model.model_language = L(EVA02CLIP)(
    clip_model="EVA02-CLIP-bigE-14-plus",
    cache_dir="models/EVA02_CLIP_E_psz14_plus_s9B.pt",
    dtype="float32",
)
model.model_vision.embed_dim_language = 1024

model.model_vision.neck = L(ChannelMapper)(
    input_shapes={
        "p2": ShapeSpec(channels=256),
        "p3": ShapeSpec(channels=256),
        "p4": ShapeSpec(channels=256),
        "p5": ShapeSpec(channels=256),
        "p6": ShapeSpec(channels=256),
    },
    in_features=["p2", "p3", "p4", "p5", "p6"],
    out_channels=256,
    num_outs=5,
    kernel_size=1,
    norm_layer=L(nn.GroupNorm)(num_groups=32, num_channels=256),
)

model.model_vision.gcp={
    "gcp_on": False,
    "gcp_weight": 0.1,
    "query_path": ''

}



model.model_vision.mask_in_features = ["p2"]
model.model_vision.input_shapes = {
    "p2": ShapeSpec(channels=256),
    "p3": ShapeSpec(channels=256),
    "p4": ShapeSpec(channels=256),
    "p5": ShapeSpec(channels=256),
    "p6": ShapeSpec(channels=256),
}

model.model_vision.transformer.encoder.num_layers = 6
model.model_vision.transformer.decoder.num_layers = 6
model.model_vision.transformer.encoder.embed_dim = 256
model.model_vision.transformer.decoder.embed_dim = 256
model.model_vision.embed_dim = 256
model.model_vision.backbone.out_channels = 256
model.model_vision.prompts_update_rate = 0.5
model.model_vision.vision_prompts_on = True

model.model_vision.vl_mode='adapter' 

# model.model_vision.vl_mode='ensemble'

model.model_vision.update(
    _target_=DeformableDETRSegmVL,
)
model.model_vision.transformer.update(
    _target_=DeformableDetrTransformerVL,
)
model.model_vision.transformer.encoder.update(
    _target_=DeformableDetrTransformerEncoderVL,
)
model.model_vision.transformer.decoder.update(
    _target_=DeformableDetrTransformerDecoderVL,
)

model.model_vision.transformer.encoder.vl_layer = L(VisionLanguageFusion)(
    v_dim="${....embed_dim}",
    l_dim="${....embed_dim_language}",
    embed_dim=2048,
    num_heads=8,
    dropout=0.1,
    drop_path=0.0,
    init_values=1.0 / 6,
    stable_softmax_2d=True,
    clamp_min_for_underflow=True,
    clamp_max_for_overflow=True,
    use_checkpoint=True,
)
model.model_vision.transformer.encoder.use_act_checkpoint = True

model.model_vision.text_feature_bank = True
model.model_vision.text_feature_reduce_before_fusion = True
model.model_vision.text_feature_batch_repeat = True
model.model_vision.expression_cumulative_gt_class = True
model.model_vision.name_prompt_fusion_type = "zero"

# model.model_vision.num_classes = 1256
# model.model_vision.num_classes = 365
model.model_vision.num_classes = 58
model.model_vision.select_box_nums_for_evaluation = 300

criterion = model.model_vision.criterion[0]
del criterion.use_fed_loss
del criterion.get_fed_loss_cls_weights
del criterion.fed_loss_num_classes
model.model_vision.criterion = [criterion for _ in range(1)]
for criterion, num_classes in zip(
    model.model_vision.criterion, [58] # [365]
):
    criterion.num_classes = num_classes


model.model_vision.stuff_dataset_learn_thing = False
model.model_vision.stuff_prob_thing = 0.9
model.model_vision.transformer.proposal_ambiguous = 1

model.model_vision.instance_on = True
model.model_vision.semantic_on = False
model.model_vision.panoptic_on = False

train.max_iter = 100000
train.eval_period = 1000 # full=500,few=300

lr_multiplier = L(WarmupParamScheduler)(
    scheduler=L(MultiStepParamScheduler)(
        values=[1.0, 0.1],
        milestones=[900000],
        num_updates=1080000,
    ),
    warmup_length=2000 / 270000,
    warmup_method="linear",
    warmup_factor=0.001,
)

for i in range(len(dataloader.train)):
    dataloader.train[i].mapper.max_num_phrase = 128
    dataloader.train[i].mapper.nms_thresh_phrase = 0.6
    dataloader.train[i].total_batch_size = 4
    dataloader.train[i].total_batch_size_list = [4]
    dataloader.train[i].num_workers = 2

train.iter_size = 1
train.iter_loop = False
train.dataset_ratio = [1]

model.model_vision.dataset_prompts = [
    "name",
]
model.model_vision.dataset_names = [
    "city_58",
]
model.model_vision.dataset_metas = [xx for x in dataloader.train for xx in x.dataset.names]

train.output_dir = "output/" + __file__[:-3]
model.model_vision.vis_period = 5120

train.fast_dev_run.enabled = False

optimizer.lr = 5e-5

train.ddp.find_unused_parameters = False

train.amp.enabled = True
train.ddp.fp16_compression = True

model.model_vision.cafo={
            "enabled": False,
            "clip_adapter_shape": (256,256),
            "dino_adapter_shape": (384,256),
            "init_alpha": 0.6,     
            "init_beta": 1.0,
            "cls_num": 13
        }
model.model_vision.support_words={
            "enabled": False,
            "words_dict": '/home/vis/dingyuning03/baidu/personal-code/dingyuning_APE/APE/models/itfeature.pth',
            "words_num": 5,
            "support_mode": 'adapt',
            'weight': 0.5
}