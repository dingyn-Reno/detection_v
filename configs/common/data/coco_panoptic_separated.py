import detectron2.data.transforms as T
from detectron2.config import LazyCall as L
from detectron2.data import (
    DatasetMapper,
    MetadataCatalog,
    build_detection_test_loader,
    build_detection_train_loader,
    get_detection_dataset_dicts,
)
from detectron2.evaluation import COCOEvaluator, COCOPanopticEvaluator, SemSegEvaluator
from omegaconf import OmegaConf
from ape.data import DatasetMapper_detr_panoptic

dataloader = OmegaConf.create()

dataloader.train = L(build_detection_train_loader)(
    dataset=L(get_detection_dataset_dicts)(
        names="coco_2017_train_panoptic_separated", filter_empty=True
    ),
    mapper=L(DatasetMapper_detr_panoptic)(
        is_train=True,
        augmentations=[
            L(T.RandomFlip)(),
            L(T.ResizeShortestEdge)(
                short_edge_length=(480, 512, 544, 576, 608, 640, 672, 704, 736, 768, 800),
                max_size=1333,
                sample_style="choice",
            ),
        ],
        augmentations_with_crop=[
            L(T.RandomFlip)(),
            L(T.ResizeShortestEdge)(
                short_edge_length=(400, 500, 600),
                sample_style="choice",
            ),
            L(T.RandomCrop)(
                crop_type="absolute_range",
                crop_size=(384, 600),
            ),
            L(T.ResizeShortestEdge)(
                short_edge_length=(480, 512, 544, 576, 608, 640, 672, 704, 736, 768, 800),
                max_size=1333,
                sample_style="choice",
            ),
        ],
        image_format="RGB",
        use_instance_mask=True,
        recompute_boxes=True,
        instance_mask_format="bitmask",
        ignore_label=MetadataCatalog.get("coco_2017_train_panoptic_stuffonly").ignore_label,
        stuff_classes_offset=80,
        stuff_classes_decomposition=True,
        dataset_names="${..dataset.names}",
    ),
    total_batch_size=16,
    aspect_ratio_grouping=True,
    num_workers=16,
)

dataloader.test = L(build_detection_test_loader)(
    dataset=L(get_detection_dataset_dicts)(
        names="coco_2017_val_panoptic_with_sem_seg", filter_empty=False
    ),
    mapper=L(DatasetMapper)(
        is_train=False,
        augmentations=[
            L(T.ResizeShortestEdge)(short_edge_length=800, max_size=1333),
        ],
        image_format="${...train.mapper.image_format}",
    ),
    num_workers=4,
)

dataloader.evaluator = [
    L(COCOEvaluator)(
        dataset_name="${...test.dataset.names}",
    ),
    L(SemSegEvaluator)(
        dataset_name="${...test.dataset.names}",
    ),
    L(COCOPanopticEvaluator)(
        dataset_name="${...test.dataset.names}",
    ),
]
