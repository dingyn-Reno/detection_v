import os

from detectron2.data.datasets.register_coco import register_coco_instances

OBJECTS365_CATEGORIES_FIXNAME = [
    {"id": 1, "name": "Person"},
    {"id": 2, "name": "Sneakers"},

]

OBJECTS365_CATEGORIES =[
    {"id": 1, "name": "Person"},
    {"id": 2, "name": "Sneakers"},

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
    return {
        "thing_dataset_id_to_contiguous_id": thing_dataset_id_to_contiguous_id,
        "thing_classes": thing_classes,
    }


_PREDEFINED_SPLITS_OBJECTS365 = {
    "objects365_train": ("objects365/train", "objects365/annotations/objects365_train.json"),
    "objects365_val": ("objects365/val", "objects365/annotations/objects365_val.json"),
    "objects365_minival": ("objects365/val", "objects365/annotations/objects365_minival.json"),
    "objects365_train_fixname": (
        "objects365/train",
        "objects365/annotations/objects365_train_fixname.json",
    ),
    "objects365_val_fixname": (
        "objects365/val",
        "objects365/annotations/objects365_val_fixname.json",
    ),
    "objects365_minival_fixname": (
        "objects365/val",
        "objects365/annotations/objects365_minival_fixname.json",
    ),
    "objects365_train_fixname_fixmiss": (
        "objects365/train",
        "objects365/annotations/objects365_train_fixname_fixmiss.json",
    ),
    "objects365_val_fixname_fixmiss": (
        "objects365/val",
        "objects365/annotations/objects365_val_fixname_fixmiss.json",
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


if __name__.endswith(".objects365"):
    # Assume pre-defined datasets live in `./datasets`.
    _root = os.getenv("DETECTRON2_DATASETS", "datasets")
    register_all_objects365(_root)
