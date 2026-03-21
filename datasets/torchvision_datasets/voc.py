import os
import xml.etree.ElementTree as ET

import torch
from PIL import Image
from torch.utils.data import Dataset

from ..coco import ConvertCocoPolysToMask, make_coco_transforms


DIOR = (
    "airplane", "baseballfield", "bridge", "groundtrackfield", "vehicle",
    "ship", "tenniscourt", "airport", "chimney", "dam",
    "basketballcourt", "Expressway-Service-area", "Expressway-toll-station", "golffield", "harbor",
    "overpass", "stadium", "storagetank", "trainstation", "windmill",
)


class DIORDataset(Dataset):
    name2idx = {name: idx for idx, name in enumerate(DIOR)}
    root = "dataset/DIOR"

    def __init__(self, mode="train"):
        super().__init__()

        self.is_train = mode == "train"
        self.img_dir = os.path.join(self.root, "JPEGImages")
        self.xml_dir = os.path.join(self.root, "Annotations")

        self.transform = make_coco_transforms("train" if self.is_train else "val")

        txt_path = os.path.join(self.root, "ImageSets", mode + ".txt")
        with open(txt_path, "r") as f:
            self.xml_list = [os.path.join(self.xml_dir, line.strip("\n") + ".xml") for line in f.readlines()]

    def __len__(self):
        return len(self.xml_list)

    def __getitem__(self, idx):
        (h, w), target = self.index(idx)
        img = Image.open(os.path.join(self.img_dir, target["file_name"])).convert("RGB")

        target["orig_size"] = torch.as_tensor([int(h), int(w)])
        target["size"] = torch.as_tensor([int(h), int(w)])
        if self.transform is not None:
            img, target = self.transform(img, target)
        return img, target

    def index(self, idx):
        xml_path = self.xml_list[idx]
        data = self.get_data(xml_path)

        boxes, labels, iscrowd = [], [], []
        for obj in data.get("object", []):
            obj_class = self.name2idx[obj["name"]]

            xmin = float(obj["bndbox"]["xmin"])
            xmax = float(obj["bndbox"]["xmax"])
            ymin = float(obj["bndbox"]["ymin"])
            ymax = float(obj["bndbox"]["ymax"])
            if xmax <= xmin or ymax <= ymin:
                continue

            boxes.append([xmin, ymin, xmax, ymax])
            labels.append(obj_class)
            iscrowd.append(0)

        boxes = torch.as_tensor(boxes, dtype=torch.float32).reshape(-1, 4)
        labels = torch.as_tensor(labels, dtype=torch.int64)
        iscrowd = torch.as_tensor(iscrowd, dtype=torch.int64)
        image_id = torch.tensor([idx])
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])

        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["image_id"] = image_id
        target["area"] = area
        target["iscrowd"] = iscrowd
        target["file_name"] = data["filename"]

        h, w = int(data["size"]["height"]), int(data["size"]["width"])

        return (h, w), target

    @staticmethod
    def get_data(xml_path):
        def parse_xml_to_dict(xml):
            """
            将xml文件解析成字典形式，参考tensorflow的recursive_parse_xml_to_dict

            Args:
                xml: xml tree obtained by parsing XML file contents using lxml.etree

            Returns:
                Python dictionary holding XML contents.
            """

            if len(xml) == 0:  # 遍历到底层，直接返回tag对应的信息
                return {xml.tag: xml.text}

            result = {}
            for child in xml:
                child_result = parse_xml_to_dict(child)  # 递归遍历标签信息
                if child.tag != "object":
                    result[child.tag] = child_result[child.tag]
                else:
                    if (
                        child.tag not in result
                    ):  # 因为object可能有多个，所以需要放入列表里
                        result[child.tag] = []
                    result[child.tag].append(child_result[child.tag])
            return {xml.tag: result}

        with open(xml_path, "r") as f:
            xml_str = f.read()
        xml = ET.fromstring(xml_str)

        return parse_xml_to_dict(xml)["annotation"]


class DIORDatasetCOCO(Dataset):
    def __init__(self, coco_ds, is_train=True):
        self.coco = coco_ds
        self.ids = list(sorted(self.coco.imgs.keys()))
        self.root = "dataset/DIOR/JPEGImages"
        self.transforms = make_coco_transforms("train" if is_train else "val")
        self.prepare = ConvertCocoPolysToMask(return_masks=False)
    
    def __getitem__(self, index):
        coco = self.coco
        img_id = self.ids[index]
        ann_ids = coco.getAnnIds(imgIds=img_id)
        target = {
            "image_id": img_id,
            "annotations": coco.loadAnns(ann_ids),
        }

        path = coco.loadImgs(img_id)[0]["file_name"]
        img = Image.open(os.path.join(self.root, path)).convert("RGB")
        img, target = self.prepare(img, target)
        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return img, target
    
    def __len__(self):
        return len(self.ids)