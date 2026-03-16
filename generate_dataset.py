import os
import random
import xml.etree.ElementTree as ET
from tqdm import tqdm
from collections import defaultdict

# ==================== 路径配置 ====================
# DIOR数据集路径
DIOR_IMAGE_DIR = "/data/my_code/dataset/DIOR/JPEGImages"  # DIOR图像目录
DIOR_ANNOTATION_DIR = "/data/my_code/dataset/DIOR/Annotations"  # DIOR标注目录
DIOR_TRAIN_TXT = "/data/my_code/dataset/DIOR/ImageSets/train.txt"  # DIOR train.txt路径
DIOR_TEST_TXT = "/data/my_code/dataset/DIOR/ImageSets/test.txt"  # DIOR test.txt路径

# DOTA数据集路径
DOTA_IMAGE_DIR = "/data/my_code/dataset/DOTA_xml/JPEGImages"  # DOTA图像目录
DOTA_ANNOTATION_DIR = "/data/my_code/dataset/DOTA_xml/Annotations"  # DOTA标注目录
DOTA_TRAIN_TXT = "/data/my_code/dataset/DOTA_xml/ImageSets/Main/trainval.txt"  # DOTA train.txt路径
DOTA_TEST_TXT = "/data/my_code/dataset/DOTA_xml/ImageSets/Main/test.txt"  # DOTA test.txt路径

# 输出路径
OUTPUT_DIR = "/data/my_code/dataset/OWDETR"  # 输出根目录

# 随机种子
RANDOM_SEED = 42

# ==================== 类别映射配置 ====================
DIOR_CLASS_MAP = {
    "airplane": "airplane",
    "ship": "ship",
    "vehicle": "vehicle",
    "storagetank": "storage-tank",
    "harbor": "harbor",
    "tenniscourt": "tennis-court",
    "bridge": "bridge",
    "baseballfield": "baseball-diamond",
    "basketballcourt": "basketball-court",
    "groundtrackfield": "ground-track-field",
    "airport": "airport",
    "chimney": "chimney",
    "dam": "dam",
    "Expressway-Service-area": "expressway-service-area",
    "Expressway-toll-station": "expressway-toll-station",
    "golffield": "golffield",
    "overpass": "overpass",
    "stadium": "stadium",
    "trainstation": "trainstation",
    "windmill": "windmill",
}

DOTA_CLASS_MAP = {
    "plane": "airplane",
    "ship": "ship",
    "small-vehicle": "vehicle",
    "large-vehicle": "vehicle",
    "storage-tank": "storage-tank",
    "harbor": "harbor",
    "tennis-court": "tennis-court",
    "bridge": "bridge",
    "baseball-diamond": "baseball-diamond",
    "basketball-court": "basketball-court",
    "ground-track-field": "ground-track-field",
    "helicopter": "helicopter",
    "swimming-pool": "swimming-pool",
    "soccer-ball-field": "soccer-ball-field",
    "roundabout": "roundabout",
}

TASK_CLASSES = {
    "task1": ["airplane", "ship", "vehicle", "harbor", "storage-tank", "tennis-court"],
    "task2": ["bridge", "baseball-diamond", "basketball-court", "ground-track-field", "swimming-pool", "windmill"],
    "task3": ["overpass", "helicopter", "expressway-service-area", "soccer-ball-field", "roundabout", "airport"],
    "task4": ["chimney", "expressway-toll-station", "stadium", "dam", "golffield", "trainstation"],
}

INSTANCES_PER_CLASS = 50


def read_txt(txt_path):
    with open(txt_path, "r") as f:
        return [line.strip() for line in f if line.strip()]


def write_txt(txt_path, file_list):
    with open(txt_path, "w") as f:
        for name in file_list:
            f.write(name + "\n")


def get_classes_in_xml(xml_path, target_classes):
    """获取XML中属于target_classes的类别（直接匹配，用于已转换的XML）"""
    tree = ET.parse(xml_path)
    root = tree.getroot()
    classes = set()
    for obj in root.findall("object"):
        name = obj.find("name").text
        if name in target_classes:
            classes.add(name)
    return classes


def get_class_instances_in_xml(xml_path):
    """获取XML中每个类别的实例数量"""
    tree = ET.parse(xml_path)
    root = tree.getroot()
    class_counts = defaultdict(int)
    for obj in root.findall("object"):
        name = obj.find("name").text
        class_counts[name] += 1
    return class_counts


def convert_xml(input_xml_path, output_xml_path, class_map):
    tree = ET.parse(input_xml_path)
    root = tree.getroot()
    for obj in root.findall("object"):
        name_elem = obj.find("name")
        if name_elem.text in class_map:
            name_elem.text = class_map[name_elem.text]
    tree.write(output_xml_path, encoding="utf-8", xml_declaration=True)


def link_images(image_dir, file_list, output_image_dir):
    for name in tqdm(file_list, desc="创建图像软链接"):
        for ext in [".jpg", ".png", ".jpeg", ".JPG", ".PNG"]:
            src = os.path.join(image_dir, name + ext)
            if os.path.exists(src):
                dst = os.path.join(output_image_dir, name + ext)
                src_abs = os.path.abspath(src)
                if not os.path.exists(dst):
                    os.symlink(src_abs, dst)
                break


def process_annotations(annotation_dir, file_list, output_annotation_dir, class_map):
    for name in tqdm(file_list, desc="处理标注"):
        src = os.path.join(annotation_dir, name + ".xml")
        dst = os.path.join(output_annotation_dir, name + ".xml")
        if os.path.exists(src):
            convert_xml(src, dst, class_map)


def generate_task_train_txt(annotation_dir, train_list, task_classes, output_txt_path):
    """从已转换的annotation_dir中筛选包含task_classes的图像"""
    filtered_list = []
    target_classes = set(task_classes)
    for name in train_list:
        xml_path = os.path.join(annotation_dir, name + ".xml")
        if os.path.exists(xml_path):
            classes = get_classes_in_xml(xml_path, target_classes)
            if classes:
                filtered_list.append(name)
    write_txt(output_txt_path, filtered_list)
    return filtered_list


def build_class_to_images(annotation_dir, image_list):
    """构建类别到图像列表的映射，同时记录每张图像中各类别的实例数"""
    class_to_images = defaultdict(list)
    image_class_counts = {}

    for name in image_list:
        xml_path = os.path.join(annotation_dir, name + ".xml")
        if os.path.exists(xml_path):
            class_counts = get_class_instances_in_xml(xml_path)
            image_class_counts[name] = class_counts
            for cls in class_counts:
                class_to_images[cls].append(name)

    return class_to_images, image_class_counts


def sample_classes(annotation_dir, sample_pool, target_classes, ft_images, class_instances, seed_offset=0):
    """
    从sample_pool中采样target_classes的图像
    ft_images: 当前已有的ft图像集合（会被修改）
    class_instances: 当前各类别实例数（会被修改）
    """
    random.seed(RANDOM_SEED + seed_offset)

    class_to_images, image_class_counts = build_class_to_images(annotation_dir, sample_pool)

    for cls in target_classes:
        if class_instances[cls] >= INSTANCES_PER_CLASS:
            continue

        needed = INSTANCES_PER_CLASS - class_instances[cls]
        candidate_images = [img for img in class_to_images.get(cls, []) if img not in ft_images]
        random.shuffle(candidate_images)

        for img in candidate_images:
            if needed <= 0:
                break
            img_counts = image_class_counts[img]
            cls_count_in_img = img_counts.get(cls, 0)
            if cls_count_in_img > 0:
                ft_images.add(img)
                for c, cnt in img_counts.items():
                    class_instances[c] += cnt
                needed -= cls_count_in_img


def generate_task2_ft(annotation_dir, task_train_lists, output_txt_path):
    """task2_ft: 从task1_train + task2_train采样，包含task1和task2类别"""
    ft_images = set()
    class_instances = defaultdict(int)

    sample_pool = list(set(task_train_lists["task1"] + task_train_lists["task2"]))
    target_classes = TASK_CLASSES["task1"] + TASK_CLASSES["task2"]

    sample_classes(annotation_dir, sample_pool, target_classes, ft_images, class_instances, seed_offset=2)

    write_txt(output_txt_path, sorted(ft_images))
    return ft_images, class_instances


def generate_task3_ft(annotation_dir, task_train_lists, output_txt_path, prev_ft_images, prev_class_instances):
    """task3_ft: 在task2_ft基础上，从task3_train采样task3类别"""
    ft_images = set(prev_ft_images)
    class_instances = defaultdict(int, prev_class_instances)

    sample_pool = task_train_lists["task3"]
    target_classes = TASK_CLASSES["task3"]

    sample_classes(annotation_dir, sample_pool, target_classes, ft_images, class_instances, seed_offset=3)

    write_txt(output_txt_path, sorted(ft_images))
    return ft_images, class_instances


def generate_task4_ft(annotation_dir, task_train_lists, output_txt_path, prev_ft_images, prev_class_instances):
    """task4_ft: 在task3_ft基础上，从task4_train采样task4类别"""
    ft_images = set(prev_ft_images)
    class_instances = defaultdict(int, prev_class_instances)

    sample_pool = task_train_lists["task4"]
    target_classes = TASK_CLASSES["task4"]

    sample_classes(annotation_dir, sample_pool, target_classes, ft_images, class_instances, seed_offset=4)

    write_txt(output_txt_path, sorted(ft_images))
    return ft_images, class_instances


def print_ft_stats(task_name, ft_images, class_instances, target_classes):
    print(f"\n{task_name}_ft.txt: {len(ft_images)}张图像")
    print(f"  类别实例统计 (目标: {INSTANCES_PER_CLASS}):")
    for cls in target_classes:
        cnt = class_instances[cls]
        status = "⚠️" if cnt < INSTANCES_PER_CLASS else "✓"
        print(f"    {cls}: {cnt} {status}")


def main():
    output_image_dir = os.path.join(OUTPUT_DIR, "JPEGImages")
    output_annotation_dir = os.path.join(OUTPUT_DIR, "Annotations")
    output_imagesets_dir = os.path.join(OUTPUT_DIR, "ImageSets")

    os.makedirs(output_image_dir, exist_ok=True)
    os.makedirs(output_annotation_dir, exist_ok=True)
    os.makedirs(output_imagesets_dir, exist_ok=True)

    dior_train = read_txt(DIOR_TRAIN_TXT)
    dior_test = read_txt(DIOR_TEST_TXT)
    dota_train = read_txt(DOTA_TRAIN_TXT)
    dota_test = read_txt(DOTA_TEST_TXT)

    print(f"DIOR: train={len(dior_train)}, test={len(dior_test)}")
    print(f"DOTA: train={len(dota_train)}, test={len(dota_test)}")

    print("\n========== 创建DIOR图像软链接 ==========")
    link_images(DIOR_IMAGE_DIR, dior_train + dior_test, output_image_dir)

    print("\n========== 创建DOTA图像软链接 ==========")
    link_images(DOTA_IMAGE_DIR, dota_train + dota_test, output_image_dir)

    print("\n========== 处理DIOR标注 ==========")
    process_annotations(DIOR_ANNOTATION_DIR, dior_train + dior_test, output_annotation_dir, DIOR_CLASS_MAP)

    print("\n========== 处理DOTA标注 ==========")
    process_annotations(DOTA_ANNOTATION_DIR, dota_train + dota_test, output_annotation_dir, DOTA_CLASS_MAP)

    print("\n========== 生成test.txt ==========")
    merged_test = dior_test + dota_test
    write_txt(os.path.join(output_imagesets_dir, "test.txt"), merged_test)
    print(f"test.txt: {len(merged_test)}张图像")

    merged_train = dior_train + dota_train

    print("\n========== 生成Task训练集 ==========")
    task_train_lists = {}
    for task_name, task_classes in TASK_CLASSES.items():
        output_txt = os.path.join(output_imagesets_dir, f"{task_name}_train.txt")
        filtered = generate_task_train_txt(
            output_annotation_dir, merged_train, task_classes, output_txt
        )
        task_train_lists[task_name] = filtered
        print(f"{task_name}_train.txt: {len(filtered)}张图像")

    print("\n========== 生成Task微调集 ==========")

    # task2_ft
    ft_images, class_instances = generate_task2_ft(
        output_annotation_dir, task_train_lists, os.path.join(output_imagesets_dir, "task2_ft.txt")
    )
    print_ft_stats("task2", ft_images, class_instances, TASK_CLASSES["task1"] + TASK_CLASSES["task2"])

    # task3_ft
    ft_images, class_instances = generate_task3_ft(
        output_annotation_dir, task_train_lists, os.path.join(output_imagesets_dir, "task3_ft.txt"),
        ft_images, class_instances
    )
    print_ft_stats("task3", ft_images, class_instances,
                   TASK_CLASSES["task1"] + TASK_CLASSES["task2"] + TASK_CLASSES["task3"])

    # task4_ft
    ft_images, class_instances = generate_task4_ft(
        output_annotation_dir, task_train_lists, os.path.join(output_imagesets_dir, "task4_ft.txt"),
        ft_images, class_instances
    )
    print_ft_stats("task4", ft_images, class_instances,
                   TASK_CLASSES["task1"] + TASK_CLASSES["task2"] + TASK_CLASSES["task3"] + TASK_CLASSES["task4"])

    print("\n========== 处理完成 ==========")
    print(f"输出目录: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()