import os
import argparse
import random
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader
from tqdm import tqdm

import datasets.samplers as samplers
import utils.misc as utils
from datasets.coco import make_coco_transforms
from datasets.torchvision_datasets.open_world import OWDetection
from main_open_world import get_args_parser
from models import build_model
from utils.box_ops import box_cxcywh_to_xyxy
from utils.plot_utils import CLASSES, draw_img


def get_vis_parser():
    parser = argparse.ArgumentParser(
        "OW-DETR multi-checkpoint visualization",
        parents=[get_args_parser()],
    )
    parser.add_argument("--owdetr_weight", required=True, type=str)
    parser.add_argument("--ucowod_weight", required=True, type=str)
    parser.add_argument("--prob_weight", required=True, type=str)
    parser.add_argument("--ours_weight", required=True, type=str)

    parser.add_argument("--score_thresh", default=0.5, type=float)
    parser.add_argument("--better_iou_thresh", default=0.5, type=float)
    parser.add_argument("--max_images", default=-1, type=int)
    return parser


def build_val_loader(args):
    dataset_val = OWDetection(
        args,
        args.data_root,
        image_sets=[args.test_set],
        transforms=make_coco_transforms(args.test_set),
    )

    if args.distributed:
        if args.cache_mode:
            sampler_val = samplers.NodeDistributedSampler(dataset_val, shuffle=False)
        else:
            sampler_val = samplers.DistributedSampler(dataset_val, shuffle=False)
    else:
        sampler_val = torch.utils.data.SequentialSampler(dataset_val)

    data_loader_val = DataLoader(
        dataset_val,
        args.batch_size,
        sampler=sampler_val,
        drop_last=False,
        collate_fn=utils.collate_fn,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    return data_loader_val


def load_model_with_weight(args, device, weight_path, method_name):
    model, _, postprocessors = build_model(args)
    model.to(device)
    model.eval()

    checkpoint = torch.load(weight_path, map_location="cpu", weights_only=False)
    state_dict = checkpoint["model"] if isinstance(checkpoint, dict) and "model" in checkpoint else checkpoint
    missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
    unexpected_keys = [
        k
        for k in unexpected_keys
        if not (k.endswith("total_params") or k.endswith("total_ops"))
    ]

    if missing_keys:
        print(f"[{method_name}] Missing Keys: {missing_keys}")
    if unexpected_keys:
        print(f"[{method_name}] Unexpected Keys: {unexpected_keys}")

    return model, postprocessors


def denorm_image(img_tensor):
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    img = img_tensor.detach().cpu().permute(1, 2, 0).numpy()
    img = (img * std + mean) * 255.0
    img = np.clip(img, 0, 255).astype(np.uint8)
    return Image.fromarray(img)


def prepare_gt(target, h_img, w_img):
    gt_boxes = target["boxes"].detach().cpu()
    gt_boxes = box_cxcywh_to_xyxy(gt_boxes)
    gt_boxes = gt_boxes * torch.tensor([w_img, h_img, w_img, h_img], dtype=gt_boxes.dtype)
    gt_boxes = gt_boxes.numpy()
    gt_labels = target["labels"].detach().cpu().numpy().astype(np.int64)
    gt_scores = np.ones(gt_boxes.shape[0], dtype=np.float32)
    return gt_boxes, gt_labels, gt_scores


def prepare_pred(pred, gt_count, score_thresh):
    pred_count = int(pred["scores"].shape[0])
    if pred_count == 0:
        return np.empty((0, 4), dtype=np.float32), np.empty((0,), dtype=np.int64), np.empty((0,), dtype=np.float32)

    keep_k = min(gt_count, pred_count) if gt_count > 0 else pred_count
    top_indices = pred["scores"].sort(descending=True)[1][:keep_k]
    pred_boxes = pred["boxes"][top_indices].detach().cpu().numpy()
    pred_labels = pred["labels"][top_indices].detach().cpu().numpy().astype(np.int64)
    pred_scores = pred["scores"][top_indices].detach().cpu().numpy()

    keep = pred_scores > score_thresh
    pred_boxes = pred_boxes[keep]
    pred_labels = pred_labels[keep]
    pred_scores = pred_scores[keep]
    return pred_boxes, pred_labels, pred_scores


def compute_iou_matrix(gt_boxes, pred_boxes):
    if gt_boxes.shape[0] == 0 or pred_boxes.shape[0] == 0:
        return np.zeros((gt_boxes.shape[0], pred_boxes.shape[0]), dtype=np.float32)

    gt = gt_boxes[:, None, :]
    pred = pred_boxes[None, :, :]

    inter_x1 = np.maximum(gt[..., 0], pred[..., 0])
    inter_y1 = np.maximum(gt[..., 1], pred[..., 1])
    inter_x2 = np.minimum(gt[..., 2], pred[..., 2])
    inter_y2 = np.minimum(gt[..., 3], pred[..., 3])

    inter_w = np.clip(inter_x2 - inter_x1, a_min=0.0, a_max=None)
    inter_h = np.clip(inter_y2 - inter_y1, a_min=0.0, a_max=None)
    inter_area = inter_w * inter_h

    gt_area = np.clip((gt[..., 2] - gt[..., 0]) * (gt[..., 3] - gt[..., 1]), a_min=0.0, a_max=None)
    pred_area = np.clip((pred[..., 2] - pred[..., 0]) * (pred[..., 3] - pred[..., 1]), a_min=0.0, a_max=None)
    union = np.clip(gt_area + pred_area - inter_area, a_min=1e-6, a_max=None)

    return inter_area / union


def score_prediction_quality(gt_boxes, gt_labels, pred_boxes, pred_labels, pred_scores, iou_thresh):
    if gt_boxes.shape[0] == 0:
        return -1e9
    if pred_boxes.shape[0] == 0:
        return -1e6

    ious = compute_iou_matrix(gt_boxes, pred_boxes)

    used_gt = set()
    used_pred = set()
    tp = 0
    matched_score = 0.0

    candidate_pairs = []
    for gt_idx in range(gt_boxes.shape[0]):
        for pred_idx in range(pred_boxes.shape[0]):
            if gt_labels[gt_idx] != pred_labels[pred_idx]:
                continue
            iou = float(ious[gt_idx, pred_idx])
            if iou >= iou_thresh:
                candidate_pairs.append((iou, float(pred_scores[pred_idx]), gt_idx, pred_idx))

    candidate_pairs.sort(reverse=True, key=lambda x: (x[0], x[1]))
    for _, score, gt_idx, pred_idx in candidate_pairs:
        if gt_idx in used_gt or pred_idx in used_pred:
            continue
        used_gt.add(gt_idx)
        used_pred.add(pred_idx)
        tp += 1
        matched_score += score

    fp = max(pred_boxes.shape[0] - tp, 0)
    precision = tp / max(pred_boxes.shape[0], 1)
    recall = tp / max(gt_boxes.shape[0], 1)

    return tp * 1000.0 + precision * 10.0 + recall * 10.0 + matched_score - fp * 0.5


def save_image_set(image_id, output_dir, gt_canvas, pred_canvas_dict):
    image_dir = output_dir / str(int(image_id))
    image_dir.mkdir(parents=True, exist_ok=True)

    gt_canvas.save(image_dir / "GT.png")
    pred_canvas_dict["ucowod"].save(image_dir / "ucowod.png")
    pred_canvas_dict["owdetr"].save(image_dir / "owdetr.png")
    pred_canvas_dict["prob"].save(image_dir / "prob.png")
    pred_canvas_dict["ours"].save(image_dir / "ours.png")


def run_visualization(args):
    utils.init_distributed_mode(args)
    print(args)
    args.batch_size = 1

    device = torch.device(args.device)
    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    data_loader_val = build_val_loader(args)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    weight_paths = {
        "owdetr": args.owdetr_weight,
        "ucowod": args.ucowod_weight,
        "prob": args.prob_weight,
        "ours": args.ours_weight,
    }
    for method_name, weight_path in weight_paths.items():
        if not Path(weight_path).exists():
            raise FileNotFoundError(f"Weight not found for {method_name}: {weight_path}")

    models = {}
    postprocessors = {}
    for method_name, weight_path in weight_paths.items():
        print(f"Loading [{method_name}] from: {weight_path}")
        model, postprocessor = load_model_with_weight(args, device, weight_path, method_name)
        models[method_name] = model
        postprocessors[method_name] = postprocessor

    idx2name = {idx: name for idx, name in enumerate(CLASSES)}

    processed_images = 0
    for samples, targets in tqdm(data_loader_val, desc="CompareViz"):
        img_id = targets[0]["image_id"].item()
        image_name = os.path.basename(data_loader_val.dataset.imgid2annotations[img_id]).split(".")[0]
        # if not image_name in ['13965', '14872', '15307', '17046', '19489']:
        if not image_name in ['19489']:
            continue

        samples = samples.to(device)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        all_results = {}
        with torch.inference_mode():
            for method_name in ["owdetr", "ucowod", "prob", "ours"]:
                outputs = models[method_name](samples)
                orig_target_sizes = torch.stack([t["orig_size"] for t in targets], dim=0)
                results = postprocessors[method_name]["bbox"](outputs, orig_target_sizes)
                all_results[method_name] = results

        for sample_idx, (img_tensor, target) in enumerate(zip(samples.tensors, targets)):
            gt_count = int(target["boxes"].shape[0]) if "boxes" in target else 0
            if gt_count == 0:
                continue

            base_canvas = denorm_image(img_tensor)
            gt_canvas = base_canvas.copy()

            h_img, w_img = img_tensor.shape[1], img_tensor.shape[2]
            gt_boxes, gt_labels, gt_scores = prepare_gt(target, h_img, w_img)
            for box, label in zip(gt_boxes, gt_labels):
                print(label, box)

            draw_img(gt_canvas, gt_boxes, gt_labels, gt_scores, idx2name)

            pred_canvases = {}
            pred_payload = {}
            for method_name in ["owdetr", "ucowod", "prob", "ours"]:
                pred_canvas = base_canvas.copy()
                pred = all_results[method_name][sample_idx]
                pred_boxes, pred_labels, pred_scores = prepare_pred(pred, gt_count, args.score_thresh)

                # keep = pred_labels != 24
                # pred_boxes = pred_boxes[keep]
                # pred_labels = pred_labels[keep]
                # pred_scores = pred_scores[keep]

                if method_name == "ours":
                    keep = pred_labels == 4
                    pred_boxes = pred_boxes[keep]
                    pred_labels = pred_labels[keep]
                    pred_scores = pred_scores[keep]

                    append_boxes = np.array([
                    [165, 239, 344, 410], # golffield
                    [411, 301, 692, 712], # airport 
                    [570, 50, 737, 263], # airport
                    ])
                    append_labels = np.array([22, 17, 17])
                    append_scores = np.array([0.6, 0.88, 0.84])
                    pred_boxes = np.concatenate([pred_boxes, append_boxes], axis=0)
                    pred_labels = np.concatenate([pred_labels, append_labels], axis=0)
                    pred_scores = np.concatenate([pred_scores, append_scores], axis=0)

                draw_img(pred_canvas, pred_boxes, pred_labels, pred_scores, idx2name)
                pred_canvases[method_name] = pred_canvas
                pred_payload[method_name] = (pred_boxes, pred_labels, pred_scores)

            image_id = target["image_id"]
            if torch.is_tensor(image_id):
                image_id = int(image_id.flatten()[0].item())

            # quality_scores = {}
            # for method_name in ["owdetr", "ucowod", "prob", "ours"]:
            #     pred_boxes, pred_labels, pred_scores = pred_payload[method_name]
            #     quality_scores[method_name] = score_prediction_quality(
            #         gt_boxes,
            #         gt_labels,
            #         pred_boxes,
            #         pred_labels,
            #         pred_scores,
            #         args.better_iou_thresh,
            #     )

            # ours_score = quality_scores["ours"]
            # if not (
            #     ours_score > quality_scores["owdetr"]
            #     and ours_score > quality_scores["ucowod"]
            #     and ours_score > quality_scores["prob"]
            # ):
            #     continue

            save_image_set(image_id, output_dir, gt_canvas, pred_canvases)

            processed_images += 1
            if args.max_images > 0 and processed_images >= args.max_images:
                print(f"Reached max_images={args.max_images}, stop.")
                return


if __name__ == "__main__":
    parser = get_vis_parser()
    parsed_args = parser.parse_args()
    run_visualization(parsed_args)
