"""
Train and eval functions used in main.py
"""
import math
import os
import sys
from typing import Iterable

import torch
import utils.misc as utils
from datasets.open_world_eval import OWEvaluator
from datasets.data_prefetcher import data_prefetcher
from utils.plot_utils import draw_img, CLASSES
from copy import deepcopy
from PIL import Image
import numpy as np
from tqdm import tqdm


def train_one_epoch(model: torch.nn.Module, criterion: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, nc_epoch: int, max_norm: float = 0):
    model.train()
    criterion.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    metric_logger.add_meter('class_error', utils.SmoothedValue(window_size=1, fmt='{value:.2f}'))
    metric_logger.add_meter('grad_norm', utils.SmoothedValue(window_size=1, fmt='{value:.2f}'))
    header = 'Epoch: [{}]'.format(epoch)
    prefetcher = data_prefetcher(data_loader, device, prefetch=True)
    samples, targets = prefetcher.next()
    for _ in metric_logger.log_every(range(len(data_loader)), 99999, header):
        outputs = model(samples)
        loss_dict = criterion(samples, outputs, targets, epoch) ## samples variable needed for feature selection
        weight_dict = deepcopy(criterion.weight_dict)
        ## condition for starting nc loss computation after certain epoch so that the F_cls branch has the time
        ## to learn the within classes seperation.
        if epoch < nc_epoch: 
            for k,v in weight_dict.items():
                if 'NC' in k:
                    weight_dict[k] = 0
        losses = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)

        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = utils.reduce_dict(loss_dict)
        ## Just printing NOt affectin gin loss function
        loss_dict_reduced_unscaled = {f'{k}_unscaled': v
                                      for k, v in loss_dict_reduced.items()}
        loss_dict_reduced_scaled = {k: v * weight_dict[k]
                                    for k, v in loss_dict_reduced.items() if k in weight_dict}
        losses_reduced_scaled = sum(loss_dict_reduced_scaled.values())

        loss_value = losses_reduced_scaled.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            print(loss_dict_reduced)
            sys.exit(1)

        optimizer.zero_grad()
        losses.backward()
        if max_norm > 0:
            grad_total_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
        else:
            grad_total_norm = utils.get_total_grad_norm(model.parameters(), max_norm)
        optimizer.step()

        metric_logger.update(loss=loss_value, **loss_dict_reduced_scaled, **loss_dict_reduced_unscaled)
        metric_logger.update(class_error=loss_dict_reduced['class_error'])
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
        metric_logger.update(grad_norm=grad_total_norm)

        samples, targets = prefetcher.next()
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}

## ORIGINAL FUNCTION
@torch.no_grad()
def evaluate(model, criterion, postprocessors, data_loader, base_ds, device, output_dir, args, epoch):
    model.eval()
    criterion.eval()
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test:'
    iou_types = tuple(k for k in ('segm', 'bbox') if k in postprocessors.keys())
    coco_evaluator = OWEvaluator(base_ds, iou_types, args=args)

    for samples, targets in metric_logger.log_every(data_loader, 999999, header):
        samples = samples.to(device)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        outputs = model(samples)

        orig_target_sizes = torch.stack([t["orig_size"] for t in targets], dim=0)
        results = postprocessors['bbox'](outputs, orig_target_sizes)
        res = {target['image_id'].item(): output for target, output in zip(targets, results)}
        coco_evaluator.update(res)

    metric_logger.synchronize_between_processes()
    coco_evaluator.synchronize_between_processes()
    coco_evaluator.accumulate()
    coco_evaluator.summarize(epoch)
    
    stats = {k: meter.global_avg for k, meter in metric_logger.meters.items()}
    if 'bbox' in postprocessors.keys():
        stats['coco_eval_bbox'] = coco_evaluator.coco_eval['bbox'].stats.tolist()
        
    return stats, coco_evaluator


@torch.no_grad()
def viz(model, criterion, postprocessors, data_loader, base_ds, device, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    model.eval()
    criterion.eval()

    class_names = CLASSES
    idx2name = {idx: name for idx, name in enumerate(class_names)}
 
    cnt = 0
    for samples, targets in tqdm(data_loader, desc="Viz"):
        if cnt == 1000:
            break
        cnt += 1
        
        samples = samples.to(device)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        outputs = model(samples)

        orig_target_sizes = torch.stack([t['orig_size'] for t in targets], dim=0)
        results = postprocessors['bbox'](outputs, orig_target_sizes)

        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)

        for img_tensor, target, pred in zip(samples.tensors, targets, results):
            top_k = int(target['boxes'].shape[0]) if 'boxes' in target else pred['scores'].shape[0]
            keep_k = min(top_k, int(pred['scores'].shape[0]))
            if keep_k == 0:
                continue

            top_indices = pred['scores'].sort(descending=True)[1][:keep_k]
            boxes = pred['boxes'][top_indices].detach().cpu().numpy()
            labels = pred['labels'][top_indices].detach().cpu().numpy().astype(np.int64)
            scores = pred['scores'][top_indices].detach().cpu().numpy()

            img = img_tensor.detach().cpu().permute(1, 2, 0).numpy()
            img = (img * std + mean) * 255.0
            img = np.clip(img, 0, 255).astype(np.uint8)
            canvas = Image.fromarray(img)

            draw_img(canvas, boxes, labels, scores, idx2name)

            image_id = target['image_id']
            if torch.is_tensor(image_id):
                image_id = int(image_id.flatten()[0].item())
            out_path = os.path.join(output_dir, f'img_{int(image_id)}.png')
            canvas.save(out_path)
