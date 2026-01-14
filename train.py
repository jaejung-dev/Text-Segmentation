import os
import argparse
import sys

import json

import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import matplotlib.pyplot as plt
import cv2
import random
from typing import Dict, List, Tuple
import time
import datetime
from pathlib import Path
from PIL import Image
from tqdm import tqdm

from hi_sam.modeling.build import model_registry
from hi_sam.modeling.loss import loss_masks, loss_hi_masks, loss_iou_mse, loss_hi_iou_mse
from hi_sam.modeling.predictor import SamPredictor
from hi_sam.data.dataloader import get_im_gt_name_dict, create_dataloaders, train_transforms, eval_transforms, custom_collate_fn
from hi_sam.evaluation import Evaluator
import utils.misc as misc
import warnings
warnings.filterwarnings("ignore")


def tensor_to_uint8_image(tensor: torch.Tensor) -> np.ndarray:
    array = tensor.detach().cpu().permute(1, 2, 0).float().numpy()
    if array.max() <= 1.0:
        array = array * 255.0
    array = np.clip(array, 0.0, 255.0).astype(np.uint8)
    return array


def mask_tensor_to_bool(mask_tensor: torch.Tensor, threshold: float) -> np.ndarray:
    mask = mask_tensor.detach().cpu()
    if mask.dim() == 3:
        mask = mask.squeeze(0)
    if mask.dim() == 3:
        mask, _ = torch.max(mask, dim=0)
    mask_np = mask.float().numpy()
    return mask_np > threshold


def save_pair_visualization(image_rgb: np.ndarray, mask_bool: np.ndarray, save_path: str, pair: bool = True) -> None:
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    mask_uint8 = (mask_bool.astype(np.uint8)) * 255
    if pair:
        mask_rgb = np.repeat(mask_uint8[:, :, None], 3, axis=2)
        pair_image = np.concatenate([image_rgb, mask_rgb], axis=1)
        Image.fromarray(pair_image).save(save_path)
    else:
        Image.fromarray(mask_uint8).save(save_path)


def resize_long_side(image: np.ndarray, max_long_side: int = 2048) -> Tuple[np.ndarray, float]:
    h, w = image.shape[:2]
    max_dim = max(h, w)
    if max_dim <= max_long_side:
        return image, 1.0
    scale = max_long_side / float(max_dim)
    new_h = max(int(round(h * scale)), 1)
    new_w = max(int(round(w * scale)), 1)
    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    return resized, scale


def resize_mask_to_shape(mask: np.ndarray, shape: Tuple[int, int]) -> np.ndarray:
    target_h, target_w = shape
    if mask.shape[0] == target_h and mask.shape[1] == target_w:
        return mask
    resized = cv2.resize(mask.astype(np.uint8), (target_w, target_h), interpolation=cv2.INTER_NEAREST)
    return resized.astype(bool)


def patchify_sliding(image: np.ndarray, patch_size: int = 512, stride: int = 256):
    h, w = image.shape[:2]
    patch_list = []
    h_slice_list = []
    w_slice_list = []
    for j in range(0, h, stride):
        start_h, end_h = j, j + patch_size
        if end_h > h:
            start_h = max(h - patch_size, 0)
            end_h = h
        for i in range(0, w, stride):
            start_w, end_w = i, i + patch_size
            if end_w > w:
                start_w = max(w - patch_size, 0)
                end_w = w
            h_slice = slice(start_h, end_h)
            w_slice = slice(start_w, end_w)
            h_slice_list.append(h_slice)
            w_slice_list.append(w_slice)
            patch_list.append(image[h_slice, w_slice])
    return patch_list, h_slice_list, w_slice_list


def unpatchify_sliding(patch_list, h_slice_list, w_slice_list, ori_size):
    assert len(ori_size) == 2
    whole_logits = np.zeros(ori_size)
    assert len(patch_list) == len(h_slice_list)
    assert len(h_slice_list) == len(w_slice_list)
    for idx in range(len(patch_list)):
        h_slice = h_slice_list[idx]
        w_slice = w_slice_list[idx]
        whole_logits[h_slice, w_slice] += patch_list[idx]
    return whole_logits


def run_patch_inference(predictor, image_rgb: np.ndarray) -> np.ndarray:
    processed_image, _ = resize_long_side(image_rgb, max_long_side=2048)
    processed_shape = processed_image.shape[:2]
    max_dim = max(processed_shape)
    patch_config = None
    if max_dim > 1024:
        patch_config = {"patch_size": 1024, "stride": 768}
    else:
        patch_config = {"patch_size": 512, "stride": 384}

    mask_logits = None
    if patch_config:
        patch_list, h_slice_list, w_slice_list = patchify_sliding(
            processed_image,
            patch_size=patch_config["patch_size"],
            stride=patch_config["stride"],
        )
        logits_list = []
        ones_list = []
        for patch in patch_list:
            predictor.set_image(patch)
            _, hr_mask, _, _ = predictor.predict(multimask_output=False, return_logits=True)
            logits = hr_mask[0]
            logits_list.append(logits)
            ones_list.append(np.ones_like(logits, dtype=np.float32))
        if logits_list:
            logits_sum = unpatchify_sliding(logits_list, h_slice_list, w_slice_list, processed_shape)
            counts = unpatchify_sliding(ones_list, h_slice_list, w_slice_list, processed_shape)
            counts[counts == 0] = 1.0
            mask_logits = logits_sum / counts

    predictor.set_image(processed_image)
    _, hr_mask_full, _, _ = predictor.predict(multimask_output=False, return_logits=True)
    full_logits = hr_mask_full[0]
    mask_logits = full_logits if mask_logits is None else (mask_logits + full_logits)

    mask_bool = mask_logits > predictor.model.mask_threshold
    if processed_shape != image_rgb.shape[:2]:
        mask_bool = resize_mask_to_shape(mask_bool, image_rgb.shape[:2])
    return mask_bool


def get_model_device(model) -> torch.device:
    if hasattr(model, "device"):
        return model.device
    if hasattr(model, "module") and hasattr(model.module, "device"):
        return model.module.device
    return next(model.parameters()).device


SUPPORTED_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff', '.webp')


def collect_sample_image_paths(im_dir: str, im_ext: str, limit: int) -> List[str]:
    """
    Utility to grab up to `limit` random image paths for qualitative previews.
    """
    if limit <= 0:
        return []
    directory = Path(im_dir)
    if not directory.is_dir():
        return []

    suffix = (im_ext or "").lower()
    candidates: List[Path] = []
    try:
        for entry in directory.iterdir():
            if not entry.is_file():
                continue
            if suffix and entry.suffix.lower() != suffix:
                continue
            candidates.append(entry)
    except FileNotFoundError:
        return []

    if not candidates:
        return []

    if limit >= len(candidates):
        chosen = candidates
    else:
        chosen = random.sample(candidates, k=limit)

    return [str(path) for path in sorted(chosen)]


def generate_validation_preview(
    args,
    model,
    val_dataset,
    global_step: int,
    output_dir: str,
) -> None:
    if val_dataset is None or len(val_dataset) == 0:
        return

    sample = val_dataset[random.randint(0, len(val_dataset) - 1)]
    image_tensor = sample['image']
    ori_label = sample.get('ori_label')
    original_size = ori_label.shape[-2:] if ori_label is not None else image_tensor.shape[-2:]

    base_model = model.module if hasattr(model, "module") else model
    device = get_model_device(base_model)
    batched_input = [{
        'image': image_tensor.to(device).contiguous(),
        'original_size': original_size,
    }]

    was_training = base_model.training
    base_model.eval()
    with torch.no_grad():
        results = base_model(batched_input, multimask_output=False)
    if was_training:
        base_model.train()

    if isinstance(results, tuple):
        if len(results) >= 5:
            hr_masks = results[4]
        elif len(results) >= 2:
            hr_masks = results[1]
        else:
            hr_masks = results[0]
    else:
        hr_masks = results

    threshold = float(
        getattr(model.module, "mask_threshold", getattr(model, "mask_threshold", 0.0))
    )
    mask_bool = mask_tensor_to_bool(hr_masks[0], threshold)

    image_path = sample.get('ori_im_path')
    if image_path and os.path.isfile(image_path):
        image_bgr = cv2.imread(image_path)
        if image_bgr is not None:
            image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        else:
            image_rgb = tensor_to_uint8_image(image_tensor)
    else:
        image_rgb = tensor_to_uint8_image(image_tensor)

    save_path = os.path.join(output_dir, f"step_{global_step:06d}.png")
    save_pair_visualization(image_rgb, mask_bool, save_path, pair=True)


def run_epoch_sample_inference(
    args,
    model,
    image_paths: List[str],
    output_dir: str,
) -> None:
    if not image_paths:
        return

    os.makedirs(output_dir, exist_ok=True)
    base_model = model.module if hasattr(model, "module") else model
    was_training = base_model.training
    base_model.eval()
    predictor = SamPredictor(base_model)
    predictor.model.to(get_model_device(base_model))

    with torch.no_grad():
        for image_path in image_paths:
            image_bgr = cv2.imread(str(image_path))
            if image_bgr is None:
                continue
            image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
            mask_bool = run_patch_inference(predictor, image_rgb)
            stem = Path(image_path).stem
            save_path = os.path.join(output_dir, f"{stem}.png")
            save_pair_visualization(image_rgb, mask_bool, save_path, pair=True)

    if was_training:
        base_model.train()


def get_args_parser():
    parser = argparse.ArgumentParser('Hi-SAM', add_help=False)

    parser.add_argument("--output", type=str, default="work_dirs/", 
                        help="Path to the directory where masks and checkpoints will be output")
    parser.add_argument("--model-type", type=str, default="vit_h", 
                        help="The type of model to load, in ['vit_h', 'vit_l', 'vit_b']")
    parser.add_argument("--checkpoint", type=str, default=os.path.join('pretrained_checkpoint', 'sam_tss_h_textseg.pth'),
                        help="The path to the SAM checkpoint to use for mask generation.")
    parser.add_argument("--device", type=str, default="cuda", 
                        help="The device to run generation on.")
    parser.add_argument("--train_datasets", type=str, nargs='+', default=['crello_train'])
    parser.add_argument("--val_datasets", type=str, nargs='+', default=['crello_val'])
    parser.add_argument("--test_datasets", type=str, nargs='+', default=[],
                        help="Dataset names used purely for reporting test metrics each eval period.")
    parser.add_argument("--hier_det", action='store_true',
                        help="If False, only text stroke segmentation.")

    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--lr', default=1e-4, type=float)
    parser.add_argument('--lr_charac_mask_decoder_name', default=["mask_decoder"], type=str, nargs='+')
    parser.add_argument('--lr_charac_mask_decoder', default=1e-4, type=float)
    parser.add_argument('--start_epoch', default=0, type=int)
    parser.add_argument('--lr_drop_epoch', default=70, type=int)
    parser.add_argument('--max_epoch_num', default=70, type=int)
    parser.add_argument('--input_size', default=[1024, 1024], type=list)
    parser.add_argument('--batch_size_train', default=8, type=int)
    parser.add_argument('--batch_size_valid', default=1, type=int)
    parser.add_argument('--valid_period', default=1, type=int)
    parser.add_argument('--model_save_fre', default=100, type=int)

    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')
    parser.add_argument('--rank', default=0, type=int,
                        help='number of distributed processes')
    parser.add_argument('--local_rank', type=int, help='local rank for dist')
    parser.add_argument('--find_unused_params', action='store_true')

    parser.add_argument('--eval', action='store_true')

    # self-prompting
    parser.add_argument('--attn_layers', default=1, type=int,
                        help='The number of image to token cross attention layers in model_aligner')
    parser.add_argument('--prompt_len', default=12, type=int, help='The number of prompt token')
    parser.add_argument('--crello_root', type=str, default="/mnt/local/crello_Hi-SAM",
                        help='Root directory for the Crello Hi-SAM dataset.')
    parser.add_argument('--lica_root', type=str, default="/mnt/local/lica_Hi-SAM",
                        help='Root directory for the LiCA Hi-SAM dataset.')
    parser.add_argument('--lica_test_im_dir', type=str, default="/home/ubuntu/data/text_segmentation_dataset/val_images",
                        help='Image directory for the LiCA shared test split.')
    parser.add_argument('--lica_test_gt_dir', type=str, default="/home/ubuntu/data/text_segmentation_dataset/vali_gt",
                        help='Mask directory for the LiCA shared test split.')
    parser.add_argument('--test_preview_limit', type=int, default=50,
                        help='Number of qualitative predictions to store per test dataset (0 disables).')

    return parser.parse_args()


def main(train_datasets, valid_datasets, test_datasets, args):

    misc.init_distributed_mode(args)
    print('world size: {}'.format(args.world_size))
    print('rank: {}'.format(args.rank))
    print('local_rank: {}'.format(args.local_rank))
    print("args: " + str(args) + '\n')

    seed = args.seed + misc.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    ### --- Step 1: Train or Valid dataset ---
    if not args.eval:
        print("--- create training dataloader ---")
        train_datasets_names = [train_ds["name"] for train_ds in train_datasets]
        train_im_gt_list = get_im_gt_name_dict(train_datasets, flag="train")
        train_dataloaders, train_datasets = create_dataloaders(
            train_im_gt_list,
            my_transforms=train_transforms,
            batch_size=args.batch_size_train,
            training=True,
            hier_det=args.hier_det,
            collate_fn=custom_collate_fn
        )
        print(len(train_dataloaders), " train dataloaders created")

    print("--- create valid dataloader ---")
    valid_datasets_names = [val_ds["name"] for val_ds in valid_datasets]
    valid_im_gt_list = get_im_gt_name_dict(valid_datasets, flag="valid")
    valid_dataloaders, valid_datasets = create_dataloaders(
        valid_im_gt_list,
        my_transforms=eval_transforms,
        batch_size=args.batch_size_valid,
        training=False
    )
    print(len(valid_dataloaders), " valid dataloaders created")

    test_datasets_names: List[str] = []
    test_dataloaders = []
    test_sample_image_paths: Dict[str, List[str]] = {}
    test_preview_dir = os.path.join(args.output, "test_preview")
    if test_datasets:
        print("--- create test dataloader ---")
        test_datasets_names = [test_ds["name"] for test_ds in test_datasets]
        test_im_gt_list = get_im_gt_name_dict(test_datasets, flag="test")
        test_dataloaders, _ = create_dataloaders(
            test_im_gt_list,
            my_transforms=eval_transforms,
            batch_size=args.batch_size_valid,
            training=False
        )
        print(len(test_dataloaders), " test dataloaders created")
        if args.test_preview_limit > 0:
            for test_ds in test_datasets:
                sample_paths = collect_sample_image_paths(
                    test_ds["im_dir"],
                    test_ds.get("im_ext", ".png"),
                    args.test_preview_limit,
                )
                if sample_paths:
                    test_sample_image_paths[test_ds["name"]] = sample_paths
    else:
        test_preview_dir = None
    
    ### --- Step 2: DistributedDataParallel---
    model = model_registry[args.model_type](args=args)
    if torch.cuda.is_available():
        model.cuda()
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], find_unused_parameters=args.find_unused_params)
    model_without_ddp = model.module
 
    ### --- Step 3: Train or Evaluate ---
    if not args.eval:
        n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print('number of trainable params: ' + str(n_parameters))

        def match_name_keywords(n, name_keywords):
            out = False
            for b in name_keywords:
                if b in n:
                    out = True
                    break
            return out

        param_dicts = [
            {
                "params": [p for n, p in model_without_ddp.named_parameters()
                           if not match_name_keywords(n, args.lr_charac_mask_decoder_name) and p.requires_grad],
                "lr": args.lr
            },
            {
                "params": [p for n, p in model_without_ddp.named_parameters()
                           if match_name_keywords(n, args.lr_charac_mask_decoder_name) and p.requires_grad],
                "lr": args.lr_charac_mask_decoder
            }
        ]
        optimizer = optim.AdamW(param_dicts, lr=args.lr, betas=(0.9, 0.999), weight_decay=0.05)
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, args.lr_drop_epoch)
        lr_scheduler.last_epoch = args.start_epoch
        train(
            args,
            model,
            optimizer,
            train_dataloaders,
            train_datasets_names,
            lr_scheduler,
            valid_dataloaders,
            valid_datasets_names,
            valid_datasets,
            test_dataloaders,
            test_datasets_names,
            test_sample_image_paths,
            test_preview_dir,
        )
    else:
        print("restore model from:", args.checkpoint)
        evaluate(args, model, valid_dataloaders, valid_datasets_names)
        if test_dataloaders:
            evaluate(args, model, test_dataloaders, test_datasets_names)


def train(
    args,
    model,
    optimizer,
    train_dataloaders,
    train_datasets_names,
    lr_scheduler,
    valid_dataloaders,
    valid_datasets_names,
    valid_datasets,
    test_dataloaders=None,
    test_datasets_names=None,
    test_sample_image_paths=None,
    test_preview_dir=None,
):
    if misc.is_main_process():
        os.makedirs(args.output, exist_ok=True)

    epoch_start = args.start_epoch
    epoch_num = args.max_epoch_num
    train_num = len(train_dataloaders)
    best_iou = [-1 for _ in range(len(valid_datasets_names))]
    test_dataloaders = test_dataloaders or []
    test_datasets_names = test_datasets_names or []
    test_sample_image_paths = test_sample_image_paths or {}
    test_preview_dir = test_preview_dir or os.path.join(args.output, "test_preview")

    val_preview_dataset = valid_datasets[0] if isinstance(valid_datasets, list) and len(valid_datasets) > 0 else None
    val_preview_dir = os.path.join(args.output, "val_preview")
    epoch_samples_root = os.path.join(args.output, "epoch_samples")
    sample_image_dir = Path("/home/ubuntu/jjseol/data/100_sample/images")
    sample_image_paths: List[str] = []
    if sample_image_dir.is_dir():
        sample_image_paths = sorted(
            str(p)
            for p in sample_image_dir.iterdir()
            if p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS
        )
    # fallback: collect up to 100 samples from train datasets if local folder is absent/empty
    if not sample_image_paths:
        for train_ds in train_datasets:
            remaining = 100 - len(sample_image_paths)
            if remaining <= 0:
                break
            sample_image_paths.extend(
                collect_sample_image_paths(
                    train_ds.get("im_dir", ""),
                    train_ds.get("im_ext", ".png"),
                    remaining,
                )
            )
        sample_image_paths = sample_image_paths[:100]
    train_preview_dir = os.path.join(args.output, "train_preview")

    if misc.is_main_process():
        os.makedirs(val_preview_dir, exist_ok=True)
        os.makedirs(epoch_samples_root, exist_ok=True)
        os.makedirs(train_preview_dir, exist_ok=True)
        if test_sample_image_paths:
            os.makedirs(test_preview_dir, exist_ok=True)

    model.train()
    _ = model.to(device=args.device)
    from torch.cuda.amp import autocast, GradScaler
    gradsclaler = GradScaler()

    num_batches_per_epoch = len(train_dataloaders)
    global_step = epoch_start * num_batches_per_epoch

    if misc.is_main_process():
        # Quick sanity check: save a few samples from the first training batch (post-augmentation).
        try:
            # Save up to 100 previews from several early batches without consuming
            # the main training iterator (new iterator here).
            saved = 0
            preview_iter = iter(train_dataloaders)
            while saved < 100:
                try:
                    batch = next(preview_iter)
                except StopIteration:
                    break
                imgs = batch.get("image")
                lbls = batch.get("label")
                if imgs is None or lbls is None:
                    continue
                for b_idx in range(len(imgs)):
                    if saved >= 100:
                        break
                    img_np = tensor_to_uint8_image(imgs[b_idx])
                    mask_bool = mask_tensor_to_bool(lbls[b_idx], threshold=0.5)
                    save_path = os.path.join(train_preview_dir, f"sample_{saved:02d}.png")
                    save_pair_visualization(img_np, mask_bool, save_path, pair=True)
                    saved += 1
        except Exception as exc:
            print(f"[warning] Failed to save train preview batch: {exc}")

        if sample_image_paths:
            initial_epoch_dir = os.path.join(epoch_samples_root, "epoch_start")
            try:
                needs_generation = (
                    not os.path.isdir(initial_epoch_dir)
                    or not any(Path(initial_epoch_dir).iterdir())
                )
            except FileNotFoundError:
                needs_generation = True
            if needs_generation:
                try:
                    run_epoch_sample_inference(
                        args,
                        model,
                        sample_image_paths,
                        initial_epoch_dir,
                    )
                except Exception as exc:
                    print(f"[warning] Initial 100-sample inference failed: {exc}")
        if val_preview_dataset is not None:
            try:
                generate_validation_preview(
                    args,
                    model,
                    val_preview_dataset,
                    global_step,
                    val_preview_dir,
                )
            except Exception as exc:
                print(f"[warning] Initial validation preview failed: {exc}")
        if test_sample_image_paths:
            for dataset_name, image_paths in test_sample_image_paths.items():
                initial_test_dir = os.path.join(test_preview_dir, dataset_name, "epoch_start")
                try:
                    needs_generation = (
                        not os.path.isdir(initial_test_dir)
                        or not any(Path(initial_test_dir).iterdir())
                    )
                except FileNotFoundError:
                    needs_generation = True
                if needs_generation:
                    try:
                        run_epoch_sample_inference(
                            args,
                            model,
                            image_paths,
                            initial_test_dir,
                        )
                    except Exception as exc:
                        print(f"[warning] Initial test preview failed for {dataset_name}: {exc}")
    misc.synchronize()

    for epoch in range(epoch_start, epoch_num):
        print("epoch: ", epoch, " lr: ", optimizer.param_groups[0]["lr"])
        metric_logger = misc.MetricLogger(delimiter="  ")
        train_dataloaders.batch_sampler.sampler.set_epoch(epoch)

        tqdm_disable = not misc.is_main_process()
        train_iterator = metric_logger.log_every(train_dataloaders, 50)
        progress_bar = tqdm(
            train_iterator,
            total=len(train_dataloaders),
            desc=f"Epoch {epoch:03d}",
            leave=False,
            disable=tqdm_disable,
        )
        for data in progress_bar:
            inputs, labels = data['image'], data['label'].to(model.device)  # (bs,3,1024,1024), (bs,1,1024,1024)
            batched_input = []
            if args.hier_det:
                para_masks, line_masks, word_masks = data['paragraph_masks'], data['line_masks'], data['word_masks']
                line2para_idx = data['line2paragraph_index']
                fg_points, para_masks, line_masks, word_masks = misc.sample_foreground_points(labels, para_masks, line_masks, word_masks, line2para_idx)
            for b_i in range(len(inputs)):
                dict_input = dict()
                dict_input['image'] = inputs[b_i].to(model.device).contiguous()
                dict_input['original_size'] = inputs[b_i].shape[-2:]
                if args.hier_det:
                    point_coords = fg_points[b_i][:, None, :]
                    dict_input['point_coords'] = point_coords
                    dict_input['point_labels'] = torch.ones((point_coords.shape[0], point_coords.shape[1]), device=point_coords.device)
                batched_input.append(dict_input)

            with autocast():
                if args.hier_det:
                    (up_masks_logits, up_masks, iou_output, hr_masks_logits, hr_masks, hr_iou_output,
                     hi_masks_logits, hi_iou_output, word_masks_logits) = model(batched_input, multimask_output=False)
                    loss_focal, loss_dice = loss_masks(up_masks_logits, labels / 255.0, len(up_masks_logits))
                    loss_mse = loss_iou_mse(iou_output, up_masks, labels)
                    loss_lr = loss_focal * 20 + loss_dice + loss_mse

                    loss_focal_hr, loss_dice_hr = loss_masks(hr_masks_logits, labels / 255.0, len(up_masks_logits))
                    loss_mse_hr = loss_iou_mse(hr_iou_output, hr_masks, labels)
                    loss_hr = loss_focal_hr * 20 + loss_dice_hr + loss_mse_hr

                    if word_masks is not None:
                        loss_focal_word, loss_dice_word = loss_hi_masks(
                            hi_masks_logits[:, 0:1, :, :], word_masks, len(hi_masks_logits)
                        )
                        loss_focal_word_384, loss_dice_word_384 = loss_hi_masks(
                            word_masks_logits[:, 0:1, :, :], word_masks, len(hi_masks_logits),
                        )
                        loss_word = loss_focal_word + loss_dice_word
                        loss_word_384 = loss_focal_word_384 + loss_dice_word_384

                        loss_focal_line, loss_dice_line = loss_hi_masks(
                            hi_masks_logits[:, 1:2, :, :], line_masks, len(hi_masks_logits)
                        )
                        loss_mse_line = loss_hi_iou_mse(
                            hi_iou_output[:, 1:2], hi_masks_logits[:, 1:2, :, :], model.module.mask_threshold, line_masks
                        )
                        loss_line = loss_focal_line + loss_dice_line + loss_mse_line

                        loss_focal_para, loss_dice_para = loss_hi_masks(
                            hi_masks_logits[:, 2:3, :, :], para_masks, len(hi_masks_logits)
                        )
                        loss_mse_para = loss_hi_iou_mse(
                            hi_iou_output[:, 2:3], hi_masks_logits[:, 2:3, :, :], model.module.mask_threshold, para_masks
                        )
                        loss_para = loss_focal_para + loss_dice_para + loss_mse_para

                        loss = loss_lr + loss_hr + loss_word + loss_word_384 + loss_line + loss_para * 0.5
                        loss_dict = {
                            "loss_lr_mask": loss_lr,
                            "loss_hr_mask": loss_hr,
                            "loss_word": loss_word,
                            "loss_word_384": loss_word_384,
                            "loss_line": loss_line,
                            "loss_para": loss_para * 0.5
                        }
                    else:
                        raise NotImplementedError
                else:
                    up_masks_logits, up_masks, iou_output, hr_masks_logits, hr_masks, hr_iou_output = model(
                        batched_input, multimask_output=False
                    )
                    loss_focal, loss_dice = loss_masks(up_masks_logits, labels / 255.0, len(up_masks_logits))
                    loss_focal_hr, loss_dice_hr = loss_masks(hr_masks_logits, labels / 255.0, len(up_masks_logits))
                    loss_mse = loss_iou_mse(iou_output, up_masks, labels)
                    loss_mse_hr = loss_iou_mse(hr_iou_output, hr_masks, labels)
                    loss = loss_focal * 20 + loss_dice + loss_mse + loss_focal_hr * 20 + loss_dice_hr + loss_mse_hr
                    loss_dict = {
                        "loss_iou_mse": loss_mse,
                        "loss_dice": loss_dice,
                        "loss_focal": loss_focal * 20,
                        "loss_iou_mse_hr": loss_mse_hr,
                        "loss_dice_hr": loss_dice_hr,
                        "loss_focal_hr": loss_focal_hr * 20,
                    }
                # reduce losses over all GPUs for logging purposes
                loss_dict_reduced = misc.reduce_dict(loss_dict)
                losses_reduced_scaled = sum(loss_dict_reduced.values())
                loss_value = losses_reduced_scaled.item()

            optimizer.zero_grad()
            gradsclaler.scale(loss).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0, norm_type=2)
            gradsclaler.step(optimizer)
            gradsclaler.update()
            metric_logger.update(training_loss=loss_value, **loss_dict_reduced)
            if not tqdm_disable:
                progress_bar.set_postfix({"loss": f"{loss_value:.4f}"})

            global_step += 1
            if global_step % 50 == 0:
                if misc.is_main_process() and val_preview_dataset is not None:
                    try:
                        generate_validation_preview(
                            args,
                            model,
                            val_preview_dataset,
                            global_step,
                            val_preview_dir,
                        )
                    except Exception as exc:
                        print(f"[warning] Validation preview failed at step {global_step}: {exc}")
                misc.synchronize()
        progress_bar.close()

        metric_logger.synchronize_between_processes()
        train_stats = {k: meter.global_avg for k, meter in metric_logger.meters.items() if meter.count > 0}

        if (epoch - epoch_start) % args.valid_period == 0 or (epoch + 1) == epoch_num:
            if args.hier_det:
                model.module.hier_det = False  # disable hi_decoder temporally
            val_stats = evaluate(args, model, valid_dataloaders, valid_datasets_names)
            test_eval_stats = {}
            if test_dataloaders:
                test_eval_stats = evaluate(args, model, test_dataloaders, test_datasets_names)
            if args.hier_det:
                model.module.hier_det = True
            if misc.is_main_process():
                for ds_idx, (ds_name, ds_results) in enumerate(val_stats.items()):
                    iou_result = ds_results.get('001-text-IOU')
                    iou_result_hs = ds_results.get('001-text-IOU_hr')
                    iou_result = max(iou_result, iou_result_hs)
                    if iou_result > best_iou[ds_idx]:
                        checkpoint = {
                            'model': model.module.state_dict(),
                            'optimizer': optimizer.state_dict(),
                            'lr_scheduler': lr_scheduler.state_dict(),
                            'epoch': epoch
                        }
                        torch.save(checkpoint, os.path.join(args.output, ds_name+"_best.pth"))
                        best_iou[ds_idx] = iou_result
            train_stats.update(val_stats)
            if test_eval_stats:
                train_stats.update(test_eval_stats)
            if misc.is_main_process():
                metrics_path = os.path.join(args.output, "eval_metrics.jsonl")
                with open(metrics_path, "a", encoding="utf-8") as fp:
                    record = {
                        "epoch": epoch,
                        "global_step": global_step,
                        "timestamp": datetime.datetime.utcnow().isoformat(),
                    }
                    record.update(
                        {
                            f"val::{ds_name}": ds_results
                            for ds_name, ds_results in val_stats.items()
                        }
                    )
                    if test_eval_stats:
                        record.update(
                            {
                                f"test::{ds_name}": ds_results
                                for ds_name, ds_results in test_eval_stats.items()
                            }
                        )
                    fp.write(json.dumps(record, ensure_ascii=False) + "\n")
            if test_eval_stats and test_sample_image_paths and test_preview_dir and misc.is_main_process():
                for dataset_name, image_paths in test_sample_image_paths.items():
                    epoch_dir = os.path.join(test_preview_dir, dataset_name, f"epoch_{epoch:03d}")
                    try:
                        run_epoch_sample_inference(
                            args,
                            model,
                            image_paths,
                            epoch_dir,
                        )
                    except Exception as exc:
                        print(f"[warning] Test preview failed for {dataset_name} at epoch {epoch}: {exc}")
            if test_eval_stats and test_sample_image_paths:
                misc.synchronize()
            model.train()
        lr_scheduler.step()

        if sample_image_paths:
            if misc.is_main_process():
                epoch_dir = os.path.join(epoch_samples_root, f"epoch_{epoch:03d}")
                try:
                    run_epoch_sample_inference(
                        args,
                        model,
                        sample_image_paths,
                        epoch_dir,
                    )
                except Exception as exc:
                    print(f"[warning] Epoch sample inference failed at epoch {epoch}: {exc}")
            misc.synchronize()

    # Finish training
    print("Training Reaches The Maximum Epoch Number")
    if misc.is_main_process():
        model_name = "/final_epoch_" + str(epoch_num) + ".pth"
        torch.save(model.module.state_dict(), args.output + model_name)


def inference_on_dataset(model, data_loader, data_name, evaluator, args):
    print("Start inference on {}, {} batches".format(data_name, len(data_loader)))
    num_devices = misc.get_world_size()
    total = len(data_loader)
    evaluator.reset()
    num_warmup = min(5, total - 1)
    start_time = time.perf_counter()
    total_data_time = 0
    total_compute_time = 0
    total_eval_time = 0

    progress_disable = not misc.is_main_process()
    progress_bar = tqdm(
        data_loader,
        total=total,
        desc=f"[{data_name}] eval",
        leave=False,
        disable=progress_disable,
    )

    start_data_time = time.perf_counter()
    for idx_val, data_val in enumerate(progress_bar):
        inputs_val, labels_ori = data_val['image'], data_val['ori_label']
        ignore_mask = data_val.get('ignore_mask', None)
        if torch.cuda.is_available():
            labels_ori = labels_ori.cuda()
        batched_input = []
        for b_i in range(len(inputs_val)):
            dict_input = dict()
            dict_input['image'] = inputs_val[b_i].to(model.device).contiguous()
            dict_input['original_size'] = labels_ori[b_i].shape[-2:]
            batched_input.append(dict_input)

        total_data_time += time.perf_counter() - start_data_time
        if idx_val == num_warmup:
            start_time = time.perf_counter()
            total_data_time = 0
            total_compute_time = 0
            total_eval_time = 0

        start_compute_time = time.perf_counter()
        with torch.no_grad():
            up_masks_logits, up_masks, iou_output, hr_masks_logits, hr_masks, hr_iou_output = model(
                batched_input, multimask_output=False
            )
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        total_compute_time += time.perf_counter() - start_compute_time

        start_eval_time = time.perf_counter()
        evaluator.process(up_masks, hr_masks, labels_ori, ignore_mask)
        total_eval_time += time.perf_counter() - start_eval_time

        iters_after_start = idx_val + 1 - num_warmup * int(idx_val >= num_warmup)
        data_seconds_per_iter = total_data_time / iters_after_start
        compute_seconds_per_iter = total_compute_time / iters_after_start
        eval_seconds_per_iter = total_eval_time / iters_after_start
        total_seconds_per_iter = (time.perf_counter() - start_time) / iters_after_start

        start_data_time = time.perf_counter()
        if not progress_disable:
            progress_bar.set_postfix(
                {
                    "data": f"{data_seconds_per_iter:.3f}s",
                    "infer": f"{compute_seconds_per_iter:.3f}s",
                    "eval": f"{eval_seconds_per_iter:.3f}s",
                    "eta": str(datetime.timedelta(seconds=int(total_seconds_per_iter * (total - idx_val - 1)))),
                }
            )

    progress_bar.close()

    total_time = time.perf_counter() - start_time
    total_time_str = str(datetime.timedelta(seconds=total_time))
    print(
        "Total inference time: {} ({:.6f} s / iter per device, on {} devices)".format(
            total_time_str, total_time / (total - num_warmup), num_devices
        )
    )
    total_compute_time_str = str(datetime.timedelta(seconds=int(total_compute_time)))
    print(
        "Total inference pure compute time: {} ({:.6f} s / iter per device, on {} devices)".format(
            total_compute_time_str, total_compute_time / (total - num_warmup), num_devices
        )
    )

    results = evaluator.evaluate()
    if results is None:
        results = {}

    return results


def evaluate(args, model, valid_dataloaders, valid_datasets_names):
    model.eval()
    test_stats = {}

    for k in range(len(valid_dataloaders)):
        metric_logger = misc.MetricLogger(delimiter="  ")
        valid_dataloader = valid_dataloaders[k]
        valid_dataset_name = valid_datasets_names[k]
        evaluator = Evaluator(valid_dataset_name, args, True)
        print('============================')
        results_k = inference_on_dataset(model, valid_dataloader, valid_dataset_name, evaluator, args)
        print("Evaluation results for {}:".format(valid_dataset_name))
        for task, res in results_k.items():
            if '_hr' not in task:
                print(f"copypaste: {task}={res}, {task}_hr={results_k[task+'_hr']}")
        print('============================')
        test_stats.update({valid_dataset_name: results_k})

    return test_stats


if __name__ == "__main__":

    args = get_args_parser()

    # train
    totaltext_train = {
        "name": "TotalText-train",
        "im_dir": "./datasets/TotalText/Images/Train",
        "gt_dir": "./datasets/TotalText/groundtruth_pixel/Train",
        "im_ext": ".jpg",
        "gt_ext": ".jpg",
    }
    hiertext_train = {
        "name": "HierText-train",
        "im_dir": "./datasets/HierText/train",
        "gt_dir": "./datasets/HierText/train_gt",
        "im_ext": ".jpg",
        "gt_ext": ".png",
        "json_dir": "./datasets/HierText/train_shrink_vert.json"
    }
    textseg_train = {
        "name": "TextSeg-train",
        "im_dir": "./datasets/TextSeg/train_images",
        "gt_dir": "./datasets/TextSeg/train_gt",
        "im_ext": ".jpg",
        "gt_ext": ".png"
    }
    cocots_train = {
        "name": "COCO_TS-train",
        "im_dir": "./datasets/COCO_TS/train_images",
        "gt_dir": "./datasets/COCO_TS/COCO_TS_labels",
        "im_ext": ".jpg",
        "gt_ext": ".png"
    }
    cocots_train_hier = {
        "name": "COCO_TS-train",
        "im_dir": "./datasets/COCO_TS/train_images",
        "gt_dir": "./datasets/COCO_TS/hier-model_labels",
        "im_ext": ".jpg",
        "gt_ext": ".png"
    }
    cocots_train_tt = {
        "name": "COCO_TS-train",
        "im_dir": "./datasets/COCO_TS/train_images",
        "gt_dir": "./datasets/COCO_TS/tt-model_labels",
        "im_ext": ".jpg",
        "gt_ext": ".png"
    }
    cocots_train_textseg = {
        "name": "COCO_TS-train",
        "im_dir": "./datasets/COCO_TS/train_images",
        "gt_dir": "./datasets/COCO_TS/textseg-model_labels",
        "im_ext": ".jpg",
        "gt_ext": ".png"
    }
    crello_train = {
        "name": "CrelloTextSeg-train",
        "im_dir": os.path.join(args.crello_root, "train_images"),
        "gt_dir": os.path.join(args.crello_root, "train_gt"),
        "im_ext": ".png",
        "gt_ext": ".png"
    }
    lica_train = {
        "name": "LiCATextSeg-train",
        "im_dir": os.path.join(args.lica_root, "train_images"),
        "gt_dir": os.path.join(args.lica_root, "train_gt"),
        "im_ext": ".png",
        "gt_ext": ".png"
    }

    train_dataset_map = {
        'totaltext_train': totaltext_train,
        'hiertext_train': hiertext_train,
        'textseg_train': textseg_train,
        'cocots_train': cocots_train,
        'cocots_train_hier': cocots_train_hier,
        'cocots_train_tt': cocots_train_tt,
        'cocots_train_textseg': cocots_train_textseg,
        'crello_train': crello_train,
        'lica_train': lica_train,
    }

    # validation and test
    totaltext_test = {
        "name": "TotalText-test",
        "im_dir": "./datasets/TotalText/Images/Test",
        "gt_dir": "./datasets/TotalText/groundtruth_pixel/Test",
        "im_ext": ".jpg",
        "gt_ext": ".jpg"
    }
    hiertext_val = {
        "name": "HierText-val",
        "im_dir": "./datasets/HierText/validation",
        "gt_dir": "./datasets/HierText/validation_gt",
        "im_ext": ".jpg",
        "gt_ext": ".png"
    }
    hiertext_test = {
        "name": "HierText-test",
        "im_dir": "./datasets/HierText/test",
        "gt_dir": "./datasets/HierText/test_gt",
        "im_ext": ".jpg",
        "gt_ext": ".png"
    }
    textseg_val = {
        "name": "TextSeg-val",
        "im_dir": "./datasets/TextSeg/val_images",
        "gt_dir": "./datasets/TextSeg/val_gt",
        "im_ext": ".jpg",
        "gt_ext": ".png"
    }
    textseg_test = {
        "name": "TextSeg-test",
        "im_dir": "./datasets/TextSeg/test_images",
        "gt_dir": "./datasets/TextSeg/test_gt",
        "im_ext": ".jpg",
        "gt_ext": ".png"
    }
    crello_val = {
        "name": "CrelloTextSeg-val",
        "im_dir": os.path.join(args.crello_root, "val_images"),
        "gt_dir": os.path.join(args.crello_root, "val_gt"),
        "im_ext": ".png",
        "gt_ext": ".png"
    }
    crello_test = {
        "name": "CrelloTextSeg-test",
        "im_dir": os.path.join(args.crello_root, "test_images"),
        "gt_dir": os.path.join(args.crello_root, "test_gt"),
        "im_ext": ".png",
        "gt_ext": ".png"
    }
    lica_val = {
        "name": "LiCATextSeg-val",
        "im_dir": os.path.join(args.lica_root, "val_images"),
        "gt_dir": os.path.join(args.lica_root, "val_gt"),
        "im_ext": ".png",
        "gt_ext": ".png"
    }
    lica_test = {
        "name": "LiCATextSeg-test",
        "im_dir": args.lica_test_im_dir,
        "gt_dir": args.lica_test_gt_dir,
        "im_ext": ".png",
        "gt_ext": ".png"
    }
    val_dataset_map = {
        'totaltext_test': totaltext_test,
        'hiertext_val': hiertext_val,
        'hiertext_test': hiertext_test,
        'textseg_val': textseg_val,
        'textseg_test': textseg_test,
        'crello_val': crello_val,
        'crello_test': crello_test,
        'lica_val': lica_val,
        'lica_test': lica_test,
    }

    train_datasets = [train_dataset_map[ds_name] for ds_name in args.train_datasets]
    val_datasets = [val_dataset_map[ds_name] for ds_name in args.val_datasets]

    available_eval_map = dict(val_dataset_map)
    if args.test_datasets:
        missing = [name for name in args.test_datasets if name not in available_eval_map]
        if missing:
            raise ValueError(f"Unknown test dataset specifiers: {missing}")
        test_datasets = [available_eval_map[name] for name in args.test_datasets]
    else:
        test_datasets = []

    main(train_datasets, val_datasets, test_datasets, args)
