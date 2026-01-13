import argparse
import os
from types import SimpleNamespace

import cv2
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

from hi_sam.modeling.build import model_registry
from hi_sam.modeling.predictor import SamPredictor


def patchify_sliding(image: np.ndarray, patch_size: int = 512, stride: int = 384):
    """Split image into overlapping patches."""
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
    whole_logits = np.zeros(ori_size, dtype=np.float32)
    assert len(patch_list) == len(h_slice_list) == len(w_slice_list)
    for patch, h_slice, w_slice in zip(patch_list, h_slice_list, w_slice_list):
        whole_logits[h_slice, w_slice] += patch
    return whole_logits


def resolve_device(device_str: str) -> torch.device:
    device_str = device_str.lower()
    if device_str.startswith("cuda") and not torch.cuda.is_available():
        print("CUDA not available. Falling back to CPU.")
        return torch.device("cpu")
    return torch.device(device_str)


def load_predictor(checkpoint: str, model_type: str, device: torch.device) -> SamPredictor:
    build_args = SimpleNamespace(
        checkpoint=checkpoint,
        model_type=model_type,
        hier_det=False,
        attn_layers=1,
        prompt_len=12,
    )
    model = model_registry[model_type](build_args)
    model.eval()
    model.to(device)
    return SamPredictor(model)


def generate_mask(predictor: SamPredictor, image: np.ndarray, patch_size: int, stride: int) -> np.ndarray:
    ori_size = image.shape[:2]
    patch_list, h_slice_list, w_slice_list = patchify_sliding(image, patch_size, stride)
    logits = []
    with torch.no_grad():
        for patch in patch_list:
            predictor.set_image(patch)
            _, hr_mask, _, _ = predictor.predict(multimask_output=False, return_logits=True)
            logits.append(hr_mask[0])
    logits = unpatchify_sliding(logits, h_slice_list, w_slice_list, ori_size)
    threshold = predictor.model.mask_threshold
    return logits > threshold


def save_binary_mask(mask: np.ndarray, filename: str):
    mask_image = (mask.astype(np.uint8)) * 255
    Image.fromarray(mask_image).save(filename)


def list_images(image_dir: str):
    supported_exts = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
    image_paths = []
    for entry in sorted(os.listdir(image_dir)):
        full_path = os.path.join(image_dir, entry)
        if not os.path.isfile(full_path):
            continue
        if os.path.splitext(entry)[1].lower() in supported_exts:
            image_paths.append(full_path)
    return image_paths


def parse_args():
    parser = argparse.ArgumentParser("Hi-SAM text mask union generator")
    parser.add_argument(
        "--image-dir",
        type=str,
        default="/home/ubuntu/jjseol/data/100_sample/images",
        help="Directory containing input images.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="/home/ubuntu/jjseol/data/100_sample/Hi-SAM_Mask",
        help="Directory to store union masks.",
    )
    parser.add_argument(
        "--checkpoint-h",
        type=str,
        default="/home/ubuntu/jjseol/data/ml-hub/inpainting/Hi-SAM/pretrained_checkpoint/sam_tss_h_textseg.pth",
        help="Checkpoint path for the ViT-H text segmentation model.",
    )
    parser.add_argument(
        "--checkpoint-l",
        type=str,
        default="/home/ubuntu/jjseol/data/ml-hub/inpainting/Hi-SAM/pretrained_checkpoint/sam_tss_l_textseg.pth",
        help="Checkpoint path for the ViT-L text segmentation model.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device to run inference on (e.g., cuda, cuda:0, cpu).",
    )
    parser.add_argument(
        "--patch-size",
        type=int,
        default=512,
        help="Sliding window patch size.",
    )
    parser.add_argument(
        "--stride",
        type=int,
        default=384,
        help="Sliding window stride.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    device = resolve_device(args.device)

    os.makedirs(args.output_dir, exist_ok=True)

    print("Loading ViT-H predictor...")
    predictor_h = load_predictor(args.checkpoint_h, "vit_h", device)
    print("Loading ViT-L predictor...")
    predictor_l = load_predictor(args.checkpoint_l, "vit_l", device)

    image_paths = list_images(args.image_dir)
    if not image_paths:
        print(f"No supported images found in {args.image_dir}.")
        return

    for image_path in tqdm(image_paths, desc="Processing images"):
        image_bgr = cv2.imread(image_path)
        if image_bgr is None:
            print(f"Skipping unreadable file: {image_path}")
            continue
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

        mask_h = generate_mask(predictor_h, image_rgb, args.patch_size, args.stride)
        mask_l = generate_mask(predictor_l, image_rgb, args.patch_size, args.stride)
        union_mask = np.logical_or(mask_h, mask_l)

        output_path = os.path.join(args.output_dir, os.path.basename(image_path))
        save_binary_mask(union_mask, output_path)


if __name__ == "__main__":
    main()

