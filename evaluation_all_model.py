import argparse
import base64
import io
import json
import os
import re
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from dataclasses import replace as dataclass_replace
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

from hi_sam.modeling.build import model_registry
from hi_sam.modeling.predictor import SamPredictor
from demo_hisam import patchify_sliding, unpatchify_sliding, resize_binary_mask


DEFAULT_IMAGE_DIR = "/home/ubuntu/jjseol/data/data/text_segmentation_dataset/val_images"
DEFAULT_GT_DIR = "/home/ubuntu/jjseol/data/data/text_segmentation_dataset/val_gt"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DEFAULT_OBB_MODEL_PATH = "/home/ubuntu/yolo11xOBB-obb80_best_f881f849-5985-4a45-bb26-6d6247118262.pt"


@dataclass
class ModelConfig:
    name: str
    runner: str = "sam"
    model_type: Optional[str] = None
    checkpoint: Optional[str] = None
    patch_mode: bool = False
    metadata: Dict[str, str] = field(default_factory=dict)


@dataclass
class OBBMetricConfig:
    model_path: Optional[str] = None
    imgsz: int = 512
    conf: float = 0.3
    iou: float = 0.7
    max_det: int = 300
    class_id: int = 0
    device: str = "0"


DEFAULT_GEMINI_PROMPT = (
    "Give the segmentation masks for all the text in the image.\n"
    "Output a JSON list of segmentation masks where each entry contains the 2D "
    'bounding box in the key "box_2d", the segmentation mask in key "mask", and '
    'the text label in the key "label". Use descriptive labels. Please detail as much as possible.\n'
    "Return each mask strictly as a base64 PNG data URL (prefix with data:image/png;base64,).\n"
    "Do not output RLE, <start_of_mask>, or any format other than base64 PNG."
)

DEFAULT_GEMINI_API_KEY = "AIzaSyCAQZZYqk1gO7FAqpDV7JHUJt0PnkumU8c"


AVAILABLE_MODELS: Dict[str, ModelConfig] = {
    "crello_vit_h_patch": ModelConfig(
        name="Hi-SAM Crello vit_h (patch)",
        runner="sam",
        model_type="vit_h",
        checkpoint="/home/ubuntu/jjseol/data/Hi-SAM/work_dirs/crello_vit_h/CrelloTextSeg-val_best.pth",
        patch_mode=True,
    ),
    "crello_lica_vit_h_patch": ModelConfig(
        name="Hi-SAM Crello LiCA vit_h (patch)",
        runner="sam",
        model_type="vit_h",
        checkpoint="/home/ubuntu/jjseol/data/checkpoints/LiCATextSeg-val_best.pth",
        patch_mode=True,
    ),
    "pretrained_vit_h": ModelConfig(
        name="Hi-SAM Pretrained vit_h",
        runner="sam",
        model_type="vit_h",
        checkpoint="/home/ubuntu/jjseol/data/Hi-SAM/pretrained_checkpoint/sam_tss_h_textseg.pth",
    ),
    "pretrained_vit_l": ModelConfig(
        name="Hi-SAM Pretrained vit_l",
        runner="sam",
        model_type="vit_l",
        checkpoint="/home/ubuntu/jjseol/data/Hi-SAM/pretrained_checkpoint/sam_tss_l_textseg.pth",
    ),
    "texrnet_hrnet_textseg": ModelConfig(
        name="TexRNet HRNet (TextSeg)",
        runner="texrnet",
        checkpoint="/home/ubuntu/jjseol/data/Hi-SAM/pretrained_checkpoint/texrnet_hrnet_textseg.pth",
        metadata={"repo_path": "/home/ubuntu/Rethinking-Text-Segmentation"},
    ),
    "gemini_flash_textseg": ModelConfig(
        name="Gemini 2.5 Flash TextSeg",
        runner="gemini",
        metadata={
            "model": "gemini-2.5-flash",
            "max_items": 40,
            "lang_hint": "ko,en",
            "threshold": 127,
            "temperature": 0.0,
            "api_key": DEFAULT_GEMINI_API_KEY,
            "prompt": DEFAULT_GEMINI_PROMPT,
        },
    ),
    "gemini_pro_textseg": ModelConfig(
        name="Gemini 2.5 Pro TextSeg",
        runner="gemini",
        metadata={
            "model": "gemini-2.5-pro",
            "max_items": 40,
            "lang_hint": "ko,en",
            "threshold": 127,
            "temperature": 0.0,
            "thinking_budget": 2,
            "api_key": DEFAULT_GEMINI_API_KEY,
            "prompt": DEFAULT_GEMINI_PROMPT,
        },
    ),
}

DEFAULT_MODEL_KEYS = list(AVAILABLE_MODELS.keys())


def collect_samples(image_dir: Path, gt_dir: Path) -> List[Tuple[Path, Path]]:
    gt_map: Dict[str, Path] = {}
    for gt_path in gt_dir.iterdir():
        if gt_path.is_file():
            gt_map[gt_path.stem] = gt_path

    samples: List[Tuple[Path, Path]] = []
    missing_gt = []
    for image_path in image_dir.iterdir():
        if not image_path.is_file():
            continue
        stem = image_path.stem
        gt_path = gt_map.get(stem)
        if gt_path is None:
            missing_gt.append(stem)
            continue
        samples.append((image_path, gt_path))

    if missing_gt:
        print(f"Warning: {len(missing_gt)} images missing GT masks. Skipping: {missing_gt[:5]}")
    if not samples:
        raise RuntimeError("No valid image/mask pairs were found.")
    return sorted(samples)


def load_mask(path: Path) -> np.ndarray:
    mask = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise FileNotFoundError(f"Failed to read mask: {path}")
    return mask > 0


def load_image(path: Path) -> np.ndarray:
    image = cv2.imread(str(path))
    if image is None:
        raise FileNotFoundError(f"Failed to read image: {path}")
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def slugify(name: str) -> str:
    slug = re.sub(r'[^A-Za-z0-9._-]+', '_', name).strip('_')
    return slug or "model"


def extract_json_payload(raw_text: str) -> str:
    lines = raw_text.splitlines()
    start = None
    for idx, line in enumerate(lines):
        if line.strip().lower() == "```json":
            start = idx + 1
            break
    if start is None:
        return raw_text.strip()
    payload_lines = []
    for line in lines[start:]:
        if line.strip().startswith("```"):
            break
        payload_lines.append(line)
    return "\n".join(payload_lines).strip()


def build_sam_predictor(config: ModelConfig, device: str) -> SamPredictor:
    if not config.checkpoint or not os.path.exists(config.checkpoint):
        raise FileNotFoundError(f"Checkpoint not found: {config.checkpoint}")
    args = SimpleNamespace(
        model_type=config.model_type,
        checkpoint=config.checkpoint,
        device=device,
        hier_det=False,
        input_size=[1024, 1024],
        patch_mode=config.patch_mode,
        pair_visualization=False,
        attn_layers=1,
        prompt_len=12,
    )
    model = model_registry[config.model_type](args)
    model.eval()
    model.to(device)
    return SamPredictor(model)


def run_patch_inference(predictor: SamPredictor, image: np.ndarray) -> np.ndarray:
    ori_size = image.shape[:2]
    inference_image = image
    scale = 1.0
    max_dim = max(ori_size)
    if max_dim > 2048:
        scale = 2048 / max_dim
        new_h = max(1, int(round(ori_size[0] * scale)))
        new_w = max(1, int(round(ori_size[1] * scale)))
        inference_image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    proc_size = inference_image.shape[:2]
    stride = 384
    patch_size = 512
    if max(proc_size) > 1024:
        patch_size = 1024
        stride = 768

    patch_list, h_slice_list, w_slice_list = patchify_sliding(inference_image, patch_size, stride)
    mask_tiles = []
    with torch.no_grad():
        for patch in patch_list:
            predictor.set_image(patch)
            _, hr_m, _, _ = predictor.predict(multimask_output=False, return_logits=True)
            assert hr_m.shape[0] == 1
            mask_tiles.append(hr_m[0])
    mask_logits = unpatchify_sliding(mask_tiles, h_slice_list, w_slice_list, proc_size)

    with torch.no_grad():
        predictor.set_image(inference_image)
        _, full_hr_m, _, _ = predictor.predict(multimask_output=False, return_logits=True)
        assert full_hr_m.shape[0] == 1
        mask_logits += full_hr_m[0]

    mask = mask_logits > predictor.model.mask_threshold
    if scale != 1.0:
        mask = resize_binary_mask(mask, ori_size)
    return mask.astype(bool)


def run_full_inference(predictor: SamPredictor, image: np.ndarray) -> np.ndarray:
    ori_size = image.shape[:2]
    inference_image = image
    scale = 1.0
    max_dim = max(ori_size)
    if max_dim > 2048:
        scale = 2048 / max_dim
        new_h = max(1, int(round(ori_size[0] * scale)))
        new_w = max(1, int(round(ori_size[1] * scale)))
        inference_image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    with torch.no_grad():
        predictor.set_image(inference_image)
        _, hr_m, _, _ = predictor.predict(multimask_output=False, return_logits=True)
        assert hr_m.shape[0] == 1
        mask = hr_m[0] > predictor.model.mask_threshold

    if scale != 1.0:
        mask = resize_binary_mask(mask, ori_size)
    return mask.astype(bool)


def build_texrnet_runner(config: ModelConfig, device: str):
    repo_path = config.metadata.get("repo_path")
    if repo_path:
        repo_path = os.path.expanduser(repo_path)
        if not os.path.isdir(repo_path):
            raise FileNotFoundError(
                f"TexRNet repository not found at '{repo_path}'. Clone it from "
                "https://github.com/SHI-Labs/Rethinking-Text-Segmentation and update the path."
            )
        if repo_path not in sys.path:
            sys.path.append(repo_path)
    try:
        from inference import TextRNet_HRNet_Wrapper
    except ImportError as exc:
        raise ImportError(
            "Failed to import TextRNet inference utilities. "
            "Ensure the TexRNet repository is cloned and its dependencies are installed."
        ) from exc

    if not config.checkpoint or not os.path.exists(config.checkpoint):
        raise FileNotFoundError(
            f"TexRNet checkpoint not found: {config.checkpoint}. "
            "Download the pretrained weights from the official repository and place them at this path."
        )

    torch_device = torch.device(device)
    runner = TextRNet_HRNet_Wrapper(torch_device, config.checkpoint)
    return runner


def run_texrnet_inference(runner, image: np.ndarray) -> np.ndarray:
    pil_image = Image.fromarray(image)
    mask_image = runner.process(pil_image)
    mask_arr = np.array(mask_image)
    if mask_arr.ndim == 3:
        mask_arr = mask_arr[:, :, 0]
    return (mask_arr > 127).astype(bool)


class GeminiSegmentationRunner:
    def __init__(self, metadata: Dict[str, str]):
        try:
            from google import genai
            from google.genai import types
        except ImportError as exc:
            raise ImportError(
                "google-genai package is required for Gemini-based segmentation. Install it via `pip install google-genai`."
            ) from exc

        self.types = types
        self.model_name = metadata.get("model", "gemini-2.5-flash")
        self.prompt = metadata.get("prompt", DEFAULT_GEMINI_PROMPT)
        self.max_items = int(metadata.get("max_items", 40))
        self.lang_hint = metadata.get("lang_hint", "en")
        self.threshold = int(metadata.get("threshold", 127))
        self.temperature = float(metadata.get("temperature", 0.0))
        self.top_p = metadata.get("top_p")
        self.top_k = metadata.get("top_k")
        self.candidate_count = metadata.get("candidate_count")

        api_key = metadata.get("api_key") or os.environ.get("GEMINI_API_KEY")
        if not api_key:
            raise ValueError(
                "Gemini API key not provided. Supply metadata['api_key'] or set GEMINI_API_KEY environment variable."
            )

        self.client = genai.Client(api_key=api_key)

        config_kwargs: Dict[str, object] = {"response_mime_type": "application/json"}
        if self.temperature is not None:
            config_kwargs["temperature"] = self.temperature
        if self.top_p is not None:
            config_kwargs["top_p"] = float(self.top_p)
        if self.top_k is not None:
            config_kwargs["top_k"] = int(self.top_k)
        if self.candidate_count is not None:
            config_kwargs["candidate_count"] = int(self.candidate_count)
        self.generation_config = types.GenerateContentConfig(**config_kwargs)

    def __call__(self, image: np.ndarray) -> np.ndarray:
        h, w = image.shape[:2]
        pil_image = Image.fromarray(image)
        response = self.client.models.generate_content(
            model=self.model_name,
            contents=[self.prompt, pil_image],
            config=self.generation_config,
        )

        raw_text = getattr(response, "text", "")
        try:
            items = json.loads(extract_json_payload(raw_text)) if raw_text else []
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"Failed to parse Gemini response for model {self.model_name}. Raw output (truncated): {raw_text[:200]}"
            ) from exc

        full_mask = np.zeros((h, w), dtype=bool)
        for item in items or []:
            if not isinstance(item, dict):
                continue
            box = item.get("box_2d")
            if not box or len(box) != 4:
                continue
            y0, x0, y1, x1 = box
            x0 = max(0, min(w, int(round((x0 / 1000.0) * w))))
            x1 = max(0, min(w, int(round((x1 / 1000.0) * w))))
            y0 = max(0, min(h, int(round((y0 / 1000.0) * h))))
            y1 = max(0, min(h, int(round((y1 / 1000.0) * h))))
            if x1 <= x0 or y1 <= y0:
                continue

            mask_b64 = item.get("mask", "")
            if mask_b64.startswith("data:image/png;base64,"):
                mask_b64 = mask_b64.split(",", 1)[-1]
            try:
                mask_prob = Image.open(io.BytesIO(base64.b64decode(mask_b64))).convert("L")
            except Exception:
                continue

            width = x1 - x0
            height = y1 - y0
            mask_prob = mask_prob.resize((width, height), Image.Resampling.BILINEAR)
            mask_np = np.asarray(mask_prob, dtype=np.uint8)
            full_mask[y0:y1, x0:x1] |= mask_np >= self.threshold

        return full_mask


def build_gemini_runner(config: ModelConfig, device: str):
    metadata = dict(config.metadata or {})
    metadata.setdefault("api_key", metadata.get("api_key") or os.environ.get("GEMINI_API_KEY"))
    return GeminiSegmentationRunner(metadata)


def run_gemini_inference(runner: GeminiSegmentationRunner, image: np.ndarray) -> np.ndarray:
    return runner(image)


def polygon_to_mask(poly: np.ndarray, shape: Tuple[int, int]) -> np.ndarray:
    mask = np.zeros(shape, dtype=np.uint8)
    pts = np.asarray(poly, dtype=np.float32).reshape(-1, 2)
    if pts.shape[0] < 3:
        return mask.astype(bool)
    pts = np.round(pts).astype(np.int32)
    h, w = shape
    pts[:, 0] = np.clip(pts[:, 0], 0, w - 1)
    pts[:, 1] = np.clip(pts[:, 1], 0, h - 1)
    cv2.fillPoly(mask, [pts], 1)
    return mask.astype(bool)


def compute_obb_metrics(polygons: np.ndarray, mask: np.ndarray) -> Dict[str, float]:
    if polygons is None or len(polygons) == 0:
        return {
            "obb_global_containment": 0.0,
            "obb_global_purity": 0.0,
            "obb_zero_purity_ratio": 0.0,
            "obb_box_count": 0,
        }

    mask_bool = mask.astype(bool)
    mask_pixels = int(mask_bool.sum())
    boxes_union = np.zeros_like(mask_bool, dtype=bool)
    zero_count = 0

    for poly in polygons:
        obb_mask = polygon_to_mask(poly, mask_bool.shape)
        if not obb_mask.any():
            zero_count += 1
            continue
        if not np.logical_and(obb_mask, mask_bool).any():
            zero_count += 1
        boxes_union |= obb_mask

    union_pixels = int(boxes_union.sum())
    intersection_pixels = int(np.logical_and(boxes_union, mask_bool).sum())
    global_containment = intersection_pixels / mask_pixels if mask_pixels else 0.0
    global_purity = intersection_pixels / union_pixels if union_pixels else 0.0
    zero_ratio = zero_count / len(polygons) if len(polygons) else 0.0

    return {
        "obb_global_containment": float(global_containment),
        "obb_global_purity": float(global_purity),
        "obb_zero_purity_ratio": float(zero_ratio),
        "obb_box_count": int(len(polygons)),
    }


class OBBMetricRunner:
    def __init__(self, config: OBBMetricConfig):
        if not config.model_path:
            raise ValueError("OBB model path must be provided to compute OBB metrics.")
        if not os.path.exists(config.model_path):
            raise FileNotFoundError(f"OBB model checkpoint not found: {config.model_path}")
        try:
            from ultralytics import YOLO  # type: ignore[import-untyped]
        except ImportError as exc:
            raise ImportError(
                "ultralytics package is required to compute OBB metrics. Install it via `pip install ultralytics`."
            ) from exc

        self.config = config
        self.model = YOLO(config.model_path)
        self._cache: Dict[Path, np.ndarray] = {}

    def _extract_polygons(self, result) -> np.ndarray:
        obb = getattr(result, "obb", None)
        if obb is None or getattr(obb, "data", None) is None:
            return np.empty((0, 8), dtype=np.float32)

        cls = obb.cls.detach().cpu().numpy().astype(int)
        if cls.size == 0:
            return np.empty((0, 8), dtype=np.float32)
        mask = cls == int(self.config.class_id)
        if not np.any(mask):
            return np.empty((0, 8), dtype=np.float32)

        polys_tensor = getattr(obb, "xyxyxyxy", None)
        if polys_tensor is None:
            return np.empty((0, 8), dtype=np.float32)
        polys = polys_tensor.detach().cpu().numpy()
        return polys[mask]

    def predict_polygons(self, image_path: Path) -> np.ndarray:
        image_path = Path(image_path)
        cached = self._cache.get(image_path)
        if cached is not None:
            return cached

        results = self.model.predict(
            source=str(image_path),
            imgsz=int(self.config.imgsz),
            conf=float(self.config.conf),
            iou=float(self.config.iou),
            max_det=int(self.config.max_det),
            device=self.config.device,
            save=False,
            verbose=False,
        )
        if not results:
            return np.empty((0, 8), dtype=np.float32)
        polygons = self._extract_polygons(results[0])
        self._cache[image_path] = polygons
        return polygons

    def compute_metrics(self, image_path: Path, mask: np.ndarray) -> Dict[str, float]:
        polygons = self.predict_polygons(image_path)
        return compute_obb_metrics(polygons, mask)


def build_obb_metric_runner(config: OBBMetricConfig) -> Optional[OBBMetricRunner]:
    if not config.model_path:
        return None
    return OBBMetricRunner(config)


def compute_confusion(pred: np.ndarray, target: np.ndarray) -> Dict[str, int]:
    pred_flat = pred.astype(bool).ravel()
    target_flat = target.astype(bool).ravel()
    tp = np.logical_and(pred_flat, target_flat).sum()
    fp = np.logical_and(pred_flat, np.logical_not(target_flat)).sum()
    fn = np.logical_and(np.logical_not(pred_flat), target_flat).sum()
    tn = np.logical_not(np.logical_or(pred_flat, target_flat)).sum()
    return {"tp": int(tp), "fp": int(fp), "fn": int(fn), "tn": int(tn)}


def metrics_from_confusion(conf: Dict[str, int]) -> Dict[str, float]:
    tp = conf["tp"]
    fp = conf["fp"]
    fn = conf["fn"]
    denom_iou = tp + fp + fn
    denom_precision = tp + fp
    denom_recall = tp + fn
    denom_dice = 2 * tp + fp + fn
    eps = 1e-8
    return {
        "iou": tp / denom_iou if denom_iou > 0 else 0.0,
        "dice": (2 * tp) / denom_dice if denom_dice > 0 else 0.0,
        "precision": tp / denom_precision if denom_precision > 0 else 0.0,
        "recall": tp / denom_recall if denom_recall > 0 else 0.0,
    }


def summarize_metrics(per_image_metrics: List[Dict[str, float]]) -> Dict[str, float]:
    summary = {}
    for key in ["iou", "dice", "precision", "recall"]:
        values = [metrics[key] for metrics in per_image_metrics]
        summary[f"{key}_mean"] = float(np.mean(values)) if values else 0.0
        summary[f"{key}_std"] = float(np.std(values)) if values else 0.0
    return summary


def summarize_optional_metrics(per_image_metrics: List[Dict[str, float]], metric_keys: List[str]) -> Dict[str, float]:
    summary = {}
    for key in metric_keys:
        values = [metrics[key] for metrics in per_image_metrics if key in metrics]
        if values:
            summary[f"{key}_mean"] = float(np.mean(values))
            summary[f"{key}_std"] = float(np.std(values))
    return summary


def ensure_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)


def overlay_mask(image: np.ndarray, mask: np.ndarray, color: Tuple[int, int, int], alpha: float = 0.5) -> np.ndarray:
    overlay = image.copy().astype(np.float32)
    mask_indices = mask.astype(bool)
    if not np.any(mask_indices):
        return image
    overlay_mask = np.zeros_like(overlay)
    overlay_mask[mask_indices] = color
    overlay[mask_indices] = (alpha * overlay_mask[mask_indices] + (1 - alpha) * overlay[mask_indices])
    return overlay.astype(np.uint8)


def save_prediction_artifacts(
    image: np.ndarray,
    gt_mask: np.ndarray,
    pred_mask: np.ndarray,
    model_dir: Path,
    image_id: str,
    model_slug: str,
):
    ensure_dir(model_dir)
    pred_dir = model_dir / "pred_masks"
    viz_dir = model_dir / "visualizations"
    ensure_dir(pred_dir)
    ensure_dir(viz_dir)

    gt_uint8 = (gt_mask.astype(np.uint8) * 255)
    pred_uint8 = (pred_mask.astype(np.uint8) * 255)

    pred_path = pred_dir / f"{image_id}_{model_slug}_pred.png"
    cv2.imwrite(str(pred_path), pred_uint8)

    base_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    gt_overlay = overlay_mask(image, gt_mask, color=(0, 255, 0))
    pred_overlay = overlay_mask(image, pred_mask, color=(255, 0, 0))
    gt_overlay_bgr = cv2.cvtColor(gt_overlay, cv2.COLOR_RGB2BGR)
    pred_overlay_bgr = cv2.cvtColor(pred_overlay, cv2.COLOR_RGB2BGR)
    gt_mask_bgr = cv2.cvtColor(gt_uint8, cv2.COLOR_GRAY2BGR)
    pred_mask_bgr = cv2.cvtColor(pred_uint8, cv2.COLOR_GRAY2BGR)

    composite = np.concatenate(
        [base_bgr, gt_overlay_bgr, pred_overlay_bgr, gt_mask_bgr, pred_mask_bgr],
        axis=1,
    )
    viz_path = viz_dir / f"{image_id}_{model_slug}_viz.png"
    cv2.imwrite(str(viz_path), composite)


def evaluate_model(
    config: ModelConfig,
    samples: List[Tuple[Path, Path]],
    device: str,
    save_root: Path,
    quality_thresholds: Dict[str, float],
    obb_runner: Optional[OBBMetricRunner] = None,
) -> Dict:
    model_label = config.model_type or config.metadata.get("model") or config.runner
    print(f"Evaluating {config.name} ({model_label})")
    if config.runner == "sam":
        runner = build_sam_predictor(config, device)
    elif config.runner == "texrnet":
        runner = build_texrnet_runner(config, device)
    elif config.runner == "gemini":
        runner = build_gemini_runner(config, device)
    else:
        raise ValueError(f"Unsupported runner type '{config.runner}' for model '{config.name}'.")

    per_image_metrics: List[Dict[str, float]] = []
    total_conf = defaultdict(int)
    model_slug = slugify(config.name)
    model_dir = save_root / model_slug

    for image_path, gt_path in tqdm(samples, desc=config.name):
        image = load_image(image_path)
        gt_mask = load_mask(gt_path)

        if config.runner == "sam":
            if config.patch_mode:
                pred_mask = run_patch_inference(runner, image)
            else:
                pred_mask = run_full_inference(runner, image)
        elif config.runner == "texrnet":
            pred_mask = run_texrnet_inference(runner, image)
        elif config.runner == "gemini":
            pred_mask = run_gemini_inference(runner, image)
        else:
            raise ValueError(f"Unsupported runner type '{config.runner}'.")

        if pred_mask.shape != gt_mask.shape:
            pred_mask = cv2.resize(
                pred_mask.astype(np.uint8),
                (gt_mask.shape[1], gt_mask.shape[0]),
                interpolation=cv2.INTER_NEAREST,
            ).astype(bool)

        obb_metrics: Optional[Dict[str, float]] = None
        if obb_runner is not None:
            obb_mask = pred_mask
            if obb_mask.shape != image.shape[:2]:
                obb_mask = cv2.resize(
                    obb_mask.astype(np.uint8),
                    (image.shape[1], image.shape[0]),
                    interpolation=cv2.INTER_NEAREST,
                ).astype(bool)
            obb_metrics = obb_runner.compute_metrics(image_path, obb_mask)

        conf = compute_confusion(pred_mask, gt_mask)
        for key, value in conf.items():
            total_conf[key] += value
        image_metrics = metrics_from_confusion(conf)
        if obb_metrics is not None:
            image_metrics.update(obb_metrics)
        image_metrics["image_id"] = image_path.stem
        threshold_iou = quality_thresholds.get("iou")
        if threshold_iou is not None:
            image_metrics["low_quality"] = bool(image_metrics["iou"] < threshold_iou)
        per_image_metrics.append(image_metrics)

        save_prediction_artifacts(
            image=image,
            gt_mask=gt_mask,
            pred_mask=pred_mask,
            model_dir=model_dir,
            image_id=image_path.stem,
            model_slug=model_slug,
        )

    aggregate_metrics = metrics_from_confusion(total_conf)
    summary_stats = summarize_metrics(per_image_metrics)
    aggregate_metrics.update(summary_stats)
    obb_summary = summarize_optional_metrics(
        per_image_metrics,
        ["obb_global_containment", "obb_global_purity", "obb_zero_purity_ratio"],
    )
    aggregate_metrics.update(obb_summary)

    return {
        "model_name": config.name,
        "model_type": model_label,
        "num_images": len(samples),
        "aggregate_metrics": aggregate_metrics,
        "per_image": per_image_metrics,
    }


def print_results(results: List[Dict]):
    header = f"{'Model':40s} {'IoU':>8s} {'Dice':>8s} {'Precision':>10s} {'Recall':>8s}"
    print(header)
    print("-" * len(header))
    for result in results:
        metrics = result["aggregate_metrics"]
        print(
            f"{result['model_name'][:40]:40s} "
            f"{metrics['iou']:.4f} "
            f"{metrics['dice']:.4f} "
            f"{metrics['precision']:.4f} "
            f"{metrics['recall']:.4f}"
        )
        obb_keys = [
            ("obb_global_containment_mean", "Containment"),
            ("obb_global_purity_mean", "Purity"),
            ("obb_zero_purity_ratio_mean", "ZeroRatio"),
        ]
        obb_parts = [
            f"{label}={metrics[key]:.4f}"
            for key, label in obb_keys
            if key in metrics
        ]
        if obb_parts:
            print(f"    OBB: {', '.join(obb_parts)}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate multiple text segmentation models.")
    parser.add_argument("--image-dir", type=str, default=DEFAULT_IMAGE_DIR, help="Directory with validation images.")
    parser.add_argument("--gt-dir", type=str, default=DEFAULT_GT_DIR, help="Directory with ground-truth masks.")
    parser.add_argument("--device", type=str, default=DEVICE, help="torch device to run inference on.")
    parser.add_argument(
        "--save-dir",
        type=str,
        default="evaluation_results",
        help="Root directory to store predictions, visualizations, and summaries.",
    )
    parser.add_argument(
        "--results-json",
        type=str,
        default=None,
        help="Custom path for summary JSON. Defaults to <save-dir>/evaluation_results.json.",
    )
    parser.add_argument(
        "--bad-iou-threshold",
        type=float,
        default=0.2,
        help="IoU threshold used to flag low-quality samples across all models.",
    )
    parser.add_argument(
        "--bad-samples-file",
        type=str,
        default=None,
        help="Optional custom path for the low-quality sample list. Defaults to <save-dir>/low_quality_samples.txt.",
    )
    parser.add_argument(
        "--models",
        type=str,
        nargs="+",
        default=None,
        help=f"List of model keys to evaluate. Available: {', '.join(sorted(AVAILABLE_MODELS.keys()))}",
    )
    parser.add_argument(
        "--obb-model",
        type=str,
        default=DEFAULT_OBB_MODEL_PATH,
        help=(
            "Path to a YOLO OBB checkpoint used for containment/purity metrics. "
            "Set to an empty string to skip OBB evaluation."
        ),
    )
    parser.add_argument("--obb-imgsz", type=int, default=512, help="Input dimension used for OBB inference.")
    parser.add_argument("--obb-conf", type=float, default=0.3, help="Confidence threshold for OBB inference.")
    parser.add_argument("--obb-iou", type=float, default=0.7, help="IoU threshold for OBB NMS.")
    parser.add_argument("--obb-max-det", type=int, default=300, help="Maximum number of OBB detections per image.")
    parser.add_argument("--obb-class-id", type=int, default=0, help="Class id considered as text for OBB metrics.")
    parser.add_argument(
        "--obb-device",
        type=str,
        default="0",
        help="Device string passed to Ultralytics YOLO (e.g., '0' for GPU 0, 'cpu' for CPU-only OBB inference).",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    image_dir = Path(args.image_dir).expanduser()
    gt_dir = Path(args.gt_dir).expanduser()
    save_root = Path(args.save_dir).expanduser()
    ensure_dir(save_root)

    if not image_dir.exists():
        raise FileNotFoundError(f"Image directory not found: {image_dir}")
    if not gt_dir.exists():
        raise FileNotFoundError(f"GT directory not found: {gt_dir}")

    samples = collect_samples(image_dir, gt_dir)
    quality_thresholds = {"iou": args.bad_iou_threshold}
    obb_runner: Optional[OBBMetricRunner] = None
    obb_model_path = (args.obb_model or "").strip()
    if obb_model_path:
        obb_config = OBBMetricConfig(
            model_path=obb_model_path,
            imgsz=args.obb_imgsz,
            conf=args.obb_conf,
            iou=args.obb_iou,
            max_det=args.obb_max_det,
            class_id=args.obb_class_id,
            device=args.obb_device,
        )
        obb_runner = build_obb_metric_runner(obb_config)
    selected_model_keys = args.models if args.models is not None else DEFAULT_MODEL_KEYS
    configs: List[ModelConfig] = []
    for key in selected_model_keys:
        if key not in AVAILABLE_MODELS:
            raise ValueError(f"Unknown model key '{key}'. Available options: {', '.join(sorted(AVAILABLE_MODELS.keys()))}")
        configs.append(dataclass_replace(AVAILABLE_MODELS[key]))

    results = []
    for config in configs:
        try:
            result = evaluate_model(
                config,
                samples,
                args.device,
                save_root,
                quality_thresholds,
                obb_runner=obb_runner,
            )
            results.append(result)
        except Exception as exc:
            print(f"[Error] Failed to evaluate {config.name}: {exc}")

    print_results(results)

    joined_slug = "_".join(slugify(cfg.name) for cfg in configs) if configs else "no_models"
    results_json = (
        Path(args.results_json).expanduser()
        if args.results_json
        else save_root / f"evaluation_results_{joined_slug}.json"
    )
    image_quality_map: Dict[str, List[Dict[str, float]]] = {}
    for result in results:
        for metrics in result["per_image"]:
            entry = {
                "model_name": result["model_name"],
                "iou": float(metrics["iou"]),
                "dice": float(metrics["dice"]),
                "precision": float(metrics["precision"]),
                "recall": float(metrics["recall"]),
                "low_quality": bool(metrics.get("low_quality", False)),
            }
            image_quality_map.setdefault(metrics["image_id"], []).append(entry)

    bad_samples_info = []
    for image_id, metrics_list in sorted(image_quality_map.items()):
        if len(metrics_list) == len(results) and all(m["low_quality"] for m in metrics_list):
            bad_samples_info.append({"image_id": image_id, "metrics": metrics_list})

    bad_samples_path = (
        Path(args.bad_samples_file).expanduser()
        if args.bad_samples_file
        else save_root / f"low_quality_samples_{joined_slug}.txt"
    )
    with open(bad_samples_path, "w", encoding="utf-8") as f:
        f.write(f"# Images with IoU < {quality_thresholds['iou']} across all evaluated models\n")
        f.write("# image_id | model_name: IoU, Dice, Precision, Recall\n\n")
        for item in bad_samples_info:
            f.write(f"{item['image_id']}\n")
            for metric in item["metrics"]:
                f.write(
                    f"  {metric['model_name']}: "
                    f"IoU={metric['iou']:.4f}, Dice={metric['dice']:.4f}, "
                    f"Precision={metric['precision']:.4f}, Recall={metric['recall']:.4f}\n"
                )
            f.write("\n")

    payload = {
        "quality_thresholds": quality_thresholds,
        "models": results,
        "low_quality_samples": bad_samples_info,
    }
    with open(results_json, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    print(f"Saved detailed results to {results_json}")
    print(f"Flagged {len(bad_samples_info)} low-quality samples -> {bad_samples_path}")


if __name__ == "__main__":
    main()

