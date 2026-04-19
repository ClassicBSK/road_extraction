import os
import sys
import threading
import logging
import zipfile
import time
import types
from datetime import datetime
from pathlib import Path
from typing import Generator, Optional
from urllib.parse import urlparse, urlunparse

import cv2
import numpy as np
import torch
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel, Field


PROJECT_ROOT = Path(__file__).resolve().parents[1]
BACKEND_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BACKEND_DIR / "outputs" / "hsi_msi"
DEFAULT_WEIGHTS_PATH = PROJECT_ROOT / "adi_code" / "best_material_unet.pth"

NUM_CLASSES = 9
MSI_CHANNELS = 8
CLASS_NAMES = [
    "Background",
    "Asphalt",
    "Meadows",
    "Gravel",
    "Trees",
    "Metal",
    "Soil",
    "Bitumen",
    "Bricks",
]

ROAD_MODEL_INPUT_SIZE = 512

# tab20-like palette for 9 classes
CLASS_COLORS = np.array(
    [
        [0, 0, 0],
        [220, 20, 60],
        [34, 139, 34],
        [255, 165, 0],
        [0, 128, 0],
        [169, 169, 169],
        [139, 69, 19],
        [75, 0, 130],
        [178, 34, 34],
    ],
    dtype=np.uint8,
)


class HSISegmentRequest(BaseModel):
    hsi_mat_path: str = Field(..., description="Path to .mat file containing HSI cube")
    hsi_key: str = Field("paviaU", description="Key name inside HSI .mat file")
    gt_mat_path: Optional[str] = Field(None, description="Optional path to GT .mat file")
    gt_key: str = Field("paviaU_gt", description="Key name inside GT .mat file")
    save_output: bool = Field(True, description="Save output files under backend/outputs/hsi_msi")
    output_prefix: Optional[str] = Field(None, description="Optional prefix for output filenames")


class VideoRequest(BaseModel):
    # Keep field names aligned with main.py for drop-in route compatibility.
    video_url: str = Field(..., description="URL of video stream or HSI .mat file path")
    frame_skip: int = Field(1, ge=1, le=10, description="Process every Nth frame")
    confidence_threshold: float = Field(0.0, ge=0.0, le=1.0, description="Confidence threshold for material classification")
    save_output: bool = False
    output_filename: Optional[str] = None
    hsi_key: str = Field("paviaU", description="Key name inside HSI .mat file")
    gt_mat_path: Optional[str] = Field(None, description="Optional path to GT .mat file")
    gt_key: str = Field("paviaU_gt", description="Key name inside GT .mat file")


app = FastAPI(title="HSI-MSI Material Segmentation API", version="1.0.0")
logger = logging.getLogger("hsi-msi-api")
if not logging.getLogger().handlers:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s :: %(message)s")
logger.setLevel(logging.INFO)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

MODEL: Optional[torch.nn.Module] = None
ROAD_MODEL: Optional[torch.nn.Module] = None
DEVICE = torch.device("cpu")
ROAD_DEVICE = torch.device("cpu")
MODEL_LOAD_ERROR: Optional[str] = None
MODEL_LOADING = False
MODEL_LOCK = threading.Lock()


def _log_stage(stage: str, **kwargs) -> None:
    if kwargs:
        details = " ".join(f"{k}={v}" for k, v in kwargs.items())
        logger.info("stage=%s %s", stage, details)
    else:
        logger.info("stage=%s", stage)


def _resolve_weights_path() -> Path:
    env_path = os.getenv("HSI_MSI_MODEL_PATH")
    if env_path:
        return Path(env_path).expanduser().resolve()

    # Prefer direct checkpoints to avoid repacking extracted torch-save folders.
    if DEFAULT_WEIGHTS_PATH.exists():
        return DEFAULT_WEIGHTS_PATH.resolve()
    raise FileNotFoundError(
        "No direct checkpoint found. Expected: "
        f"{DEFAULT_WEIGHTS_PATH}. "
        "Set HSI_MSI_MODEL_PATH to a .pth/.pt/.ckpt file to override."
    )


def _choose_checkpoint_file(path: Path) -> Path:
    if path.is_file():
        return path

    if not path.exists():
        raise FileNotFoundError(f"Weights path does not exist: {path}")

    if (path / "data.pkl").exists() and (path / "data").is_dir():
        return _repack_extracted_torch_checkpoint(path)

    candidates = []
    for pattern in ("*.pth", "*.pt", "*.ckpt"):
        candidates.extend(path.rglob(pattern))

    if not candidates:
        raise FileNotFoundError(
            "No checkpoint file found. Set HSI_MSI_MODEL_PATH to a .pth/.pt file "
            f"or a folder containing one. Current path: {path}"
        )

    candidates.sort()
    return candidates[0]


def _repack_extracted_torch_checkpoint(extracted_dir: Path) -> Path:
    """
    Repack an extracted torch.save zip archive directory into a .pth file.
    Expected structure includes data.pkl and data/.
    """
    cache_dir = BACKEND_DIR / ".cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    repacked_path = cache_dir / f"{extracted_dir.name}_repacked.pth"

    # Rebuild each startup to avoid stale/invalid repacks when format logic changes.
    if repacked_path.exists():
        repacked_path.unlink()

    with zipfile.ZipFile(repacked_path, mode="w", compression=zipfile.ZIP_STORED) as zf:
        for file_path in sorted(extracted_dir.rglob("*")):
            if not file_path.is_file():
                continue
            relative = file_path.relative_to(extracted_dir).as_posix()
            arcname = f"{extracted_dir.name}/{relative}"
            zf.write(file_path, arcname=arcname)

    _log_stage("model.repacked_checkpoint", source=str(extracted_dir), target=str(repacked_path))
    return repacked_path


def _build_model(device: torch.device) -> torch.nn.Module:
    try:
        import segmentation_models_pytorch as smp
    except Exception as exc:
        raise RuntimeError(
            "segmentation-models-pytorch is required but not available"
        ) from exc

    model = smp.Unet(
        encoder_name="resnet18",
        encoder_weights=None,
        in_channels=MSI_CHANNELS,
        classes=NUM_CLASSES,
    ).to(device)
    return model


def _normalize_state_dict_for_loading(state_dict: dict) -> dict:
    fixed = {}
    for key, value in state_dict.items():
        normalized = str(key)
        if normalized.startswith("module."):
            normalized = normalized[len("module."):]
        if normalized.startswith("model."):
            normalized = normalized[len("model."):]
        fixed[normalized] = value
    return fixed


def _safe_torch_load(checkpoint_file: Path, device: torch.device):
    # Prefer tensor-only loading to avoid failures when old pickled module paths are missing.
    try:
        return torch.load(checkpoint_file, map_location=device, weights_only=True)
    except Exception:
        return torch.load(checkpoint_file, map_location=device, weights_only=False)


def _load_model() -> tuple[torch.nn.Module, torch.device]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    weights_path = _resolve_weights_path()
    checkpoint_file = _choose_checkpoint_file(weights_path)

    model = _build_model(device)

    checkpoint = _safe_torch_load(checkpoint_file, device)

    state_dict = checkpoint
    if isinstance(checkpoint, torch.nn.Module):
        state_dict = checkpoint.state_dict()
    elif isinstance(checkpoint, dict):
        if "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]
        elif "model_state_dict" in checkpoint:
            state_dict = checkpoint["model_state_dict"]

    if isinstance(state_dict, dict):
        state_dict = _normalize_state_dict_for_loading(state_dict)

    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing or unexpected:
        _log_stage(
            "model.partial_load",
            missing=len(missing),
            unexpected=len(unexpected),
            checkpoint=str(checkpoint_file),
        )

    model.eval()
    _log_stage("model.loaded", device=device, checkpoint=str(checkpoint_file))
    return model, device


def _load_road_model() -> tuple[torch.nn.Module, torch.device]:
    road_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Reuse the existing road-speed segmentation model implementation from main.py.
    backend_dir_str = str(BACKEND_DIR.resolve())
    if backend_dir_str not in sys.path:
        sys.path.append(backend_dir_str)

    import importlib

    road_main = importlib.import_module("main")
    _load_model_impl = getattr(road_main, "_load_model_impl")

    road_model = _load_model_impl(road_device)
    road_model.eval()
    _log_stage("road_model.loaded", device=str(road_device))
    return road_model, road_device


def _load_model_once() -> None:
    global MODEL, ROAD_MODEL, DEVICE, ROAD_DEVICE, MODEL_LOAD_ERROR, MODEL_LOADING
    with MODEL_LOCK:
        if MODEL is not None or MODEL_LOADING:
            return
        MODEL_LOADING = True

    _log_stage("model.init_start")
    try:
        model, device = _load_model()
        road_model, road_device = _load_road_model()

        MODEL = model
        ROAD_MODEL = road_model
        DEVICE = device
        ROAD_DEVICE = road_device
        MODEL_LOAD_ERROR = None
        _log_stage("model.init_success", material_device=str(DEVICE), road_device=str(ROAD_DEVICE))
    except Exception as exc:
        MODEL = None
        ROAD_MODEL = None
        DEVICE = torch.device("cpu")
        ROAD_DEVICE = torch.device("cpu")
        MODEL_LOAD_ERROR = str(exc)
        _log_stage("model.init_failed", error=MODEL_LOAD_ERROR)
    finally:
        MODEL_LOADING = False


def _start_background_load() -> None:
    if MODEL is not None or MODEL_LOADING:
        return
    threading.Thread(target=_load_model_once, daemon=True).start()


@app.on_event("startup")
def _startup() -> None:
    _log_stage("app.startup")
    _start_background_load()


def _load_mat_array(mat_path: Path, key: str) -> np.ndarray:
    try:
        import scipy.io
    except Exception as exc:
        raise RuntimeError("scipy is required to read .mat files") from exc

    if not mat_path.exists():
        raise FileNotFoundError(f"MAT file not found: {mat_path}")

    payload = scipy.io.loadmat(str(mat_path))
    if key not in payload:
        available = [k for k in payload.keys() if not k.startswith("__")]
        raise KeyError(f"Key '{key}' not found in {mat_path}. Available keys: {available}")

    arr = np.asarray(payload[key])
    return arr


def hsi_to_msi_road_optimized(hsi: np.ndarray) -> np.ndarray:
    """Convert 103-band PaviaU HSI to 8 road-focused MSI bands."""
    if hsi.ndim != 3:
        raise ValueError(f"Expected HSI with 3 dims (H, W, C), got shape: {hsi.shape}")
    if hsi.shape[2] < 103:
        raise ValueError(f"Expected at least 103 spectral bands, got: {hsi.shape[2]}")

    h, w, _ = hsi.shape
    msi = np.zeros((h, w, MSI_CHANNELS), dtype=np.float32)

    road_groups = {
        0: range(8, 18),
        1: range(25, 35),
        2: range(42, 52),
        3: range(55, 65),
        4: range(70, 80),
        5: range(85, 95),
        6: range(95, 100),
        7: range(100, 103),
    }

    for i, band_range in road_groups.items():
        msi[:, :, i] = np.mean(hsi[:, :, list(band_range)], axis=2)

    return msi


def normalize_msi(msi: np.ndarray) -> np.ndarray:
    """Per-band min-max normalization to [0, 1], mirroring notebook behavior."""
    msi = msi.astype(np.float32)
    out = np.zeros_like(msi, dtype=np.float32)

    for b in range(msi.shape[2]):
        band = msi[:, :, b]
        b_min = float(np.min(band))
        b_max = float(np.max(band))
        denom = b_max - b_min
        if denom <= 1e-12:
            out[:, :, b] = 0.0
        else:
            out[:, :, b] = (band - b_min) / denom

    return out


def _predict_full_scene(msi_norm: np.ndarray) -> np.ndarray:
    if MODEL is None:
        raise RuntimeError("Model is not loaded")

    tensor = torch.from_numpy(msi_norm.transpose(2, 0, 1)).unsqueeze(0).float().to(DEVICE)
    with torch.inference_mode():
        logits = MODEL(tensor)
        pred = torch.argmax(logits, dim=1).squeeze(0).detach().cpu().numpy().astype(np.uint8)
    return pred


def _predict_mask_from_msi(msi_norm: np.ndarray) -> np.ndarray:
    pred = _predict_full_scene(msi_norm)
    return CLASS_COLORS[pred]


def _predict_labels_from_msi(msi_norm: np.ndarray) -> np.ndarray:
    return _predict_full_scene(msi_norm)


def _rgb_frame_to_msi_surrogate(frame_bgr: np.ndarray) -> np.ndarray:
    """
    Build an 8-channel surrogate MSI tensor from RGB video frames.
    This enables real-time URL stream inference with the HSI-trained model.
    """
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    r = rgb[:, :, 0]
    g = rgb[:, :, 1]
    b = rgb[:, :, 2]
    gray = 0.299 * r + 0.587 * g + 0.114 * b

    msi = np.stack(
        [
            b,
            g,
            r,
            0.5 * (r + g),
            0.5 * (g + b),
            0.5 * (r + b),
            gray,
            np.clip(g - r, 0.0, 1.0),
        ],
        axis=2,
    ).astype(np.float32)
    return normalize_msi(msi)


def _predict_mask_from_frame(frame_bgr: np.ndarray) -> np.ndarray:
    msi = _rgb_frame_to_msi_surrogate(frame_bgr)
    return _predict_mask_from_msi(msi)


def _predict_labels_from_frame(frame_bgr: np.ndarray) -> np.ndarray:
    msi = _rgb_frame_to_msi_surrogate(frame_bgr)
    return _predict_labels_from_msi(msi)


def _predict_road_binary_mask(frame_bgr: np.ndarray, confidence_threshold: float = 0.0) -> np.ndarray:
    if ROAD_MODEL is None:
        raise RuntimeError("Road model is not loaded")

    original_h, original_w = frame_bgr.shape[:2]
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(rgb, (ROAD_MODEL_INPUT_SIZE, ROAD_MODEL_INPUT_SIZE), interpolation=cv2.INTER_LINEAR)

    tensor = torch.from_numpy(resized).float().permute(2, 0, 1).unsqueeze(0) / 255.0
    tensor = tensor.to(ROAD_DEVICE)

    with torch.inference_mode():
        logits = ROAD_MODEL(tensor)
        probs = torch.sigmoid(logits)
        conf, _ = torch.max(probs, dim=1)
        threshold = confidence_threshold if confidence_threshold > 0 else 0.5

    conf_np = conf.squeeze(0).detach().cpu().numpy()
    road_mask = np.where(conf_np >= threshold, 1, 0).astype(np.uint8)
    road_mask = cv2.resize(road_mask, (original_w, original_h), interpolation=cv2.INTER_NEAREST)
    return road_mask


def _render_output_frame(frame_bgr: np.ndarray, request: VideoRequest) -> np.ndarray:
    labels = _predict_labels_from_frame(frame_bgr)
    road_mask = _predict_road_binary_mask(frame_bgr, confidence_threshold=request.confidence_threshold)
    masked_labels = np.where(road_mask > 0, labels, 0).astype(np.uint8)
    return cv2.cvtColor(CLASS_COLORS[masked_labels], cv2.COLOR_RGB2BGR)


def _normalize_video_url(raw_url: str) -> str:
    url = (raw_url or "").strip()
    if not url:
        raise HTTPException(status_code=400, detail="video_url is required")

    if url.startswith("//"):
        url = "http:" + url
    if "://" not in url:
        url = "http://" + url

    parsed = urlparse(url)
    if not parsed.hostname:
        raise HTTPException(status_code=400, detail="Invalid video_url")

    if parsed.hostname in {"127.0.0.1", "localhost"} and parsed.port is None:
        parsed = parsed._replace(netloc=f"{parsed.hostname}:9090")

    return urlunparse(parsed)


def _looks_like_url(value: str) -> bool:
    raw = (value or "").strip().lower()
    return raw.startswith("http://") or raw.startswith("https://") or raw.startswith("localhost") or raw.startswith("127.0.0.1")


def _validate_video_source(video_url: str) -> None:
    cap = cv2.VideoCapture(video_url)
    try:
        if not cap.isOpened():
            raise HTTPException(status_code=400, detail="Unable to open video URL")

        ok = False
        for _ in range(10):
            ok, _ = cap.read()
            if ok:
                break
            time.sleep(0.05)

        if not ok:
            raise HTTPException(status_code=400, detail="Video URL opened but no frames were readable")
    finally:
        cap.release()


def _frame_stream_url(request: VideoRequest) -> Generator[bytes, None, None]:
    normalized_url = _normalize_video_url(request.video_url)
    cap = cv2.VideoCapture(normalized_url)
    if not cap.isOpened():
        return

    writer = None
    frame_index = 0

    if request.save_output:
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps <= 0 or np.isnan(fps):
            fps = 20.0

        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        filename = request.output_filename or f"hsi_stream_{datetime.now().strftime('%Y%m%d_%H%M%S')}.mp4"
        output_path = OUTPUT_DIR / filename
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            frame_index += 1
            if request.frame_skip > 1 and (frame_index % request.frame_skip != 0):
                continue

            output_frame = _render_output_frame(frame, request)

            if writer is not None:
                writer.write(output_frame)

            success, encoded = cv2.imencode(".jpg", output_frame)
            if not success:
                continue

            yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + encoded.tobytes() + b"\r\n"
    finally:
        cap.release()
        if writer is not None:
            writer.release()


def _compute_iou_metrics(pred: np.ndarray, gt: np.ndarray) -> dict:
    metrics = {}
    per_class = {}
    ious = []

    for cls in range(NUM_CLASSES):
        gt_mask = gt == cls
        pred_mask = pred == cls
        union = np.logical_or(gt_mask, pred_mask).sum()
        if union == 0:
            iou = None
        else:
            intersection = np.logical_and(gt_mask, pred_mask).sum()
            iou = float(intersection / (union + 1e-8))
            if cls != 0:
                ious.append(iou)

        per_class[CLASS_NAMES[cls]] = iou

    metrics["per_class_iou"] = per_class
    metrics["miou_no_background"] = float(np.mean(ious)) if ious else 0.0
    return metrics


def _save_outputs(pred: np.ndarray, output_prefix: Optional[str]) -> dict:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    stem = output_prefix or f"hsi_msi_pred_{ts}"

    pred_npy_path = OUTPUT_DIR / f"{stem}.npy"
    pred_png_path = OUTPUT_DIR / f"{stem}.png"

    np.save(pred_npy_path, pred)
    color_mask = CLASS_COLORS[pred]
    cv2.imwrite(str(pred_png_path), cv2.cvtColor(color_mask, cv2.COLOR_RGB2BGR))

    return {
        "prediction_npy": str(pred_npy_path),
        "prediction_png": str(pred_png_path),
    }


@app.get("/health")
def health() -> dict:
    if MODEL_LOADING:
        status = "loading"
    elif MODEL is not None:
        status = "ok"
    else:
        status = "error"

    return {
        "status": status,
        "device": str(DEVICE),
        "road_device": str(ROAD_DEVICE),
        "model_loading": MODEL_LOADING,
        "model_loaded": MODEL is not None,
        "road_model_loaded": ROAD_MODEL is not None,
        "model_error": MODEL_LOAD_ERROR,
        "weights_path": str(_resolve_weights_path()),
        "cuda_available": torch.cuda.is_available(),
    }


@app.post("/segment/hsi")
def segment_hsi(request: HSISegmentRequest) -> dict:
    return _segment_core(
        hsi_mat_path=request.hsi_mat_path,
        hsi_key=request.hsi_key,
        gt_mat_path=request.gt_mat_path,
        gt_key=request.gt_key,
        save_output=request.save_output,
        output_prefix=request.output_prefix,
    )


def _segment_core(
    hsi_mat_path: str,
    hsi_key: str = "paviaU",
    gt_mat_path: Optional[str] = None,
    gt_key: str = "paviaU_gt",
    save_output: bool = False,
    output_prefix: Optional[str] = None,
) -> dict:
    if MODEL_LOADING:
        raise HTTPException(status_code=503, detail="Model is still loading, please retry shortly")
    if MODEL is None or ROAD_MODEL is None:
        raise HTTPException(status_code=500, detail=f"Model not loaded: {MODEL_LOAD_ERROR}")

    try:
        hsi_path = Path(hsi_mat_path).expanduser().resolve()
        hsi = _load_mat_array(hsi_path, hsi_key)

        msi_raw = hsi_to_msi_road_optimized(hsi)
        msi_norm = normalize_msi(msi_raw)
        pred = _predict_full_scene(msi_norm)

        output_files = _save_outputs(pred, output_prefix) if save_output else {}

        result = {
            "hsi_shape": tuple(int(v) for v in hsi.shape),
            "msi_shape": tuple(int(v) for v in msi_norm.shape),
            "prediction_shape": tuple(int(v) for v in pred.shape),
            "classes": CLASS_NAMES,
            "outputs": output_files,
        }

        if gt_mat_path:
            gt_path = Path(gt_mat_path).expanduser().resolve()
            gt = _load_mat_array(gt_path, gt_key)
            if gt.shape != pred.shape:
                raise ValueError(
                    f"GT shape {gt.shape} does not match prediction shape {pred.shape}"
                )
            result["metrics"] = _compute_iou_metrics(pred, gt)

        return result

    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except KeyError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Unexpected server error: {exc}") from exc


@app.post("/segment/stream")
def segment_stream(request: VideoRequest):
    # Route/method intentionally match main.py.
    if _looks_like_url(request.video_url):
        if MODEL_LOADING:
            raise HTTPException(status_code=503, detail="Model is still loading, please retry shortly")
        if MODEL is None or ROAD_MODEL is None:
            raise HTTPException(status_code=500, detail=f"Model not loaded: {MODEL_LOAD_ERROR}")

        normalized_url = _normalize_video_url(request.video_url)
        _validate_video_source(normalized_url)
        request.video_url = normalized_url
        return StreamingResponse(
            _frame_stream_url(request),
            media_type="multipart/x-mixed-replace; boundary=frame",
        )

    # For non-URL input, keep HSI MAT processing mode.
    output_prefix = request.output_filename if request.output_filename else None
    return _segment_core(
        hsi_mat_path=request.video_url,
        hsi_key=request.hsi_key,
        gt_mat_path=request.gt_mat_path,
        gt_key=request.gt_key,
        save_output=request.save_output,
        output_prefix=output_prefix,
    )


@app.get("/segment/stream-url")
def segment_stream_url(
    video_url: str = Query(..., description="URL of video stream or HSI .mat file path"),
    frame_skip: int = Query(1, ge=1, le=10),
    confidence_threshold: float = Query(0.0, ge=0.0, le=1.0),
    hsi_key: str = Query("paviaU", description="Key name inside HSI .mat file"),
    gt_mat_path: Optional[str] = Query(None, description="Optional path to GT .mat file"),
    gt_key: str = Query("paviaU_gt", description="Key name inside GT .mat file"),
    save_output: bool = Query(False),
    output_filename: Optional[str] = Query(None),
) -> StreamingResponse:
    # Keep the same query params as main.py plus optional HSI-specific params.
    if _looks_like_url(video_url):
        if MODEL_LOADING:
            raise HTTPException(status_code=503, detail="Model is still loading, please retry shortly")
        if MODEL is None:
            raise HTTPException(status_code=500, detail=f"Model not loaded: {MODEL_LOAD_ERROR}")

        normalized_url = _normalize_video_url(video_url)
        _validate_video_source(normalized_url)

        request = VideoRequest(
            video_url=normalized_url,
            frame_skip=frame_skip,
            confidence_threshold=confidence_threshold,
            save_output=save_output,
            output_filename=output_filename,
            hsi_key=hsi_key,
            gt_mat_path=gt_mat_path,
            gt_key=gt_key,
        )
        return StreamingResponse(
            _frame_stream_url(request),
            media_type="multipart/x-mixed-replace; boundary=frame",
        )

    _ = frame_skip
    _ = confidence_threshold
    return _segment_core(
        hsi_mat_path=video_url,
        hsi_key=hsi_key,
        gt_mat_path=gt_mat_path,
        gt_key=gt_key,
        save_output=save_output,
        output_prefix=output_filename,
    )


@app.get("/segment/download/{filename}")
def download_mask_video(filename: str) -> FileResponse:
    file_path = OUTPUT_DIR / filename
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="File not found")

    media_type = "application/octet-stream"
    if file_path.suffix.lower() == ".png":
        media_type = "image/png"
    elif file_path.suffix.lower() == ".npy":
        media_type = "application/octet-stream"

    return FileResponse(path=file_path, filename=filename, media_type=media_type)

