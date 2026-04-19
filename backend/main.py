import os
import sys
import threading
import time
import logging
import json
import csv
from datetime import datetime
from functools import partial
from pathlib import Path
from typing import Generator, Optional, Literal
from urllib.parse import urlparse, urlunparse

import cv2
import numpy as np
import torch
import torch.nn as nn
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel, Field


PROJECT_ROOT = Path(__file__).resolve().parents[1]
BACKEND_DIR = Path(__file__).resolve().parent
DEFAULT_ENSEMBLE_DIR = PROJECT_ROOT / "backend"  / "spacenet5"
DEFAULT_MODEL_WEIGHTS = DEFAULT_ENSEMBLE_DIR / "spacenet_irv_unet_inceptionresnetv2_1_best_dice"
OUTPUT_DIR = BACKEND_DIR / "outputs"

try:
	from dotenv import load_dotenv
	load_dotenv(BACKEND_DIR / ".env")
except Exception:
	pass

MODEL_OUTPUT_CHANNELS = 12
MODEL_INPUT_SIZE = 512
SPEED_CONVERSION_CSV = BACKEND_DIR / "selim_sef" / "speed_conversion_binned10.csv"

MATERIAL_CLASS_NAMES = [
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
MATERIAL_COLOR_PALETTE = np.array(
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

# Material IDs aligned with MATERIAL_CLASS_NAMES.
MAT_BACKGROUND = 0
MAT_ASPHALT = 1
MAT_GRAVEL = 3
MAT_SOIL = 6
MAT_BITUMEN = 7
MAT_BRICKS = 8


class VideoRequest(BaseModel):
	video_url: str = Field(..., description="Direct URL of the video stream or file")
	frame_skip: int = Field(1, ge=1, le=10, description="Process every Nth frame")
	confidence_threshold: float = Field(0.0, ge=0.0, le=1.0)
	save_output: bool = False
	output_filename: Optional[str] = None


app = FastAPI(title="Road Segmentation API", version="1.0.0")

logger = logging.getLogger("road-seg-api")
if not logging.getLogger().handlers:
	logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s :: %(message)s")
logger.setLevel(logging.INFO)
logger.info("module_import_complete")


def _log_stage(stage: str, **kwargs) -> None:
	if kwargs:
		details = " ".join(f"{k}={v}" for k, v in kwargs.items())
		logger.info("stage=%s %s", stage, details)
	else:
		logger.info("stage=%s", stage)

app.add_middleware(
	CORSMiddleware,
	allow_origins=["*"],
	allow_credentials=True,
	allow_methods=["*"],
	allow_headers=["*"],
)


def _resolve_model_path() -> Path:
	env_path = os.getenv("MODEL_PATH")
	if env_path:
		return Path(env_path).expanduser().resolve()
	return DEFAULT_MODEL_WEIGHTS.resolve()


def _build_channel_to_speed_mapping() -> dict[int, float]:
	"""
	Build a representative speed value (mph-like scale from dataset) per channel.
	Uses speed_conversion_binned10.csv when available and falls back to synthetic bins.
	"""
	channel_values: dict[int, list[float]] = {}
	if SPEED_CONVERSION_CSV.exists():
		try:
			with open(SPEED_CONVERSION_CSV, "r", encoding="utf-8") as f:
				reader = csv.DictReader(f)
				for row in reader:
					ch = int(row["channel"])
					speed = float(row["speed"])
					channel_values.setdefault(ch, []).append(speed)
		except Exception as exc:
			_log_stage("speed_mapping.csv_read_failed", error=repr(exc), path=SPEED_CONVERSION_CSV)

	if channel_values:
		return {
			channel: float(np.mean(values))
			for channel, values in channel_values.items()
		}

	# Fallback for environments without the speed conversion CSV.
	return {ch: float(5 + ch * 5) for ch in range(MODEL_OUTPUT_CHANNELS)}


def _speed_to_material_id(speed_value: float) -> int:
	"""
	Map representative speed to likely road material.
	Low speeds tend to unpaved/rough surfaces, higher speeds tend to paved roads.
	"""
	if speed_value <= 12:
		return MAT_SOIL
	if speed_value <= 22:
		return MAT_GRAVEL
	if speed_value <= 32:
		return MAT_BRICKS
	if speed_value <= 44:
		return MAT_BITUMEN
	return MAT_ASPHALT


def _build_speed_channel_to_material_mapping() -> dict[int, int]:
	channel_speed = _build_channel_to_speed_mapping()
	mapping: dict[int, int] = {}
	for channel in range(MODEL_OUTPUT_CHANNELS):
		speed_value = channel_speed.get(channel)
		if speed_value is None:
			speed_value = channel_speed[max(channel_speed.keys())] if channel_speed else float(5 + channel * 5)
		mapping[channel] = _speed_to_material_id(speed_value)
	return mapping


SPEED_CHANNEL_TO_MATERIAL = _build_speed_channel_to_material_mapping()


def _normalize_state_dict_keys(state_dict: dict) -> dict:
	if any(str(k).startswith("module.") for k in state_dict.keys()):
		return {str(k)[len("module."):]: v for k, v in state_dict.items()}
	return state_dict


def _find_model_impl_root() -> Path:
	"""Locate model implementation folder by expected config and module files."""
	override = os.getenv("MODEL_IMPL_ROOT")
	if override:
		override_path = Path(override).expanduser().resolve()
		if override_path.exists():
			return override_path

	for candidate in BACKEND_DIR.iterdir():
		if not candidate.is_dir():
			continue
		has_config = (candidate / "configs" / "irv2.json").exists()
		has_models = (candidate / "models").is_dir() and (candidate / "models" / "unet.py").exists()
		if has_config and has_models:
			return candidate

	raise FileNotFoundError("Model implementation folder not found")


def _disable_pretrained_downloads(unet_module, encoder_name: str) -> None:
	encoder_params = unet_module.encoder_params
	if encoder_name not in encoder_params:
		return
	init_op = encoder_params[encoder_name].get("init_op")
	if init_op is not None:
		try:
			encoder_params[encoder_name]["init_op"] = partial(init_op, pretrained=None)
		except Exception:
			pass
	encoder_params[encoder_name]["url"] = None


def _load_model_impl(device: torch.device) -> nn.Module:
	ckpt_path = _resolve_model_path()
	model_impl_root = _find_model_impl_root()
	default_config_path = model_impl_root / "configs" / "irv2.json"
	config_path = Path(os.getenv("MODEL_CONFIG", str(default_config_path))).expanduser().resolve()

	if not ckpt_path.exists():
		raise FileNotFoundError(f"model checkpoint not found: {ckpt_path}")
	if not config_path.exists():
		raise FileNotFoundError(f"model config not found: {config_path}")

	impl_root_str = str(model_impl_root.resolve())
	if impl_root_str not in sys.path:
		sys.path.append(impl_root_str)

	import importlib

	model_lib = importlib.import_module("models")
	unet_module = importlib.import_module("models.unet")

	with open(config_path, "r", encoding="utf-8") as f:
		conf = json.load(f)

	encoder_name = conf["encoder"]
	_disable_pretrained_downloads(unet_module, encoder_name)

	model = model_lib.__dict__[conf["network"]](
		seg_classes=MODEL_OUTPUT_CHANNELS,
		backbone_arch=encoder_name,
	).to(device)

	checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
	state_dict = checkpoint["state_dict"] if isinstance(checkpoint, dict) and "state_dict" in checkpoint else checkpoint
	state_dict = _normalize_state_dict_keys(state_dict)
	missing, unexpected = model.load_state_dict(state_dict, strict=True)
	if missing or unexpected:
		raise RuntimeError(
			"model state dict mismatch: "
			f"missing={len(missing)} unexpected={len(unexpected)}"
		)

	model.eval()
	_log_stage(
		"model.loaded",
		checkpoint=ckpt_path,
		config=config_path,
		channels=MODEL_OUTPUT_CHANNELS,
	)
	return model

def _load_model() -> tuple[torch.nn.Module, torch.device]:
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	_log_stage("model.backend", backend="custom", device=device)
	model = _load_model_impl(device)
	return model, device


MODEL: Optional[torch.nn.Module] = None
# Keep import-time initialization lightweight; runtime device is resolved in _load_model().
DEVICE = torch.device("cpu")
MODEL_LOAD_ERROR: Optional[str] = None
MODEL_LOADING = False
MODEL_LOAD_LOCK = threading.Lock()


def _load_model_once() -> None:
	global MODEL, DEVICE, MODEL_LOAD_ERROR, MODEL_LOADING
	with MODEL_LOAD_LOCK:
		if MODEL is not None:
			return
		if MODEL_LOADING:
			return
		MODEL_LOADING = True
		_log_stage("model.init_start")

	try:
		loaded_model, loaded_device = _load_model()
		MODEL = loaded_model
		DEVICE = loaded_device
		MODEL_LOAD_ERROR = None
		_log_stage("model.init_success", device=DEVICE)
	except Exception as exc:
		MODEL = None
		DEVICE = torch.device("cpu")
		MODEL_LOAD_ERROR = str(exc)
		_log_stage("model.init_failed", error=MODEL_LOAD_ERROR)
	finally:
		MODEL_LOADING = False


def _start_model_loading_background() -> None:
	if MODEL is not None or MODEL_LOADING:
		return
	threading.Thread(target=_load_model_once, daemon=True).start()


@app.on_event("startup")
def _startup() -> None:
	# Start model loading in the background so API startup is responsive.
	_log_stage("app.startup")
	_start_model_loading_background()


def _predict_mask(frame_bgr: np.ndarray, confidence_threshold: float = 0.0) -> np.ndarray:
	original_h, original_w = frame_bgr.shape[:2]
	rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
	resized = cv2.resize(rgb, (MODEL_INPUT_SIZE, MODEL_INPUT_SIZE), interpolation=cv2.INTER_LINEAR)

	tensor = torch.from_numpy(resized).float().permute(2, 0, 1).unsqueeze(0) / 255.0
	tensor = tensor.to(DEVICE)

	with torch.inference_mode():
		logits = MODEL(tensor)
		probs = torch.sigmoid(logits)
		conf, pred = torch.max(probs, dim=1)
		# Use 0.5 as default threshold for black background when API threshold is not provided.
		threshold = confidence_threshold if confidence_threshold > 0 else 0.5

	pred_np = pred.squeeze(0).detach().cpu().numpy().astype(np.uint8)
	conf_np = conf.squeeze(0).detach().cpu().numpy()
	label_np = np.where(conf_np >= threshold, pred_np + 1, 0).astype(np.uint8)
	return cv2.resize(label_np, (original_w, original_h), interpolation=cv2.INTER_NEAREST)


def _map_speed_labels_to_material_labels(speed_labels: np.ndarray) -> np.ndarray:
	material_labels = np.zeros_like(speed_labels, dtype=np.uint8)
	for speed_channel, material_id in SPEED_CHANNEL_TO_MATERIAL.items():
		speed_label_id = speed_channel + 1
		material_labels[speed_labels == speed_label_id] = material_id
	return material_labels


def _colorize_labels(label_map: np.ndarray) -> np.ndarray:
	material_labels = _map_speed_labels_to_material_labels(label_map)
	return MATERIAL_COLOR_PALETTE[material_labels]


def _normalize_video_url(raw_url: str) -> str:
	"""
	Normalize friendly inputs like "127.0.0.1/video-feed" to a URL OpenCV can open.
	Defaults localhost without port to VIDEO_SOURCE_PORT or 6969.
	"""
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
		default_port = int(os.getenv("VIDEO_SOURCE_PORT", "6969"))
		netloc = f"{parsed.hostname}:{default_port}"
		parsed = parsed._replace(netloc=netloc)

	return urlunparse(parsed)


def _build_video_url_candidates(raw_url: str) -> list[str]:
	"""Generate likely stream URL variants for localhost-style inputs."""
	normalized = _normalize_video_url(raw_url)
	parsed = urlparse(normalized)

	candidates = [normalized]
	if parsed.hostname in {"127.0.0.1", "localhost"}:
		base = f"{parsed.scheme}://{parsed.netloc}"
		path = parsed.path or ""
		# If user passed a bare host or unexpected path, try common local feed routes.
		if path in {"", "/"}:
			candidates.extend([
				base + "/video-feed",
				base + "/video-stream.mp4",
			])
		elif path == "/video-feed":
			candidates.append(base + "/video-stream.mp4")
		elif path == "/video-stream.mp4":
			candidates.append(base + "/video-feed")

	# De-duplicate while preserving order.
	seen = set()
	ordered: list[str] = []
	for c in candidates:
		if c not in seen:
			seen.add(c)
			ordered.append(c)
	return ordered


def _validate_video_source(video_url: str) -> None:
	"""Open and close once to fail fast before starting a chunked response."""
	_log_stage("video.validate_start", url=video_url)
	cap = cv2.VideoCapture(video_url)
	try:
		if not cap.isOpened():
			_log_stage("video.validate_open_failed", url=video_url)
			raise HTTPException(status_code=400, detail="Unable to open video URL")
		ok = False
		# Some streams need a short warm-up before the first frame is available.
		for _ in range(10):
			ok, _ = cap.read()
			if ok:
				break
			time.sleep(0.05)
		if not ok:
			_log_stage("video.validate_read_failed", url=video_url)
			raise HTTPException(status_code=400, detail="Video URL opened but no frames were readable")
		_log_stage("video.validate_ok", url=video_url)
	finally:
		cap.release()


def _select_working_video_url(raw_video_url: str) -> str:
	last_error: Optional[Exception] = None
	tried: list[str] = []
	for candidate in _build_video_url_candidates(raw_video_url):
		tried.append(candidate)
		_log_stage("video.candidate_try", url=candidate)
		try:
			_validate_video_source(candidate)
			_log_stage("video.candidate_selected", url=candidate)
			return candidate
		except Exception as exc:
			last_error = exc
			_log_stage("video.candidate_rejected", url=candidate, error=repr(exc))
			continue

	if isinstance(last_error, HTTPException):
		raise HTTPException(
			status_code=last_error.status_code,
			detail=f"Could not open video source. Tried: {tried}",
		)
	raise HTTPException(status_code=400, detail=f"Could not open video source. Tried: {tried}")


def _frame_stream(request: VideoRequest) -> Generator[bytes, None, None]:
	_log_stage(
		"stream.open",
		url=request.video_url,
		frame_skip=request.frame_skip,
		threshold=request.confidence_threshold,
	)
	cap = cv2.VideoCapture(request.video_url)
	if not cap.isOpened():
		_log_stage("stream.open_failed", url=request.video_url)
		return

	writer = None
	frame_index = 0
	processed_frames = 0

	if request.save_output:
		OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
		fps = cap.get(cv2.CAP_PROP_FPS)
		if fps <= 0 or np.isnan(fps):
			fps = 20.0

		width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
		height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

		filename = request.output_filename or f"mask_{datetime.now().strftime('%Y%m%d_%H%M%S')}.mp4"
		output_path = OUTPUT_DIR / filename
		fourcc = cv2.VideoWriter_fourcc(*"mp4v")
		writer = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))

	try:
		while True:
			ok, frame = cap.read()
			if not ok:
				_log_stage("stream.read_end", url=request.video_url, processed=processed_frames)
				break

			frame_index += 1
			if request.frame_skip > 1 and (frame_index % request.frame_skip != 0):
				continue

			speed_label_map = _predict_mask(frame, confidence_threshold=request.confidence_threshold)
			mask_frame_rgb = _colorize_labels(speed_label_map)
			# OpenCV encode/write expect BGR channel order.
			mask_frame_bgr = cv2.cvtColor(mask_frame_rgb, cv2.COLOR_RGB2BGR)
			processed_frames += 1
			if processed_frames == 1:
				_log_stage("stream.first_frame_processed", url=request.video_url, frame_shape=frame.shape)
			elif processed_frames % 60 == 0:
				_log_stage("stream.progress", url=request.video_url, processed=processed_frames)

			if writer is not None:
				writer.write(mask_frame_bgr)

			success, encoded = cv2.imencode(".jpg", mask_frame_bgr)
			if not success:
				continue

			yield (
				b"--frame\r\n"
				b"Content-Type: image/jpeg\r\n\r\n" + encoded.tobytes() + b"\r\n"
			)
	except Exception as exc:
		# Keep stream termination graceful so clients do not get abrupt chunking failures.
		_log_stage("stream.error", url=request.video_url, processed=processed_frames, error=repr(exc))
	finally:
		cap.release()
		if writer is not None:
			writer.release()
		_log_stage("stream.closed", url=request.video_url, processed=processed_frames)


@app.get("/health")
def health_check() -> dict:
	if MODEL_LOADING:
		status = "loading"
	elif MODEL is not None:
		status = "ok"
	else:
		status = "error"

	return {
		"status": status,
		"device": str(DEVICE),
		"model_path": str(_resolve_model_path()),
		"speed_to_material_mapping": {
			str(ch): MATERIAL_CLASS_NAMES[mid]
			for ch, mid in SPEED_CHANNEL_TO_MATERIAL.items()
		},
		"model_error": MODEL_LOAD_ERROR,
		"model_loading": MODEL_LOADING,
		"model_loaded": MODEL is not None,
		"cuda_available": torch.cuda.is_available(),
	}


@app.post("/segment/stream")
def segment_stream(request: VideoRequest) -> StreamingResponse:
	_log_stage(
		"request.segment_stream",
		url=request.video_url,
		frame_skip=request.frame_skip,
		threshold=request.confidence_threshold,
	)
	if MODEL_LOADING:
		raise HTTPException(status_code=503, detail="Model is still loading, please retry shortly")
	if MODEL is None:
		raise HTTPException(status_code=500, detail=f"Model not loaded: {MODEL_LOAD_ERROR}")

	request.video_url = _select_working_video_url(request.video_url)

	return StreamingResponse(
		_frame_stream(request),
		media_type="multipart/x-mixed-replace; boundary=frame",
	)


@app.get("/segment/stream-url")
def segment_stream_url(
	video_url: str = Query(..., description="Direct URL of the video stream or file"),
	frame_skip: int = Query(1, ge=1, le=10),
	confidence_threshold: float = Query(0.0, ge=0.0, le=1.0),
) -> StreamingResponse:
	_log_stage(
		"request.segment_stream_url",
		url=video_url,
		frame_skip=frame_skip,
		threshold=confidence_threshold,
	)
	if MODEL_LOADING:
		raise HTTPException(status_code=503, detail="Model is still loading, please retry shortly")
	if MODEL is None:
		raise HTTPException(status_code=500, detail=f"Model not loaded: {MODEL_LOAD_ERROR}")

	normalized_video_url = _select_working_video_url(video_url)

	request = VideoRequest(
		video_url=normalized_video_url,
		frame_skip=frame_skip,
		confidence_threshold=confidence_threshold,
	)

	return StreamingResponse(
		_frame_stream(request),
		media_type="multipart/x-mixed-replace; boundary=frame",
	)


@app.get("/segment/download/{filename}")
def download_mask_video(filename: str) -> FileResponse:
	file_path = OUTPUT_DIR / filename
	if not file_path.exists():
		raise HTTPException(status_code=404, detail="File not found")

	return FileResponse(path=file_path, filename=filename, media_type="video/mp4")
