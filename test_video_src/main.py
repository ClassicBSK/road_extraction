import os
import time
from pathlib import Path
from typing import Generator

import cv2
import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, HTMLResponse, StreamingResponse


BASE_DIR = Path(__file__).resolve().parent
DEFAULT_IMAGE_PATH = BASE_DIR / "test_image.jpg"
DEFAULT_VIDEO_PATH = BASE_DIR / "test_video.mp4"

FPS = int(os.getenv("TEST_VIDEO_FPS", "12"))
FRAME_WIDTH = int(os.getenv("TEST_VIDEO_WIDTH", "1280"))
FRAME_HEIGHT = int(os.getenv("TEST_VIDEO_HEIGHT", "720"))
VIDEO_DURATION_SECONDS = int(os.getenv("TEST_VIDEO_DURATION_SECONDS", "30"))

app = FastAPI(title="Temporary Test Video Source", version="1.0.0")


def _resolve_input_image_path() -> Path:
	configured = os.getenv("TEST_IMAGE_PATH")
	if configured:
		return Path(configured).expanduser().resolve()

	preferred_candidates = [
		BASE_DIR / "test_image.tif",
		BASE_DIR / "test_image.tiff",
		BASE_DIR / "test_image.jpg",
		BASE_DIR / "test_image.png",
	]
	for candidate in preferred_candidates:
		if candidate.exists():
			return candidate.resolve()

	return DEFAULT_IMAGE_PATH.resolve()


def _to_bgr_uint8(image: np.ndarray) -> np.ndarray:
	if image is None:
		raise RuntimeError("Loaded image is None")

	if image.ndim == 2:
		# Single-channel TIFF/PNG/JPG -> convert to 3-channel BGR.
		if image.dtype != np.uint8:
			image = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
		return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

	if image.ndim == 3:
		if image.dtype != np.uint8:
			image = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

		channels = image.shape[2]
		if channels == 1:
			return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
		if channels >= 3:
			return image[:, :, :3]

	raise RuntimeError(f"Unsupported image shape: {image.shape}")


def _create_default_image_if_missing(image_path: Path) -> None:
	if image_path.exists():
		return

	canvas = np.zeros((FRAME_HEIGHT, FRAME_WIDTH, 3), dtype=np.uint8)
	for y in range(FRAME_HEIGHT):
		blue = int(30 + 120 * (y / max(1, FRAME_HEIGHT - 1)))
		green = int(40 + 140 * (y / max(1, FRAME_HEIGHT - 1)))
		canvas[y, :, :] = (blue, green, 18)

	cv2.putText(
		canvas,
		"TEST VIDEO SOURCE",
		(80, 130),
		cv2.FONT_HERSHEY_SIMPLEX,
		2.0,
		(240, 240, 240),
		4,
		cv2.LINE_AA,
	)
	cv2.putText(
		canvas,
		"Replace test_image.jpg with your own image",
		(80, 200),
		cv2.FONT_HERSHEY_SIMPLEX,
		1.0,
		(220, 220, 220),
		2,
		cv2.LINE_AA,
	)

	cv2.rectangle(canvas, (70, 260), (1210, 650), (255, 255, 255), 3)
	cv2.circle(canvas, (250, 455), 95, (0, 210, 255), -1)
	cv2.rectangle(canvas, (450, 350), (1140, 560), (45, 85, 215), -1)

	image_path.parent.mkdir(parents=True, exist_ok=True)
	cv2.imwrite(str(image_path), canvas)


def _read_base_image() -> np.ndarray:
	image_path = _resolve_input_image_path()
	_create_default_image_if_missing(image_path)

	frame = cv2.imread(str(image_path), cv2.IMREAD_UNCHANGED)
	if frame is None:
		raise RuntimeError(f"Unable to read test image: {image_path}")
	return _to_bgr_uint8(frame)


def _build_overlayed_frame(base_frame: np.ndarray, frame_idx: int) -> np.ndarray:
	frame = base_frame.copy()
	x_pos = 40 + (frame_idx * 14) % (FRAME_WIDTH - 80)
	cv2.line(frame, (x_pos, 0), (x_pos, FRAME_HEIGHT), (0, 255, 255), 2)

	now = time.strftime("%Y-%m-%d %H:%M:%S")
	cv2.putText(
		frame,
		f"LIVE TEST FEED | frame={frame_idx}",
		(40, FRAME_HEIGHT - 40),
		cv2.FONT_HERSHEY_SIMPLEX,
		0.9,
		(255, 255, 255),
		2,
		cv2.LINE_AA,
	)
	cv2.putText(
		frame,
		now,
		(40, FRAME_HEIGHT - 10),
		cv2.FONT_HERSHEY_SIMPLEX,
		0.7,
		(255, 255, 255),
		2,
		cv2.LINE_AA,
	)
	return frame


def _ensure_test_video_exists() -> Path:
	video_path = Path(os.getenv("TEST_VIDEO_PATH", str(DEFAULT_VIDEO_PATH))).expanduser().resolve()
	if video_path.exists() and video_path.stat().st_size > 0:
		return video_path

	base_frame = _read_base_image()
	resized_base = cv2.resize(base_frame, (FRAME_WIDTH, FRAME_HEIGHT), interpolation=cv2.INTER_LINEAR)
	video_path.parent.mkdir(parents=True, exist_ok=True)

	fourcc = cv2.VideoWriter_fourcc(*"mp4v")
	writer = cv2.VideoWriter(str(video_path), fourcc, max(1, FPS), (FRAME_WIDTH, FRAME_HEIGHT))
	if not writer.isOpened():
		raise RuntimeError(f"Unable to create video file: {video_path}")

	try:
		total_frames = max(1, FPS * VIDEO_DURATION_SECONDS)
		for frame_idx in range(total_frames):
			writer.write(_build_overlayed_frame(resized_base, frame_idx))
	finally:
		writer.release()

	return video_path


def _frame_generator() -> Generator[bytes, None, None]:
	try:
		base_frame = _read_base_image()
	except Exception as exc:
		raise HTTPException(status_code=500, detail=str(exc)) from exc

	resized_base = cv2.resize(base_frame, (FRAME_WIDTH, FRAME_HEIGHT), interpolation=cv2.INTER_LINEAR)
	interval = 1.0 / max(1, FPS)
	frame_idx = 0

	while True:
		frame = _build_overlayed_frame(resized_base, frame_idx)

		ok, encoded = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 88])
		if ok:
			yield (
				b"--frame\r\n"
				b"Content-Type: image/jpeg\r\n\r\n" + encoded.tobytes() + b"\r\n"
			)

		frame_idx += 1
		time.sleep(interval)


@app.get("/", response_class=HTMLResponse)
def root() -> HTMLResponse:
	image_path = _resolve_input_image_path()
	html = f"""
<!doctype html>
<html lang=\"en\">
<head>
	<meta charset=\"utf-8\" />
	<meta name=\"viewport\" content=\"width=device-width, initial-scale=1\" />
	<title>Temporary Test Video Source</title>
	<style>
		body {{ background:#08141d; color:#d9eef2; font-family:Segoe UI,Tahoma,sans-serif; margin:0; }}
		.wrap {{ max-width:1100px; margin:24px auto; padding:0 16px; }}
		.card {{ border:1px solid #305867; border-radius:12px; background:#0d202b; padding:16px; }}
		h1 {{ margin:0 0 8px; font-size:1.4rem; }}
		p {{ margin:6px 0; color:#b8d4da; }}
		img {{ width:100%; aspect-ratio:16/9; object-fit:cover; border-radius:10px; border:1px solid #2d5160; background:#061019; }}
		.links a {{ color:#86dfff; text-decoration:none; margin-right:12px; }}
	</style>
</head>
<body>
	<main class=\"wrap\">
		<section class=\"card\">
			<h1>Temporary Test Video Source</h1>
			<p>Current image: {image_path}</p>
			<p>Video endpoint: /video-stream.mp4</p>
			<div class=\"links\">
				<a href=\"/health\" target=\"_blank\">Health</a>
				<a href=\"/video-stream.mp4\" target=\"_blank\">Open MP4 Stream</a>
				<a href=\"/video-feed\" target=\"_blank\">Open Raw Stream</a>
			</div>
		</section>
		<section class=\"card\" style=\"margin-top:12px;\">
			<video controls autoplay muted loop playsinline style=\"width:100%; aspect-ratio:16/9; object-fit:cover; border-radius:10px; border:1px solid #2d5160; background:#061019;\">
				<source src=\"/video-stream.mp4\" type=\"video/mp4\" />
			</video>
		</section>
	</main>
</body>
</html>
"""
	return HTMLResponse(content=html)


@app.get("/health")
def health() -> dict:
	return {"status": "ok", "fps": FPS, "size": [FRAME_WIDTH, FRAME_HEIGHT]}


@app.get("/video-feed")
def video_feed() -> StreamingResponse:
	return StreamingResponse(
		_frame_generator(),
		media_type="multipart/x-mixed-replace; boundary=frame",
	)


@app.get("/video-stream.mp4")
def video_stream_mp4() -> FileResponse:
	try:
		video_path = _ensure_test_video_exists()
	except Exception as exc:
		raise HTTPException(status_code=500, detail=str(exc)) from exc

	return FileResponse(path=str(video_path), media_type="video/mp4", filename=video_path.name)


if __name__ == "__main__":
	import uvicorn

	uvicorn.run("main:app", host="0.0.0.0", port=9000, reload=True)
