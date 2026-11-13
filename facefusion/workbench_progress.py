import base64
import json
import os
import time
from threading import Lock

import cv2


ENABLED = os.environ.get('FACEFUSION_WORKBENCH_PROGRESS') == '1'
PREFIX = 'VR180_PROGRESS '
_lock = Lock()
_phase = None
_completed = 0
_total = 0
_last_event = 0.0
_last_preview = 0.0


def encode_preview(frame):
	height, width = frame.shape[:2]
	scale = min(1, 720 / width, 360 / height)
	if scale < 1:
		frame = cv2.resize(frame, (max(1, round(width * scale)), max(1, round(height * scale))), interpolation=cv2.INTER_AREA)
	ok, data = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 78])
	return base64.b64encode(data).decode('ascii') if ok else None


def _emit(**data):
	# tqdm can leave a partial line on the merged stderr/stdout stream.
	print('\n' + PREFIX + json.dumps(data, separators=(',', ':')), flush=True)


def begin(phase, total=0):
	if ENABLED:
		global _phase, _completed, _total, _last_event, _last_preview
		with _lock:
			_phase, _completed, _last_event, _last_preview = phase, 0, 0.0, 0.0
			_total = total
			_emit(phase=phase)


def video_count(completed, total):
	if ENABLED:
		_emit(phase=_phase, completed=completed, total=total)


def processed_frame(frame, index):
	if ENABLED:
		global _completed, _last_event, _last_preview
		with _lock:
			_completed += 1
			now = time.monotonic()
			if now - _last_event < 0.35 and _completed != _total:
				return
			data = dict(phase=_phase, completed=_completed, total=_total)
			if now - _last_preview >= 0.9 or _completed == _total:
				data.update(frame=index, jpeg=encode_preview(frame))
				_last_preview = now
			_last_event = now
			_emit(**data)
