from __future__ import annotations

import base64
import json
import math
import os
import shutil
import sys
import threading
import time
import uuid
from collections import OrderedDict, deque
from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np

# A directly launched Conda interpreter may not inherit the activation PATH.
if os.name == 'nt':
    conda_bin = Path(sys.prefix) / 'Library' / 'bin'
    if conda_bin.is_dir():
        os.environ['PATH'] = str(conda_bin) + os.pathsep + os.environ.get('PATH', '')

import vr180_sbs_pipeline as pipeline


ROOT = Path(__file__).resolve().parent
MODEL = ROOT.parent / '.assets' / 'models' / 'yoloface_12n-vr180.onnx'
DATA = ROOT / '.workbench'
VIDEO_SUFFIXES = {'.mp4', '.mov', '.mkv', '.webm', '.avi', '.m4v'}
FACE_SUFFIXES = {'.jpg', '.jpeg', '.png', '.webp', '.bmp'}
JOBS: dict = {}
LOCK = threading.Lock()
DETECT_LOCK = threading.Lock()
PLAYBACK: OrderedDict[str, Path] = OrderedDict()


def local_file(value, suffixes):
    if not isinstance(value, str) or not value.strip():
        raise ValueError('Please select a file.')
    path = Path(value.strip().strip('"')).resolve()
    if not path.is_file() or path.suffix.lower() not in suffixes:
        raise ValueError(f'File does not exist or has an unsupported format: {path}')
    return path


def playback_url(path):
    with LOCK:
        token = next((token for token, file in PLAYBACK.items() if file == path), None) or uuid.uuid4().hex
        PLAYBACK[token] = path
        PLAYBACK.move_to_end(token)
        if len(PLAYBACK) > 32:
            PLAYBACK.popitem(last=False)
    return f'/api/playback/{token}'


def playback_file(token):
    with LOCK:
        if token not in PLAYBACK:
            raise FileNotFoundError('Playback source is no longer available.')
        PLAYBACK.move_to_end(token)
        return PLAYBACK[token]


def open_video(path):
    cap = cv2.VideoCapture(str(path))
    try:
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if not cap.isOpened() or width < 4 or width % 2 or height < 2 or height % 2 or not math.isfinite(fps) or fps <= 0 or count < 1:
            raise ValueError('Could not read an even-sized SBS video with a valid frame rate.')
        return cap, {'width': width, 'height': height, 'fps': fps, 'count': count}
    except Exception:
        cap.release()
        raise


def read_frame(path, number):
    cap, info = open_video(path)
    try:
        if not 0 <= number < info['count']:
            raise ValueError('Frame number is outside the video.')
        if number:
            cap.set(cv2.CAP_PROP_POS_FRAMES, number)
        ok, frame = cap.read()
        if not ok:
            raise ValueError(f'Could not decode frame {number}.')
        return frame, info
    finally:
        cap.release()


def image_url(image, max_width=1440):
    if image.shape[1] > max_width:
        image = cv2.resize(image, (max_width, round(image.shape[0] * max_width / image.shape[1])), interpolation=cv2.INTER_AREA)
    ok, data = cv2.imencode('.jpg', image, [cv2.IMWRITE_JPEG_QUALITY, 88])
    if not ok:
        raise RuntimeError('Could not encode preview.')
    return 'data:image/jpeg;base64,' + base64.b64encode(data).decode('ascii')


@lru_cache(maxsize=2)
def detector(provider):
    import onnxruntime as ort

    if not MODEL.is_file():
        raise ValueError(f'Missing YOLOFace VR180 weights: {MODEL}')
    selected = 'CUDAExecutionProvider' if provider == 'cuda' else 'CPUExecutionProvider'
    if selected not in ort.get_available_providers():
        raise ValueError(f'{selected} unavailable. Select CPU or fix the CUDA environment.')
    providers = [(selected, {'cudnn_conv_algo_search': 'HEURISTIC'}), 'CPUExecutionProvider'] if provider == 'cuda' else [selected]
    session = ort.InferenceSession(str(MODEL), providers=providers)
    if session.get_providers()[0] != selected:
        raise ValueError('CUDA failed to load for the detector. Select CPU or fix the CUDA environment.')
    return session


def detect_eye(image, session, threshold):
    model_input = session.get_inputs()[0]
    _, _, height, width = model_input.shape
    height = height if isinstance(height, int) else 640
    width = width if isinstance(width, int) else 640
    h, w = image.shape[:2]
    scale = min(width / w, height / h)
    rw, rh = max(1, round(w * scale)), max(1, round(h * scale))
    rgb = cv2.cvtColor(cv2.resize(image, (rw, rh)), cv2.COLOR_BGR2RGB)
    canvas = np.zeros((height, width, 3), np.uint8)
    canvas[:rh, :rw] = rgb
    blob = np.ascontiguousarray(canvas.transpose(2, 0, 1)[None], dtype=np.float32) / 255.0
    rows = session.run(None, {model_input.name: blob})[0]
    if rows.ndim != 3 or rows.shape[0] != 1 or 5 not in rows.shape[1:]:
        raise RuntimeError(f'Unexpected YOLOFace VR180 output shape: {rows.shape}')
    rows = rows[0].T if rows.shape[1] == 5 else rows[0]
    rows = rows[np.isfinite(rows).all(axis=1) & (rows[:, 4] >= threshold)]
    boxes, scores = [], []
    for cx, cy, bw, bh, score in rows:
        x1, x2 = np.clip([(cx - bw / 2) / rw, (cx + bw / 2) / rw], 0, 1)
        y1, y2 = np.clip([(cy - bh / 2) / rh, (cy + bh / 2) / rh], 0, 1)
        if x2 - x1 > 0.003 and y2 - y1 > 0.003:
            boxes.append([float(x1), float(y1), float(x2 - x1), float(y2 - y1)])
            scores.append(float(score))
    if not boxes:
        return []
    keep = cv2.dnn.NMSBoxes(boxes, scores, 0.0, 0.45, top_k=40)
    return [{'box': boxes[i], 'score': scores[i]} for i in np.asarray(keep).ravel().astype(int)[:20]]


def validate_box(box):
    if not isinstance(box, list) or len(box) != 4:
        raise ValueError('Select one face box in each eye.')
    x, y, w, h = (float(v) for v in box)
    if not all(math.isfinite(v) for v in (x, y, w, h)) or min(x, y) < 0 or min(w, h) < 0.003 or max(x + w, y + h) > 1.000001:
        raise ValueError('Invalid face box.')
    return x, y, w, h


def camera_from_boxes(boxes, padding, start, end):
    if not isinstance(boxes, list) or len(boxes) != 2:
        raise ValueError('Select one face box in each eye.')
    if not 1.1 <= padding <= 3:
        raise ValueError('Crop margin must be between 1.1 and 3.')
    centers, fovs = [], []
    for box in boxes:
        x, y, w, h = validate_box(box)
        pan, tilt = (x + w / 2 - .5) * 180, (.5 - y - h / 2) * 180
        centers.append((pan, tilt))
        rotation = pipeline._build_eye_rotation_matrix(pan, tilt, 0)
        theta = (np.array([x, x + w, x, x + w]) - .5) * np.pi
        phi = (.5 - np.array([y, y, y + h, y + h])) * np.pi
        rays = np.stack((np.cos(phi) * np.sin(theta), -np.sin(phi), np.cos(phi) * np.cos(theta)), axis=1)
        local = rays @ rotation
        if np.any(local[:, 2] <= 0):
            raise ValueError('Box is too wide for a single perspective view.')
        needed = 2 * np.rad2deg(np.arctan(np.max(np.abs(local[:, :2] / local[:, 2:]), axis=0) * padding))
        fovs.append(needed)
    (pan_l, tilt_l), (pan_r, tilt_r) = centers
    hfov, vfov = np.clip(np.max(fovs, axis=0), 22, 145)
    yaw = ((-pan_l - 180 + 180) % 360) - 180
    return (
        pipeline.CameraParams(yaw, tilt_l, 0, float(hfov), float(vfov), start, end),
        pan_r - pan_l,
        tilt_r - tilt_l,
    )


def inspect_video(payload):
    path = local_file(payload.get('video'), VIDEO_SUFFIXES)
    number = int(payload.get('frame', 0))
    frame, info = read_frame(path, number)
    result = {'frame': number, 'info': info}
    if not payload.get('detect_only'):
        result['image'] = image_url(frame)
    if payload.get('playback'):
        result['playback'] = playback_url(path)
    if payload.get('detect'):
        provider = execution_provider(payload)
        threshold = float(payload.get('threshold', .25))
        if not .05 <= threshold <= .95:
            raise ValueError('Invalid detection threshold.')
        with DETECT_LOCK:
            session = detector(provider)
            eyes = np.hsplit(frame, 2)
            result['faces'] = [detect_eye(eye, session, threshold) for eye in eyes]
        result['provider'] = session.get_providers()[0]
    return result


def preview_crops(payload):
    path = local_file(payload.get('video'), VIDEO_SUFFIXES)
    number = int(payload.get('frame', 0))
    frame, info = read_frame(path, number)
    params, dyaw, dpitch = camera_from_boxes(payload.get('boxes'), float(payload.get('padding', 1.8)), 0, 0)
    width, height = pipeline.resolve_extract_size(pipeline.VideoInfo(info['width'], info['height'], info['fps']), params)
    scale = 384 / max(width, height)
    width, height = max(2, round(width * scale)), max(2, round(height * scale))
    fx = width / (2 * math.tan(math.radians(params.h_fov) / 2))
    fy = height / (2 * math.tan(math.radians(params.v_fov) / 2))
    crops = []
    for eye, rotation in zip(np.hsplit(frame, 2), pipeline.eye_rotations(params, dyaw, dpitch)):
        maps = pipeline._build_extract_maps(info['width'] // 2, info['height'], rotation, fx, fy, width / 2 - .5, height / 2 - .5, width, height)
        crop = cv2.remap(eye, *maps, cv2.INTER_LINEAR)
        crops.append(image_url(crop))
    return {'images': crops, 'h_fov': params.h_fov, 'v_fov': params.v_fov}


def execution_provider(payload):
    provider = payload.get('provider', 'cuda')
    if provider not in {'cuda', 'cpu'}:
        raise ValueError('Invalid execution provider.')
    return provider


def write_monitor(path, frame, boxes):
    if frame.shape[1] > 1920:
        frame = cv2.resize(frame, (1920, round(frame.shape[0] * 1920 / frame.shape[1])), interpolation=cv2.INTER_AREA)
    h, full_w = frame.shape[:2]
    w = full_w // 2
    for i, box in enumerate(boxes):
        x, y, bw, bh = validate_box(box)
        p1, p2 = (round((x + i) * w), round(y * h)), (round((x + bw + i) * w), round((y + bh) * h))
        cv2.rectangle(frame, p1, p2, (190, 229, 108), 2)
        cv2.putText(frame, ('L' if i == 0 else 'R'), (p1[0], max(18, p1[1] - 7)), cv2.FONT_HERSHEY_SIMPLEX, .6, (190, 229, 108), 2)
    ok, data = cv2.imencode('.png', frame)
    if not ok:
        raise RuntimeError('Could not encode monitoring PNG.')
    path.write_bytes(data.tobytes())


def ensure_idle():
    if any(job['state'] == 'running' for job in JOBS.values()):
        raise ValueError('已有任务正在运行，请等待完成。')


def start_job(payload, staged=False):
    video = local_file(payload.get('video'), VIDEO_SUFFIXES)
    face = None if staged else local_file(payload.get('face'), FACE_SUFFIXES)
    provider = execution_provider(payload)
    if not shutil.which('ffmpeg') or not shutil.which('ffprobe'):
        raise ValueError('FFmpeg and FFprobe must be on PATH.')
    reference, start, end = (int(payload.get(key, 0)) for key in ('frame', 'start', 'end'))
    sample, info = read_frame(video, reference)
    if not 0 <= start <= reference <= end < info['count']:
        raise ValueError('The reference frame must be inside the selected clip.')
    boxes = payload.get('boxes')
    params, dyaw, dpitch = camera_from_boxes(boxes, float(payload.get('padding', 1.8)), start, end)
    job_id = uuid.uuid4().hex
    folder = DATA / 'jobs' / job_id
    job = {'id': job_id, 'state': 'running', 'stage': 'export', 'started': time.time(),
           'frames': end - start + 1, 'enhance': payload.get('enhance') is True,
           'staged': staged, 'times': {}, 'inputs': (video, params, dyaw, dpitch, reference),
           'provider': provider, 'range': [start, end], 'progress': {}, 'logs': deque(maxlen=60)}
    with LOCK:
        ensure_idle()
        folder.mkdir(parents=True)
        write_monitor(folder / 'monitor.png', sample, boxes)
        pipeline.write_camera_params_json(params, folder / 'camera_params.json')
        (folder / 'selection.json').write_text(json.dumps({
            'reference_frame': reference, 'boxes': boxes, 'right_yaw_offset': dyaw, 'right_pitch_offset': dpitch,
        }, indent=2), encoding='utf-8')
        JOBS[job_id] = job
    launch(job, face, provider)
    return job_status(job_id)


def advance_job(job_id, payload):
    stage = payload.get('stage')
    if stage not in ('swap', 'refill'):
        raise ValueError('无效的测试步骤。')
    face = local_file(payload.get('face'), FACE_SUFFIXES) if stage == 'swap' else None
    provider = execution_provider(payload)
    with LOCK:
        ensure_idle()
        job = JOBS.get(job_id)
        if not job or not job['staged'] or job['state'] != 'paused' or job['stage'] != stage:
            raise ValueError('测试步骤未就绪，请先完成前一步。')
        if stage == 'swap':
            job['enhance'] = payload.get('enhance') is True
            job['provider'] = provider
        job['progress'] = {}
        job['state'] = 'running'
    launch(job, face, provider)
    return job_status(job_id)


def launch(job, face, provider):
    job_id = job['id']
    folder = DATA / 'jobs' / job_id
    video, params, dyaw, dpitch, reference = job['inputs']
    settings = folder / 'camera_params.json'
    last_preview = 0.0
    last_console = 0.0
    stage_order = {'export': 1, 'swap': 2, 'refill': 3}
    phase_labels = {'prepare': '准备', 'project': '透视导出', 'finalize': '收尾',
                    'analyse': '分析', 'extract': '提取帧', 'merge': '合成', 'audio': '音频',
                    'face_swapper': '换脸', 'face_enhancer': '人脸增强'}
    eye_labels = {'left': '左眼', 'right': '右眼'}

    def report(phase=None, completed=None, total=None, frame=None, images=None, jpeg=None, eye=None, message=None):
        nonlocal last_preview, last_console
        now = time.monotonic()
        preview = base64.b64decode(jpeg) if jpeg else None
        if images and (now - last_preview >= .9 or completed == total):
            thumbnails = []
            for image in images:
                h, w = image.shape[:2]
                scale = min(1, 720 / len(images) / w, 360 / h)
                thumbnails.append(cv2.resize(image, (max(1, round(w * scale)), max(1, round(h * scale))), interpolation=cv2.INTER_AREA))
            ok, encoded = cv2.imencode('.jpg', np.hstack(thumbnails), [cv2.IMWRITE_JPEG_QUALITY, 78])
            if ok:
                preview = encoded.tobytes()
                last_preview = now
        with LOCK:
            if message:
                job['logs'].append({'time': time.time(), 'message': message})
            if phase:
                previous = job['progress']
                same_phase = previous.get('phase') == phase and previous.get('eye') == eye
                if not same_phase:
                    job['progress'] = {'phase': phase, 'eye': eye, 'started': time.time()}
                if completed is not None and total and total > 0:
                    job['progress'].update(completed=min(completed, total), total=total)
            if preview is not None:
                sequence = job.get('preview', {}).get('sequence', 0) + 1
                job.setdefault('preview_bytes', deque(maxlen=5)).append((str(sequence), preview))
                job['preview'] = {'sequence': sequence, 'stage': job['stage'], 'phase': phase,
                                  'eye': eye, 'frame': params.start_frame + frame}
        if completed is not None and total and total > 0 and (now - last_console >= 2.0 or completed >= total):
            last_console = now
            step = stage_order.get(job['stage'], 0)
            elapsed = sum(job['times'].values()) + max(0, time.time() - job.get('stage_started', time.time()))
            speed = completed / elapsed if elapsed > 0 else 0.0
            pct = round(100 * completed / total)
            bits = [f'步骤 {step}/3']
            if eye in eye_labels:
                bits.append(f'{eye_labels[eye]}（{"1" if eye == "left" else "2"}/2）')
            bits.append(phase_labels.get(phase, phase))
            bits.append(f'{completed}/{total} 帧 · {pct}%')
            if speed >= 0.05:
                bits.append(f'{speed:.1f} 帧/秒')
            print('[工作台] ' + ' · '.join(bits), flush=True)

    def worker():
        nonlocal last_preview
        try:
            while True:
                stage = job['stage']
                started = time.perf_counter()
                last_preview = 0.0
                with LOCK:
                    job['stage_started'] = time.time()
                report(phase='prepare')
                if stage == 'export':
                    pipeline.export_perspective_segments(video, settings, folder, dyaw, dpitch, progress=report)
                    next_stage = 'swap'
                elif stage == 'swap':
                    pipeline.replace_enhance_segments(
                        folder / 'left_flat.mp4', folder / 'right_flat.mp4', face, folder, job['enhance'],
                        provider=provider, reference_frame=reference - params.start_frame, progress=report,
                    )
                    next_stage = 'refill'
                else:
                    pipeline.inverse_refill_to_vr180(
                        video, settings, folder / 'left_swapped.mp4', folder / 'right_swapped.mp4',
                        folder / 'output.mp4', dyaw, dpitch, segment_only=True, progress=report,
                    )
                    next_stage = 'done'
                with LOCK:
                    job['times'][stage] = time.perf_counter() - started
                    job['stage'] = next_stage
                    job['progress'] = {}
                    if next_stage == 'done':
                        job.update(state='done', finished=time.time())
                        print(f'[工作台] 任务完成：{job_id} · 输出 output.mp4', flush=True)
                        return
                    if job['staged']:
                        job['state'] = 'paused'
                        return
        except Exception as exc:
            print(f'[workbench] {job_id}: {exc}', flush=True)
            with LOCK:
                job.update(state='error', error=str(exc), finished=time.time())

    threading.Thread(target=worker, daemon=True, name=f'vr-{job_id[:8]}').start()
    print(f'[工作台] 任务 {job_id[:8]} 开始 · 推理 {provider} · 人脸增强 {"开" if job.get("enhance") else "关"} · {job.get("frames", 0)} 帧', flush=True)


def job_status(job_id):
    with LOCK:
        if job_id not in JOBS:
            raise FileNotFoundError('Job not found in this server session.')
        job = {key: value for key, value in JOBS[job_id].items() if key not in ('inputs', 'preview_bytes')}
        job['times'] = dict(job['times'])
        job['progress'] = dict(job['progress'])
        job['logs'] = list(job['logs'])
    job['elapsed'] = sum(job['times'].values())
    if job['state'] == 'running' and job.get('stage_started'):
        job['elapsed'] += max(0, time.time() - job['stage_started'])
    base = f'/media/{job_id}'
    if 'preview' in job:
        job['preview'] = {**job['preview'], 'url': f'/api/jobs/{job_id}/preview?seq={job["preview"]["sequence"]}'}
    job['monitor'] = f'{base}/monitor.png'
    files = ['selection.json']
    if 'export' in job['times']:
        files.extend(('left_flat.mp4', 'right_flat.mp4'))
    if 'swap' in job['times']:
        files.extend(('left_swapped.mp4', 'right_swapped.mp4'))
    job['files'] = {name: f'{base}/{name}' for name in files}
    if job['state'] == 'done':
        job['output'] = f'{base}/output.mp4'
    return job


def job_preview(job_id, sequence):
    with LOCK:
        job = JOBS.get(job_id, {})
        for key, data in job.get('preview_bytes', ()):
            if key == sequence:
                return data
    raise FileNotFoundError('Preview no longer available.')
