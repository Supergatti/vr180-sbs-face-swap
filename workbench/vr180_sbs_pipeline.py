from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import threading
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

import cv2
import numpy as np



# ---------------------------------------------------------------------------
# GPU acceleration: PyTorch grid_sample > cv2.cuda > CPU
# ---------------------------------------------------------------------------

def _torch_cuda_available() -> bool:
    """Check if PyTorch with CUDA is available for grid_sample remap."""
    try:
        import torch
        return torch.cuda.is_available()
    except ImportError:
        return False


def _cv2_cuda_available() -> bool:
    """Check if OpenCV was built with CUDA."""
    try:
        if cv2.cuda.getCudaEnabledDeviceCount() <= 0:
            return False
        cv2.cuda_GpuMat(np.zeros((2, 2), dtype=np.float32))
        return True
    except (AttributeError, cv2.error, Exception):
        return False


_USE_TORCH_CUDA = _torch_cuda_available()
_USE_CV2_CUDA = _cv2_cuda_available()
_GPU_BACKEND = 'torch' if _USE_TORCH_CUDA else ('cv2.cuda' if _USE_CV2_CUDA else 'CPU')
print(f'[gpu_accel] remap/blend backend: {_GPU_BACKEND}')


class _TorchRemapper:
    """GPU remap using torch.nn.functional.grid_sample (works with pip-installed PyTorch)."""

    def __init__(self, map_x: np.ndarray, map_y: np.ndarray, src_h: int, src_w: int):
        import torch
        self._torch = torch
        self._device = torch.device('cuda')
        # Pre-compute normalized grid on GPU (done once, reused every frame)
        mx = torch.as_tensor(map_x, dtype=torch.float32, device=self._device)
        my = torch.as_tensor(map_y, dtype=torch.float32, device=self._device)
        x_norm = 2.0 * (mx + 0.5) / float(src_w) - 1.0
        y_norm = 2.0 * (my + 0.5) / float(src_h) - 1.0
        self._grid = torch.stack((x_norm, y_norm), dim=-1).unsqueeze(0)  # (1, H_out, W_out, 2)

    def __call__(self, src: np.ndarray) -> np.ndarray:
        torch = self._torch
        # BGR uint8 -> float32 NCHW on GPU
        t_in = torch.from_numpy(src).permute(2, 0, 1).unsqueeze(0).float().to(self._device)
        t_in.mul_(1.0 / 255.0)
        out = torch.nn.functional.grid_sample(
            t_in, self._grid, mode='bilinear', padding_mode='zeros', align_corners=False,
        )
        # Back to uint8 BGR HWC
        out = (out.squeeze(0).permute(1, 2, 0).clamp(0, 1) * 255.0).byte().cpu().numpy()
        return np.ascontiguousarray(out)


class _CpuRemapper:
    """CPU fallback remap using cv2.remap."""
    __slots__ = ('_mx', '_my')

    def __init__(self, map_x: np.ndarray, map_y: np.ndarray):
        self._mx = map_x
        self._my = map_y

    def __call__(self, src: np.ndarray) -> np.ndarray:
        return cv2.remap(src, self._mx, self._my, cv2.INTER_LINEAR,
                         borderMode=cv2.BORDER_CONSTANT)


def _make_remapper(map_x: np.ndarray, map_y: np.ndarray, src_h: int = 0, src_w: int = 0):
    """Create best available remapper: torch.cuda > cv2.cuda > CPU."""
    if _USE_TORCH_CUDA:
        try:
            return _TorchRemapper(map_x, map_y, src_h, src_w)
        except Exception as e:
            print(f'[gpu_accel] torch remapper failed ({e}), trying next')
    if _USE_CV2_CUDA:
        try:
            gpu_mx = cv2.cuda_GpuMat(map_x)
            gpu_my = cv2.cuda_GpuMat(map_y)

            class _Cv2CudaRemapper:
                def __call__(self, src):
                    g = cv2.cuda_GpuMat(src)
                    return cv2.cuda.remap(g, gpu_mx, gpu_my, cv2.INTER_LINEAR,
                                          borderMode=cv2.BORDER_CONSTANT).download()
            return _Cv2CudaRemapper()
        except Exception as e:
            print(f'[gpu_accel] cv2.cuda remapper failed ({e}), falling back to CPU')
    return _CpuRemapper(map_x, map_y)


# ---------------------------------------------------------------------------
# NVENC detection (h264_nvenc with libx264 fallback)
# ---------------------------------------------------------------------------

def _nvenc_available() -> bool:
    """Quick probe: can ffmpeg encode with h264_nvenc?"""
    try:
        proc = subprocess.run(
            ['ffmpeg', '-hide_banner', '-loglevel', 'error',
             '-f', 'lavfi', '-i', 'nullsrc=s=128x128:d=0.04',
             '-c:v', 'h264_nvenc', '-f', 'null', '-'],
            capture_output=True, timeout=15,
        )
        return proc.returncode == 0
    except Exception:
        return False


_USE_NVENC = _nvenc_available()
print(f'[gpu_accel] encoder: {"h264_nvenc" if _USE_NVENC else "libx264 (CPU)"}')


def _h264_encode_args(crf_or_cq: int = 16) -> list[str]:
    """Return encoder flags: NVENC if available, else libx264."""
    if _USE_NVENC:
        return ['-c:v', 'h264_nvenc', '-preset', 'p4', '-cq', str(crf_or_cq), '-pix_fmt', 'yuv420p']
    return ['-c:v', 'libx264', '-preset', 'fast', '-crf', str(crf_or_cq), '-pix_fmt', 'yuv420p']


# ---------------------------------------------------------------------------
# GPU alpha blending (PyTorch > cv2.cuda > CPU)
# ---------------------------------------------------------------------------

class _TorchBlender:
    """GPU alpha blend using PyTorch: result = region*(1-alpha) + warped*alpha."""

    def __init__(self, alpha_hwc: np.ndarray):
        import torch
        self._torch = torch
        self._device = torch.device('cuda')
        # alpha_hwc is (H, W, 1) float32 -> (1, 1, H, W) for broadcasting with NCHW
        a = torch.from_numpy(alpha_hwc.squeeze(-1)).float().to(self._device).unsqueeze(0).unsqueeze(0)
        self._alpha = a       # (1, 1, H, W)
        self._inv_alpha = 1.0 - a

    def __call__(self, region_u8: np.ndarray, warped_u8: np.ndarray) -> np.ndarray:
        torch = self._torch
        dev = self._device
        r = torch.from_numpy(region_u8).permute(2, 0, 1).unsqueeze(0).float().to(dev)
        w = torch.from_numpy(warped_u8).permute(2, 0, 1).unsqueeze(0).float().to(dev)
        out = r * self._inv_alpha + w * self._alpha
        return out.clamp(0, 255).byte().squeeze(0).permute(1, 2, 0).cpu().numpy().copy()


class _CpuBlender:
    """CPU fallback alpha blend."""
    __slots__ = ('_alpha', '_inv_alpha')

    def __init__(self, alpha_hwc: np.ndarray):
        self._alpha = alpha_hwc
        self._inv_alpha = 1.0 - alpha_hwc

    def __call__(self, region_u8: np.ndarray, warped_u8: np.ndarray) -> np.ndarray:
        blended = (region_u8.astype(np.float32) * self._inv_alpha
                   + warped_u8.astype(np.float32) * self._alpha)
        return blended.clip(0, 255).astype(np.uint8)


def _make_blender(alpha_hwc: np.ndarray):
    """Create best available blender: torch.cuda > cv2.cuda > CPU."""
    if _USE_TORCH_CUDA:
        try:
            return _TorchBlender(alpha_hwc)
        except Exception as e:
            print(f'[gpu_accel] torch blender failed ({e}), falling back to CPU')
    if _USE_CV2_CUDA:
        try:
            a3 = np.broadcast_to(alpha_hwc, (*alpha_hwc.shape[:2], 3)).copy().astype(np.float32)
            gpu_a = cv2.cuda_GpuMat(a3)
            gpu_inv = cv2.cuda_GpuMat((1.0 - a3).astype(np.float32))

            class _Cv2CudaBlender:
                def __call__(self, region_u8, warped_u8):
                    rf = cv2.cuda_GpuMat(region_u8).convertTo(cv2.CV_32FC3)
                    wf = cv2.cuda_GpuMat(warped_u8).convertTo(cv2.CV_32FC3)
                    t1 = cv2.cuda.multiply(rf, gpu_inv)
                    t2 = cv2.cuda.multiply(wf, gpu_a)
                    return cv2.cuda.add(t1, t2).convertTo(cv2.CV_8UC3).download()
            return _Cv2CudaBlender()
        except Exception as e:
            print(f'[gpu_accel] cv2.cuda blender failed ({e}), falling back to CPU')
    return _CpuBlender(alpha_hwc)


def _start_stderr_drain(proc: subprocess.Popen) -> tuple[threading.Thread, list[bytes]]:
    """Drain subprocess stderr in a background thread to prevent pipe deadlock."""
    chunks: list[bytes] = []
    def _drain():
        try:
            data = proc.stderr.read()
            if data:
                chunks.append(data)
        except Exception:
            pass
    t = threading.Thread(target=_drain, daemon=True)
    t.start()
    return t, chunks


@dataclass
class CameraParams:
    yaw: float
    pitch: float
    roll: float
    h_fov: float
    v_fov: float
    start_frame: int
    end_frame: int


@dataclass
class VideoInfo:
    width: int
    height: int
    fps: float


def _to_even(value: float | int, min_value: int = 2) -> int:
    n = max(min_value, int(round(value)))
    if n % 2 != 0:
        n += 1
    return n


def parse_frontend_json(json_path: str | Path) -> CameraParams:
    data = json.loads(Path(json_path).read_text(encoding='utf-8'))
    return CameraParams(
        yaw=float(data['yaw']),
        pitch=float(data['pitch']),
        roll=float(data['roll']),
        h_fov=float(data['h_fov']),
        v_fov=float(data['v_fov']),
        start_frame=int(data['start_frame']),
        end_frame=int(data['end_frame']),
    )


def probe_video_info(input_video: str | Path) -> VideoInfo:
    cmd = ['ffprobe', '-v', 'error', '-select_streams', 'v:0',
           '-show_entries', 'stream=width,height,r_frame_rate', '-of', 'json', str(input_video)]
    result = subprocess.run(cmd, check=True, capture_output=True, text=True)
    payload = json.loads(result.stdout)
    stream = payload['streams'][0]

    width = int(stream['width'])
    height = int(stream['height'])

    num, den = stream['r_frame_rate'].split('/')
    fps = float(num) / float(den)

    return VideoInfo(width=width, height=height, fps=fps)


def resolve_extract_size(video_info: VideoInfo, params: CameraParams, supersample: float = 1.5) -> Tuple[int, int]:
    left_half_width = video_info.width / 2.0
    out_w = left_half_width * (params.h_fov / 180.0) * supersample
    out_h = video_info.height * (params.v_fov / 180.0) * supersample
    return _to_even(out_w), _to_even(out_h)


def write_camera_params_json(params: CameraParams, output_json: str | Path) -> Path:
    payload = {
        'yaw': params.yaw,
        'pitch': params.pitch,
        'roll': params.roll,
        'h_fov': params.h_fov,
        'v_fov': params.v_fov,
        'start_frame': params.start_frame,
        'end_frame': params.end_frame,
    }
    output_path = Path(output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding='utf-8')
    return output_path


# ---------------------------------------------------------------------------
# Core OpenCV Math (Shared by Extract and Backfill)
# ---------------------------------------------------------------------------

def _build_eye_rotation_matrix(pan_deg: float, tilt_deg: float, roll_deg: float) -> np.ndarray:
    """Build a 3x3 camera rotation matrix R_cam.
    Standard pinhole convention: +x = right, +y = down, +z = forward.
    Columns of R_cam represent the [right_img, down_img, fwd] basis vectors in world space.
    """
    pan  = np.deg2rad(pan_deg)
    tilt = np.deg2rad(tilt_deg)

    fwd: np.ndarray = np.array([
        np.sin(pan) * np.cos(tilt),
        -np.sin(tilt),
        np.cos(pan) * np.cos(tilt),
    ], dtype=np.float64)
    fwd /= np.linalg.norm(fwd)

    # World up represents vertical geometry.
    # To get standard right (+x) and down (+y) for OpenCV pinhole:
    world_up = np.array([0.0, -1.0, 0.0], dtype=np.float64)
    right = np.cross(world_up, fwd)
    nr = float(np.linalg.norm(right))
    right = right / nr if nr > 1e-12 else np.array([1.0, 0.0, 0.0], dtype=np.float64)
    up = np.cross(fwd, right)
    up /= np.linalg.norm(up)

    right_img = -right
    down_img = -up

    if abs(roll_deg) > 1e-9:
        r = np.deg2rad(roll_deg)
        cos_r, sin_r = np.cos(r), np.sin(r)
        right_img, down_img = (
            cos_r * right_img + sin_r * down_img,
            -sin_r * right_img + cos_r * down_img,
        )

    return np.column_stack((right_img, down_img, fwd)).astype(np.float64)


def _build_extract_maps(
    erp_half_w: int,
    erp_full_h: int,
    r_cam: np.ndarray,
    fx: float,
    fy: float,
    cx: float,
    cy: float,
    roi_w: int,
    roi_h: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Precompute OpenCV maps for extracting a flat ROI from ERP."""
    u, v = np.meshgrid(
        np.arange(roi_w, dtype=np.float64),
        np.arange(roi_h, dtype=np.float64),
        indexing='xy',
    )
    # Image to Camera
    x_cam = (u - cx) / fx
    y_cam = (v - cy) / fy
    z_cam = np.ones_like(x_cam)
    
    norm = np.sqrt(x_cam**2 + y_cam**2 + z_cam**2)
    x_cam /= norm
    y_cam /= norm
    z_cam /= norm
    
    d_cam_flat = np.stack((x_cam, y_cam, z_cam), axis=-1).reshape(-1, 3)
    
    # Camera to World (D_world = R_cam @ D_cam)
    # in rows: d_world_row = d_cam_row @ R_cam^T
    d_world_flat = d_cam_flat @ r_cam.T
    
    dx = d_world_flat[:, 0]
    dy = d_world_flat[:, 1]
    dz = d_world_flat[:, 2]
    
    theta = np.arctan2(dx, dz)
    phi = np.arcsin(np.clip(-dy, -1.0, 1.0))
    
    # World to ERP (fractional pixel coords)
    ex = (theta / np.pi + 0.5) * float(erp_half_w)
    ey = (-phi / np.pi + 0.5) * float(erp_full_h)
    
    map_x = ex.reshape(roi_h, roi_w).astype(np.float32)
    map_y = ey.reshape(roi_h, roi_w).astype(np.float32)
    return map_x, map_y


def _build_backfill_maps(
    erp_half_w: int,
    erp_full_h: int, # Using full_h here.
    r_cam: np.ndarray,
    fx: float,
    fy: float,
    cx: float,
    cy: float,
    roi_w: int,
    roi_h: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Precompute OpenCV maps for writing a flat ROI back into ERP."""
    ex, ey = np.meshgrid(
        np.arange(erp_half_w, dtype=np.float64),
        np.arange(erp_full_h, dtype=np.float64),
        indexing='xy',
    )
    theta = (ex / float(erp_half_w) - 0.5) * np.pi
    phi   = -((ey / float(erp_full_h)) - 0.5) * np.pi

    cos_phi = np.cos(phi)
    dx = cos_phi * np.sin(theta)
    dy = -np.sin(phi)
    dz = cos_phi * np.cos(theta)
    d_world = np.stack((dx, dy, dz), axis=-1)

    d_flat = d_world.reshape(-1, 3)
    d_cam  = d_flat @ r_cam

    xc, yc, zc = d_cam[:, 0], d_cam[:, 1], d_cam[:, 2]

    eps   = 1e-9
    valid = zc > eps
    safe_z = np.where(valid, zc, 1.0)
    u = xc / safe_z * fx + cx
    v = yc / safe_z * fy + cy
    valid &= (u >= 0.0) & (u <= float(roi_w - 1)) & (v >= 0.0) & (v <= float(roi_h - 1))

    u[~valid] = -1.0
    v[~valid] = -1.0

    map_x = u.reshape(erp_full_h, erp_half_w).astype(np.float32)
    map_y = v.reshape(erp_full_h, erp_half_w).astype(np.float32)
    return map_x, map_y, valid.reshape(erp_full_h, erp_half_w)


def _feather_weight_map(valid_mask: np.ndarray, feather_px: int = 15) -> np.ndarray:
    dist = cv2.distanceTransform(
        (valid_mask.astype(np.uint8) * 255),
        cv2.DIST_L2,
        maskSize=5,
    )
    return np.clip(dist / float(max(feather_px, 1)), 0.0, 1.0).astype(np.float32)


# ---------------------------------------------------------------------------
# Stage 1: Extraction
# ---------------------------------------------------------------------------

def eye_rotations(params, right_yaw_offset, right_pitch_offset=0.0):
    pan = -params.yaw - 180.0
    return (
        _build_eye_rotation_matrix(pan, params.pitch, params.roll),
        _build_eye_rotation_matrix(pan + right_yaw_offset, params.pitch + right_pitch_offset, params.roll),
    )


def extract_perspective_with_opencv(
    input_video: str | Path,
    left_flat: str | Path,
    right_flat: str | Path,
    params: CameraParams,
    video_info: VideoInfo,
    extract_size: tuple[int, int],
    right_yaw_offset: float = 1.2,
    right_pitch_offset: float = 0.0,
    progress=None,
) -> None:
    half_w = video_info.width // 2
    full_h = video_info.height
    roi_w, roi_h = extract_size

    # Negate yaw: Three.js +yaw = look left, but _build_eye_rotation_matrix +pan = look right
    r_left, r_right = eye_rotations(params, right_yaw_offset, right_pitch_offset)

    fx = (roi_w / 2.0) / np.tan(np.deg2rad(params.h_fov) / 2.0)
    fy = (roi_h / 2.0) / np.tan(np.deg2rad(params.v_fov) / 2.0)
    cx = roi_w / 2.0 - 0.5
    cy = roi_h / 2.0 - 0.5

    l_map_x, l_map_y = _build_extract_maps(half_w, full_h, r_left, fx, fy, cx, cy, roi_w, roi_h)
    r_map_x, r_map_y = _build_extract_maps(half_w, full_h, r_right, fx, fy, cx, cy, roi_w, roi_h)

    # Build remappers (GPU if available, else CPU)
    print(f'[opencv_extract] remap backend: {_GPU_BACKEND}')
    l_remapper = _make_remapper(l_map_x, l_map_y, src_h=full_h, src_w=half_w)
    r_remapper = _make_remapper(r_map_x, r_map_y, src_h=full_h, src_w=half_w)

    cap_base = cv2.VideoCapture(str(input_video))
    if not cap_base.isOpened():
        raise RuntimeError(f"Could not open input video: {input_video}")

    cap_base.set(cv2.CAP_PROP_POS_FRAMES, params.start_frame)

    fps = video_info.fps
    
    # We pipe to ffmpeg to encode. We could use cv2.VideoWriter but h264 via ffmpeg is more reliable.
    enc_tag = 'h264_nvenc' if _USE_NVENC else 'libx264'
    print(f'[opencv_extract] encoder: {enc_tag}')

    def build_ffmpeg_writer(out_path):
        cmd = [
            'ffmpeg', '-y', '-hide_banner', '-loglevel', 'error',
            '-f', 'rawvideo', '-pix_fmt', 'bgr24',
            '-s', f'{roi_w}x{roi_h}',
            '-r', str(fps),
            '-i', 'pipe:0',
            *_h264_encode_args(16),
            str(out_path)
        ]
        proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
        return proc, _start_stderr_drain(proc)

    proc_l, (drain_l, stderr_l) = build_ffmpeg_writer(left_flat)
    proc_r, (drain_r, stderr_r) = build_ffmpeg_writer(right_flat)

    try:
        frames_to_read = params.end_frame - params.start_frame + 1
        print(f"[opencv_extract] Extracting {frames_to_read} frames at {roi_w}x{roi_h}...")
        t_start = time.perf_counter()

        for idx in range(frames_to_read):
            ret, frame = cap_base.read()
            if not ret or frame is None:
                raise RuntimeError(f"Unexpected end of input at clip frame {idx + 1}/{frames_to_read}")

            eye_l = frame[:, :half_w]
            eye_r = frame[:, half_w:half_w*2]

            ext_l = l_remapper(eye_l)
            ext_r = r_remapper(eye_r)

            proc_l.stdin.write(ext_l.tobytes())
            proc_r.stdin.write(ext_r.tobytes())
            if progress:
                progress(phase='project', completed=idx + 1, total=frames_to_read, frame=idx, images=(ext_l, ext_r))

            if (idx + 1) == 1 or (idx + 1) % 50 == 0 or (idx + 1) == frames_to_read:
                elapsed = time.perf_counter() - t_start
                fps_proc = (idx + 1) / elapsed if elapsed > 0 else 0
                print(f"[opencv_extract] Done {idx+1}/{frames_to_read} ({fps_proc:.1f} fps)")

    finally:
        if progress:
            progress(phase='finalize')
        cap_base.release()
        proc_l.stdin.close()
        proc_r.stdin.close()
        proc_l.wait()
        proc_r.wait()
        drain_l.join(timeout=5)
        drain_r.join(timeout=5)

        if proc_l.returncode != 0:
            err = b''.join(stderr_l).decode(errors='replace')
            raise RuntimeError(f"Error encoding left eye flat: {err}")
        if proc_r.returncode != 0:
            err = b''.join(stderr_r).decode(errors='replace')
            raise RuntimeError(f"Error encoding right eye flat: {err}")


def export_perspective_segments(
    input_video: str | Path,
    params_json: str | Path,
    work_dir: str | Path,
    right_yaw_offset: float = 1.2,
    right_pitch_offset: float = 0.0,
    progress=None,
) -> dict:
    work_path = Path(work_dir)
    work_path.mkdir(parents=True, exist_ok=True)

    params = parse_frontend_json(params_json)
    info = probe_video_info(input_video)
    extract_size = resolve_extract_size(info, params)

    saved_params_json = write_camera_params_json(params, work_path / 'camera_params.json')
    left_flat = work_path / 'left_flat.mp4'
    right_flat = work_path / 'right_flat.mp4'

    extract_perspective_with_opencv(
        input_video, left_flat, right_flat, params, info, extract_size, right_yaw_offset, right_pitch_offset, progress
    )

    return {
        'left_flat': str(left_flat),
        'right_flat': str(right_flat),
        'params_json': str(saved_params_json),
        'work_dir': str(work_path),
        'extract_width': extract_size[0],
        'extract_height': extract_size[1],
    }


# ---------------------------------------------------------------------------
# Stage 2: FaceFusion Replace & Enhance
# ---------------------------------------------------------------------------

def _facefusion_headless_run(
    source_face: str | Path,
    target_video: str | Path,
    output_video: str | Path,
    use_face_enhancer: bool,
    face_swapper_model: str,
    provider: str = 'cuda',
    reference_frame: int = 0,
    progress=None,
) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    facefusion_entry = repo_root / 'facefusion.py'

    processors = ['face_swapper']
    if use_face_enhancer:
        processors.append('face_enhancer')

    cmd = [
        sys.executable,
        str(facefusion_entry),
        'headless-run',
        '-s',
        str(source_face),
        '-t',
        str(target_video),
        '-o',
        str(output_video),
        '--processors',
        *processors,
        '--face-swapper-model',
        face_swapper_model,
        '--face-swapper-pixel-boost',
        '128x128',
        '--execution-providers',
        provider,
        '--face-detector-model', 'yoloface',
        '--face-selector-mode', 'reference',
        '--reference-frame-number', str(reference_frame),
        '--reference-face-position', '0',
        '--execution-thread-count', '2',
        '--temp-frame-format', 'jpg',
    ]

    print(subprocess.list2cmdline(cmd))
    env = dict(os.environ, PYTHONIOENCODING='utf-8')
    if progress:
        env.update(FACEFUSION_WORKBENCH_PROGRESS='1', FACEFUSION_SESSION_PROGRESS='1', FACEFUSION_EFFECTIVE_PROVIDERS='1')
    recent = deque(maxlen=16)
    prefix = 'VR180_PROGRESS '
    with subprocess.Popen(cmd, cwd=str(repo_root), env=env, stdout=subprocess.PIPE,
                          stderr=subprocess.STDOUT, encoding='utf-8', errors='replace') as process:
        for raw in process.stdout:
            line = re.sub(r'\x1b\[[0-9;]*[A-Za-z]', '', raw.replace('\x00', '')).strip()
            if line.startswith(prefix):
                if progress:
                    progress(**json.loads(line[len(prefix):]))
            elif line and not re.search(r'\d+%\|', line):
                recent.append(line[-1000:])
                if progress:
                    progress(message=line[-1000:])
    if process.returncode or not Path(output_video).is_file():
        details = '\n'.join(recent)
        raise RuntimeError(f'FaceFusion did not produce {Path(output_video).name}: {details[-2400:]}')


def replace_enhance_segments(
    left_flat: str | Path,
    right_flat: str | Path,
    source_face: str | Path,
    work_dir: str | Path,
    use_face_enhancer: bool = True,
    face_swapper_model: str = 'inswapper_128_fp16',
    provider: str = 'cuda',
    reference_frame: int = 0,
    progress=None,
) -> dict:
    work_path = Path(work_dir)
    work_path.mkdir(parents=True, exist_ok=True)

    swapped_left = work_path / 'left_swapped.mp4'
    swapped_right = work_path / 'right_swapped.mp4'

    for eye, flat, swapped in (('left', left_flat, swapped_left), ('right', right_flat, swapped_right)):
        def report(**event):
            progress(eye=eye, **event)

        if progress:
            report(phase='prepare')
        _facefusion_headless_run(source_face, flat, swapped, use_face_enhancer, face_swapper_model, provider, reference_frame,
                                 report if progress else None)

    return {
        'left_swapped': str(swapped_left),
        'right_swapped': str(swapped_right),
    }


# ---------------------------------------------------------------------------
# Stage 3: Backfill (OpenCV Exact Inverse Remap)
# ---------------------------------------------------------------------------

def inverse_refill_to_vr180(
    base_video: str | Path,
    params_json: str | Path,
    left_swapped_flat: str | Path,
    right_swapped_flat: str | Path,
    output_video: str | Path,
    right_yaw_offset: float = 1.2,
    right_pitch_offset: float = 0.0,
    segment_only: bool = False,
    progress=None,
) -> Path:
    params = parse_frontend_json(params_json)
    info = probe_video_info(base_video)
    
    half_w  = info.width  // 2
    full_h  = info.height
    start_f = params.start_frame
    end_f   = params.end_frame

    # Negate yaw: Three.js +yaw = look left, but _build_eye_rotation_matrix +pan = look right
    r_left, r_right = eye_rotations(params, right_yaw_offset, right_pitch_offset)

    _cap_probe = cv2.VideoCapture(str(left_swapped_flat))
    roi_w = int(_cap_probe.get(cv2.CAP_PROP_FRAME_WIDTH))
    roi_h = int(_cap_probe.get(cv2.CAP_PROP_FRAME_HEIGHT))
    _cap_probe.release()

    fx = (roi_w / 2.0) / np.tan(np.deg2rad(params.h_fov) / 2.0)
    fy = (roi_h / 2.0) / np.tan(np.deg2rad(params.v_fov) / 2.0)
    cx = roi_w  / 2.0 - 0.5
    cy = roi_h  / 2.0 - 0.5

    l_map_x, l_map_y, l_valid = _build_backfill_maps(half_w, full_h, r_left,  fx, fy, cx, cy, roi_w, roi_h)
    r_map_x, r_map_y, r_valid = _build_backfill_maps(half_w, full_h, r_right, fx, fy, cx, cy, roi_w, roi_h)

    l_alpha = _feather_weight_map(l_valid)[:, :, np.newaxis]
    r_alpha = _feather_weight_map(r_valid)[:, :, np.newaxis]

    cap_base  = cv2.VideoCapture(str(base_video))
    cap_left  = cv2.VideoCapture(str(left_swapped_flat))
    cap_right = cv2.VideoCapture(str(right_swapped_flat))

    total_frames = int(cap_base.get(cv2.CAP_PROP_FRAME_COUNT))
    fps          = info.fps
    first_frame = start_f if segment_only else 0
    last_frame = end_f + 1 if segment_only else total_frames
    if segment_only:
        cap_base.set(cv2.CAP_PROP_POS_FRAMES, first_frame)

    l_rows, l_cols = np.where(l_valid)
    r_rows, r_cols = np.where(r_valid)
    l_bbox = (l_rows.min(), l_rows.max() + 1, l_cols.min(), l_cols.max() + 1) if l_rows.size else None
    r_bbox = (r_rows.min(), r_rows.max() + 1, r_cols.min(), r_cols.max() + 1) if r_rows.size else None

    if l_bbox:
        ly0, ly1, lx0, lx1 = l_bbox
        l_map_x_c = l_map_x[ly0:ly1, lx0:lx1]
        l_map_y_c = l_map_y[ly0:ly1, lx0:lx1]
        l_alpha_c = l_alpha[ly0:ly1, lx0:lx1]
        l_valid_c = l_valid[ly0:ly1, lx0:lx1]
        l_remapper = _make_remapper(l_map_x_c, l_map_y_c, src_h=roi_h, src_w=roi_w)
        l_blender = _make_blender(l_alpha_c)
    if r_bbox:
        ry0, ry1, rx0, rx1 = r_bbox
        r_map_x_c = r_map_x[ry0:ry1, rx0:rx1]
        r_map_y_c = r_map_y[ry0:ry1, rx0:rx1]
        r_alpha_c = r_alpha[ry0:ry1, rx0:rx1]
        r_valid_c = r_valid[ry0:ry1, rx0:rx1]
        r_remapper = _make_remapper(r_map_x_c, r_map_y_c, src_h=roi_h, src_w=roi_w)
        r_blender = _make_blender(r_alpha_c)

    enc_tag = 'h264_nvenc' if _USE_NVENC else 'libx264'
    print(f'[opencv_backfill] remap/blend: {_GPU_BACKEND}  encoder: {enc_tag}')

    encode_cmd = [
        'ffmpeg', '-y', '-hide_banner', '-loglevel', 'error',
        '-f', 'rawvideo', '-pix_fmt', 'bgr24',
        '-s', f'{info.width}x{full_h}',
        '-r', str(fps),
        '-i', 'pipe:0',
        *(['-ss', str(start_f / fps)] if segment_only else []),
        '-i', str(base_video),
        '-map', '0:v:0', '-map', '1:a?',
        '-map_metadata', '1',
        *_h264_encode_args(18),
        *(['-t', str((end_f - start_f + 1) / fps), '-c:a', 'aac'] if segment_only else ['-c:a', 'copy']),
        str(output_video),
    ]
    encode_proc = subprocess.Popen(encode_cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
    encode_drain, encode_stderr = _start_stderr_drain(encode_proc)

    t_start = time.perf_counter()
    swap_count = end_f - start_f + 1

    try:
        print(f"[opencv_backfill] Total frames {total_frames}. Swapping from {start_f} to {end_f}.")
        for frame_idx in range(first_frame, last_frame):
            ret_b, frame = cap_base.read()
            if not ret_b:
                raise RuntimeError(f"Unexpected end of original video at frame {frame_idx}")

            if start_f <= frame_idx <= end_f:
                ret_l, roi_left  = cap_left.read()
                ret_r, roi_right = cap_right.read()
                if not ret_l or not ret_r:
                    raise RuntimeError(f"A swapped eye video ended early at frame {frame_idx}")

                if ret_l and ret_r:
                    if l_bbox:
                        region_l = frame[ly0:ly1, lx0:lx1].copy()
                        warped_l = l_remapper(roi_left)
                        frame[ly0:ly1, lx0:lx1] = l_blender(region_l, warped_l)

                    if r_bbox:
                        region_r = frame[ry0 : ry1, half_w + rx0 : half_w + rx1].copy()
                        warped_r = r_remapper(roi_right)
                        frame[ry0 : ry1, half_w + rx0 : half_w + rx1] = r_blender(region_r, warped_r)

                done = frame_idx - start_f + 1
                if done == 1 or done % 50 == 0 or frame_idx == end_f:
                    elapsed = time.perf_counter() - t_start
                    fps_proc = done / elapsed if elapsed > 0 else 0
                    print(f'[opencv_backfill] frame {frame_idx}/{total_frames}  '
                          f'swaps: {done}/{swap_count}  {fps_proc:.1f} fps', flush=True)

            assert encode_proc.stdin is not None
            encode_proc.stdin.write(frame.tobytes())
            if progress:
                progress(phase='composite', completed=frame_idx - first_frame + 1, total=last_frame - first_frame,
                         frame=frame_idx - start_f, images=(frame,))
    finally:
        if progress:
            progress(phase='finalize')
        cap_base.release()
        cap_left.release()
        cap_right.release()
        if encode_proc.stdin:
            encode_proc.stdin.close()

        encode_proc.wait()
        encode_drain.join(timeout=5)
        if encode_proc.returncode and encode_proc.returncode != 0:
            err = b''.join(encode_stderr).decode(errors='replace')
            raise RuntimeError(
                f'FFmpeg encode failed (code={encode_proc.returncode}):\n{err}'
            )
        elapsed_total = time.perf_counter() - t_start
        print(f'[opencv_backfill] done in {elapsed_total:.1f}s', flush=True)

    return Path(output_video)


if __name__ == '__main__':
    pass
