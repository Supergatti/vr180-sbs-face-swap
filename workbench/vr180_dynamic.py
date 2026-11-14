"""Bounded, in-memory stereo rendering for the production workbench."""

from __future__ import annotations

import math
import subprocess
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from inspect import signature

import cv2
import numpy as np

import facefusion_runtime
import facefusion_settings as tuning
import vr180_sbs_pipeline as pipeline


STEREO_DISTANCE = .45
VIEW_SIZE = 512


@dataclass
class View:
    image: np.ndarray
    maps: tuple[np.ndarray, np.ndarray]
    rotation: np.ndarray
    fx: float
    fy: float


@dataclass
class Match:
    view: View
    face: object
    box: np.ndarray
    recovered: bool


def center(box):
    return box[:2] + box[2:] * .5


def shifted_box(box, offset):
    lo = np.clip(box[:2] + offset, 0., 1.)
    hi = np.clip(box[:2] + box[2:] + offset, 0., 1.)
    return np.r_[lo, hi - lo]


def make_view(eye, box, padding):
    if min(box[2:]) < .003:
        return None
    x, y = center(box)
    rotation = pipeline._build_eye_rotation_matrix((x - .5) * 180, (.5 - y) * 180, 0)
    theta = (x + np.array([-1, 1, -1, 1]) * box[2] / 2 - .5) * np.pi
    phi = (.5 - y + np.array([-1, -1, 1, 1]) * box[3] / 2) * np.pi
    rays = np.stack((np.cos(phi) * np.sin(theta), -np.sin(phi), np.cos(phi) * np.cos(theta)), axis=1)
    local = rays @ rotation
    z = np.maximum(local[:, 2:], 1e-4)
    tangent = np.clip(np.max(np.abs(local[:, :2] / z)) * padding,
                      math.tan(math.radians(11)), math.tan(math.radians(72.5)))
    focal = (VIEW_SIZE / 2) / tangent
    c = (VIEW_SIZE - 1) / 2
    height, width = eye.shape[:2]
    maps = pipeline._build_extract_maps(width, height, rotation, focal, focal, c, c, VIEW_SIZE, VIEW_SIZE)
    return View(cv2.remap(eye, *maps, cv2.INTER_LINEAR), maps, rotation, focal, focal)


def face_box(view, face, shape):
    """Map the detected perspective face back to normalized eye coordinates."""
    x0, y0, x1, y1 = np.rint(face.bounding_box).astype(int).clip(0, VIEW_SIZE - 1)
    if x1 <= x0 or y1 <= y0:
        return None
    # Include edge midpoints; projected rectangle extrema need not be at corners.
    xs, ys = np.meshgrid([x0, (x0 + x1) // 2, x1], [y0, (y0 + y1) // 2, y1])
    h, w = shape[:2]
    xy = np.array([view.maps[0][ys, xs] / w, view.maps[1][ys, xs] / h])
    lo, hi = np.clip(xy.min(axis=(1, 2)), 0, 1), np.clip(xy.max(axis=(1, 2)), 0, 1)
    return np.r_[lo, hi - lo]


def distance(a, b):
    score = float(1 - np.dot(a.normed_embedding, b.normed_embedding))
    return score if math.isfinite(score) else math.inf


class StereoSwap:
    def __init__(self, wb, source, sample, boxes, padding, provider, enhance, report, *, identity_mode='strict', options=None):
        facefusion_runtime.configure()
        from facefusion import choices, content_analyser, face_detector, face_landmarker, face_masker, face_recognizer, state_manager
        from facefusion.execution import get_available_execution_providers
        from facefusion.face_analyser import get_many_faces
        from facefusion.processors.modules import face_enhancer, face_swapper
        from facefusion.vision import read_image

        if 'embed' not in signature(get_many_faces).parameters:
            raise RuntimeError('FaceFusion 缺少工作台兼容补丁，请运行 setup.ps1。')

        available = get_available_execution_providers()
        if provider not in available:
            raise ValueError(f'FaceFusion 推理设备 {provider.upper()} 不可用。')
        providers = [provider]
        options = tuning.DEFAULTS if options is None else options
        settings = {
            'execution_device_id': '0', 'execution_providers': providers,
            'download_providers': [],
            'face_detector_model': 'yoloface', 'face_detector_size': '640x640',
            'face_landmarker_model': '2dfan4',
            'source_paths': [str(source)],
            **{key: value for key, value in options.items() if key != 'vr_detector_score'},
        }
        masks = options['face_mask_types']
        settings['face_occluder_model'] = tuning.MASK_MODELS['occlusion'] if 'occlusion' in masks else None
        settings['face_parser_model'] = tuning.MASK_MODELS['region'] if 'region' in masks else None
        # Pools are keyed by processor/provider, not by model. Clear the old
        # model before changing the state, including when switching back to a provider.
        previous = state_manager.get_item('execution_providers')
        if previous:
            for module, keys in (
                (face_swapper, ('face_swapper_model',)),
                (face_enhancer, ('face_enhancer_model',)),
                (face_masker, ('face_occluder_model', 'face_parser_model')),
            ):
                if previous != providers or any(state_manager.get_item(key) != settings[key] for key in keys):
                    module.clear_inference_pool()
        for key, value in settings.items():
            state_manager.init_item(key, value)

        models = [face_detector, face_landmarker, face_recognizer, face_swapper, content_analyser]
        if enhance:
            models.append(face_enhancer)
        if settings['face_occluder_model'] or settings['face_parser_model']:
            models.append(face_masker)
        expected = {choices.execution_provider_set[p] for p in providers}
        for model in models:
            for name, session in model.get_inference_pool().items():
                actual = session.get_providers()
                if not actual or actual[0] not in expected:
                    raise RuntimeError(f'{name} 的 {provider.upper()} 推理初始化失败：{actual}')
                report(message=f'{name}: {", ".join(actual)}')

        self.identity_mode = identity_mode
        self.reference_distance = options['reference_face_distance']
        self.detector_score = options['vr_detector_score']
        self.detector_angles = options['face_detector_angles']
        self.analyse_faces = lambda image, embed: get_many_faces([image], classify=False, embed=embed)
        self.content_check = content_analyser.analyse_frame
        self.content_rate_limit = content_analyser.RATE_LIMIT
        self.swap = face_swapper.swap_face
        self.enhance = face_enhancer.enhance_face if enhance else None
        image = read_image(str(source))
        source_faces = get_many_faces([image], classify=False) if image is not None else []
        if not source_faces:
            raise ValueError('来源图片未检测到人脸。')
        self.source = max(source_faces, key=lambda f: np.prod(f.bounding_box[2:] - f.bounding_box[:2]))
        self.padding = padding
        self.boxes = [np.asarray(box, dtype=np.float64) for box in boxes]
        self.disparity = center(self.boxes[1]) - center(self.boxes[0])
        self.wb = wb
        with wb.DETECT_LOCK:
            self.detector = wb.detector(provider)

        # The selected stereo pair is the fixed identity library, not the source face.
        self.targets = []
        if identity_mode == 'off':
            report(message='身份匹配已关闭；按目标位置跟随，不运行目标人脸特征比对。')
            return
        for eye, box in zip(np.hsplit(sample, 2), self.boxes):
            match = self.match(eye, box, padding)
            if match is None:
                if identity_mode == 'strict':
                    raise ValueError('选中的参考框内没有找到可对齐的人脸。请换一帧或扩大取景余量。')
            else:
                self.targets.append(match.face)
        if not self.targets:
            raise ValueError('左右眼参考框都没有找到可对齐的人脸。请换一帧或扩大取景余量。')
        if len(self.targets) == 2 and distance(*self.targets) > STEREO_DISTANCE:
            raise ValueError('左右眼参考脸比对不通过。请选择同一人物的清晰一帧。')
        report(message=f'已从参考帧提取 {len(self.targets)} 张目标脸。')

    def identity_distance(self, face):
        return min(distance(face, target) for target in self.targets)

    def match(self, eye, box, padding, index=None, partner=None, recovered=False, *, verify=True):
        view = make_view(eye, box, padding)
        if view is None:
            return None
        best, best_cost = None, math.inf
        embed = verify and self.identity_mode != 'off'
        for face in self.analyse_faces(view.image, embed):
            measured = face_box(view, face, eye.shape)
            if measured is None or min(measured[2:]) < .003:
                continue
            offset = np.linalg.norm((center(measured) - center(box)) / np.maximum(box[2:], .018))
            if offset > (.65 if recovered else .52):
                continue
            score = 0
            if embed and index is not None:
                score = self.identity_distance(face)
                if score > self.reference_distance or (self.identity_mode == 'strict' and partner is not None
                                                  and distance(face, partner.face) > STEREO_DISTANCE):
                    continue
            cost = score + .16 * offset
            if cost < best_cost:
                best = Match(view, face, measured, recovered)
                best_cost = cost
        return best

    def candidates(self, detections, anchor):
        ordered = sorted(detections, key=lambda item:
                         np.linalg.norm((center(np.asarray(item['box'])) - center(anchor)) /
                                        np.maximum(anchor[2:], .035)) - .4 * item['score'])
        return [np.asarray(item['box']) for item in ordered[:3]]

    @staticmethod
    def near_prediction(box, predicted):
        delta = center(box) - center(predicted)
        return np.linalg.norm(delta) <= .12 and np.all(np.abs(delta) <= np.maximum(predicted[2:] * .65, .018))

    def relaxed_matches(self, eyes, detections):
        confirmed = [None, None]
        for i in range(2):
            for box in self.candidates(detections[i], self.boxes[i]):
                confirmed[i] = self.match(eyes[i], box, self.padding, i)
                if confirmed[i] is not None:
                    break
        if not any(confirmed):
            for i in range(2):
                confirmed[i] = self.match(eyes[i], self.boxes[i], min(3., self.padding * 1.3), i, recovered=True)
        anchors = [i for i in range(2) if confirmed[i] is not None]
        if not anchors:
            return confirmed
        if len(anchors) == 2:
            observed = center(confirmed[1].box) - center(confirmed[0].box)
            if np.linalg.norm(observed - self.disparity) <= .12:
                self.update_stereo(confirmed, observed)
                return confirmed

        i = min(anchors, key=lambda eye: self.identity_distance(confirmed[eye].face))
        j = 1 - i
        primary = confirmed[i]
        predicted = shifted_box(primary.box, self.disparity if j else -self.disparity)
        partner = None
        for box in self.candidates(detections[j], predicted):
            if not self.near_prediction(box, predicted):
                continue
            candidate = self.match(eyes[j], box, self.padding, j, verify=False)
            if candidate is not None and self.near_prediction(candidate.box, predicted):
                partner = candidate
                break
        if partner is None:
            # No usable coarse detection: run FaceFusion on the predicted perspective crop.
            partner = self.match(eyes[j], predicted, min(3., self.padding * 1.4), j, recovered=True, verify=False)
            if partner is not None and not self.near_prediction(partner.box, predicted):
                partner = None

        self.boxes[i] = primary.box
        if min(predicted[2:]) >= .003:
            self.boxes[j] = predicted
        confirmed[j] = partner
        return confirmed

    def update_stereo(self, matches, observed):
        # Recovery may be noisy. Only direct detections can shift the stereo offset.
        if not any(match.recovered for match in matches):
            self.disparity = .8 * self.disparity + .2 * observed
        self.boxes = [match.box for match in matches]

    def render_matches(self, matches):
        edits = {}
        for i, match in enumerate(matches):
            if match is None:
                continue
            image = self.swap(self.source, match.face, match.view.image)
            if self.enhance:
                image = self.enhance(match.face, image)
            edits[i] = (match.view, image)
        return edits

    def flagged_content(self, eyes, matches, sample_content):
        return sample_content and any(self.content_check(match.view.image if match else eye)
                                      for eye, match in zip(eyes, matches))

    def edits(self, frame, sample_content=False):
        eyes = np.hsplit(frame, 2)
        with self.wb.DETECT_LOCK:
            detections = [self.wb.detect_eye(eye, self.detector, self.detector_score, self.detector_angles) for eye in eyes]
        if self.identity_mode == 'relaxed':
            matches = self.relaxed_matches(eyes, detections)
            flagged = self.flagged_content(eyes, matches, sample_content)
            recovered = any(match is not None and match.recovered for match in matches)
            return self.render_matches(matches), recovered, flagged

        matches = [None, None]
        for i, candidates in enumerate(detections):
            for box in self.candidates(candidates, self.boxes[i]):
                matches[i] = self.match(eyes[i], box, self.padding, i, matches[1 - i])
                if matches[i] is not None:
                    break

        # When detectors miss both eyes, start at the last verified positions.
        if not any(matches):
            for i in range(2):
                matches[i] = self.match(eyes[i], self.boxes[i], min(3., self.padding * 1.3), i, recovered=True)
                if matches[i] is not None:
                    break
        for i in range(2):
            partner = matches[1 - i]
            if matches[i] is None and partner is not None:
                predicted = shifted_box(partner.box, self.disparity if i else -self.disparity)
                matches[i] = self.match(eyes[i], predicted, min(3., self.padding * 1.4), i, partner, recovered=True)

        flagged = self.flagged_content(eyes, matches, sample_content)
        if not all(matches):
            return {}, False, flagged
        if self.identity_mode == 'strict' and distance(matches[0].face, matches[1].face) > STEREO_DISTANCE:
            return {}, False, flagged

        observed = center(matches[1].box) - center(matches[0].box)
        if np.linalg.norm(observed - self.disparity) > .12:
            return {}, False, flagged
        self.update_stereo(matches, observed)
        return self.render_matches(matches), any(match.recovered for match in matches), flagged


class BackProject:
    def __init__(self, height, width):
        theta = (np.arange(width, dtype=np.float32) / width - .5) * np.pi
        phi = (.5 - np.arange(height, dtype=np.float32) / height) * np.pi
        self.sin_t, self.cos_t = np.sin(theta), np.cos(theta)
        self.sin_p, self.cos_p = np.sin(phi), np.cos(phi)

    def apply(self, eye, view, changed):
        # Warp the signed color delta, not the original crop: untouched pixels stay untouched.
        delta = changed.astype(np.int16) - view.image
        x, y, w, h = cv2.boundingRect(cv2.convertScaleAbs(delta).max(axis=2))
        if not w:
            return
        x0, y0 = max(0, x - 2), max(0, y - 2)
        x1, y1 = min(VIEW_SIZE, x + w + 2), min(VIEW_SIZE, y + h + 2)
        erp_x, erp_y = (m[y0:y1, x0:x1] for m in view.maps)
        height, width = eye.shape[:2]
        l = max(0, math.floor(float(erp_x.min())) - 2)
        r = min(width, math.ceil(float(erp_x.max())) + 3)
        t = max(0, math.floor(float(erp_y.min())) - 2)
        b = min(height, math.ceil(float(erp_y.max())) + 3)
        if r <= l or b <= t:
            return
        dx = self.cos_p[t:b, None] * self.sin_t[None, l:r]
        dy = -self.sin_p[t:b, None]
        dz = self.cos_p[t:b, None] * self.cos_t[None, l:r]
        rotation = view.rotation.astype(np.float32)
        xc, yc, zc = (dx * i + dy * j + dz * k for i, j, k in rotation.T)
        valid = zc > 1e-6
        safe = np.where(valid, zc, 1)
        c = (VIEW_SIZE - 1) / 2
        u = np.where(valid, xc / safe * view.fx + c - x0, -1)
        v = np.where(valid, yc / safe * view.fy + c - y0, -1)
        warped = cv2.remap(delta[y0:y1, x0:x1], u.astype(np.float32), v.astype(np.float32), cv2.INTER_LINEAR)
        region = eye[t:b, l:r]
        region[:] = (region.astype(np.int16) + warped).clip(0, 255).astype(np.uint8)


def render(wb, video, source, output, start, end, reference, boxes, padding, provider, enhance, report, *,
           identity_mode='strict', options=None):
    report(phase='reference')
    sample, info = wb.read_frame(video, reference)
    swap = StereoSwap(wb, source, sample, boxes, padding, provider, enhance, report,
                      identity_mode=identity_mode, options=options)
    del sample
    width, height, fps = info['width'], info['height'], info['fps']
    total = end - start + 1
    project = BackProject(height, width // 2)
    stats = {'swapped': 0, 'single_eye': 0, 'recovered': 0, 'unchanged': 0}
    report(phase='dynamic', completed=0, total=total, stats=stats)
    temp = output.with_name('rendering.mp4')
    command = [
        'ffmpeg', '-y', '-hide_banner', '-loglevel', 'error',
        '-f', 'rawvideo', '-pix_fmt', 'bgr24', '-s', f'{width}x{height}', '-r', str(fps), '-i', 'pipe:0',
        '-ss', str(start / fps), '-i', str(video), '-map', '0:v:0', '-map', '1:a?',
        '-map_metadata', '1', *pipeline._h264_encode_args(18), '-c:a', 'aac',
        '-t', str(total / fps), '-movflags', '+faststart', str(temp),
    ]
    cap, _ = wb.open_video(video)
    try:
        if start and not cap.set(cv2.CAP_PROP_POS_FRAMES, start):
            raise RuntimeError(f'无法定位到起始帧 {start}。')
        with subprocess.Popen(command, stdin=subprocess.PIPE, stderr=subprocess.PIPE) as encoder:
            drain, errors = pipeline._start_stderr_drain(encoder)

            def decode(index):
                ok, frame = cap.read()
                if not ok:
                    raise RuntimeError(f'无法解码视频第 {start + index} 帧。')
                return frame

            def composite_encode(index, frame, edits, counts):
                eyes = np.hsplit(frame, 2)
                for eye, (view, changed) in edits.items():
                    project.apply(eyes[eye], view, changed)
                try:
                    encoder.stdin.write(memoryview(frame).cast('B'))
                except OSError as exc:
                    encoder.wait(timeout=5)
                    drain.join()
                    details = b''.join(errors).decode(errors='replace')
                    raise RuntimeError(f'FFmpeg 无法接收第 {start + index} 帧：{details or exc}') from exc
                report(phase='dynamic', completed=index + 1, total=total,
                       frame=index, images=(frame,), stats=counts)

            try:
                flagged, interval = 0, max(1, round(fps))
                with ThreadPoolExecutor(1) as reader, ThreadPoolExecutor(1) as writer:
                    read = reader.submit(decode, 0)
                    pending = None
                    for index in range(total):
                        frame = read.result()
                        if index + 1 < total:
                            read = reader.submit(decode, index + 1)
                        edits, recovered, flagged_frame = swap.edits(frame, index % interval == 0)
                        flagged += flagged_frame
                        if flagged * interval / total * 100 > swap.content_rate_limit:
                            raise ValueError('FaceFusion 输入内容检查未通过。')
                        if len(edits) == 2:
                            stats['swapped'] += 1
                            stats['recovered'] += recovered
                        elif edits:
                            stats['single_eye'] += 1
                        else:
                            stats['unchanged'] += 1
                        if pending is not None:
                            pending.result()
                        pending = writer.submit(composite_encode, index, frame, edits, dict(stats))
                    if pending is not None:
                        pending.result()
                report(phase='finalize')
                encoder.stdin.close()
                code = encoder.wait()
                drain.join()
                if code:
                    raise RuntimeError('FFmpeg 编码失败：' + b''.join(errors).decode(errors='replace'))
            finally:
                if encoder.poll() is None:
                    encoder.kill()
        temp.replace(output)
        if not stats['swapped'] and not stats['single_eye']:
            report(message='本片段没有可换脸的帧，输出保留原画。')
    finally:
        cap.release()
        temp.unlink(missing_ok=True)
