from __future__ import annotations

import argparse
import json
import mimetypes
import re
import subprocess
import sys
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from importlib.util import find_spec
from pathlib import Path
from urllib.parse import parse_qs, urlsplit


if __name__ == '__main__' and find_spec('onnxruntime') is None:
    prefix = Path(sys.prefix)
    conda_root = prefix.parent.parent if prefix.parent.name == 'envs' else prefix
    env = conda_root / 'envs' / 'facefusion'
    python = env / ('python.exe' if sys.platform == 'win32' else 'bin/python')
    if python.is_file() and python.resolve() != Path(sys.executable).resolve():
        print(f'Starting workbench with the FaceFusion environment: {python}', flush=True)
        raise SystemExit(subprocess.call([str(python), str(Path(__file__).resolve()), *sys.argv[1:]]))
    raise SystemExit(f'ONNX Runtime is missing in {sys.executable}. Activate the FaceFusion environment before starting the workbench.')

import workbench as wb


STATIC = {'/': 'index.html', '/index.html': 'index.html', '/test': 'index.html',
          '/facefusion': 'index.html', '/facefusion/': 'index.html',
          '/workbench.css': 'workbench.css', '/workbench.js': 'workbench.js'}
ARTIFACTS = {'output.mp4', 'monitor.png', 'left_flat.mp4', 'right_flat.mp4', 'left_swapped.mp4', 'right_swapped.mp4', 'selection.json'}
JOB_PATH = re.compile(r'/api/jobs/([a-f0-9]{32})')
PREVIEW_PATH = re.compile(r'/api/jobs/([a-f0-9]{32})/preview')
MEDIA_PATH = re.compile(r'/media/([a-f0-9]{32})/([a-z_.0-9]+)')
PLAYBACK_PATH = re.compile(r'/api/playback/([a-f0-9]{32})')


class Handler(BaseHTTPRequestHandler):
    def send_json(self, data, status=200):
        body = json.dumps(data, ensure_ascii=False).encode('utf-8')
        self.send_response(status)
        self.send_header('Content-Type', 'application/json; charset=utf-8')
        self.send_header('Cache-Control', 'no-store')
        self.send_header('Content-Length', str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def local_request(self):
        host = self.headers.get('Host', '')
        accepted = {f'127.0.0.1:{self.server.server_port}', f'localhost:{self.server.server_port}'}
        origin = self.headers.get('Origin')
        if host not in accepted or (origin and origin != f'http://{host}'):
            self.send_json({'error': 'Only same-origin localhost requests are accepted.'}, 403)
            return False
        return True

    def log_message(self, format, *args):
        if self.command == 'GET' and (self.path.startswith('/api/jobs/') or self.path == '/api/health'):
            return
        super().log_message(format, *args)

    def body_length(self, limit):
        length = int(self.headers.get('Content-Length', '0'))
        if not 0 < length <= limit:
            raise ValueError('Invalid request size.')
        return length

    def upload(self, query):
        kind = query.get('kind', [''])[0]
        name = query.get('name', [''])[0]
        suffix = Path(name).suffix.lower()
        if kind not in {'video', 'face'} or suffix not in (wb.VIDEO_SUFFIXES if kind == 'video' else wb.FACE_SUFFIXES):
            raise ValueError('Unsupported file type.')
        remaining = self.body_length(16 * 1024**3 if kind == 'video' else 32 * 1024**2)
        root = wb.DATA / 'uploads'
        root.mkdir(parents=True, exist_ok=True)
        path = root / (uuid.uuid4().hex + suffix)
        try:
            with path.open('xb') as stream:
                while remaining:
                    chunk = self.rfile.read(min(remaining, 1024 * 1024))
                    if not chunk:
                        raise ValueError('Upload interrupted.')
                    stream.write(chunk)
                    remaining -= len(chunk)
        except Exception:
            path.unlink(missing_ok=True)
            raise
        return {'path': str(path)}

    def file(self, path, download=False):
        size = path.stat().st_size
        first, last = 0, size - 1
        request_range = self.headers.get('Range')
        if request_range:
            match = re.fullmatch(r'bytes=(\d*)-(\d*)', request_range)
            if match:
                start, end = match.groups()
                if start:
                    first, last = int(start), min(int(end), last) if end else last
                elif end:
                    first = max(0, size - int(end))
                else:
                    first = size
            if not match or not 0 <= first <= last:
                self.send_response(416)
                self.send_header('Content-Range', f'bytes */{size}')
                self.end_headers()
                return
        self.send_response(206 if request_range else 200)
        self.send_header('Content-Type', mimetypes.guess_type(path.name)[0] or 'application/octet-stream')
        self.send_header('X-Content-Type-Options', 'nosniff')
        self.send_header('Cache-Control', 'no-store')
        self.send_header('Content-Length', str(last - first + 1))
        self.send_header('Accept-Ranges', 'bytes')
        if request_range:
            self.send_header('Content-Range', f'bytes {first}-{last}/{size}')
        if download:
            self.send_header('Content-Disposition', f'attachment; filename="{path.name}"')
        self.end_headers()
        with path.open('rb') as stream:
            stream.seek(first)
            remaining = last - first + 1
            while remaining:
                chunk = stream.read(min(1024 * 1024, remaining))
                if not chunk:
                    break
                self.wfile.write(chunk)
                remaining -= len(chunk)

    def do_GET(self):
        if not self.local_request():
            return
        url = urlsplit(self.path)
        path = url.path
        try:
            if path in STATIC:
                self.file(wb.ROOT / STATIC[path])
            elif path == '/api/health':
                self.send_json({'ok': True, 'model': wb.MODEL.is_file()})
            elif path == '/api/facefusion/options':
                self.send_json(wb.tuning.catalog())
            elif match := PLAYBACK_PATH.fullmatch(path):
                self.file(wb.playback_file(match[1]))
            elif match := JOB_PATH.fullmatch(path):
                self.send_json(wb.job_status(match[1]))
            elif match := PREVIEW_PATH.fullmatch(path):
                data = wb.job_preview(match[1], parse_qs(url.query).get('seq', [''])[0])
                self.send_response(200)
                self.send_header('Content-Type', 'image/jpeg')
                self.send_header('Cache-Control', 'no-store')
                self.send_header('Content-Length', str(len(data)))
                self.end_headers()
                self.wfile.write(data)
            elif (match := MEDIA_PATH.fullmatch(path)) and match[2] in ARTIFACTS:
                self.file(wb.DATA / 'jobs' / match[1] / match[2], download='download=1' in url.query)
            else:
                self.send_json({'error': 'Not found.'}, 404)
        except FileNotFoundError:
            self.send_json({'error': 'File or job not found.'}, 404)
        except (BrokenPipeError, ConnectionResetError):
            pass
        except RuntimeError as exc:
            self.send_json({'error': str(exc)}, 503)

    def do_POST(self):
        if not self.local_request():
            return
        url = urlsplit(self.path)
        actions = {'/api/inspect': wb.inspect_video, '/api/preview': wb.preview_crops, '/api/jobs': wb.start_job}
        try:
            if url.path == '/api/upload' and self.headers.get_content_type() == 'application/octet-stream':
                self.send_json(self.upload(parse_qs(url.query)), 201)
            elif self.headers.get_content_type() == 'application/json' and (url.path in actions or url.path == '/api/tests' or JOB_PATH.fullmatch(url.path)):
                data = json.loads(self.rfile.read(self.body_length(64 * 1024)))
                if not isinstance(data, dict):
                    raise ValueError('Expected a JSON object.')
                if url.path == '/api/tests':
                    self.send_json(wb.start_job(data, staged=True), 202)
                elif match := JOB_PATH.fullmatch(url.path):
                    self.send_json(wb.advance_job(match[1], data), 202)
                else:
                    self.send_json(actions[url.path](data), 202 if url.path == '/api/jobs' else 200)
            else:
                self.send_json({'error': 'Unknown endpoint or invalid request type.'}, 404)
        except (BrokenPipeError, ConnectionResetError):
            pass
        except (TypeError, ValueError) as exc:
            self.send_json({'error': str(exc)}, 400)
        except Exception as exc:
            self.log_error('%s', exc)
            self.send_json({'error': str(exc)}, 500)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Local VR180 SBS workbench')
    parser.add_argument('--port', type=int, default=8080)
    args = parser.parse_args()
    with ThreadingHTTPServer(('127.0.0.1', args.port), Handler) as server:
        print(f'VR180 workbench: http://127.0.0.1:{server.server_port}', flush=True)
        server.serve_forever()
