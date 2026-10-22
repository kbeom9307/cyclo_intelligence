"""Local JSON bridge. HTTP threads queue work; only the simulation thread touches MuJoCo.

One leased owner; sequenced mutations never execute twice. No DDS/ROS publishing.
"""
import json
import queue
from urllib.parse import urlsplit, parse_qs
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

PROTOCOL = 'cyclo-nist-hil-v1'


# overview width/height, robot-camera width/height, JPEG quality.
FRAME_PROFILES = {'low': (320, 240, 320, 180, 60),
                  'standard': (640, 480, 480, 270, 82),
                  'high': (960, 720, 640, 360, 95),
                  'ultra': (1920, 1440, 640, 360, 98)}

class Bridge:
    def __init__(self, adapter, port=7126, lease_seconds=30):
        self.adapter = adapter
        self.queue = queue.Queue(maxsize=8)
        self.lock = threading.Lock()
        self.owner = None
        self.expires = 0
        self.sequence = -1
        self.last_response = None
        self.lease_seconds = lease_seconds
        self.alive = True
        self.heartbeat = time.monotonic()
        self.details = adapter.describe()
        self.frame_lock = threading.Lock()
        self.frame = None
        self.frame_at = 0.
        self.frame_id = 0
        self.viewer_at = 0.
        self.frame_variants = {}
        self.camera_frames = {}
        self.camera_viewers = {}
        bridge = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass

            def reply(self, code, payload):
                body = json.dumps(payload, allow_nan=False).encode()
                self.send_response(code)
                self.send_header('Content-Type', 'application/json')
                self.send_header('Content-Length', str(len(body)))
                self.end_headers()
                try:
                    self.wfile.write(body)
                except (BrokenPipeError, ConnectionResetError):
                    pass

            def do_GET(self):
                if urlsplit(self.path).path == '/frame.jpg':
                    query = parse_qs(urlsplit(self.path).query)
                    camera = query.get('camera', ['spectator'])[0]
                    quality = query.get('quality', ['standard'])[0]
                    if quality not in FRAME_PROFILES:
                        return self.reply(400, {'error': 'Unknown quality'})
                    if camera not in ('spectator', 'head', 'left_wrist', 'right_wrist'):
                        return self.reply(400, {'error': 'Unknown camera'})
                    with bridge.frame_lock:
                        bridge.camera_viewers[camera] = time.monotonic()
                        if camera == 'spectator':
                            bridge.viewer_at = time.monotonic()
                            frame, at, frame_id = bridge.frame, bridge.frame_at, bridge.frame_id
                        else:
                            frame, at, frame_id = bridge.camera_frames.get(camera, (None, 0., 0))
                        if quality != 'standard':
                            frame, at, frame_id = bridge.frame_variants.get((camera, quality), (None, 0., 0))
                    age = time.monotonic() - at
                    if not frame or age > 3:
                        return self.reply(503, {'error': 'Visualization frame not ready'})
                    self.send_response(200)
                    self.send_header('Content-Type', 'image/jpeg')
                    self.send_header('Cache-Control', 'no-store')
                    self.send_header('Content-Length', str(len(frame)))
                    self.send_header('X-Frame-Id', str(frame_id))
                    self.send_header('X-Frame-Quality', quality)
                    profile = FRAME_PROFILES[quality]
                    width, height = profile[:2] if camera == 'spectator' else profile[2:4]
                    self.send_header('X-Frame-Resolution', f'{width}x{height}')
                    self.send_header('X-Frame-Age-Ms', str(round(age * 1000)))
                    self.end_headers()
                    try:
                        self.wfile.write(frame)
                    except (BrokenPipeError, ConnectionResetError):
                        pass
                    return
                if self.path != '/health':
                    return self.reply(404, {'error': 'Not found'})
                with bridge.lock:
                    payload = {**bridge.details, 'protocol': PROTOCOL,
                               'ready': bridge.alive and time.monotonic() - bridge.heartbeat < 10,
                               'busy': bridge.owner is not None and time.monotonic() < bridge.expires}
                self.reply(200, payload)

            def do_POST(self):
                if self.path != '/command':
                    return self.reply(404, {'error': 'Not found'})
                try:
                    length = int(self.headers.get('Content-Length', '0'))
                    if not 0 < length <= 16384:
                        raise ValueError('Invalid request size')
                    body = json.loads(self.rfile.read(length), parse_constant=lambda _: (_ for _ in ()).throw(ValueError('Non-finite JSON')))
                    if not isinstance(body, dict):
                        raise ValueError('Expected object')
                    event = threading.Event()
                    job = {'request': body, 'done': event, 'cancelled': False}
                    bridge.queue.put_nowait(job)
                    if not event.wait(180 if body.get('operation') in ('acquire', 'reset') else 25):
                        job['cancelled'] = True
                        return self.reply(504, {'error': 'Simulation did not acknowledge command; do not retry with a new sequence'})
                    self.reply(*job['response'])
                except (ValueError, TypeError, queue.Full) as exc:
                    self.reply(400, {'error': str(exc)})

        self.server = ThreadingHTTPServer(('127.0.0.1', port), Handler)
        self.server.daemon_threads = True
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)

    def wants_frame(self, camera='spectator'):
        with self.frame_lock:
            return time.monotonic() - (self.viewer_at if camera == 'spectator' else self.camera_viewers.get(camera, 0.)) < 3

    def clear_frames(self, camera):
        with self.frame_lock:
            if camera == 'spectator':
                self.frame, self.frame_at = None, 0.
            self.camera_frames.pop(camera, None)
            self.frame_variants = {key: value for key, value in self.frame_variants.items() if key[0] != camera}

    def publish_frame(self, frame, camera='spectator', quality='standard'):
        with self.frame_lock:
            if quality != 'standard':
                previous = self.frame_variants.get((camera, quality), (None, 0., 0))
                self.frame_variants[(camera, quality)] = (frame, time.monotonic(), previous[2] + 1)
                return
            if camera == 'spectator':
                self.frame = frame
                self.frame_at = time.monotonic()
                self.frame_id += 1
            else:
                previous = self.camera_frames.get(camera, (None, 0., 0))
                self.camera_frames[camera] = (frame, time.monotonic(), previous[2] + 1)

    def start(self):
        self.thread.start()

    def dispatch(self, body):
        operation = body.get('operation')
        owner = body.get('owner')
        sequence = body.get('sequence')
        if not isinstance(owner, str) or not 16 <= len(owner) <= 128 or type(sequence) is not int or sequence < 0:
            return 400, {'error': 'Owner and nonnegative integer sequence required'}
        now = time.monotonic()
        if self.owner is not None and now >= self.expires:
            self.owner = None
            self.adapter.release()
        if operation == 'acquire':
            if self.owner not in (None, owner):
                return 409, {'error': 'Simulator is owned by another session'}
            if self.owner == owner:
                return self.last_response if sequence == self.sequence else (409, {'error': 'Already acquired'})
            config = body.get('config', {})
            try:
                result = self.adapter.configure(config)
            except (ValueError, KeyError, TypeError) as exc:
                return 422, {'error': str(exc)}
            self.owner = owner
        else:
            if self.owner != owner:
                return 409, {'error': 'Lease missing or expired; start a new session'}
            if sequence == self.sequence:
                return self.last_response
            if sequence != self.sequence + 1:
                return 409, {'error': 'Out-of-order command'}
            if operation == 'step':
                result = self.adapter.step(body.get('action'))
            elif operation == 'reset':
                result = self.adapter.reset()
            elif operation == 'capture':
                result = {'observation': self.adapter.observation(), 'info': self.adapter.info()}
            elif operation == 'heartbeat':
                result = {'ok': True}
            elif operation == 'release':
                self.adapter.release()
                self.owner = None
                result = {'ok': True}
            else:
                return 400, {'error': 'Unknown operation'}
        if isinstance(result, dict) and 'observation' in result and getattr(self.adapter, 'c', {}).get('observation_mode') == 'pixels':
            result['pixels'] = self.adapter.capture_observation()
        self.sequence = sequence
        self.last_response = (200, result)
        self.expires = time.monotonic() + self.lease_seconds
        return self.last_response

    def pump(self, timeout=.02):
        self.heartbeat = time.monotonic()
        if self.owner and self.heartbeat >= self.expires:
            with self.lock:
                self.owner = None
                self.adapter.release()
        try:
            job = self.queue.get(timeout=timeout)
        except queue.Empty:
            return
        if job['cancelled']:
            return
        try:
            with self.lock:
                job['response'] = self.dispatch(job['request'])
        except (ValueError, KeyError, TypeError) as exc:
            job['response'] = (422, {'error': str(exc)})
        except Exception as exc:
            self.owner = None
            job['response'] = (500, {'error': f'{type(exc).__name__}: {exc}'})
        finally:
            job['done'].set()
            self.heartbeat = time.monotonic()

    def close(self):
        self.alive = False
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=2)
