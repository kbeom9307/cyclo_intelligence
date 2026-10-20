"""JSON client for the simulator-owned NIST bridge (no Isaac dependency here)."""
import json
import math
import os
import time
from urllib.error import URLError, HTTPError
from urllib.request import Request, urlopen
from urllib.parse import urlparse
from uuid import uuid4

PROTOCOL = 'cyclo-nist-hil-v1'
SCHEMA = 'nist-sg2-state39-delta6-fixedgripper-v2'


def bridge_url():
    # Deployment-owned setting, never a browser-supplied URL.
    url = os.environ.get('CYCLO_HILSERL_ISAAC_URL', 'http://127.0.0.1:7125').rstrip('/')
    parsed = urlparse(url)
    if parsed.scheme != 'http' or parsed.hostname not in ('127.0.0.1', 'localhost'):
        raise ValueError('Isaac bridge must use local loopback HTTP')
    return url


def request(path, body=None, timeout=3):
    req = Request(bridge_url() + path, data=None if body is None else json.dumps(body, allow_nan=False).encode(),
                  headers={'Content-Type': 'application/json'})
    try:
        with urlopen(req, timeout=timeout) as response:
            return json.load(response)
    except HTTPError as exc:
        try:
            detail = json.load(exc).get('error', str(exc))
        except (ValueError, AttributeError):
            detail = str(exc)
        raise RuntimeError(f'Isaac bridge: {detail}') from exc
    except (OSError, URLError, ValueError) as exc:
        raise RuntimeError(f'Isaac bridge unavailable: {exc}') from exc


def health():
    try:
        data = request('/health', timeout=.5)
        if data.get('protocol') != PROTOCOL or data.get('schema') != SCHEMA or data.get('observation_dim') != 39 or data.get('action_dim') != 6:
            return {'ready': False, 'error': 'Incompatible Isaac bridge protocol/schema'}
        return data
    except (RuntimeError, ValueError) as exc:
        return {'ready': False, 'error': str(exc)}


class IsaacNistEnv:
    observation_dim = 39
    action_dim = 6
    schema = SCHEMA

    def __init__(self, config):
        self.owner = str(uuid4())
        self.sequence = -1
        self.config = config
        self.pixel_mode = config.get('observation_mode', 'state') == 'pixels'
        self.images = None
        self.capture_metadata = None
        if self.pixel_mode:
            from hilserl_profiles import pixel_schema
            self.schema, self.observation_dim = pixel_schema(config), 22
        self.info = {}
        self._obs = None
        self.closed = False
        self.last_heartbeat = 0
        result = self._command('acquire', config=config)
        self._accept(result)

    def _command(self, operation, **kwargs):
        self.sequence += 1
        # No automatic retry of mutations; ambiguous network failure stops Actor.
        result = request('/command', {'owner': self.owner, 'sequence': self.sequence,
                                     'operation': operation, **kwargs}, timeout=185 if operation in ('acquire', 'reset') else 28)
        self.last_heartbeat = time.monotonic()
        return result

    def _accept(self, result):
        import numpy as np
        obs = np.asarray(result['observation'], dtype=np.float32)
        if obs.shape != (39,) or not np.isfinite(obs).all():
            raise ValueError('Isaac returned an invalid observation')
        self._raw_obs = obs.copy()
        if self.pixel_mode:
            import base64, zlib
            from hilserl_profiles import image_shape, CAMERAS
            IMAGE_SHAPE = image_shape(getattr(self, 'config', {}))
            pixels = result.get('pixels', {})
            if pixels.get('encoding') != 'zlib-uint8' or pixels.get('shape') != list(IMAGE_SHAPE) or pixels.get('cameras') != CAMERAS:
                raise ValueError('Missing or incompatible synchronized pixel observation')
            if not isinstance(pixels.get('sequence'), int) or pixels['sequence'] <= (self.capture_metadata or {}).get('sequence', -1):
                raise ValueError('Stale pixel observation sequence')
            if not math.isfinite(pixels.get('timestamp', float('nan'))) or not 0 <= time.time() - pixels['timestamp'] < 10:
                raise ValueError('Stale pixel observation timestamp')
            decoder = zlib.decompressobj()
            raw = decoder.decompress(base64.b64decode(pixels['data'], validate=True), int(np.prod(IMAGE_SHAPE)) + 1)
            if not decoder.eof or decoder.unused_data:
                raise ValueError('Invalid or oversized pixel observation')
            if len(raw) != int(np.prod(IMAGE_SHAPE)):
                raise ValueError('Invalid pixel observation size')
            from hilserl_cameras import selected, route
            raw_images = np.frombuffer(raw, dtype=np.uint8).reshape(IMAGE_SHAPE)
            config = getattr(self, 'config', {})
            self.images = route(raw_images, selected(config, 'observation'))
            self.classifier_images = route(raw_images, selected(config, 'classifier'))
            self.capture_metadata = {k: v for k, v in pixels.items() if k != 'data'}
            # Robot proprioception only. Excludes peg pose/velocity, target and success progress.
            obs = np.concatenate((obs[:21], obs[37:38]))
        self._obs = obs
        self.info = result['info']
        self.position = np.asarray(self.info['position_m'], dtype=np.float32)
        return obs

    def capture(self):
        return self._accept(self._command('capture'))

    def observation(self):
        return self._obs.copy()

    def reset(self):
        return self._accept(self._command('reset'))

    def tcp_to_action(self, action):
        """Map manual TCP deltas into the policy's fixed-root action frame.

        This scene fixes the articulation root at identity world orientation.
        Keeping replay actions in the original frame preserves policy semantics.
        """
        import numpy as np
        value = np.asarray(action, dtype=np.float32).copy()
        q = self._obs[17:21].astype(np.float64)
        q /= np.linalg.norm(q)
        w, xyz = q[0], q[1:]
        for start in (0, 3):
            vector = value[start:start + 3]
            rotated = vector + 2 * np.cross(xyz, np.cross(xyz, vector) + w * vector)
            value[start:start + 3] = rotated / max(1., float(np.max(np.abs(rotated))))
        return value

    def step(self, action):
        import numpy as np
        action = np.asarray(action, dtype=np.float32)
        if action.shape != (6,) or not np.isfinite(action).all() or (np.abs(action) > 1).any():
            raise ValueError('Expected six finite normalized actions in [-1,1]')
        result = self._command('step', action=action.tolist())
        obs = self._accept(result)
        reward = float(result['reward'])
        if not math.isfinite(reward):
            raise ValueError('Non-finite Isaac reward')
        return obs, reward, bool(result['terminated']), bool(result['truncated'])

    def heartbeat(self):
        if time.monotonic() - self.last_heartbeat > 5:
            self._command('heartbeat')

    def close(self):
        if not self.closed:
            self.closed = True
            self._command('release')


def frame(camera='spectator', quality='standard'):
    """Proxy a bounded cached JPEG; never enqueue a physics command."""
    if quality not in ('low', 'standard', 'high', 'ultra'):
        raise ValueError('Unknown quality')
    if camera not in ('spectator', 'head', 'left_wrist', 'right_wrist'):
        raise ValueError('Unknown camera')
    try:
        with urlopen(bridge_url() + '/frame.jpg?camera=' + camera + '&quality=' + quality, timeout=2) as response:
            content = response.read(8 * 1024 * 1024 + 1)
            if len(content) > 8 * 1024 * 1024 or response.headers.get_content_type() != 'image/jpeg' or not content.startswith(b'\xff\xd8'):
                raise ValueError('Invalid visualization frame')
            return content, {'Cache-Control': 'no-store',
                             'X-Frame-Id': response.headers.get('X-Frame-Id', ''),
                             'X-Frame-Quality': response.headers.get('X-Frame-Quality', ''),
                             'X-Frame-Resolution': response.headers.get('X-Frame-Resolution', ''),
                             'X-Frame-Age-Ms': response.headers.get('X-Frame-Age-Ms', '')}
    except (OSError, URLError, ValueError) as exc:
        raise RuntimeError('Isaac live view is unavailable; start the bridge with visualization enabled') from exc
