"""Safe numeric NPZ transport for the external learner (no pickle)."""
import io
import json
import os
from pathlib import Path
from urllib.request import Request, build_opener, HTTPRedirectHandler
from urllib.error import HTTPError
from urllib.parse import urlsplit
import zipfile
import numpy as np
from hilserl_data import FIELDS
from hilserl_profiles import image_shape

PROTOCOL = 'cyclo-hilserl-npz-v2'
MAX_PACKET = 96 * 1024 * 1024


def pack(meta, arrays=None):
    stream = io.BytesIO()
    np.savez(stream, __meta__=np.array(json.dumps(meta, allow_nan=False)), **(arrays or {}))
    return stream.getvalue()


def unpack(payload):
    if len(payload) > MAX_PACKET:
        raise ValueError('Learner packet exceeds 96 MiB')
    with zipfile.ZipFile(io.BytesIO(payload)) as archive:
        if len(archive.infolist()) > 512 or sum(x.file_size for x in archive.infolist()) > MAX_PACKET:
            raise ValueError('Expanded learner packet exceeds limit')
    with np.load(io.BytesIO(payload), allow_pickle=False) as data:
        meta = json.loads(str(data['__meta__']))
        arrays = {k: data[k] for k in data.files if k != '__meta__'}
    if any(a.dtype.kind not in 'buif' or not np.isfinite(a).all() for a in arrays.values()):
        raise ValueError('Learner arrays must be finite numeric values')
    return meta, arrays


def rows_to_arrays(rows, pixel):
    fields = FIELDS + (('images', 'next_images') if pixel else ())
    return {key: np.stack([row.get(key, False) for row in rows]) for key in fields} if rows else {}


def arrays_to_rows(arrays, spec):
    if not arrays:
        return []
    count = len(arrays.get('reward', []))
    if not 1 <= count <= 32:
        raise ValueError('Transition batch must contain 1–32 rows')
    shapes = {'observation': (count, spec['observation_dim']), 'next_observation': (count, spec['observation_dim']),
              'action': (count, spec['action_dim']), **{k: (count,) for k in ('reward', 'terminal', 'human', 'truncated')}}
    if spec['pixel']:
        shapes.update(images=(count, *image_shape(spec['config'])), next_images=(count, *image_shape(spec['config'])))
    if set(arrays) != set(shapes) or any(arrays[k].shape != shape for k, shape in shapes.items()):
        raise ValueError('Transition schema does not match learner')
    if spec['pixel'] and any(arrays[k].dtype != np.uint8 for k in ('images', 'next_images')):
        raise ValueError('RGB images must be uint8')
    if np.abs(arrays['action']).max() > 1 or any(not np.isin(arrays[k], [0, 1]).all() for k in ('terminal', 'human', 'truncated')):
        raise ValueError('Invalid action or terminal/human flags')
    return [{k: a[i] for k, a in arrays.items()} for i in range(count)]


def validate_url(value):
    url = urlsplit(value)
    if url.scheme not in ('http', 'https') or not url.hostname or url.username or url.password or url.query or url.fragment:
        raise ValueError('Use an http(s) learner URL without credentials, query or fragment')
    if url.scheme != 'https' and url.hostname not in ('127.0.0.1', 'localhost', '::1'):
        raise ValueError('Use HTTPS for an external host, or HTTP localhost with an SSH tunnel')
    return value.rstrip('/')


def remote_token(config):
    token = os.environ.get('CYCLO_HIL_LEARNER_TOKEN', '').strip()
    if not token:
        try:
            token = Path(config.get('learner_token_file', '/workspace/hilserl/learner-token')).read_text().strip()
        except OSError:
            raise ValueError('Learner token file is missing; copy the server token to the configured local file') from None
    if len(token) < 32:
        raise ValueError('Learner token must contain at least 32 characters')
    return token


class NoRedirect(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None  # Never forward the bearer token to another endpoint.


class RemoteEndpoint:
    def __init__(self, config):
        self.url, self.token = validate_url(config['learner_url']), remote_token(config)
        self.run_id = None
        self.sequence = 0
        self.pixel = False

    def request(self, path, body=None, content_type='application/x-npz', timeout=10):
        request = Request(self.url + path, data=body, headers={'Authorization': 'Bearer ' + self.token,
                          'Content-Type': content_type}, method='POST' if body is not None else 'GET')
        try:
            with build_opener(NoRedirect).open(request, timeout=timeout) as response:
                data = response.read(MAX_PACKET + 1)
                if len(data) > MAX_PACKET:
                    raise RuntimeError('Oversized learner response')
                return data
        except HTTPError as exc:
            detail = exc.read(2048).decode(errors='replace')
            raise RuntimeError(f'Learner HTTP {exc.code}: {detail}') from None
        except OSError as exc:
            raise RuntimeError(f'Learner connection failed: {type(exc).__name__}') from None

    def open(self, spec, weights=None):
        self.pixel = spec['pixel']
        result = json.loads(self.request('/runs', json.dumps(spec).encode(), 'application/json', timeout=300))
        self.run_id = result['run_id']
        try:
            if weights:
                self.request(f'/runs/{self.run_id}/weights', pack({}, weights))
            return unpack(self.request(f'/runs/{self.run_id}/exchange', pack({'sequence': 0, 'learning': False, 'version': -1})))
        except Exception:
            self.close()
            raise

    def exchange(self, rows, learning, version, offline=False):
        self.sequence += 1
        return unpack(self.request(f'/runs/{self.run_id}/exchange',
            pack({'sequence': self.sequence, 'learning': learning, 'version': version, 'offline': offline},
                 rows_to_arrays(rows, self.pixel)), timeout=300 if learning else 30))

    def close(self):
        if self.run_id:
            return unpack(self.request(f'/runs/{self.run_id}/stop', b'', timeout=20))
        return {}, {}
