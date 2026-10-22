"""MuJoCo implementation of the simulator-neutral NIST observation contract."""
import json
import os
from urllib.error import HTTPError, URLError
from urllib.parse import urlparse
from urllib.request import Request, urlopen

from hilserl_isaac import IsaacNistEnv, PROTOCOL, SCHEMA, frame as simulator_frame


def bridge_url():
    url = os.environ.get("CYCLO_HILSERL_MUJOCO_URL", "http://127.0.0.1:7126").rstrip("/")
    parsed = urlparse(url)
    if parsed.scheme != "http" or parsed.hostname not in ("127.0.0.1", "localhost"):
        raise ValueError("MuJoCo bridge must use local loopback HTTP")
    return url


def request(path, body=None, timeout=3):
    req = Request(bridge_url()+path, data=None if body is None else json.dumps(body, allow_nan=False).encode(),
                  headers={"Content-Type": "application/json"})
    try:
        with urlopen(req, timeout=timeout) as response:
            return json.load(response)
    except HTTPError as exc:
        try:
            detail = json.load(exc).get("error", str(exc))
        except (ValueError, AttributeError):
            detail = str(exc)
        raise RuntimeError(f"MuJoCo bridge: {detail}") from exc
    except (OSError, URLError, ValueError) as exc:
        raise RuntimeError(f"MuJoCo bridge unavailable: {exc}") from exc


def health():
    try:
        data = request("/health", timeout=.5)
        if (data.get("environment") != "mujoco" or data.get("protocol") != PROTOCOL
                or data.get("schema") != SCHEMA or data.get("observation_dim") != 39 or data.get("action_dim") != 6):
            return {"ready": False, "error": "Incompatible MuJoCo bridge protocol/schema"}
        return data
    except (RuntimeError, ValueError) as exc:
        return {"ready": False, "error": str(exc)}


class MujocoNistEnv(IsaacNistEnv):
    def _request(self, path, body, timeout):
        return request(path, body, timeout)


def frame(camera="spectator", quality="standard"):
    try:
        return simulator_frame(camera, quality, endpoint=bridge_url())
    except RuntimeError as exc:
        raise RuntimeError("MuJoCo live view is unavailable; start the MuJoCo bridge") from exc
