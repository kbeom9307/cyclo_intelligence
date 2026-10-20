#!/bin/sh
set -eu
source_dir=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
python3 -m venv --without-pip --system-site-packages /opt/hil-serl-venv
python3 -m pip --python /opt/hil-serl-venv/bin/python install --no-cache-dir \
  -r "$source_dir/requirements-hilserl-upstream.txt"
/opt/hil-serl-venv/bin/python - <<'PY'
from pathlib import Path
from urllib.request import urlopen
import hashlib
path = Path.home() / '.serl' / 'resnet10_params.pkl'
digest = '175745d43d30233eb01b5369465d1c24c11b8ee71ccb734cc1c1bca13e07f57b'
if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != digest:
    with urlopen('https://github.com/rail-berkeley/serl/releases/download/resnet10/resnet10_params.pkl', timeout=90) as response:
        content = response.read(32 * 1024 * 1024)
    if hashlib.sha256(content).hexdigest() != digest:
        raise RuntimeError('Unexpected upstream ImageNet ResNet-10 checkpoint checksum')
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.tmp')
    temporary.write_bytes(content)
    temporary.replace(path)
print('Pinned HIL-SERL runtime and verified ImageNet checkpoint ready')
PY
