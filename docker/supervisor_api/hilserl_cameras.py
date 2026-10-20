"""Independent camera routing with stable RGB channel slots and artifact metadata."""
import json
import numpy as np

CAMERAS = ['head', 'left_wrist', 'right_wrist']


def selected(config, purpose):
    values = config.get(purpose + '_cameras', CAMERAS)
    if not values or len(set(values)) != len(values) or any(v not in CAMERAS for v in values):
        raise ValueError('Select at least one distinct supported camera')
    return [camera for camera in CAMERAS if camera in values]


def route(images, cameras):
    # Preserve the established 9-channel format; excluded views contain no image information.
    value = images.copy()
    for index, camera in enumerate(CAMERAS):
        if camera not in cameras:
            value[index * 3:(index + 1) * 3] = 0
    return value


def validate_artifact(data, cameras, config=None):
    saved = json.loads(str(data['__cameras__'])) if '__cameras__' in data else CAMERAS
    if saved != cameras:
        raise ValueError(f'Artifact cameras {saved} do not match selected cameras {cameras}')
    if config is not None:
        crops = json.loads(str(data['__crops__'])) if '__crops__' in data else {}
        if crops != config.get('camera_crops', {}):
            raise ValueError('Artifact camera crops differ from this environment; use matching crops or recollect data')
