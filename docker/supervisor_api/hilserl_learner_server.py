"""Authenticated external SAC learner. Launch one process; use HTTPS or an SSH tunnel."""
import asyncio
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
import gc
import json
import os
os.environ.setdefault('XLA_PYTHON_CLIENT_PREALLOCATE', 'false')
os.environ.setdefault('XLA_PYTHON_CLIENT_MEM_FRACTION', '.90')
from pathlib import Path
import secrets
import time
from uuid import uuid4, UUID

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import Response
from pydantic import BaseModel, ConfigDict, Field, ValidationError
from hilserl_profiles import prepare_jax_libraries
prepare_jax_libraries()
import torch

from hilserl import ExperimentConfig
from hilserl_devices import hardware_status, device_ready
from hilserl_learner import LearnerEngine
from hilserl_remote import PROTOCOL, MAX_PACKET, pack, unpack, arrays_to_rows
from hilserl_pixels import PIXEL_SCHEMA
from hilserl_profiles import pixel_schema, upstream
from hilserl_isaac import SCHEMA


class Exchange(BaseModel):
    model_config = ConfigDict(extra='forbid', strict=True)
    sequence: int = Field(ge=0)
    learning: bool
    version: int = Field(ge=-1)
    offline: bool = False


def validate_spec(spec):
    if set(spec) != {'protocol', 'pixel', 'schema', 'observation_dim', 'action_dim', 'config'} or spec['protocol'] != PROTOCOL:
        raise ValueError('Incompatible learner protocol')
    if type(spec['pixel']) is not bool:
        raise ValueError('Invalid pixel flag')
    config = ExperimentConfig(**{'learning_backend': 'legacy_torch', **spec['config']}).model_dump()
    expected = ((pixel_schema(config), 22, 6) if spec['pixel'] else (SCHEMA, 39, 6)) if config['environment'] in ('isaac_sim', 'mujoco') else ('mock-xyz-state4-delta3-v1', 4, 3)
    if (spec['schema'], spec['observation_dim'], spec['action_dim']) != expected or (spec['pixel'] and config['environment'] not in ('isaac_sim', 'mujoco')):
        raise ValueError('Incompatible observation/action schema')
    # Bound server RAM and reject arbitrary actor-side paths/credentials.
    allowed = {'environment', 'robot_type', 'arm', 'target_part', 'observation_cameras', 'seed', 'learning_rate',
               'discount', 'learner_device', 'pixel_batch_size', 'pixel_replay_capacity', 'batch_size', 'replay_capacity',
               'learning_backend', 'cta_ratio', 'training_starts', 'policy_publish_interval', 'camera_crops', 'image_size'}
    if set(spec['config']) - allowed or config['replay_capacity'] > 1000000:
        raise ValueError('Unsupported learner configuration or excessive replay capacity')
    return {**spec, 'config': {key: config[key] for key in allowed}}


def create_app(root=None, token=None):
    root = Path(root or os.environ.get('CYCLO_HIL_LEARNER_DATA_DIR', '/workspace/learner'))
    token = token or os.environ.get('CYCLO_HIL_LEARNER_TOKEN', '')
    if not token and os.environ.get('CYCLO_HIL_LEARNER_TOKEN_FILE'):
        token = Path(os.environ['CYCLO_HIL_LEARNER_TOKEN_FILE']).read_text().strip()
    if len(token) < 32:
        raise ValueError('Set CYCLO_HIL_LEARNER_TOKEN to a random token of at least 32 characters')
    pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix='gpu-learner')
    state = {'active': None, 'finished': {}}

    def finish():
        active = state['active']
        if not active:
            return None
        try:
            meta, weights = active['engine'].close()
            result = pack({**meta, 'run_id': active['id']}, weights)
            state['finished'][active['id']] = result
            while len(state['finished']) > 4:
                state['finished'].pop(next(iter(state['finished'])))
            return result
        finally:
            state['active'] = None
            del active
            gc.collect()
            if torch.cuda.is_initialized():
                torch.cuda.empty_cache()

    async def execute(fn):
        try:
            return await asyncio.get_running_loop().run_in_executor(pool, fn)
        except HTTPException:
            raise
        except (ValueError, ValidationError, KeyError) as exc:
            raise HTTPException(422, str(exc)) from None
        except Exception as exc:
            # Preserve checkpoint when possible, then release a failed run.
            try:
                await asyncio.get_running_loop().run_in_executor(pool, finish)
            except Exception:
                pass
            raise HTTPException(503, f'{type(exc).__name__}: {exc}') from None

    @asynccontextmanager
    async def lifespan(app):
        torch.set_num_threads(1)
        async def reap():
            while True:
                await asyncio.sleep(2)
                def expired():
                    if state['active'] and time.monotonic() - state['active']['heartbeat'] > 60:
                        finish()
                await execute(expired)
        task = asyncio.create_task(reap())
        yield
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass
        await execute(finish)
        pool.shutdown(wait=True)

    app = FastAPI(title='Cyclo external GPU learner', lifespan=lifespan)

    @app.middleware('http')
    async def authorize(request, call_next):
        supplied = request.headers.get('authorization', '')
        if not secrets.compare_digest(supplied.encode(), ('Bearer ' + token).encode()):
            return Response('Unauthorized', status_code=401)
        return await call_next(request)

    async def body(request):
        value = bytearray()
        async for chunk in request.stream():
            value.extend(chunk)
            if len(value) > MAX_PACKET:
                raise HTTPException(413, 'Learner packet exceeds 96 MiB')
        return bytes(value)

    def active(run_id):
        run = state['active']
        if not run or run['id'] != run_id:
            raise HTTPException(404, 'Learner run does not exist or its 60-second lease expired')
        run['heartbeat'] = time.monotonic()
        return run

    @app.get('/health')
    async def health():
        return {'protocol': PROTOCOL, **hardware_status(), 'busy': state['active'] is not None}

    @app.post('/runs')
    async def start(request: Request):
        payload = await body(request)
        def begin():
            value = json.loads(payload)
            resume_id = value.pop('resume_run_id', None)
            resume_counts = value.pop('resume_counts', None)
            spec = validate_spec(value)
            resume_folder = None
            if resume_id:
                resume_folder = root / str(UUID(resume_id))
                if resume_folder.is_symlink():
                    raise ValueError('Invalid resume directory')
            if state['active']:
                raise HTTPException(409, 'Learner already has an active run')
            name = spec['config']['learner_device']
            if not device_ready(name, hardware_status()):
                raise HTTPException(409, f'Remote learner device {name} is unavailable')
            torch.manual_seed(spec['config']['seed'])
            identifier = str(uuid4())
            engine = LearnerEngine(spec, root / identifier, resume_folder=resume_folder, resume_counts=resume_counts)
            state['active'] = {'id': identifier, 'engine': engine, 'heartbeat': time.monotonic(), 'sequence': -1}
            return {'run_id': identifier}
        return await execute(begin)

    @app.post('/runs/{run_id}/weights')
    async def weights(run_id: str, request: Request):
        payload = await body(request)
        def restore():
            run = active(run_id)
            if run['sequence'] >= 0:
                raise HTTPException(409, 'Initial weights can only be supplied before the first exchange')
            _, arrays = unpack(payload)
            learner = run['engine'].learner
            if run['engine'].upstream:
                learner.restore_policy(arrays)
            else:
                learner.policy.restore(arrays)
            return {'restored': True}
        return await execute(restore)

    @app.post('/runs/{run_id}/exchange')
    async def exchange(run_id: str, request: Request):
        payload = await body(request)
        def update():
            run = active(run_id)
            metadata, arrays = unpack(payload)
            command = Exchange(**metadata)
            if command.sequence != run['sequence'] + 1:
                raise HTTPException(409, 'Out-of-order transition batch; reconnect with a new run')
            rows = arrays_to_rows(arrays, run['engine'].spec)
            result, weights_out = run['engine'].exchange(rows, command.learning, command.version, offline=command.offline)
            run['heartbeat'] = time.monotonic()
            run['sequence'] = command.sequence
            return pack({**result, 'run_id': run_id}, weights_out)
        return Response(await execute(update), media_type='application/x-npz')

    @app.post('/runs/{run_id}/stop')
    async def stop(run_id: str):
        def close():
            if run_id in state['finished']:
                return state['finished'][run_id]
            active(run_id)
            return finish()
        return Response(await execute(close), media_type='application/x-npz')

    return app


if __name__ == '__main__':
    import argparse
    import uvicorn
    parser = argparse.ArgumentParser()
    parser.add_argument('--host', default='127.0.0.1')
    parser.add_argument('--port', default=7130, type=int)
    parser.add_argument('--data-dir', default=os.environ.get('CYCLO_HIL_LEARNER_DATA_DIR', '/workspace/learner'))
    parser.add_argument('--ssl-certfile')
    parser.add_argument('--ssl-keyfile')
    args = parser.parse_args()
    if args.host not in ('127.0.0.1', '::1', 'localhost') and not (args.ssl_certfile and args.ssl_keyfile):
        parser.error('Non-loopback listening requires TLS; otherwise use the default with an SSH tunnel')
    uvicorn.run(create_app(args.data_dir), host=args.host, port=args.port,
                ssl_certfile=args.ssl_certfile, ssl_keyfile=args.ssl_keyfile)
