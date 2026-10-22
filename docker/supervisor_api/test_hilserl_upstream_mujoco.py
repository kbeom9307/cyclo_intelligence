"""Opt-in real MuJoCo RGB / restored CUDA policy integration, using temporary data."""
import json
import os
from pathlib import Path
import time
import numpy as np
import pytest

pytestmark = pytest.mark.skipif(os.environ.get("CYCLO_TEST_UPSTREAM_MUJOCO") != "1",
                               reason="requires an idle running MuJoCo bridge and CUDA")


def test_mujoco_images_collection_and_upstream_gpu_inference(tmp_path):
    from hilserl_profiles import prepare_jax_libraries
    prepare_jax_libraries()
    from hilserl import ExperimentConfig
    from hilserl_mujoco import MujocoNistEnv, health
    from hilserl_upstream import UpstreamPolicy, UpstreamClassifier
    from hilserl_data import Collection, task_key
    from hilserl_devices import hardware_status
    assert health()["ready"] and not health()["busy"]
    config = ExperimentConfig(environment="mujoco", learner_placement="remote",
                               translation_limit_m=.001, rotation_limit_rad=.01).model_dump()
    actor = UpstreamPolicy(config)
    classifier = UpstreamClassifier(config["classifier_cameras"], "cuda:0")
    env = MujocoNistEnv(config)
    times = []
    try:
        assert env.images.shape == (9, 128, 128)
        assert env.observation().shape == (22,)
        assert all(env.images[i:i+3].std() > 1 for i in (0, 3, 6))
        assert env.info["grasp_verified"]
        for _ in range(4):
            start = time.monotonic()
            action = actor.action(env.observation(), env.images)
            logit, probability = classifier.predict(env.classifier_images)
            times.append((time.monotonic()-start)*1000)
            assert action.shape == (6,) and np.isfinite(action).all()
            assert np.isfinite(logit) and 0 <= probability <= 1
        collection = Collection(tmp_path, config, "classifier", env.schema)
        sequence = env.capture_metadata["sequence"]
        collection.capture(env, 0, 1, fresh=False)
        obs, images = env.observation(), env.images.copy()
        next_obs, reward, terminal, truncated = env.step(np.zeros(6, np.float32))
        assert env.capture_metadata["sequence"] > sequence
        collection.capture(env, 1, 1, fresh=False)
        assert collection.state()["positive_samples"] == collection.state()["negative_samples"] == 1
        demo = Collection(tmp_path, config, "demo", env.schema)
        demo.rows.append(dict(observation=obs, next_observation=next_obs, action=np.zeros(6),
                              images=images, next_images=env.images.copy(), reward=reward,
                              terminal=terminal, truncated=truncated, human=True))
        path = demo.save_demo(1)
        with np.load(path, allow_pickle=False) as data:
            assert json.loads(str(data["__task__"])) == task_key(config)
            assert data["images"].shape == (1, 9, 128, 128)
        result = dict(environment="mujoco", image_shape=list(env.images.shape), schema=env.schema,
                      warm_actor_classifier_ms=float(np.median(times[1:])),
                      actor_memory=actor.memory(), hardware=hardware_status(), grasp_drift_m=env.info["grasp_drift_m"])
        Path("/tmp/hil-mujoco-gpu-integration.json").write_text(json.dumps(result, indent=2))
    finally:
        env.close()
    assert not health()["busy"]
