# HIL-SERL source provenance

Source: https://github.com/rail-berkeley/hil-serl
Commit: c32939bccb65f3b8c43a9f9add3d322d4ab0264a
The serl_launcher Python package and its LICENSE are vendored here.

Local compatibility changes: deprecated jax.tree_map/tree_leaves aliases use
jax.tree_util; TensorFlow, wandb and checkpoint imports in utility functions are
lazy (including optional imageio/matplotlib), so image policy inference does not initialize an unrelated GPU framework.
No SAC loss, network architecture or augmentation logic is replaced.
Cyclo integration is in ../../hilserl_upstream.py.
TensorFlow is omitted from runtime-only Array type aliases.


## Cyclo integration contract

- JAX 0.6.2 / Flax 0.10.7 run in /opt/hil-serl-venv; the ROS base Python
  and Isaac Python environments remain separate. The venv preloads its cuDNN
  before compatibility imports of PyTorch to avoid the base image's cuDNN 9.7.
- Fixed-gripper SACAgent.create_pixels uses the upstream 128x128 frozen ImageNet
  ResNet-10, per-camera spatial embeddings, 256/256 heads, LayerNorm, random
  crop padding 4, learned temperature initialized at .01, discount .97, and
  backup_entropy=False. Target parameters are synchronized after pretrained
  weights load so the initial target backbone matches the online backbone.
- The actor contains only the original actor parameters. In external-learner
  mode its GPU convolution autotuning is disabled to avoid a large warmup cache.
  This changes kernel selection, not model size, precision, or policy outputs'
  semantics. Performance measurements are in docs/hilserl/vram-2026-10-20.json.
- Learner batch 256 is 128 online + 128 demo/intervention transitions; CTA=2.
  Offline demos populate only the demo buffer. Interventions populate both.
  Learning starts after 100 online transitions and a nonempty demo buffer,
  independently of environment ticks. Policies publish every 50 actor updates.
- Replay uses two disk-backed CPU ring buffers of 200,000 transitions. Unlike
  upstream's memory-efficient RAM replay, current and next frames are both stored;
  fully populated three-camera buffers occupy about 110 GiB on disk. No replay
  array is kept resident in VRAM. Buffers grow via sparse memory-mapped files.
- Isaac keeps its existing 20 Hz, 6-action fixed-gripper contract and 22 robot
  proprioception values. Configurable normalized camera crops are applied to
  native renderer frames before resize to 128x128; full-frame crops are default.
  This is an Isaac/AI Worker adapter, not a reproduction of Franka hardware.
- Restored rewards are binary 0/1. The existing classifier threshold/hold and
  episode-separated classifier validation remain configurable Lab behaviors.
  Classifier architecture, batch 256, augmentation and Adam 1e-4 use upstream;
  offline training stays on CPU, and runtime reward inference uses actor CUDA.
- NPZ data/transport and Lab UI are retained. Old 64x64 datasets/checkpoints use
  legacy_torch and cannot be silently upscaled or loaded into the restored model.
  Saved model metadata includes task, camera selection and crop coordinates.
- Full learner state is learner.msgpack; replay is a directory of .npy arrays
  and index.json files, with spec.json beside the policy checkpoint. The external
  service retains full learner/replay state and returns policy.npz to the actor.
