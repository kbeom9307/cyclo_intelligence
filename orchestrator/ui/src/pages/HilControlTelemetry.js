import React from 'react';

const number = value => Number.isFinite(value) ? value.toFixed(3) : '—';
export function HilActionTelemetry({ session, axes }) {
  const runtime = session.runtime;
  return <section className="hil-policy-values" aria-label="Current policy values">
    <h2>Policy / action values</h2>
    <p className="hil-help">{['preview', 'demo', 'classifier'].includes(session.mode) ? `${session.mode === 'preview' ? 'Preview' : 'Collection'}: policy disabled; policy proposal is zero.` : `Policy v${session.metrics.policy_version ?? '—'}`} · Last step {runtime?.episode_step ?? 0} · {session.state}<br />Normalized −1…+1, robot root frame. Selected command is after intervention; motion limits may block it.</p>
    <table><thead><tr><th>Axis</th><th>Policy proposal</th><th>Selected command</th></tr></thead>
      <tbody>{axes.map((axis, i) => <tr key={axis}><th>{axis}</th><td>{number(runtime?.last_policy_action?.[i])}</td><td>{number(runtime?.last_action?.[i])}</td></tr>)}</tbody>
    </table>
    {runtime?.observation_mode === 'pixels' && <p className="hil-help">Policy input: {(session.config?.observation_cameras || ['head', 'left_wrist', 'right_wrist']).length} RGB cameras ({runtime.image_shape?.[1] || 64}×{runtime.image_shape?.[2] || 64}) + robot proprioception · capture {runtime.capture_metadata?.sequence ?? '—'}{runtime.classifier_probability != null ? ` · Success probability: ${runtime.classifier_probability.toFixed(3)}` : ''}</p>}
    <p className="hil-help">Source: {runtime?.last_action_source ?? '—'}{runtime?.environment_info?.arm_joint_hold ? ' · Arm: joint position hold' : ''} · Reward: {number(runtime?.last_reward)}{runtime?.environment_info?.motion_blocked ? ' · Movement blocked by limit guard' : ''}</p>
    {session.config?.reward_source === 'classifier' && <section aria-label="Live success classifier">
      <h3>Success classifier</h3>
      <p>{runtime?.classifier_enabled ? (session.state === 'running' ? 'Live · updated each simulation step' : 'Last sampled value · simulation paused/stopped') : 'Classifier inactive / waiting for runtime'}</p>
      <p>Logit: {number(runtime?.classifier_logit)} · Success probability: {number(runtime?.classifier_probability)}</p>
      <p>Probability threshold: {number(session.config.classifier_threshold)} · Consecutive success steps: {runtime?.classifier_hold_steps ?? 0} / {session.config.classifier_hold_steps ?? '—'}</p>
      <p>Detected success: {runtime?.environment_info?.classifier_success ? 'Yes' : 'No'} · Cameras: {(session.config.classifier_cameras || []).join(', ')}</p>
    </section>}
  </section>;
}

const groups = [
  ['Move', [['KeyW','W','+X'],['KeyS','S','−X'],['KeyA','A','+Y'],['KeyD','D','−Y'],['KeyR','R','+Z'],['KeyF','F','−Z']]],
  ['Rotate', [['KeyQ','Q','+Roll'],['KeyE','E','−Roll'],['ArrowUp','↑','+Pitch'],['ArrowDown','↓','−Pitch'],['ArrowLeft','←','+Yaw'],['ArrowRight','→','−Yaw']]],
];
export function HilPressedKeys({ pressed, enabled, isaac, collecting, spaceToggle }) {
  return <section className={`hil-keyboard ${enabled ? '' : 'disabled'}`} aria-label="Keyboard input visualization">
    <p role="status">{!enabled ? 'Keyboard inactive — resume the session to control.' : pressed.length ? `Pressed: ${pressed.map(key => key.replace('Key','').replace('Arrow','')).join(' + ')}` : 'No keys pressed'}</p>
    {groups.filter((_, i) => isaac || i === 0).map(([name, keys]) => <div className="hil-key-group" key={name}><span>{name}</span><div>{keys.map(([code, label, axis]) => <div className={`hil-key ${pressed.includes(code) ? 'pressed' : ''}`} key={code} data-pressed={pressed.includes(code)} aria-label={`${label} ${pressed.includes(code) ? 'pressed' : 'released'}`}><kbd>{label}</kbd><small>{axis}</small></div>)}</div></div>)}
    <small>{collecting ? (spaceToggle ? 'Space: toggle success/failure · Esc: clear input' : 'Hold Space: success · Otherwise: failure · Esc: release input') : 'Space / Esc: release all input'}</small>
  </section>;
}


const gib = mib => Number.isFinite(mib) ? `${(mib / 1024).toFixed(2)} GiB` : '—';
export function HilGpuInventory({ hardware }) {
  if (!hardware) return null;
  return <section aria-label="Local GPU memory"><h2>Local GPU memory</h2>
    {hardware.error && <p>{hardware.error}</p>}
    {hardware.gpus?.map(gpu => <div key={gpu.device}>
      <strong>{gpu.name} · {gpu.device}</strong>
      <p>Total {gib(gpu.total_mib)} · Used {gib(gpu.used_mib)} · Free {gib(gpu.free_mib)}</p>
      {gpu.free_mib < 1024 && <p className="hil-help">Less than 1 GiB free. Local GPU training may run out of memory. An external learner still needs room for actor inference on this GPU.</p>}
    </div>)}
  </section>;
}

export function HilComputeTelemetry({ runtime }) {
  const compute = runtime?.compute;
  if (!compute) return null;
  const learner = compute.learner || {};
  const rows = [['Actor process', compute.actor_device, compute.actor_memory],
    ['Learner', learner.learner_memory?.device || 'inactive', learner.learner_memory]];
  return <section className="hil-policy-values" aria-label="GPU execution telemetry">
    <h2>Compute / GPU memory</h2>
    <p>{learner.placement === 'remote' ? 'External learner connected' : learner.learner_memory ? 'Local learner' : 'Learner inactive'} · Queued transitions: {learner.pending_transitions ?? 0}</p>
    <table><thead><tr><th>Process / role</th><th>Device</th><th>Tensor allocated</th><th>Reserved</th><th>GPU free</th></tr></thead>
      <tbody>{rows.map(([label, device, memory]) => <tr key={label}><th>{label}</th><td>{device}</td><td>{gib(memory?.allocated_mib)}</td><td>{gib(memory?.reserved_mib)}</td><td>{gib(memory?.free_mib)}</td></tr>)}</tbody>
    </table>
    <p className="hil-help">{compute.shared_process ? 'Actor and learner share one process: these memory readings overlap; do not add them. ' : ''}Tensor allocation excludes CUDA context and other processes. Reserved memory includes the framework cache. GPU free includes all processes on that device.</p>
    {learner.run_id && <p>Remote run: <code>{learner.run_id}</code><br />Server artifacts: <code>{learner.artifact_directory}</code></p>}
  </section>;
}
