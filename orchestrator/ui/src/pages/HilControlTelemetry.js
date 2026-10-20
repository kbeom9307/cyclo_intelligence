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
    {runtime?.observation_mode === 'pixels' && <p className="hil-help">Policy input: 3 RGB cameras (64×64) + robot proprioception · capture {runtime.capture_metadata?.sequence ?? '—'}{runtime.classifier_probability != null ? ` · Success probability: ${runtime.classifier_probability.toFixed(3)}` : ''}</p>}
    <p className="hil-help">Source: {runtime?.last_action_source ?? '—'}{runtime?.environment_info?.arm_joint_hold ? ' · Arm: joint position hold' : ''} · Reward: {number(runtime?.last_reward)}{runtime?.environment_info?.motion_blocked ? ' · Movement blocked by limit guard' : ''}</p>
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
