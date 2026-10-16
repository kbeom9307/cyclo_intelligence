import React, { useEffect, useRef, useState } from 'react';
import { getHilWorkspace, saveHilConfig, createHilSession, sendHilCommand, sendHilAction } from '../utils/hilserlApi';
import './HilSerlPage.css';

const robots = { ffw_bg2_rev4: 'AI Worker · BG2 Rev4', ffw_sg2_rev1: 'AI Worker · SG2 Rev1', ffw_sh5_rev1: 'AI Worker · SH5 Rev1' };
const display = (value) => value == null ? '—' : value;

export default function HilSerlPage() {
  const [workspace, setWorkspace] = useState(null);
  const [draft, setDraft] = useState(null);
  const [editRevision, setEditRevision] = useState(0);
  const [dirty, setDirty] = useState(false);
  const [selectedId, setSelectedId] = useState(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState('');
  const [notice, setNotice] = useState('');
  const [tab, setTab] = useState('setup');
  const initialized = useRef(false);

  useEffect(() => {
    let alive = true;
    let timer;
    const load = async () => {
      try {
        const data = await getHilWorkspace();
        if (!alive) return;
        setWorkspace(data);
        if (!initialized.current) {
          setDraft(data.config); setEditRevision(data.revision); initialized.current = true;
        }
      } catch (e) {
        if (alive) setError(e.message);
      } finally {
        if (alive) timer = setTimeout(() => load(), 3000);
      }
    };
    load();
    return () => { alive = false; clearTimeout(timer); };
  }, []);

  const refresh = async (reloadDraft = false) => {
    const data = await getHilWorkspace();
    setWorkspace(data);
    if (reloadDraft || !draft) {
      setDraft(data.config); setEditRevision(data.revision); setDirty(false); initialized.current = true;
    }
  };
  const run = async (operation) => {
    setBusy(true); setError(''); setNotice('');
    try { await operation(); } catch (e) { setError(e.message); }
    finally { setBusy(false); }
  };
  const update = (key, value) => { setDraft((current) => ({ ...current, [key]: value })); setDirty(true); };
  const session = workspace?.sessions.find((item) => item.id === selectedId) ?? workspace?.sessions[0];
  const configStale = workspace && editRevision !== workspace.revision;
  const active = workspace?.sessions.some((item) => item.state !== 'stopped');
  const command = (action) => run(async () => {
    await sendHilCommand(session, action);
    await refresh();
    setNotice(`${session.mode}: ${action.replace('_', ' ')} requested.`);
  });

  const save = (event) => {
    event.preventDefault();
    run(async () => {
      const result = await saveHilConfig(draft, editRevision);
      setDraft(result.config); setEditRevision(result.revision); setDirty(false);
      await refresh(); setNotice('Experiment configuration saved. Existing sessions keep their original configuration.');
    });
  };
  const create = (mode = 'preview') => run(async () => {
    const created = await createHilSession(workspace.revision, mode);
    setSelectedId(created.id); await refresh(); setTab('session');
    setNotice(`${mode} session created. Start it from the session controls.`);
  });

  const field = (key, label, options = {}) => (
    <label className="hil-field" key={key}>
      <span>{label}</span>
      <input value={draft[key]} onChange={(e) => update(key, options.type === 'number' ? (e.target.value === '' ? '' : Number(e.target.value)) : e.target.value)} {...options} />
    </label>
  );
  const select = (key, label, options) => (
    <label className="hil-field" key={key}><span>{label}</span>
      <select value={draft[key]} onChange={(e) => update(key, e.target.value)}>
        {Object.entries(options).map(([value, title]) => <option key={value} value={value}>{title}</option>)}
      </select>
    </label>
  );

  return <div className="hil-page">
    <header className="hil-header">
      <div><div className="hil-eyebrow">CYCLO INTELLIGENCE / ONLINE LEARNING</div><h1>HIL-SERL Lab</h1><p>AI Worker × NIST Assembly Task Board</p></div>
      <span className="hil-badge">{workspace?.readiness.can_train ? 'Mock runtime ready · Agentlace / SAC' : 'Preview available · Runtime not connected'}</span>
    </header>
    <div className="hil-banner">Mock runs a CPU SAC learner with real Agentlace transport and manual intervention. It uses a simplified XYZ peg task, not Isaac Sim physics or a physical robot. Preview only rehearses controls.</div>
    {error && <div role="alert" className="hil-error">{error} <button type="button" disabled={busy} onClick={() => run(() => refresh(!draft))}>Retry connection</button></div>}
    {notice && <div role="status" className="hil-notice">{notice}</div>}
    {!workspace || !draft ? <p>Loading HIL-SERL workspace…</p> : <>
      <nav className="hil-tabs" aria-label="HIL-SERL workspace">
        <button aria-pressed={tab === 'setup'} onClick={() => setTab('setup')}>Experiment setup {dirty ? '•' : ''}</button>
        <button aria-pressed={tab === 'session'} onClick={() => setTab('session')}>Session & history</button>
      </nav>
      {tab === 'setup' ? <div className="hil-layout">
        <form onSubmit={save} className="hil-card">
          <div className="hil-card-heading"><h2>Environment & task</h2><span>Saved revision {workspace.revision}</span></div>
          <fieldset disabled={busy} className="hil-fields">
            {field('name', 'Experiment name', { required: true, maxLength: 120 })}
            {select('robot_type', 'Robot model', robots)}
            {select('environment', 'Target environment', { mock: 'Mock · no robot required', isaac_sim: 'Isaac Sim', real_robot: 'Physical AI Worker' })}
            {select('arm', 'Working arm', { right: 'Right arm', left: 'Left arm' })}
            {select('board', 'Board', { nist_atb1: 'NIST Assembly Task Board #1' })}
            {select('task', 'Task', { peg_insertion: 'Peg insertion' })}
          </fieldset>
          <p className="hil-help">Board geometry, robot asset and task physics will be supplied by the environment adapter.</p>
          <h2>Control & success criteria</h2>
          <fieldset disabled={busy} className="hil-fields">
            {field('control_hz', 'RL step frequency (Hz)', { type: 'number', required: true, min: 1, max: 50 })}
            {field('max_episode_steps', 'Episode step limit', { type: 'number', required: true, min: 1, max: 10000 })}
            {field('translation_limit_m', 'Translation limit per step (m)', { type: 'number', required: true, min: 0.0001, max: 0.05, step: 'any' })}
            {field('rotation_limit_rad', 'Rotation limit per step (rad)', { type: 'number', required: true, min: 0.0001, max: 0.3, step: 'any' })}
            {field('success_depth_m', 'Insertion depth (m)', { type: 'number', required: true, min: 0.0001, max: 0.2, step: 'any' })}
            {field('success_tolerance_m', 'Position tolerance (m)', { type: 'number', required: true, min: 0.0001, max: 0.02, step: 'any' })}
            {field('success_hold_s', 'Success hold time (s)', { type: 'number', required: true, min: 0.001, max: 10, step: 'any' })}
            {select('reward_source', 'Reward source', { environment: 'Environment ground truth', human: 'Human label', classifier: 'Success classifier' })}
          </fieldset>
          <h2>Learner settings</h2>
          <fieldset disabled={busy} className="hil-fields">
            {field('seed', 'Random seed', { type: 'number', required: true, min: 0, max: 2147483647 })}
            {field('batch_size', 'Batch size', { type: 'number', required: true, min: 1, max: 4096 })}
            {field('replay_capacity', 'Replay capacity (transitions)', { type: 'number', required: true, min: 4096, max: 10000000 })}
            {field('learning_rate', 'Learning rate', { type: 'number', required: true, min: 0.000001, max: 0.1, step: 'any' })}
            {field('discount', 'Discount factor', { type: 'number', required: true, min: 0.000001, max: 1, step: 'any' })}
          </fieldset>
          <p className="hil-help">Mock uses these settings for state-based SAC with twin critics and a fixed entropy coefficient of 0.2. Human transitions are sampled alongside online experience.</p>
          <h2>Observations & artifacts</h2>
          <fieldset disabled={busy} className="hil-fields">
            {field('camera_topic', 'Camera topic', { required: true, maxLength: 256 })}
            {field('joint_state_topic', 'Joint state topic', { required: true, maxLength: 256 })}
            {field('demonstration_path', 'Demonstration dataset path (optional)', { maxLength: 512 })}
            {field('checkpoint_path', 'Initial checkpoint path (optional)', { maxLength: 512 })}
          </fieldset>
          <p className="hil-help">Mock uses XYZ position and hold progress. Rotation, cameras and joint topics are reserved for future adapters. Load policy.npz or replay.npz from a stopped session; paths must be under the HIL data directory. Evaluation requires a policy checkpoint.</p>
          <div className="hil-actions"><button className="hil-primary" type="submit" disabled={busy || !dirty}>Save configuration</button>
            <button type="button" disabled={busy} onClick={() => run(() => refresh(true))}>Reload saved configuration</button></div>
        </form>
        <aside className="hil-card hil-readiness">
          <h2>Runtime readiness</h2><p className="hil-help">Status for the saved configuration</p>
          {workspace.readiness.checks.map((check) => <div key={check.id} className="hil-check"><span className={`hil-dot ${check.ready ? 'ready' : ''}`}  /><div><strong>{check.label}</strong><p>{check.detail}</p></div></div>)}
          <div className="hil-actions hil-stack"><button className="hil-primary" disabled={busy || dirty || configStale || active || !workspace.readiness.can_preview} onClick={() => create('preview')}>Create preview session</button>
            <button disabled={busy || dirty || configStale || active || !workspace.readiness.can_train} onClick={() => create('train')}>Create Mock training session</button>
            <button disabled={busy || dirty || configStale || active || !workspace.readiness.can_evaluate} onClick={() => create('evaluate')}>Create Mock evaluation session</button></div>
          <p className="hil-help">{configStale ? 'Configuration changed elsewhere. Reload saved configuration before creating a session.' : dirty ? 'Save or reload your changes before creating a session.' : active ? 'An existing session must be stopped before creating another.' : 'Preview lets you test pause, takeover, episode labels and reset.'}</p>
        </aside>
      </div> : <div className="hil-layout">
        <section className="hil-card">
          {!session ? <div className="hil-empty"><h2>No sessions yet</h2><p>Save your experiment settings, then create a Preview session.</p><button onClick={() => setTab('setup')}>Go to experiment setup</button></div> : <>
            <div className="hil-card-heading"><div><h2>{session.config.name}</h2><p className="hil-help">{session.mode} · {session.config.environment} · {robots[session.config.robot_type]} · {session.config.arm} arm · config v{session.config_revision}</p></div><span className="hil-badge">{session.state}</span></div>
            <div className="hil-metrics">
              {[['Episode', session.episode], ['Environment steps', session.metrics.environment_steps], ['Learner steps', session.metrics.learner_steps], ['Success rate', session.metrics.success_rate], ['Policy version', session.metrics.policy_version], ['Replay size', session.metrics.replay_size], ['Human steps', session.metrics.human_steps], ['Critic loss', session.metrics.critic_loss == null ? null : session.metrics.critic_loss.toFixed(4)]].map(([label, value]) => <div key={label}><span>{label}</span><strong>{display(value)}</strong></div>)}
            </div>
            {session.mode === 'preview' ? <p className="hil-help">Learning metrics remain unavailable in Preview. Episode labels below are workflow checks, not measured results.</p> : <>
              <p className="hil-help">{session.runtime_pending ? 'Waiting for runtime acknowledgement…' : session.runtime?.transport || 'Runtime starts when you press Start.'} · Learner warms up after {session.config.batch_size} transitions. Natural episode endings reset automatically.</p>
              {session.runtime_error && <pre role="alert" className="hil-error">{session.runtime_error}</pre>}
              {session.runtime && <div className="hil-mock-view">
                <h2>Mock peg position</h2><p>XYZ (m): {session.runtime.position_m?.map((v) => v.toFixed(4)).join(' / ')}</p>
                <svg viewBox="0 0 220 140" role="img" aria-label="Mock peg XY position relative to target hole">
                  <rect x="10" y="10" width="200" height="120" rx="8" fill="#cbd5e1" />
                  <circle cx="110" cy="70" r="8" fill="#334155" />
                  <circle cx={110 + (session.runtime.position_m?.[0] || 0) * 1800} cy={70 - (session.runtime.position_m?.[1] || 0) * 1000} r="5" fill="#2563eb" />
                </svg>
                <p className="hil-help">Top view: blue peg, dark target. Display refreshes every 3 seconds.</p>
                <p className="hil-help">Artifacts are finalized on Stop.<br />Policy: <code>{session.runtime.checkpoint_path}</code><br />Replay: <code>{session.runtime.dataset_path}</code></p>
              </div>}
            </>}
            <div className="hil-actions">
              <button className="hil-primary" disabled={busy || session.runtime_pending || session.state !== 'created'} onClick={() => command('start')}>{session.mode === 'preview' ? 'Start preview' : 'Start Mock runtime'}</button>
              <button disabled={busy || session.runtime_pending || session.state !== 'running'} onClick={() => command('pause')}>Pause</button>
              <button disabled={busy || session.runtime_pending || session.state !== 'paused' || session.episode_outcome !== null} onClick={() => command('resume')}>Resume</button>
              <button className="hil-danger" disabled={busy || session.state === 'stopped'} onClick={() => command('stop')}>{session.mode === 'preview' ? 'Stop preview' : 'Stop & save artifacts'}</button>
            </div>
            <div className="hil-control"><div><h2>Human intervention</h2><p>Control source: <strong>{session.control_source}</strong></p></div>
              <button disabled={busy || session.runtime_pending || session.state !== 'running'} onClick={() => command(session.control_source === 'human' ? 'release' : 'take_over')}>{session.control_source === 'human' ? 'Release to policy' : session.mode === 'preview' ? 'Take over (preview)' : 'Take over'}</button>
            </div>
            {session.mode !== 'preview' && session.control_source === 'human' && <div className="hil-actions" aria-label="Manual XYZ controls">
              {[[1, 0, 0, '+X'], [-1, 0, 0, '−X'], [0, 1, 0, '+Y'], [0, -1, 0, '−Y'], [0, 0, 1, '+Z'], [0, 0, -1, '−Z'], [0, 0, 0, 'Hold']].map(([x, y, z, label]) => <button key={label} disabled={busy || session.runtime_pending || session.state !== 'running'} onClick={() => run(async () => { await sendHilAction(session, [x, y, z]); await refresh(); })}>{label} pulse</button>)}
            </div>}
            <div className="hil-control"><div><h2>Episode outcome</h2><p>{session.episode_outcome || 'Not labeled'}</p></div><div className="hil-actions">
              <button disabled={busy || session.runtime_pending || session.state !== 'running'} onClick={() => command('success')}>Label success</button>
              <button disabled={busy || session.runtime_pending || session.state !== 'running'} onClick={() => command('failure')}>Label failure</button>
              <button disabled={busy || session.runtime_pending || session.state !== 'paused'} onClick={() => command('reset')}>{session.mode === 'preview' ? 'Reset preview episode' : 'Reset episode'}</button>
            </div></div>
            <h2>Session events</h2>
            {session.events.length === 0 ? <p className="hil-help">No events recorded.</p> : <ol className="hil-events">{[...session.events].reverse().map((event) => <li key={event.sequence}><time>{new Date(event.at).toLocaleTimeString()}</time><strong>{event.action.replace('_', ' ')}</strong><span>Episode {event.episode} · {session.mode}</span></li>)}</ol>}
          </>}
        </section>
        <aside className="hil-card"><h2>Session history</h2><p className="hil-help">Stored on this Cyclo instance</p>
          {workspace.sessions.map((item) => <button key={item.id} className={`hil-history ${session?.id === item.id ? 'selected' : ''}`} onClick={() => setSelectedId(item.id)}><strong>{item.config.name}</strong><span>{new Date(item.created_at).toLocaleString()}</span><span>{item.mode} · {item.state}</span></button>)}
          {workspace.sessions.length === 0 && <p className="hil-help">Your sessions will appear here.</p>}
        </aside>
      </div>}
    </>}
  </div>;
}
