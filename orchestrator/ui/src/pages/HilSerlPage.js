import React, { useEffect, useRef, useState } from 'react';
import { getHilWorkspace, saveHilConfig, createHilSession, sendHilCommand, sendHilAction, checkHilLearner } from '../utils/hilserlApi';
import './HilSerlPage.css';
import HilIsaacView from './HilIsaacView';
import { HilCollectionControls, HilDataLibrary, HilDemoData } from './HilDataPanel';
import useHilKeyboard from './useHilKeyboard';
import useHilCaptureKeyboard from './useHilCaptureKeyboard';
import { HilActionTelemetry, HilPressedKeys, HilComputeTelemetry, HilGpuInventory } from './HilControlTelemetry';

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
  const [spaceToggle, setSpaceToggle] = useState(false);
  const [tab, setTab] = useState('setup');
  const initialized = useRef(false);

  useEffect(() => {
    let alive = true;
    let timer;
    const load = async () => {
      let delay = 3000;
      try {
        const data = await getHilWorkspace();
        if (!alive) return;
        setWorkspace(data);
        if (data.classifier_jobs?.some(job => ['queued', 'training'].includes(job.state))) delay = 1000;
        if (data.sessions.some(item => item.state === 'running' || item.runtime_pending)) delay = 500;
        if (!initialized.current) {
          setDraft(data.config); setEditRevision(data.revision); initialized.current = true;
        }
      } catch (e) {
        if (alive) setError(e.message);
      } finally {
        if (alive) timer = setTimeout(() => load(), delay);
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
  const environmentLabel = draft?.environment === 'isaac_sim' ? 'Isaac Sim' : 'Mock';
  const sessionLabel = session?.config.environment === 'isaac_sim' ? 'Isaac Sim' : 'Mock';
  const isaacSession = session?.config.environment === 'isaac_sim';
  const physicalPreview = isaacSession && ['preview', 'demo', 'classifier'].includes(session?.mode);
  const keyboardEnabled = tab === 'session' && session?.state === 'running' && !session?.runtime_pending && (session?.mode !== 'preview' || isaacSession);
  const pressedKeys = useHilKeyboard(session, keyboardEnabled, setError);
  const episodeStep = session?.runtime?.episode_step ?? session?.runtime?.environment_info?.episode_step ?? 0;
  const manualDimensions = isaacSession ? 6 : 3;
  const manualAxes = isaacSession ? ['X', 'Y', 'Z', 'Roll', 'Pitch', 'Yaw'] : ['X', 'Y', 'Z'];
  const configStale = workspace && editRevision !== workspace.revision;
  const active = workspace?.sessions.some((item) => item.state !== 'stopped');
  const restartCollection = session?.mode === 'classifier' && session?.state === 'stopped';
  const startDisabled = busy || session?.runtime_pending || (restartCollection
    ? dirty || configStale || active || !workspace?.readiness.can_classifier
    : session?.state !== 'created');
  const startRuntime = () => run(async () => {
    let target = session;
    if (restartCollection) {
      target = await createHilSession(workspace.revision, 'classifier');
      setSelectedId(target.id);
      // Show the new session even if starting fails, so Start can be retried.
      await refresh();
    }
    await sendHilCommand(target, 'start');
    await refresh();
    setNotice(restartCollection ? 'New classifier collection started. Previous datasets are preserved.' : `${target.mode}: start requested.`);
  });
  const command = (action) => run(async () => {
    await sendHilCommand(session, action);
    await refresh();
    setNotice(`${session.mode}: ${action.replace('_', ' ')} requested.`);
  });

  const labelSuccessActive = useHilCaptureKeyboard(session, tab === 'session' && !session?.runtime_pending && session?.state === 'running', setError, spaceToggle);

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

  const prepareDemoTraining = async () => {
    let latest = await getHilWorkspace();
    if (latest.revision !== editRevision) throw new Error('Configuration changed. Reload saved configuration before applying demos.');
    if (latest.sessions.some(item => item.state !== 'stopped' && item.mode !== 'demo')) {
      throw new Error('Stop the existing non-demo session before creating a training session.');
    }
    const demos = latest.sessions.filter(item => item.mode === 'demo' && (item.state !== 'stopped' || item.runtime_pending));
    for (const item of demos) {
      if (item.state !== 'stopped') await sendHilCommand(item, 'stop');
    }
    if (demos.length) setNotice('Stopping demo collection and saving artifacts…');
    const deadline = Date.now() + 90000;
    while (demos.length) {
      latest = await getHilWorkspace(); setWorkspace(latest);
      const currentSessions = latest.sessions;
      const pending = demos.some(item => {
        const current = currentSessions.find(value => value.id === item.id);
        return !current || current.state !== 'stopped' || current.runtime_pending ||
          (item.state !== 'created' && !current.runtime?.artifacts_saved && !current.runtime_error);
      });
      if (!pending && !latest.readiness.bridge?.busy) break;
      if (Date.now() >= deadline) throw new Error('Demo export is still finishing. Wait, then apply the selected demos again.');
      await new Promise(resolve => setTimeout(resolve, 500));
    }
    const failed = demos.find(item => latest.sessions.find(value => value.id === item.id)?.runtime_error);
    if (failed) throw new Error('Demo runtime reported an error while stopping. Check its saved data before creating training.');
  };

  const applyArtifact = async (key, path, cameras, artifact) => {
    const config = {...draft, [key]: path, observation_mode: 'pixels',
      ...(key === 'classifier_path' ? {reward_source: 'classifier', classifier_cameras: cameras || ['head', 'left_wrist', 'right_wrist']}
        : {...(artifact?.task || {}), observation_cameras: cameras || ['head', 'left_wrist', 'right_wrist'],
          pixel_replay_capacity: Math.max(draft.pixel_replay_capacity || 2048, artifact?.steps || 0)})};
    setDraft(config); setDirty(true); setTab('setup');
    if (key !== 'demonstration_path') {
      setNotice(`Artifact selected: ${path}. Save configuration to use it in a new session.`);
      return;
    }
    try {
      const saved = await saveHilConfig(config, editRevision);
      setDraft(saved.config); setEditRevision(saved.revision); setDirty(false);
      const created = await createHilSession(saved.revision, 'train');
      setSelectedId(created.id); setTab('session');
      setNotice(`Training session created with ${artifact?.episodes ?? 'selected'} demos from ${artifact?.sourceCount ?? 'selected'} sources (${artifact?.steps ?? '—'} steps). Press Start to run training.`);
    } finally { await refresh(); }
  };

  const field = (key, label, options = {}) => (
    <label className="hil-field" key={key}>
      <span>{label}</span>
      <input value={draft[key] ?? (key === 'classifier_success_target' ? 200 : '')} onChange={(e) => update(key, options.type === 'number' ? (e.target.value === '' ? '' : Number(e.target.value)) : e.target.value)} {...options} />
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
      <span className="hil-badge">{workspace?.readiness.bridge?.ready ? 'Isaac Sim connected · Agentlace / SAC' : workspace?.readiness.can_train ? 'Mock runtime ready · Agentlace / SAC' : 'Preview available · Runtime not connected'}</span>
    </header>
    <div className="hil-banner">Actor inference and SAC learning use the selected GPU devices. The learner can run locally or on an external GPU server; local replay uses Agentlace transport. Isaac Sim connects to the FFW-SG2 NIST physics scene through its local bridge, using synchronized RGB images, robot proprioception and 6-axis control with a fixed gripper. Isaac Preview initializes the simulator for manual control without learning. Other previews rehearse controls; physical robot execution is unavailable.</div>
    {error && <div role="alert" className="hil-error">{error} <button type="button" disabled={busy} onClick={() => run(() => refresh(!draft))}>Retry connection</button></div>}
    {notice && <div role="status" className="hil-notice">{notice}</div>}
    {!workspace || !draft ? <p>Loading HIL-SERL workspace…</p> : <>
      <nav className="hil-tabs" aria-label="HIL-SERL workspace">
        <button aria-pressed={tab === 'setup'} onClick={() => setTab('setup')}>Experiment setup {dirty ? '•' : ''}</button>
        <button aria-pressed={tab === 'session'} onClick={() => setTab('session')}>Session & history</button>
        <button aria-pressed={tab === 'classifier'} onClick={() => setTab('classifier')}>Train classifier</button>
      </nav>
      {((tab === 'setup' && draft.environment === 'isaac_sim') || (tab === 'session' && isaacSession)) && <HilIsaacView />}
      {tab === 'setup' ? <div className="hil-layout">
        <form onSubmit={save} className="hil-card">
          <div className="hil-card-heading"><h2>Environment & task</h2><span>Saved revision {workspace.revision}</span></div>
          <fieldset disabled={busy} className="hil-fields">
            {field('name', 'Experiment name', { required: true, maxLength: 120 })}
            {select('robot_type', 'Robot model', robots)}
            {select('environment', 'Target environment', { mock: 'Mock · no robot required', isaac_sim: 'Isaac Sim', real_robot: 'Physical AI Worker' })}
            {select('arm', 'Working arm', { right: 'Right arm', left: 'Left arm' })}
            {select('initialization', 'Isaac start state', { near_hole: 'Already grasped near hole (direct reset)', pick_place: 'Pick from tray → above hole (vertical)', grasped: 'Selected peg grasped (verified)', kit: 'Peg in kit tray' })}
            {select('target_part', 'Target peg (Isaac Sim)', Object.fromEntries(['round', 'square'].flatMap((shape) => [4, 8, 12, 16].map((size) => [`peg_${shape}_${size}`, `${shape} ${size} mm`]))))}
            {select('board', 'Board', { nist_atb1: 'NIST Assembly Task Board #1' })}
            {select('task', 'Task', { peg_insertion: 'Peg insertion' })}
          </fieldset>
          <p className="hil-help">Isaac Sim requires the running NIST bridge, FFW-SG2 and 20 Hz. The default start places the robot and held peg directly near the hole, verifies free-physics retention, then begins at episode step 0. No tray pickup or transfer motion is performed. Preparation is excluded from episode steps. Tray pick/place and legacy starts remain optional.</p>
          <h2>Control & success criteria</h2>
          <fieldset disabled={busy} className="hil-fields">
            {field('control_hz', 'RL step frequency (Hz)', { type: 'number', required: true, min: 1, max: 50 })}
            {field('classifier_success_target', 'Classifier success sample target', { type: 'number', required: true, min: 1, max: 10000 })}
            {field('collection_episode_steps', 'Collection episode step limit', { type: 'number', required: true, min: 1, max: 1000 })}
            {field('max_episode_steps', 'Episode step limit', { type: 'number', required: true, min: 1, max: 10000 })}
            {field('translation_limit_m', 'Translation limit per step (m)', { type: 'number', required: true, min: 0.0001, max: 0.05, step: 'any' })}
            {field('rotation_limit_rad', 'Rotation limit per step (rad)', { type: 'number', required: true, min: 0.0001, max: 0.3, step: 'any' })}
            {field('success_depth_m', 'Insertion depth (m)', { type: 'number', required: true, min: 0.0001, max: 0.2, step: 'any' })}
            {field('success_tolerance_m', 'Position tolerance (m)', { type: 'number', required: true, min: 0.0001, max: 0.02, step: 'any' })}
            {field('success_hold_s', 'Success hold time (s)', { type: 'number', required: true, min: 0.001, max: 10, step: 'any' })}
            {select('reward_source', 'Reward source', { environment: 'Environment ground truth', human: 'Human label', classifier: 'Success classifier' })}
          </fieldset>
          <h2>Compute & external learner</h2>
          <fieldset disabled={busy} className="hil-fields">
            {field('actor_device', 'Actor device', { required: true, pattern: '(cpu|cuda:[0-9]+)', placeholder: 'cuda:0' })}
            {select('learner_placement', 'Learner location', { local: 'Local GPU', remote: 'External GPU server' })}
            {field('learner_device', 'Learner device', { required: true, pattern: '(cpu|cuda:[0-9]+)', placeholder: 'cuda:0' })}
            {draft.learner_placement === 'remote' && <>
              {field('learner_url', 'External learner URL', { required: true, type: 'url', placeholder: 'http://127.0.0.1:7130' })}
              {field('learner_token_file', 'Local learner token file', { required: true, placeholder: '/workspace/hilserl/learner-token' })}
            </>}
          </fieldset>
          {draft.learner_placement === 'remote' && <>
            <button type="button" disabled={busy} onClick={() => run(async () => {
              const result = await checkHilLearner(draft);
              if (!result.connected) throw new Error(result.error || 'Learner connection failed');
              const gpu = result.gpus?.find(item => item.device === draft.learner_device);
              if (draft.learner_device !== 'cpu' && !gpu) throw new Error(`Connected, but ${draft.learner_device} is unavailable on the server`);
              setNotice(`Learner connected · ${gpu?.name || 'CPU'} · ${result.busy ? 'Busy with an active run' : 'Ready'}`);
            })}>Test learner connection</button>
            <p className="hil-help">Use HTTPS, or an SSH tunnel to localhost:7130. The token stays in the local file. Selected demos and initial policy weights are uploaded automatically. Full learner checkpoints and replay are saved on the server; the final policy is also saved locally.</p>
          </>}
          <p className="hil-help">CUDA requests never fall back to CPU. Preview and classifier data collection create no policy or learner. Demo classification uses the actor device. In external mode only actor inference and reward classification use local GPU memory.</p>
          <h2>Learner settings</h2>
          <button type="button" disabled={busy} onClick={() => {
            setDraft(value => ({ ...value, learning_backend: 'hil_serl', pixel_batch_size: 256,
              pixel_replay_capacity: 200000, discount: .97, learning_rate: .0003,
              cta_ratio: 2, training_starts: 100, policy_publish_interval: 50 }));
            setDirty(true);
          }}>Restore upstream HIL-SERL settings</button>
          <fieldset disabled={busy} className="hil-fields">
            {select('learning_backend', 'Learning engine', { hil_serl: 'HIL-SERL · JAX / pretrained ResNet-10', legacy_torch: 'Legacy · small PyTorch CNN (64×64)' })}
            {select('observation_mode', 'Policy observations', { pixels: 'RGB images + robot proprioception (Isaac)', state: 'Legacy state-only policy' })}
            {field('pixel_batch_size', 'Pixel batch size', { type: 'number', min: 2, max: 1024, step: 2 })}
            {field('pixel_replay_capacity', 'Pixel replay capacity per buffer', { type: 'number', min: 64, max: 1000000 })}
            {field('cta_ratio', 'Critic updates per actor update', { type: 'number', min: 1, max: 16 })}
            {field('training_starts', 'Online transitions before learning', { type: 'number', min: 1, max: 100000 })}
            {field('policy_publish_interval', 'Publish policy every learner steps', { type: 'number', min: 1, max: 1000 })}
            {field('seed', 'Random seed', { type: 'number', required: true, min: 0, max: 2147483647 })}
            {field('batch_size', 'Batch size', { type: 'number', required: true, min: 1, max: 4096 })}
            {field('replay_capacity', 'Replay capacity (transitions)', { type: 'number', required: true, min: 4096, max: 10000000 })}
            {field('learning_rate', 'Learning rate', { type: 'number', required: true, min: 0.000001, max: 0.1, step: 'any' })}
            {field('discount', 'Discount factor', { type: 'number', required: true, min: 0.000001, max: 1, step: 'any' })}
          </fieldset>
          <p className="hil-help">{draft.learning_backend === 'hil_serl'
            ? 'HIL-SERL uses native 128×128 RGB crops, frozen ImageNet ResNet-10, separate camera embeddings, 256×256 policy/critic heads and 22 proprioception values. Batch 256 mixes 128 online and 128 demo/intervention transitions. Two critic updates per actor update, random crop padding 4 and learned entropy temperature match upstream. The learner runs independently of environment steps. Replay stays in disk-backed CPU storage; both 200,000-transition buffers together can grow to about 110 GiB for three cameras. A demo or intervention is required before learning starts.'
            : 'Legacy pixel SAC uses a small trainable CNN and 64×64 RGB images. Mock/state mode uses the separate state batch and replay settings.'}</p>
          {draft.learning_backend === 'hil_serl' && <p className="hil-help">Existing 64×64 data and small-CNN checkpoints remain in the data library. Collect new 128×128 classifier data and demonstrations, then retrain the classifier before using them with HIL-SERL. Offline classifier training uses the original ResNet-10 and batch 256 on CPU; actor and learner use the devices selected above.</p>}
          <h2>Observations & artifacts</h2>
          {['observation', 'classifier'].map(purpose => <fieldset key={purpose} disabled={busy || draft.observation_mode !== 'pixels'} className="hil-camera-selection">
            <legend>{purpose === 'observation' ? 'Policy observation cameras' : 'Success classifier cameras'}</legend>
            {[['head', 'Head'], ['left_wrist', 'Left wrist'], ['right_wrist', 'Right wrist']].map(([camera, label]) => {
              const selected = draft[`${purpose}_cameras`] || ['head', 'left_wrist', 'right_wrist'];
              return <label key={camera}><input type="checkbox" aria-label={`${purpose} camera ${label}`} checked={selected.includes(camera)}
                disabled={selected.length === 1 && selected.includes(camera)} onChange={event => update(`${purpose}_cameras`, event.target.checked ? [...selected, camera] : selected.filter(value => value !== camera))} />{label}</label>;
            })}
          </fieldset>)}
          <p className="hil-help">Choose at least one camera for each purpose. Classifier capture/reward and policy observations use independent selections. Save and start a new session to apply. Overview is for visualization only.</p>
          {draft.learning_backend === 'hil_serl' && <fieldset disabled={busy} className="hil-fields">
            <legend>Observation crop · before resizing to 128×128</legend>
            {['head', 'left_wrist', 'right_wrist'].map(camera => <div key={camera}>
              <strong>{camera}</strong>
              {['Left', 'Top', 'Right', 'Bottom'].map((edge, index) => <label key={edge}>{edge} (0–1)
                <input type="number" min="0" max="1" step=".01" aria-label={`${camera} crop ${edge}`}
                  value={(draft.camera_crops?.[camera] || [0, 0, 1, 1])[index]}
                  onChange={event => {
                    const roi = [...(draft.camera_crops?.[camera] || [0, 0, 1, 1])];
                    roi[index] = Number(event.target.value);
                    update('camera_crops', { ...draft.camera_crops, [camera]: roi });
                  }} />
              </label>)}
            </div>)}
            <p className="hil-help">Crop coordinates refer to the full camera image. The same crop is used for policy and classifier data from that camera. Existing artifacts require matching crops. Preview zoom does not change training observations.</p>
          </fieldset>}
          <fieldset disabled={busy} className="hil-fields">
            {field('camera_topic', 'Camera topic', { required: true, maxLength: 256 })}
            {field('joint_state_topic', 'Joint state topic', { required: true, maxLength: 256 })}
            {field('demonstration_path', 'Demonstration dataset path (optional)', { maxLength: 512 })}
            {field('classifier_path', 'Success classifier checkpoint (optional)', { maxLength: 512 })}
            {field('classifier_threshold', 'Classifier success threshold', { type: 'number', min: 0.01, max: 0.99, step: .01 })}
            {field('classifier_hold_steps', 'Classifier consecutive success steps', { type: 'number', min: 1, max: 100 })}
            {field('checkpoint_path', 'Initial checkpoint path (optional)', { maxLength: 512 })}
          </fieldset>
          <p className="hil-help">Isaac pixel mode uses fresh images captured after each physics step, independently of preview quality. Camera topics are reserved for future ROS integration. Legacy state datasets and checkpoints cannot be loaded into pixel mode. Isaac success uses the peg’s assembled asset pose; the insertion-depth field applies only to Mock. Load policy.npz or replay.npz from a stopped session; paths must be under the HIL data directory. Evaluation requires a policy checkpoint.</p>
          <div className="hil-actions"><button className="hil-primary" type="submit" disabled={busy || !dirty}>Save configuration</button>
            <button type="button" disabled={busy} onClick={() => run(() => refresh(true))}>Reload saved configuration</button></div>
        </form>
        <aside className="hil-card hil-readiness">
          <HilGpuInventory hardware={workspace.readiness.hardware} />
          <h2>Runtime readiness</h2><p className="hil-help">Status for the saved configuration</p>
          <button type="button" disabled={busy} onClick={() => { setDraft((value) => ({ ...value, environment: 'isaac_sim', robot_type: 'ffw_sg2_rev1', control_hz: 20, target_part: value.target_part || 'peg_round_8', translation_limit_m: Math.min(value.translation_limit_m, 0.02), rotation_limit_rad: Math.min(value.rotation_limit_rad, 0.15) })); setDirty(true); }}>Use Isaac NIST settings</button>
          {workspace.readiness.bridge?.busy && <p className="hil-help">Isaac simulator is leased by an active session.</p>}
          {workspace.readiness.checks.map((check) => <div key={check.id} className="hil-check"><span className={`hil-dot ${check.ready ? 'ready' : ''}`}  /><div><strong>{check.label}</strong><p>{check.detail}</p></div></div>)}
          <div className="hil-actions hil-stack"><button className="hil-primary" disabled={busy || dirty || configStale || active || !workspace.readiness.can_preview} onClick={() => create('preview')}>Create preview session</button>
            <button disabled={busy || dirty || configStale || active || !workspace.readiness.can_classifier} onClick={() => create('classifier')}>Get classifier data</button>
            <button disabled={busy || dirty || configStale || active || !workspace.readiness.can_demo} onClick={() => create('demo')}>Get demo</button>
            <button disabled={busy || dirty || configStale || active || !workspace.readiness.can_train} onClick={() => create('train')}>Create {environmentLabel} training session</button>
            <button disabled={busy || dirty || configStale || active || !workspace.readiness.can_evaluate} onClick={() => create('evaluate')}>Create {environmentLabel} evaluation session</button></div>
          <p className="hil-help">{configStale ? 'Configuration changed elsewhere. Reload saved configuration before creating a session.' : dirty ? 'Save or reload your changes before creating a session.' : active ? 'An existing session must be stopped before creating another.' : draft.environment === 'isaac_sim' ? 'Preview starts the simulator with manual controls and no learning. Collection sessions use the smaller of the training and collection episode limits.' : 'Preview lets you test pause, takeover, episode labels and reset.'}</p>
        </aside>
      </div> : tab === 'session' ? <div className="hil-layout hil-session-layout">
        <section className="hil-card">
          {!session ? <div className="hil-empty"><h2>No sessions yet</h2><p>Save your experiment settings, then create a Preview session.</p><button onClick={() => setTab('setup')}>Go to experiment setup</button></div> : <>
            <div className="hil-card-heading"><div><h2>{session.config.name}</h2><p className="hil-help">{session.mode} · {session.config.environment} · {robots[session.config.robot_type]} · {session.config.arm} arm · config v{session.config_revision}</p></div><span className="hil-badge">{session.state}</span></div>
            <div className="hil-episode-progress">
              <label htmlFor="hil-episode-step">Episode {session.episode} · {episodeStep} / {session.config.max_episode_steps} steps</label>
              <progress id="hil-episode-step" aria-label="Episode steps" max={session.config.max_episode_steps} value={Math.min(episodeStep, session.config.max_episode_steps)} />
              {session.runtime_pending && <span>Initializing / waiting for simulator…</span>}
            </div>
            <div className="hil-metrics">
              {[['Episode', session.episode], ['Environment steps', session.metrics.environment_steps], ['Learner steps', session.metrics.learner_steps], ['Success rate', session.metrics.success_rate], ['Policy version', session.metrics.policy_version], ['Replay size', session.metrics.replay_size], ['Human steps', session.metrics.human_steps], ['Critic loss', session.metrics.critic_loss == null ? null : session.metrics.critic_loss.toFixed(4)]].map(([label, value]) => <div key={label}><span>{label}</span><strong>{display(value)}</strong></div>)}
            </div>
            <div className="hil-session-split"><section className="hil-episode-pane" aria-label="Episode and policy">
            {session.mode === 'preview' && !physicalPreview ? <p className="hil-help">Learning metrics remain unavailable in Preview. Episode labels below are workflow checks, not measured results.</p> : <>
              <p className="hil-help">{session.runtime_pending ? 'Waiting for runtime acknowledgement…' : session.runtime?.transport || 'Runtime starts when you press Start.'}{physicalPreview ? ` · Manual ${session.mode}: no policy actions or learning. Use the controls below.` : ` · Learner warms up after ${session.config.learning_backend === 'hil_serl' ? session.config.training_starts : (session.runtime?.effective_batch_size ?? (session.config.observation_mode === 'pixels' ? session.config.pixel_batch_size : session.config.batch_size))} online transitions${session.config.learning_backend === 'hil_serl' ? ' and at least one demo/intervention transition' : ''}. Natural episode endings reset automatically.`}</p>
              {session.runtime_error && <pre role="alert" className="hil-error">{session.runtime_error}</pre>}
              {session.runtime && <div className="hil-mock-view">
                <h2>{sessionLabel} peg position</h2>
                {isaacSession && <p className="hil-help">Initial state: {session.runtime.environment_info?.initialization !== 'kit' ? (session.runtime.environment_info?.grasp_verified ? 'Grasp verified at reset' : 'Grasp verification pending') : 'Kit tray'} · {session.config.arm} arm · {session.config.target_part}</p>}<p>XYZ (m): {session.runtime.position_m?.map((v) => v.toFixed(4)).join(' / ')}</p>
                <svg viewBox="0 0 220 220" role="img" aria-label="Peg XY telemetry relative to target hole">
                  <rect x={isaacSession ? 62 : 10} y={isaacSession ? 62 : 10} width={isaacSession ? 96 : 200} height={isaacSession ? 96 : 200} rx="8" fill="#cbd5e1" />
                  <circle cx={110 + (session.runtime.environment_info?.target_position_m?.[0] || 0) * 250} cy={110 - (session.runtime.environment_info?.target_position_m?.[1] || 0) * 250} r="8" fill="#334155" />
                  <circle cx={110 + (session.runtime.position_m?.[0] || 0) * (isaacSession ? 250 : 1800)} cy={110 - (session.runtime.position_m?.[1] || 0) * (isaacSession ? 250 : 1800)} r="5" fill="#2563eb" />
                </svg>
                <p className="hil-help">XY telemetry: blue peg, dark target. Updates while running; not a rendered camera view. {session.runtime.schema}</p>
                {!physicalPreview && <p className="hil-help">Artifacts are finalized on Stop.<br />Policy: <code>{session.runtime.checkpoint_path}</code><br />Replay{session.runtime.compute?.learner?.placement === 'remote' ? ' (external server)' : ''}: <code>{session.runtime.dataset_path}</code></p>}
              </div>}
            </>}
            <HilComputeTelemetry runtime={session.runtime} />
            <HilActionTelemetry session={session} axes={manualAxes} />
            <div className="hil-actions">
              <button className="hil-primary" disabled={startDisabled} onClick={startRuntime}>{session.mode === 'preview' ? 'Start preview' : `Start ${restartCollection ? environmentLabel : sessionLabel} runtime`}</button>
              <button disabled={busy || session.runtime_pending || session.state !== 'running'} onClick={() => command('pause')}>Pause</button>
              <button disabled={busy || session.runtime_pending || session.state !== 'paused' || session.episode_outcome !== null} onClick={() => command('resume')}>Resume</button>
              <button className="hil-danger" disabled={busy || session.state === 'stopped'} onClick={() => command('stop')}>{session.mode === 'preview' ? 'Stop preview' : 'Stop & save artifacts'}</button>
            </div>
            {restartCollection && <p className="hil-help">Start creates a new classifier collection using the saved configuration, with counts starting at zero. Previous datasets are preserved.{dirty || configStale ? ' Save or reload configuration in Experiment setup first.' : active ? ' Stop the other active session first.' : !workspace.readiness.can_classifier ? ' Waiting for runtime readiness and simulator release.' : ''}</p>}
            </section><section className="hil-intervention-pane" aria-label="Human intervention panel">
            <div className="hil-control"><div><h2>Human intervention</h2><p>Control source: <strong>{physicalPreview && session.control_source === 'policy' ? `Hold (${session.mode} · policy disabled)` : session.control_source}</strong></p></div>
              <span>Automatic · intervention while input is active</span>
            </div>
            <HilCollectionControls spaceToggle={spaceToggle} setSpaceToggle={setSpaceToggle} labelSuccessActive={labelSuccessActive} session={session} busy={busy} command={command} />
            {session.mode === 'demo' && <HilDemoData workspace={workspace} task={session.config} busy={busy} run={run} refresh={refresh} applyArtifact={applyArtifact} prepareDemoTraining={prepareDemoTraining} />}
            <HilPressedKeys spaceToggle={spaceToggle} collecting={session.mode === 'classifier'} pressed={pressedKeys} enabled={keyboardEnabled} isaac={isaacSession} />
            {keyboardEnabled && <p className="hil-keyboard-help">Keyboard active · {isaacSession ? 'TCP frame' : 'XYZ frame'} · W/S: ±X · A/D: ±Y · R/F: ±Z{isaacSession && ' · Q/E: roll · ↑/↓: pitch · ←/→: yaw · Gripper fixed'} · {session.mode === 'classifier' ? (spaceToggle ? 'Space: toggle success/failure · Esc: clear success and release input.' : 'Hold Space: success each step · Release Space: failure each step · Esc: release input.') : 'Space/Esc: release input.'} Release keys or leave the window to return to policy (Preview holds position).</p>}
            {isaacSession && <p className="hil-help">Keyboard target per simulation step: XYZ ±{(session.config.translation_limit_m * 1000).toFixed(1)} mm; rotation ±{session.config.rotation_limit_rad} rad ({(session.config.rotation_limit_rad * 180 / Math.PI).toFixed(2)}°); gripper fixed at its initialized grasp target. Repeats while held at up to {session.config.control_hz} steps/s; actual motion depends on IK, joint limits and contact. Keyboard pulses refresh every 100 ms.</p>}
            {session.runtime?.environment_info?.motion_blocked && <p className="hil-help" role="status">Movement blocked: {session.runtime.environment_info.motion_block_joint || session.runtime.environment_info.motion_block_reason}. Move in another direction to leave the limit.</p>}
            {(session.mode !== 'preview' || physicalPreview) && <div className="hil-actions" aria-label="Manual action controls">
              {manualAxes.flatMap((axis, index) => [1, -1].map((sign) => {
                const label = axis === 'Gripper' ? (sign > 0 ? 'Close gripper' : 'Open gripper') : `${sign > 0 ? '+' : '−'}${axis}`;
                const value = Array(manualDimensions).fill(0); value[index] = sign;
                return <button key={label} disabled={busy || session.runtime_pending || session.state !== 'running'} onClick={() => run(async () => { await sendHilAction(session, value); await refresh(); })}>{label} pulse</button>;
              }))}
              <button disabled={busy || session.runtime_pending || session.state !== 'running'} onClick={() => run(async () => { await sendHilAction(session, Array(manualDimensions).fill(0)); await refresh(); })}>Release input</button>
            </div>}
            <div className="hil-control"><div><h2>Episode outcome</h2><p>{session.episode_outcome || 'Not labeled'}</p></div><div className="hil-actions">
              <button disabled={busy || session.runtime_pending || (!['running', 'paused'].includes(session.state) || (session.state === 'paused' && !['demo', 'classifier'].includes(session.mode))) || session.episode_outcome !== null} onClick={() => command('success')}>Label success</button>
              <button disabled={busy || session.runtime_pending || (!['running', 'paused'].includes(session.state) || (session.state === 'paused' && !['demo', 'classifier'].includes(session.mode))) || session.episode_outcome !== null} onClick={() => command('failure')}>Label failure</button>
              <button title="End this episode without labeling success or failure; reset before resuming." disabled={busy || session.runtime_pending || !['running', 'paused'].includes(session.state) || session.episode_outcome !== null} onClick={() => command('truncate')}>Truncated</button>
              <button disabled={busy || session.runtime_pending || session.state !== 'paused' || (session.mode === 'demo' && session.runtime?.collection?.pending_demo_steps > 0)} onClick={() => command('reset')}>{session.mode === 'preview' ? 'Reset preview episode' : 'Reset episode'}</button>
            </div></div>
            </section></div>
            <h2>Session events</h2>
            {session.events.length === 0 ? <p className="hil-help">No events recorded.</p> : <ol className="hil-events">{[...session.events].reverse().map((event) => <li key={event.sequence}><time>{new Date(event.at).toLocaleTimeString()}</time><strong>{event.action.replace('_', ' ')}</strong><span>Episode {event.episode} · {session.mode}</span></li>)}</ol>}
          </>}
        </section>
        <aside className="hil-card"><h2>Session history</h2><p className="hil-help">Stored on this Cyclo instance</p>
          {workspace.sessions.map((item) => <button key={item.id} className={`hil-history ${session?.id === item.id ? 'selected' : ''}`} onClick={() => setSelectedId(item.id)}><strong>{item.config.name}</strong><span>{new Date(item.created_at).toLocaleString()}</span><span>{item.mode} · {item.state}</span></button>)}
          {workspace.sessions.length === 0 && <p className="hil-help">Your sessions will appear here.</p>}
        </aside>
      </div> : null}
    {tab === 'classifier' && <HilDataLibrary workspace={workspace} busy={busy} run={run} refresh={refresh} applyArtifact={applyArtifact} prepareDemoTraining={prepareDemoTraining} />}
    </>}
  </div>;
}
