import React, { useState } from 'react';
import HilClassifierMetrics from './HilClassifierMetrics';
import { trainHilClassifier, stopHilClassifier, deleteHilDataset, deleteHilClassifier, deleteHilDemoEpisodes, combineHilDemos } from '../utils/hilserlApi';

function DeleteArtifact({ label, disabled, remove }) {
  const [confirm, setConfirm] = useState(false);
  return confirm ? <span className="hil-actions"><span>Permanently delete {label}?</span>
    <button disabled={disabled} onClick={remove}>Confirm delete</button><button onClick={() => setConfirm(false)}>Cancel</button></span>
    : <button disabled={disabled} onClick={() => setConfirm(true)}>Delete {label}</button>;
}

function DiscardDemo({ disabled, canAddFailures, steps, cameras, command }) {
  const [choosing, setChoosing] = useState(false);
  const discard = action => { setChoosing(false); command(action); };
  if (!choosing) return <button disabled={disabled} onClick={() => setChoosing(true)}>Discard pending demo</button>;
  return <div role="group" aria-label="Discard demo options">
    <p>Add this demo to classifier failure data before discarding it?</p>
    <p className="hil-help">Adds {steps} post-action images, including the final success-triggering frame, as FAILURE (0). Use this when all recorded steps are unsuccessful. Classifier cameras: {cameras.join(', ')}. Stop this session to select these images with existing datasets in Train classifier, then retrain and apply the new checkpoint.</p>
    {!canAddFailures && <p>Classifier views were not buffered by this runtime. Start a new demo session to enable failure-data collection.</p>}
    <div className="hil-actions">
      <button disabled={disabled} onClick={() => discard('discard_demo')}>Discard only</button>
      <button disabled={disabled || !canAddFailures} onClick={() => discard('discard_demo_as_failure')}>Discard &amp; add classifier failures</button>
      <button disabled={disabled} onClick={() => setChoosing(false)}>Cancel</button>
    </div>
  </div>;
}

export function HilCollectionControls({ session, busy, command, spaceToggle = false, setSpaceToggle, labelSuccessActive = false }) {
  if (!['classifier', 'demo'].includes(session.mode)) return null;
  if (session.deleted_datasets?.includes(session.mode)) return <p>Collected data deleted. Session history is retained.</p>;
  const collection = session.runtime?.collection || {};
  const ready = !busy && !session.runtime_pending && ['running', 'paused'].includes(session.state);
  const last = collection.last_capture;
  const failureStepLimit = collection.failure_step_limit ?? session.config?.collection_episode_steps ?? 1000;
  const waitingForIntervention = collection.waiting_for_intervention === true;
  return <section className="hil-collection" aria-label="Data collection">
    <h2>{session.mode === 'classifier' ? 'Get classifier data' : 'Get demo'}</h2>
    <p className="hil-help">Policy and learner are disabled. Control the arm with the keyboard; the gripper stays fixed.</p>
    {session.mode === 'classifier' ? <>
      <label className="hil-space-toggle"><input type="checkbox" checked={spaceToggle} onChange={event => { setSpaceToggle?.(event.target.checked); event.currentTarget.blur(); }} />Space key toggle mode</label>
      <p role="status">Label input: <strong>{labelSuccessActive ? 'SUCCESS (true)' : 'FAILURE (false)'}</strong></p>
      <p>Classifier cameras: {(session.config?.classifier_cameras || ['head', 'left_wrist', 'right_wrist']).join(', ')}</p>
      <p role="status">{waitingForIntervention ? 'Waiting for first intervention · No classifier images recorded' : session.state === 'running' ? `${collection.success_only ? 'Recording success only' : 'Recording each step'} · Last saved label: ${last ? (last.label ? 'SUCCESS (1)' : 'FAILURE (0)') : 'waiting'}` : 'Automatic recording paused'}</p>
      {waitingForIntervention && <p className="hil-help">While running, press a movement or rotation key to begin collection. Episode steps stay at 0 until then. Space labels success but does not start collection. Recording continues after you release the movement key.</p>}
      {collection.success_only && <p role="status">Failure collection limit reached ({failureStepLimit} steps). The episode continues; only success images are saved. Skipped failure steps: {collection.skipped_negative_samples || 0}.</p>}
      <label htmlFor="hil-success-target">Success target: {collection.positive_samples || 0} / {collection.success_target || session.config?.classifier_success_target || 200}</label>
      <progress id="hil-success-target" aria-label="Success collection progress" value={collection.positive_samples || 0} max={collection.success_target || session.config?.classifier_success_target || 200} />
      {collection.target_reached && <p role="status">Success target reached. Collection stopped and samples saved. Create a new collection to continue.</p>}
      <p>Success images: {collection.positive_samples || 0} · Failure images: {collection.negative_samples || 0}</p>
      <p className="hil-help">Capture the selected classifier cameras: {spaceToggle ? 'press Space to toggle success (1) / failure (0).' : 'hold Space for success (1), release for failure (0).'} Failure images are saved through episode step {failureStepLimit}; later failure steps are skipped while success collection and keyboard control continue. Reset starts a new failure collection window. Pause stops automatic collection. Esc, leaving this tab/window, or changing input mode clears the success input. Success means the peg is correctly inserted; failure means it is not. Collect diverse views in multiple episodes. Reset or Stop saves each episode as one NPZ file containing its labeled images. Manual capture is available while paused for an unlabeled state.</p>
      <div className="hil-actions">
        <button disabled={!ready || waitingForIntervention || session.state !== 'paused'} onClick={() => command('capture_success')}>Capture success image</button>
        <button disabled={!ready || waitingForIntervention || session.state !== 'paused' || collection.success_only} onClick={() => command('capture_failure')}>Capture failure image</button>
      </div>
      {last && <figure><img width="384" style={{maxWidth:'100%'}} src={`/api/hilserl/classifier-samples/${session.id}/${last.id}`} alt={`Last labeled sample: ${last.label ? 'success' : 'failure'} — ${(last.cameras || ['head', 'left_wrist', 'right_wrist']).join(', ')}`} /><figcaption>Last capture: {last.label ? 'success' : 'failure'} · episode {last.episode}</figcaption></figure>}
    </> : <>
      {waitingForIntervention && <>
        <p role="status">Waiting for first intervention · No demonstration steps recorded</p>
        <p className="hil-help">While running, press a movement or rotation key to begin demo collection. Episode steps stay at 0 until then. Recording continues after you release the key.</p>
      </>}
      <p>Pending steps: {collection.pending_demo_steps || 0} · Saved episodes: {collection.saved_demo_episodes || 0}</p>
      <p className="hil-help">Success pauses the episode for review. Save successful demos manually, or discard unsuccessful attempts before Reset. Stop exports saved episodes as demonstrations.npz.</p>
      <div className="hil-actions">
        <button disabled={!ready || session.state !== 'paused' || session.episode_outcome !== 'success' || !collection.pending_demo_steps} onClick={() => command('save_demo')}>Save successful demo</button>
        <DiscardDemo key={`${session.id}:${session.episode}`} command={command}
          disabled={!ready || session.state !== 'paused' || !collection.pending_demo_steps}
          canAddFailures={collection.pending_demo_classifier_samples > 0 && collection.pending_demo_classifier_samples === collection.pending_demo_steps}
          steps={collection.pending_demo_steps || 0} cameras={session.config?.classifier_cameras || ['head', 'left_wrist', 'right_wrist']} />
      </div>
      {collection.negative_samples > 0 && <p role="status">Discarded demos added {collection.negative_samples} classifier failure images. Stop this session to select them in Train classifier → Data management.</p>}
    </>}
    {collection.error && <p role="alert" className="hil-error">{collection.error}</p>}
  </section>;
}

const taskKey = task => JSON.stringify(['environment', 'robot_type', 'arm', 'target_part'].map(key => task?.[key] || 'unknown'));
const taskLabel = task => `${task?.target_part || 'Unknown target'} · ${task?.arm || 'unknown'} arm · ${task?.robot_type || 'unknown robot'} · ${task?.environment || 'unknown environment'}`;
const cameraKey = item => [...(item.cameras || ['head', 'left_wrist', 'right_wrist'])].sort().join(', ') + (item.image_size ? ` · ${item.image_size}×${item.image_size}` : '');

export function HilDataLibrary({ workspace, busy, run, refresh, applyArtifact }) {
  const [selected, setSelected] = useState([]);
  const [environmentFilter, setEnvironmentFilter] = useState('');
  const [cameraFilter, setCameraFilter] = useState('');
  const [epochs, setEpochs] = useState(20);
  const library = (workspace.data_library || []).filter(item => item.kind === 'classifier');
  const jobs = workspace.classifier_jobs || [];
  const stopped = new Set(workspace.sessions.filter(s => s.state === 'stopped' && !s.dataset_finalizing).map(s => s.id));
  const sessionInfo = id => workspace.sessions.find(session => session.id === id);
  const environmentName = item => item.environment_name || sessionInfo(item.session_id)?.config?.name?.trim() || 'Unnamed environment';
  const jobEnvironments = job => job.environment_names?.length ? job.environment_names : [environmentName(job)];
  const environments = [...new Set([...library.map(environmentName), ...jobs.flatMap(jobEnvironments)])];
  const activeEnvironment = environments.includes(environmentFilter) ? environmentFilter : environments[0] || '';
  const environmentData = library.filter(item => environmentName(item) === activeEnvironment);
  const cameraOptions = [...new Set(environmentData.map(cameraKey))];
  const activeCameras = cameraOptions.includes(cameraFilter) ? cameraFilter : cameraOptions[0] || '';
  const sources = environmentData.filter(item => cameraKey(item) === activeCameras);
  const selectedSources = sources.filter(item => stopped.has(item.session_id) && selected.includes(item.session_id));
  const selectedIds = selectedSources.map(item => item.session_id);
  const counts = selectedSources.reduce((total, item) => ({positive: total.positive + item.positive, negative: total.negative + item.negative}), {positive: 0, negative: 0});
  const visibleJobs = jobs.filter(job => jobEnvironments(job).includes(activeEnvironment));
  const compatible = new Set(selectedSources.map(item => `${taskKey(item.task)}:${item.learning_backend || 'legacy_torch'}`)).size <= 1;
  const training = jobs.some(job => ['queued', 'training'].includes(job.state));
  return <div className="hil-data-library" aria-label="Datasets and classifiers">
    <section className="hil-card" aria-label="Environment data management">
      <h2>Data management</h2>
      <div className="hil-data-filters">
        <label>Environment name<select aria-label="Dataset environment" value={activeEnvironment} disabled={!environments.length} onChange={event => {setEnvironmentFilter(event.target.value); setCameraFilter(''); setSelected([]);}}>
          {!environments.length && <option value="">No collected environments</option>}
          {environments.map(name => <option value={name} key={name}>{name}</option>)}
        </select></label>
        <label>Classifier cameras<select aria-label="Dataset cameras" value={activeCameras} disabled={!cameraOptions.length} onChange={event => {setCameraFilter(event.target.value); setSelected([]);}}>
          {!cameraOptions.length && <option value="">No classifier datasets</option>}
          {cameraOptions.map(cameras => <option key={cameras} value={cameras}>{cameras}</option>)}
        </select></label>
      </div>
      <p className="hil-help">Collections are grouped by the environment name saved at collection time. Choose compatible camera views, then select stopped collections for training. Running collections are read-only.</p>
      <p>Environment classifier library: {sources.reduce((sum,item) => sum+item.positive,0)} success / {sources.reduce((sum,item) => sum+item.negative,0)} failure samples · {sources.length} collection sources</p>
      <DeleteArtifact key={selectedIds.join(',')} label="selected classifier data" disabled={busy || training || !selectedIds.length} remove={() => run(async () => {
        try {
          for (const id of selectedIds) { await deleteHilDataset(id, 'classifier'); setSelected(value => value.filter(item => item !== id)); }
        } finally { await refresh(); }
      })} />
      <div className="hil-actions"><button disabled={busy || !sources.some(item => stopped.has(item.session_id))} onClick={() => setSelected(sources.filter(item => stopped.has(item.session_id)).map(item => item.session_id))}>Select all datasets</button>
        <button disabled={!selectedIds.length} onClick={() => setSelected([])}>Clear selection</button></div>
      {sources.length > 0 ? <div className="hil-data-table-wrap"><table className="hil-data-table"><thead><tr><th>Select</th><th>Collection source</th><th>Status</th><th>Success</th><th>Failure</th><th>Episodes / files</th><th>Manage</th></tr></thead><tbody>
      {sources.map(item => <tr key={item.session_id}>
        <td><input type="checkbox" aria-label={`Select dataset ${item.session_id}`} disabled={!stopped.has(item.session_id) || busy} checked={selectedIds.includes(item.session_id)} onChange={e => setSelected(value => e.target.checked ? [...value, item.session_id] : value.filter(id => id !== item.session_id))} /></td>
        <td><strong>{environmentName(item)}</strong><br/><code>{item.session_id}</code><small>{taskLabel(item.task)}</small><small>{item.learning_backend === 'hil_serl' ? '128×128 · HIL-SERL ResNet-10' : '64×64 · Legacy CNN'}</small>
          {sessionInfo(item.session_id)?.created_at && <small>{new Date(sessionInfo(item.session_id).created_at).toLocaleString()}</small>}</td>
        <td>{sessionInfo(item.session_id)?.dataset_finalizing ? 'Saving episode files' : sessionInfo(item.session_id)?.state || 'unknown'}</td><td>{item.positive}</td><td>{item.negative}</td><td>{item.episodes}{(item.files || []).map(file => <small key={file.episode}>{file.name || `Episode ${file.episode} · individual images`} · {file.positive} success / {file.negative} failure</small>)}</td>
        <td><DeleteArtifact label="dataset" disabled={busy || training || !stopped.has(item.session_id)} remove={() => run(async () => {
          await deleteHilDataset(item.session_id, 'classifier'); setSelected(value => value.filter(id => id !== item.session_id)); await refresh();
        })} /></td>
      </tr>)}
      </tbody></table></div> : <p>No classifier collections for this environment.</p>}
      <p role="status">Selected {selectedIds.length} datasets · {counts.positive} success / {counts.negative} failure samples</p>
    </section>
    <section className="hil-card" aria-label="Classifier training and checkpoints">
    <h2>Train classifier & checkpoints</h2>
    <p>{activeEnvironment || 'Select an environment in Data management.'}</p>
    <p className="hil-help">New classifier jobs train on local GPU cuda:0. Training uses only the selected datasets above. Collect at least 10 unique images per class. Images from all selected datasets are pooled without episode boundaries. Success and failure images are each randomly split into about 75% training and 25% validation. Failure-only episodes are included. Changing the selected images may change the split; this does not measure unseen-episode performance.</p>
    <div className="hil-actions"><label>Classifier epochs <input aria-label="Classifier epochs" type="number" min="1" max="200" value={epochs} onChange={e => setEpochs(Number(e.target.value))} /></label>
      <button disabled={busy || training || !compatible || !selectedIds.length || epochs < 1 || epochs > 200} onClick={() => run(async () => { await trainHilClassifier(selectedIds, epochs); await refresh(); })}>Train classifier</button></div>
    {!compatible && <p role="alert">Selected collections use different simulators, robots, arms, targets or model backends. Select compatible collections to train together.</p>}
    {visibleJobs.map(job => <div className="hil-artifact" key={job.id}>
      <strong>Classifier {job.id.slice(0, 8)} · {job.state}</strong><p>{job.learning_backend === 'hil_serl' ? '128×128 · ResNet-10 · batch 256' : '64×64 · Legacy CNN'} · {job.device?.startsWith('cuda:') ? `GPU ${job.device}` : job.device === 'cpu' ? 'CPU' : 'Device not recorded'}</p>
      {['queued', 'training'].includes(job.state) && job.phase && <div role="status">
        <p>{job.current_epoch ? `Epoch ${job.current_epoch} · ` : ''}{job.phase} · {job.phase_completed ?? 0} / {job.phase_total ?? 0}
          {job.phase_started_at && ` · ${Math.max(0, Math.floor(Date.now() / 1000 - job.phase_started_at))} s in this phase`}</p>
        <progress aria-label="Classifier phase progress" value={job.phase_completed ?? 0} max={Math.max(1, job.phase_total ?? 1)} />
      </div>}
      {job.loading_seconds != null && <p>Images loaded in {job.loading_seconds.toFixed(1)} s
        {job.initialization_seconds != null && ` · Model initialized in ${job.initialization_seconds.toFixed(1)} s`}
        {job.first_batch_seconds != null && ` · First batch ${job.first_batch_seconds.toFixed(1)} s`}
        {job.epoch_seconds != null && ` · Last epoch ${job.epoch_seconds.toFixed(1)} s`}</p>}
      {job.memory?.peak_allocated_mib != null && <p>Peak GPU allocation: {job.memory.peak_allocated_mib.toFixed(0)} MiB</p>}
      {['queued', 'training'].includes(job.state) && <button disabled={busy} onClick={() => run(async () => { await stopHilClassifier(job.id); await refresh(); })}>Stop training</button>}
      {job.state === 'stopped' && <p role="status">Training stopped. {job.checkpoint_path ? 'The best saved checkpoint is available below.' : 'No completed checkpoint was saved.'}</p>}
      {job.epoch != null && <p>Epoch {job.epoch}/{job.epochs} · {job.checkpoint_metrics ? 'Saved checkpoint' : 'Current epoch'} balanced accuracy {(100 * (job.checkpoint_metrics || job).balanced_accuracy).toFixed(1)}% · precision {(100 * (job.checkpoint_metrics || job).precision).toFixed(1)}% · recall {(100 * (job.checkpoint_metrics || job).recall).toFixed(1)}% · threshold {job.threshold}</p>}
      {job.train_samples != null && <p>Training {job.train_samples} / validation {job.validation_samples} images</p>}
      {job.best_epoch != null && <p>Saved checkpoint: epoch {job.best_epoch} · validation balanced accuracy {(100 * job.best_metrics.balanced_accuracy).toFixed(1)}% · training {(100 * job.best_metrics.train_balanced_accuracy).toFixed(1)}%</p>}
      {job.tp != null && <p>Latest epoch validation: TP {job.tp} · TN {job.tn} · FP {job.fp} · FN {job.fn} · loss {job.validation_loss.toFixed(4)}</p>}
      {(job.warnings || (job.state === 'complete' && job.recall === 0 ? ['No validation successes detected. Review labels, camera views and episode diversity before use.'] : [])).map(message => <p className="hil-error" key={message}>{message}</p>)}
      {job.error && <pre className="hil-error">{job.error}</pre>}
      <HilClassifierMetrics job={job} />
      <DeleteArtifact label="classifier checkpoint / job" disabled={busy || ['queued', 'training'].includes(job.state)} remove={() => run(async () => { await deleteHilClassifier(job.id); await refresh(); })} />
      {['complete', 'stopped'].includes(job.state) && job.checkpoint_path && <button disabled={busy} onClick={() => applyArtifact('classifier_path', job.checkpoint_path, job.cameras, job)}>Use classifier</button>}
    </div>)}
    {!visibleJobs.length && <p>No classifier training jobs for this environment.</p>}
    </section>
  </div>;
}

export function HilDemoData({workspace, task, busy, run, refresh, applyArtifact, prepareDemoTraining}) {
  const [exported, setExported] = useState(null);
  const [exporting, setExporting] = useState(false);
  const [copyNotice, setCopyNotice] = useState('');
  const [selected, setSelected] = useState([]);
  const [cameraFilter, setCameraFilter] = useState('');
  const taskData = (workspace.data_library || []).filter(item => item.kind === 'demo' && taskKey(item.task) === taskKey(task));
  const stopped = new Set(workspace.sessions.filter(s => s.state === 'stopped' && !s.dataset_finalizing).map(s => s.id));
  const cameras = [...new Set(taskData.map(cameraKey))];
  const camera = cameras.includes(cameraFilter) ? cameraFilter : cameras[0] || '';
  const sources = taskData.filter(item => cameraKey(item) === camera);
  const files = sources.flatMap(item => (item.files || []).map(file => ({...file, learning_backend:item.learning_backend, session_id:item.session_id, id:`${item.session_id}:${file.episode}`})));
  const chosen = files.filter(file => selected.includes(file.id));
  const selections = [...new Set(chosen.map(file => file.session_id))].map(session_id => ({session_id, episodes:chosen.filter(file => file.session_id === session_id).map(file => file.episode)}));
  const selectionKey = JSON.stringify(selections);
  const currentExport = exported?.selectionKey === selectionKey ? exported : null;
  const exportDisabled = busy || !chosen.length || chosen.reduce((sum,file) => sum+file.steps,0)>4096;
  const prepareExport = async () => {
    if (currentExport) return currentExport;
    setExporting(true);
    try {
      const artifact = await combineHilDemos(selections);
      const result = {...artifact, selectionKey, sourceCount: selections.length};
      setExported(result); setCopyNotice('');
      return result;
    } finally { setExporting(false); }
  };
  const copyPath = async path => {
    try {
      if (navigator.clipboard?.writeText) await navigator.clipboard.writeText(path);
      else {
        const input = document.createElement('textarea');
        input.value = path; input.style.position = 'fixed'; input.style.opacity = '0';
        document.body.appendChild(input); input.select();
        try { if (!document.execCommand('copy')) throw new Error('Clipboard unavailable'); }
        finally { input.remove(); }
      }
      setCopyNotice('Demo path copied to clipboard.');
    } catch { setCopyNotice('Clipboard unavailable. Select and copy the path below, or apply it directly to training settings.'); }
  };
  return <section aria-label="Demo data management">
    <h3>Demonstration data for this task</h3>
    <p className="hil-help">All matching collections share this task library. Session IDs identify the collection source. Select episodes across sources to delete or combine for learning. Exports are snapshots; regenerate after changing your selection.</p>
    <label>Demo cameras<select aria-label="Demo cameras" value={camera} onChange={event => {setCameraFilter(event.target.value); setSelected([]);}}>
      {!cameras.length && <option value="">No demonstrations</option>}
      {cameras.map(value => <option key={value} value={value}>{value}</option>)}
    </select></label>
    <p>{files.length} episodes / {files.reduce((sum, file) => sum + file.steps, 0)} steps · {sources.length} collection sources</p>
    <p role="status">Selected {chosen.length} demos from {selections.length} sources · {chosen.reduce((sum, file) => sum + file.steps, 0)} steps</p>
    <div className="hil-actions">
      <button disabled={busy || !files.length} onClick={() => setSelected(files.map(file => file.id))}>Select all demos</button>
      <button disabled={busy || !chosen.length} onClick={() => setSelected([])}>Clear demo selection</button>
      <DeleteArtifact key={chosen.map(file => file.id).join(',')} label="selected demos" disabled={busy || !chosen.length || chosen.some(file => !stopped.has(file.session_id))} remove={() => run(async () => {
        try {
          for (const selection of selections) {
            await deleteHilDemoEpisodes(selection.session_id, selection.episodes);
            setSelected(value => value.filter(id => !selection.episodes.some(episode => id === `${selection.session_id}:${episode}`)));
          }
        } finally { await refresh(); }
      })} />
      <button disabled={exportDisabled || chosen.some(file => !stopped.has(file.session_id))} onClick={() => run(async () => { const artifact = await prepareExport(); await copyPath(artifact.path); })}>Copy selected demo path</button>
      {applyArtifact && <button disabled={exportDisabled} onClick={() => run(async () => {
        await prepareDemoTraining?.();
        const artifact = await prepareExport();
        await applyArtifact('demonstration_path', artifact.path, artifact.cameras, artifact);
      })}>Apply demos to training settings</button>}

    </div>
    {exporting && <p role="status">Combining selected demo images and saving the training dataset… This may take up to two minutes.</p>}
    {currentExport && <div><label>Training demo path<input aria-label="Training demo path" readOnly value={currentExport.path} onFocus={event => event.target.select()} /></label>
      {copyNotice && <p role="status">{copyNotice}</p>}</div>}
    <p className="hil-help">Apply demos stops the demo session, combines all checked sources, saves training settings and creates a training session. Press Start in the new session to run training. Only saved, checked episodes are included; pending attempts are kept separately.</p>
    <p className="hil-help">Saved demos can be selected during collection. Stop first to copy or delete; Apply stops collection automatically. Training exports support up to 4096 steps; set replay capacity to at least the selected step count.</p>
    <table><thead><tr><th>Select</th><th>Episode</th><th>Steps</th><th>File</th><th>Collection source</th></tr></thead>
      <tbody>{files.map(file => <tr key={file.id}>
        <td><input type="checkbox" aria-label={`Select demo ${file.session_id} episode ${file.episode}`} disabled={busy} checked={chosen.some(value => value.id === file.id)} onChange={event => setSelected(value => event.target.checked ? [...value, file.id] : value.filter(id => id !== file.id))} /></td>
        <td>{file.episode}</td><td>{file.steps}</td>
        <td>{stopped.has(file.session_id) ? <a href={`/api/hilserl/datasets/${file.session_id}/demo/files/${encodeURIComponent(file.name)}`} download>{file.name}</a> : file.name}</td>
        <td><small>{file.session_id}</small><br/><small>{file.learning_backend === 'hil_serl' ? '128×128 · HIL-SERL' : '64×64 · Legacy'}</small></td>
      </tr>)}</tbody>
    </table>
    {!files.length && <p>No demonstrations for this task.</p>}
  </section>;
}
