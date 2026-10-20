import React, { useState } from 'react';
import HilClassifierMetrics from './HilClassifierMetrics';
import { trainHilClassifier, deleteHilDataset, deleteHilClassifier, deleteHilDemoEpisodes, combineHilDemos } from '../utils/hilserlApi';

function DeleteArtifact({ label, disabled, remove }) {
  const [confirm, setConfirm] = useState(false);
  return confirm ? <span className="hil-actions"><span>Permanently delete {label}?</span>
    <button disabled={disabled} onClick={remove}>Confirm delete</button><button onClick={() => setConfirm(false)}>Cancel</button></span>
    : <button disabled={disabled} onClick={() => setConfirm(true)}>Delete {label}</button>;
}

export function HilCollectionControls({ session, busy, command, spaceToggle = false, setSpaceToggle, labelSuccessActive = false }) {
  if (!['classifier', 'demo'].includes(session.mode)) return null;
  if (session.deleted_datasets?.includes(session.mode)) return <p>Collected data deleted. Session history is retained.</p>;
  const collection = session.runtime?.collection || {};
  const ready = !busy && !session.runtime_pending && ['running', 'paused'].includes(session.state);
  const last = collection.last_capture;
  return <section className="hil-collection" aria-label="Data collection">
    <h2>{session.mode === 'classifier' ? 'Get classifier data' : 'Get demo'}</h2>
    <p className="hil-help">Policy and learner are disabled. Control the arm with the keyboard; the gripper stays fixed.</p>
    {session.mode === 'classifier' ? <>
      <label className="hil-space-toggle"><input type="checkbox" checked={spaceToggle} onChange={event => { setSpaceToggle?.(event.target.checked); event.currentTarget.blur(); }} />Space key toggle mode</label>
      <p role="status">Label input: <strong>{labelSuccessActive ? 'SUCCESS (true)' : 'FAILURE (false)'}</strong></p>
      <p>Classifier cameras: {(session.config?.classifier_cameras || ['head', 'left_wrist', 'right_wrist']).join(', ')}</p>
      <p role="status">{session.state === 'running' ? `Recording each step · Last saved label: ${last ? (last.label ? 'SUCCESS (1)' : 'FAILURE (0)') : 'waiting'}` : 'Automatic recording paused'}</p>
      <label htmlFor="hil-success-target">Success target: {collection.positive_samples || 0} / {collection.success_target || session.config?.classifier_success_target || 200}</label>
      <progress id="hil-success-target" aria-label="Success collection progress" value={collection.positive_samples || 0} max={collection.success_target || session.config?.classifier_success_target || 200} />
      {collection.target_reached && <p role="status">Success target reached. Collection stopped and samples saved. Create a new collection to continue.</p>}
      <p>Success images: {collection.positive_samples || 0} · Failure images: {collection.negative_samples || 0}</p>
      <p className="hil-help">Capture the selected classifier cameras. While running, every environment step is saved: {spaceToggle ? 'press Space to toggle success (1) / failure (0).' : 'hold Space for success (1), release for failure (0).'} Pause stops automatic collection. Esc, leaving this tab/window, or changing input mode clears the success input. Success means the peg is correctly inserted; failure means it is not. Collect diverse views in multiple episodes. Manual capture is available while paused for an unlabeled state.</p>
      <div className="hil-actions">
        <button disabled={!ready || session.state !== 'paused'} onClick={() => command('capture_success')}>Capture success image</button>
        <button disabled={!ready || session.state !== 'paused'} onClick={() => command('capture_failure')}>Capture failure image</button>
      </div>
      {last && <figure><img width="384" style={{maxWidth:'100%'}} src={`/api/hilserl/classifier-samples/${session.id}/${last.id}`} alt={`Last labeled sample: ${last.label ? 'success' : 'failure'} — ${(last.cameras || ['head', 'left_wrist', 'right_wrist']).join(', ')}`} /><figcaption>Last capture: {last.label ? 'success' : 'failure'} · episode {last.episode}</figcaption></figure>}
    </> : <>

      <p>Pending steps: {collection.pending_demo_steps || 0} · Saved episodes: {collection.saved_demo_episodes || 0}</p>
      <p className="hil-help">Success pauses the episode for review. Save successful demos manually, or discard unsuccessful attempts before Reset. Stop exports saved episodes as demonstrations.npz.</p>
      <div className="hil-actions">
        <button disabled={!ready || session.state !== 'paused' || session.episode_outcome !== 'success' || !collection.pending_demo_steps} onClick={() => command('save_demo')}>Save successful demo</button>
        <button disabled={!ready || session.state !== 'paused' || !collection.pending_demo_steps} onClick={() => command('discard_demo')}>Discard pending demo</button>
      </div>
    </>}
    {collection.error && <p role="alert" className="hil-error">{collection.error}</p>}
  </section>;
}

const taskKey = task => JSON.stringify(['environment', 'robot_type', 'arm', 'target_part'].map(key => task?.[key] || 'unknown'));
const taskLabel = task => `${task?.target_part || 'Unknown target'} · ${task?.arm || 'unknown'} arm · ${task?.robot_type || 'unknown robot'} · ${task?.environment || 'unknown environment'}`;
const cameraKey = item => [...(item.cameras || ['head', 'left_wrist', 'right_wrist'])].sort().join(', ');

export function HilDataLibrary({ workspace, busy, run, refresh, applyArtifact, prepareDemoTraining }) {
  const [selected, setSelected] = useState([]);
  const [taskFilter, setTaskFilter] = useState('');
  const [cameraFilter, setCameraFilter] = useState('');
  const [epochs, setEpochs] = useState(20);
  const library = workspace.data_library || [];
  const jobs = workspace.classifier_jobs || [];
  const stopped = new Set(workspace.sessions.filter(s => s.state === 'stopped').map(s => s.id));
  const tasks = [...new Map([...library, ...jobs].filter(item => item.task).map(item => [taskKey(item.task), item.task])).entries()];
  const activeTask = tasks.some(([key]) => key === taskFilter) ? taskFilter : tasks[0]?.[0] || '';
  const taskData = library.filter(item => taskKey(item.task) === activeTask);
  const cameraOptions = [...new Set(taskData.filter(item => item.kind === 'classifier').map(cameraKey))];
  const activeCameras = cameraOptions.includes(cameraFilter) ? cameraFilter : cameraOptions[0] || '';
  const sources = taskData.filter(item => item.kind === 'classifier' && cameraKey(item) === activeCameras);
  const selectedSources = sources.filter(item => stopped.has(item.session_id) && selected.includes(item.session_id));
  const selectedIds = selectedSources.map(item => item.session_id);
  const counts = selectedSources.reduce((total, item) => ({positive: total.positive + item.positive, negative: total.negative + item.negative}), {positive: 0, negative: 0});
  const visibleJobs = jobs.filter(job => !job.task || taskKey(job.task) === activeTask);
  const sessionInfo = id => workspace.sessions.find(session => session.id === id);
  const training = jobs.some(job => ['queued', 'training'].includes(job.state));
  return <div className="hil-data-library" aria-label="Datasets and classifiers">
    <section className="hil-card" aria-label="Task data management">
      <h2>Data management</h2>
      <div className="hil-data-filters">
        <label>Task<select aria-label="Dataset task" value={activeTask} disabled={!tasks.length} onChange={event => {setTaskFilter(event.target.value); setCameraFilter(''); setSelected([]);}}>
          {!tasks.length && <option value="">No collected tasks</option>}
          {tasks.map(([key, task]) => <option value={key} key={key}>{taskLabel(task)}</option>)}
        </select></label>
        <label>Classifier cameras<select aria-label="Dataset cameras" value={activeCameras} disabled={!cameraOptions.length} onChange={event => {setCameraFilter(event.target.value); setSelected([]);}}>
          {!cameraOptions.length && <option value="">No classifier datasets</option>}
          {cameraOptions.map(cameras => <option key={cameras} value={cameras}>{cameras}</option>)}
        </select></label>
      </div>
      <p className="hil-help">Tasks are grouped by environment, robot, working arm and target part. Choose compatible camera views, then select stopped collections for training. Running collections are read-only.</p>
      <p>Task classifier library: {sources.reduce((sum,item) => sum+item.positive,0)} success / {sources.reduce((sum,item) => sum+item.negative,0)} failure samples · {sources.length} collection sources</p>
      <DeleteArtifact key={selectedIds.join(',')} label="selected classifier data" disabled={busy || training || !selectedIds.length} remove={() => run(async () => {
        try {
          for (const id of selectedIds) { await deleteHilDataset(id, 'classifier'); setSelected(value => value.filter(item => item !== id)); }
        } finally { await refresh(); }
      })} />
      <div className="hil-actions"><button disabled={busy || !sources.some(item => stopped.has(item.session_id))} onClick={() => setSelected(sources.filter(item => stopped.has(item.session_id)).map(item => item.session_id))}>Select all datasets</button>
        <button disabled={!selectedIds.length} onClick={() => setSelected([])}>Clear selection</button></div>
      {sources.length > 0 ? <div className="hil-data-table-wrap"><table className="hil-data-table"><thead><tr><th>Select</th><th>Collection source</th><th>Status</th><th>Success</th><th>Failure</th><th>Episodes</th><th>Manage</th></tr></thead><tbody>
      {sources.map(item => <tr key={item.session_id}>
        <td><input type="checkbox" aria-label={`Select dataset ${item.session_id}`} disabled={!stopped.has(item.session_id) || busy} checked={selectedIds.includes(item.session_id)} onChange={e => setSelected(value => e.target.checked ? [...value, item.session_id] : value.filter(id => id !== item.session_id))} /></td>
        <td><strong>{sessionInfo(item.session_id)?.config?.name || item.task.target_part}</strong><br/><code>{item.session_id}</code><small>{item.learning_backend === 'hil_serl' ? '128×128 · HIL-SERL ResNet-10' : '64×64 · Legacy CNN'}</small>
          {sessionInfo(item.session_id)?.created_at && <small>{new Date(sessionInfo(item.session_id).created_at).toLocaleString()}</small>}</td>
        <td>{sessionInfo(item.session_id)?.state || 'unknown'}</td><td>{item.positive}</td><td>{item.negative}</td><td>{item.episodes}</td>
        <td><DeleteArtifact label="dataset" disabled={busy || training || !stopped.has(item.session_id)} remove={() => run(async () => {
          await deleteHilDataset(item.session_id, 'classifier'); setSelected(value => value.filter(id => id !== item.session_id)); await refresh();
        })} /></td>
      </tr>)}
      </tbody></table></div> : <p>No classifier collections for this task.</p>}
      <p role="status">Selected {selectedIds.length} datasets · {counts.positive} success / {counts.negative} failure samples</p>
      <HilDemoData workspace={workspace} task={tasks.find(([key]) => key === activeTask)?.[1]} busy={busy} run={run} refresh={refresh} applyArtifact={applyArtifact} prepareDemoTraining={prepareDemoTraining} />
    </section>
    <section className="hil-card" aria-label="Classifier training and checkpoints">
    <h2>Train classifier & checkpoints</h2>
    <p>{tasks.find(([key]) => key === activeTask)?.[1] ? taskLabel(tasks.find(([key]) => key === activeTask)[1]) : 'Select a task in Data management.'}</p>
    <p className="hil-help">Training uses only the selected datasets above. Collect at least 10 unique images per class across multiple episodes. Validation uses separate episodes.</p>
    <div className="hil-actions"><label>Classifier epochs <input aria-label="Classifier epochs" type="number" min="1" max="200" value={epochs} onChange={e => setEpochs(Number(e.target.value))} /></label>
      <button disabled={busy || training || !selectedIds.length || epochs < 1 || epochs > 200} onClick={() => run(async () => { await trainHilClassifier(selectedIds, epochs); await refresh(); })}>Train classifier</button></div>
    {visibleJobs.map(job => <div className="hil-artifact" key={job.id}>
      <strong>Classifier {job.id.slice(0, 8)} · {job.state}</strong><p>{job.learning_backend === 'hil_serl' ? '128×128 · ResNet-10 · batch 256 · CPU training' : '64×64 · Legacy CNN'}</p>
      {job.epoch != null && <p>Epoch {job.epoch}/{job.epochs} · {job.checkpoint_metrics ? 'Saved checkpoint' : 'Current epoch'} balanced accuracy {(100 * (job.checkpoint_metrics || job).balanced_accuracy).toFixed(1)}% · precision {(100 * (job.checkpoint_metrics || job).precision).toFixed(1)}% · recall {(100 * (job.checkpoint_metrics || job).recall).toFixed(1)}% · threshold {job.threshold}</p>}
      {job.train_samples != null && <p>Training {job.train_samples} / validation {job.validation_samples} images</p>}
      {job.best_epoch != null && <p>Saved checkpoint: epoch {job.best_epoch} · validation balanced accuracy {(100 * job.best_metrics.balanced_accuracy).toFixed(1)}% · training {(100 * job.best_metrics.train_balanced_accuracy).toFixed(1)}%</p>}
      {job.tp != null && <p>Latest epoch validation: TP {job.tp} · TN {job.tn} · FP {job.fp} · FN {job.fn} · loss {job.validation_loss.toFixed(4)}</p>}
      {(job.warnings || (job.state === 'complete' && job.recall === 0 ? ['No validation successes detected. Review labels, camera views and episode diversity before use.'] : [])).map(message => <p className="hil-error" key={message}>{message}</p>)}
      {job.error && <pre className="hil-error">{job.error}</pre>}
      <HilClassifierMetrics job={job} />
      <DeleteArtifact label="classifier checkpoint / job" disabled={busy || ['queued', 'training'].includes(job.state)} remove={() => run(async () => { await deleteHilClassifier(job.id); await refresh(); })} />
      {job.state === 'complete' && <button disabled={busy} onClick={() => applyArtifact('classifier_path', job.checkpoint_path, job.cameras)}>Use classifier</button>}
    </div>)}
    {!visibleJobs.length && <p>No classifier training jobs for this task.</p>}
    </section>
  </div>;
}

export function HilDemoData({workspace, task, busy, run, refresh, applyArtifact, prepareDemoTraining}) {
  const [exported, setExported] = useState(null);
  const [copyNotice, setCopyNotice] = useState('');
  const [selected, setSelected] = useState([]);
  const [cameraFilter, setCameraFilter] = useState('');
  const taskData = (workspace.data_library || []).filter(item => item.kind === 'demo' && taskKey(item.task) === taskKey(task));
  const stopped = new Set(workspace.sessions.filter(s => s.state === 'stopped').map(s => s.id));
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
    const artifact = await combineHilDemos(selections);
    const result = {...artifact, selectionKey, sourceCount: selections.length};
    setExported(result); setCopyNotice('');
    return result;
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
