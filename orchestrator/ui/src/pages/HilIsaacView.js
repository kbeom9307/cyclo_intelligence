import React, { useEffect, useRef, useState } from 'react';
import HilOverviewCrop from './HilOverviewCrop';
import HilSessionCropFrame from './HilSessionCropFrame';

/** Latest-frame viewer: one request in flight, bounded memory, no physics commands. */
function CameraView({ camera, label, onClose, environment, size = {}, onResize, sessionConfig, sessionMode }) {
  const [showCrop, setShowCrop] = useState(true);
  const cropped = camera !== 'spectator' && sessionConfig?.learning_backend === 'hil_serl' && sessionConfig?.observation_mode === 'pixels';
  const imageSize = sessionConfig?.image_size || 128;
  const selectedFor = cropped ? [
    ...(sessionMode !== 'classifier' && (sessionConfig.observation_cameras || ['head','left_wrist','right_wrist']).includes(camera) ? ['Policy / demo observation'] : []),
    ...((sessionMode === 'classifier' || sessionConfig.reward_source === 'classifier') && (sessionConfig.classifier_cameras || ['head','left_wrist','right_wrist']).includes(camera) ? ['Classifier input'] : []),
  ] : [];
  const [enabled, setEnabled] = useState(true);
  const [quality, setQuality] = useState(() => {
    try { const saved = localStorage.getItem(`hil-camera-quality-${camera}`); return (camera === 'spectator' ? ['low','standard','high','ultra'] : ['low','standard','high']).includes(saved) ? saved : 'standard'; } catch (_) { return 'standard'; }
  });
  const [frame, setFrame] = useState('');
  const [status, setStatus] = useState('Connecting to simulator camera…');
  const container = useRef(null);

  useEffect(() => {
    if (!enabled) return undefined;
    let alive = true;
    let timer;
    let controller;
    let currentUrl;
    let previousUrl;
    const load = async () => {
      if (document.hidden) { timer = setTimeout(load, 500); return; }
      controller = new AbortController();
      const timeout = setTimeout(() => controller.abort(), 3000);
      let delay = 250;
      try {
        const response = await fetch(`/api/hilserl/${environment === 'mujoco' ? 'mujoco' : 'isaac'}/frame?camera=${camera}&quality=${quality}`,  { signal: controller.signal, cache: 'no-store' });
        if (!response.ok) throw new Error('Camera unavailable. Waiting for the visualization bridge…');
        if (!response.headers.get('content-type')?.startsWith('image/jpeg')) throw new Error('Invalid camera response');
        const blob = await response.blob();
        if (!alive) return;
        if (previousUrl) URL.revokeObjectURL(previousUrl);
        previousUrl = currentUrl;
        currentUrl = URL.createObjectURL(blob);
        setFrame(currentUrl);
        const age = Number(response.headers.get('X-Frame-Age-Ms') || 0);
        setStatus(`Live · ${response.headers.get('X-Frame-Resolution') || quality} · up to 4 fps · frame age ${age} ms`);
      } catch (error) {
        if (alive) {
          setFrame('');
          setStatus(error.name === 'AbortError' ? 'Camera timed out. Reconnecting…' : error.message);
          delay = 1000;
        }
      } finally {
        clearTimeout(timeout);
        if (alive) timer = setTimeout(load, delay);
      }
    };
    load();
    return () => {
      alive = false; clearTimeout(timer); controller?.abort();
      if (currentUrl) URL.revokeObjectURL(currentUrl);
      if (previousUrl) URL.revokeObjectURL(previousUrl);
    };
  }, [enabled, camera, quality, environment]);

  const fullscreen = async () => {
    try {
      if (document.fullscreenElement) await document.exitFullscreen();
      else await container.current?.requestFullscreen();
    } catch (_) { setStatus('Fullscreen is unavailable in this browser.'); }
  };

  return <section className="hil-card hil-live-view" ref={container} style={{'--hil-camera-height': size.height ? `${size.height}px` : undefined}} aria-label={`${label} live view`}>
    <div className="hil-card-heading"><div><h2>{label}</h2></div>
      <div className="hil-actions"><button onClick={() => { setEnabled((value) => !value); setFrame(''); }}>{enabled ? 'Pause video' : 'Resume video'}</button>
        <button onClick={fullscreen}>Fullscreen</button><button aria-label={`Close ${label}`} onClick={onClose}>Close</button></div>
    </div>
    <div className="hil-camera-viewport">
    {cropped && showCrop && enabled && frame ? <HilSessionCropFrame frame={frame} crop={sessionConfig.camera_crops?.[camera]} size={imageSize} label={label} /> : camera === 'spectator' && enabled && frame ? <HilOverviewCrop frame={frame} label={label} simulationLabel={environment === 'mujoco' ? 'MuJoCo' : 'Isaac'} /> : <div className="hil-video-frame">
      {enabled && frame ? <img src={frame} alt={`Live ${environment === 'mujoco' ? 'MuJoCo' : 'Isaac'} simulation · ${label}`} /> : <p>{enabled ? status : 'Video paused'}</p>}
    </div>}
    </div>
    <div className="hil-camera-options">
    <label className="hil-quality">Quality<select aria-label={`${label} quality`} value={quality} onChange={event => { const value = event.target.value; setQuality(value); try { localStorage.setItem(`hil-camera-quality-${camera}`, value); } catch (_) {} }}>
      <option value="low">Low · {camera === 'spectator' ? '320×240' : '320×180'}</option>
      <option value="standard">Standard · {camera === 'spectator' ? '640×480' : '480×270'}</option>
      <option value="high">High · {camera === 'spectator' ? '960×720' : '640×360'}</option>
      {camera === 'spectator' && <option value="ultra">Ultra · 1920×1440 · JPEG 98</option>}
    </select></label>
    <details className="hil-camera-size"><summary>View size</summary>
      <label>Width <output>{size.width ? `${size.width} px` : 'Auto'}</output>
        <input aria-label={`${label} view width`} type="range" min="240" max="1600" step="20" value={size.width || 360}
          onChange={event => onResize({...size, width: Number(event.target.value)})} /></label>
      <label>Height <output>{size.height ? `${size.height} px` : 'Auto'}</output>
        <input aria-label={`${label} view height`} type="range" min="180" max="1000" step="20" value={size.height || 260}
          onChange={event => onResize({...size, height: Number(event.target.value)})} /></label>
      <button onClick={() => onResize({})}>Reset view size</button>
    </details>
    {cropped && <div className="hil-session-crop-controls">
      <button type="button" aria-pressed={showCrop} onClick={() => setShowCrop(value => !value)}>{showCrop ? 'Show full image' : 'Show session crop'}</button>
      <p className="hil-help">{showCrop ? `Session crop · ${imageSize}×${imageSize}` : 'Full camera image'} · {selectedFor.length ? selectedFor.join(' · ') : 'Not selected as a learning input'}</p>
    </div>}
    </div>
    <p className="hil-help" aria-live="off">{enabled ? status : 'Video paused; simulation controls are unchanged.'}</p>
  </section>;
}

const CAMERAS = [['spectator', 'Overview'], ['head', 'Head camera'], ['left_wrist', 'Left eye-in-hand'], ['right_wrist', 'Right eye-in-hand']];
export default function HilIsaacView({ environment = 'isaac_sim', sessionConfig, sessionMode, sessionId = '' }) {
  const [individual, setIndividual] = useState(false);
  const [uniformHeight, setUniformHeight] = useState(() => {
    try {const n=Number(localStorage.getItem('hil-camera-uniform-height')); return n>=180 && n<=1000 ? n : 320;} catch {return 320;}
  });
  const [sizes, setSizes] = useState(() => {
    try {
      const saved = JSON.parse(localStorage.getItem('hil-camera-sizes') || '{}');
      return Object.fromEntries(CAMERAS.map(([camera]) => {
        const value = saved?.[camera];
        return [camera, {
          width: Number.isFinite(value?.width) && value.width >= 240 && value.width <= 1600 ? value.width : undefined,
          height: Number.isFinite(value?.height) && value.height >= 180 && value.height <= 1000 ? value.height : undefined,
        }];
      }));
    } catch { return {}; }
  });
  const resize = (camera, size) => {
    const next = {...sizes, [camera]: size}; setSizes(next);
    try { localStorage.setItem('hil-camera-sizes', JSON.stringify(next)); } catch {}
  };
  const [closed, setClosed] = useState(() => {
    try { const value = JSON.parse(localStorage.getItem('hil-closed-cameras') || '[]'); return Array.isArray(value) ? value : []; } catch { return []; }
  });
  const updateClosed = value => { setClosed(value); try { localStorage.setItem('hil-closed-cameras', JSON.stringify(value)); } catch {} };
  return <section aria-label={`${environment === 'mujoco' ? 'MuJoCo' : 'Isaac Sim'} live views`}><p className="hil-help">Close unused views to stop video requests. Overview rendering resources are released after all viewers stop requesting it. Robot cameras remain available for learning. View size adjusts the display only; Quality controls preview resolution.</p>
    {sessionConfig && <p className="hil-help">Robot camera previews use this session's saved crop and training resolution. These are cropped live JPEG previews, not the exact synchronized frames stored in the dataset. Overview remains a full scene view.</p>}
    <div className="hil-camera-layout-controls">
      <label><input type="checkbox" checked={individual} onChange={e => setIndividual(e.target.checked)} />Individual camera sizes</label>
      {!individual && <label>All viewport heights · {uniformHeight} px<input aria-label="All camera viewport heights" type="range" min="180" max="1000" step="20" value={uniformHeight}
        onChange={e => {const value=Number(e.target.value); setUniformHeight(value); try {localStorage.setItem('hil-camera-uniform-height',String(value));} catch {}}} /></label>}
    </div>
    <div className="hil-actions">{CAMERAS.filter(([camera]) => closed.includes(camera)).map(([camera,label]) => <button key={camera} onClick={() => updateClosed(closed.filter(value => value !== camera))}>Open {label}</button>)}</div>
    <div className={`hil-camera-grid${individual ? '' : ' hil-camera-grid-uniform'}`} style={{'--hil-uniform-height':`${uniformHeight}px`,gridTemplateColumns:CAMERAS.filter(([camera]) => !closed.includes(camera)).map(([camera]) => individual && sizes[camera]?.width ? `${sizes[camera].width}px` : 'minmax(240px, 1fr)').join(' ')}}>
      {CAMERAS.filter(([camera]) => !closed.includes(camera)).map(([camera, label]) => <CameraView key={`${environment}-${sessionId}-${camera}`} environment={environment} sessionConfig={sessionConfig} sessionMode={sessionMode} camera={camera} label={label} size={individual ? sizes[camera] : {height:uniformHeight}} onResize={size => {setIndividual(true); resize(camera, size);}} onClose={() => updateClosed([...closed,camera])} />)}
    </div>
  </section>;
}
