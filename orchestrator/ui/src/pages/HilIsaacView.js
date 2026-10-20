import React, { useEffect, useRef, useState } from 'react';
import HilOverviewCrop from './HilOverviewCrop';

/** Latest-frame viewer: one request in flight, bounded memory, no physics commands. */
function CameraView({ camera, label, onClose }) {
  const [enabled, setEnabled] = useState(true);
  const [quality, setQuality] = useState(() => {
    try { const saved = localStorage.getItem(`hil-camera-quality-${camera}`); return (camera === 'spectator' ? ['low','standard','high','ultra'] : ['low','standard','high']).includes(saved) ? saved : 'standard'; } catch (_) { return 'standard'; }
  });
  const [frame, setFrame] = useState('');
  const [status, setStatus] = useState('Connecting to Isaac camera…');
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
        const response = await fetch(`/api/hilserl/isaac/frame?camera=${camera}&quality=${quality}`,  { signal: controller.signal, cache: 'no-store' });
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
  }, [enabled, camera, quality]);

  const fullscreen = async () => {
    try {
      if (document.fullscreenElement) await document.exitFullscreen();
      else await container.current?.requestFullscreen();
    } catch (_) { setStatus('Fullscreen is unavailable in this browser.'); }
  };

  return <section className="hil-card hil-live-view" ref={container} aria-label={`${label} live view`}>
    <div className="hil-card-heading"><div><h2>{label}</h2></div>
      <div className="hil-actions"><button onClick={() => { setEnabled((value) => !value); setFrame(''); }}>{enabled ? 'Pause video' : 'Resume video'}</button>
        <button onClick={fullscreen}>Fullscreen</button><button aria-label={`Close ${label}`} onClick={onClose}>Close</button></div>
    </div>
    <label className="hil-quality">Quality<select aria-label={`${label} quality`} value={quality} onChange={event => { const value = event.target.value; setQuality(value); try { localStorage.setItem(`hil-camera-quality-${camera}`, value); } catch (_) {} }}>
      <option value="low">Low · {camera === 'spectator' ? '320×240' : '320×180'}</option>
      <option value="standard">Standard · {camera === 'spectator' ? '640×480' : '480×270'}</option>
      <option value="high">High · {camera === 'spectator' ? '960×720' : '640×360'}</option>
      {camera === 'spectator' && <option value="ultra">Ultra · 1920×1440 · JPEG 98</option>}
    </select></label>
    {camera === 'spectator' && enabled && frame ? <HilOverviewCrop frame={frame} label={label} /> : <div className="hil-video-frame">
      {enabled && frame ? <img src={frame} alt={`Live Isaac simulation · ${label}`} /> : <p>{enabled ? status : 'Video paused'}</p>}
    </div>}
    <p className="hil-help" aria-live="off">{enabled ? status : 'Video paused; simulation controls are unchanged.'}</p>
  </section>;
}

const CAMERAS = [['spectator', 'Overview'], ['head', 'Head camera'], ['left_wrist', 'Left eye-in-hand'], ['right_wrist', 'Right eye-in-hand']];
export default function HilIsaacView() {
  const [closed, setClosed] = useState(() => {
    try { const value = JSON.parse(localStorage.getItem('hil-closed-cameras') || '[]'); return Array.isArray(value) ? value : []; } catch { return []; }
  });
  const updateClosed = value => { setClosed(value); try { localStorage.setItem('hil-closed-cameras', JSON.stringify(value)); } catch {} };
  return <section aria-label="Isaac Sim live views"><p className="hil-help">Close unused views to stop video requests. Overview rendering resources are released after all viewers stop requesting it. Robot cameras remain available for learning.</p>
    <div className="hil-actions">{CAMERAS.filter(([camera]) => closed.includes(camera)).map(([camera,label]) => <button key={camera} onClick={() => updateClosed(closed.filter(value => value !== camera))}>Open {label}</button>)}</div>
    <div className="hil-camera-grid" style={{gridTemplateColumns:`repeat(${Math.max(1, CAMERAS.length-closed.filter(value=>CAMERAS.some(([camera])=>camera===value)).length)}, minmax(240px, 1fr))`}}>
      {CAMERAS.filter(([camera]) => !closed.includes(camera)).map(([camera, label]) => <CameraView key={camera} camera={camera} label={label} onClose={() => updateClosed([...closed,camera])} />)}
    </div>
  </section>;
}
