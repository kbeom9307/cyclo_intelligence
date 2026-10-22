import React, {useEffect, useRef, useState} from 'react';

const CAMERAS = [['head', 'Head'], ['left_wrist', 'Left wrist'], ['right_wrist', 'Right wrist']];
const FULL = [0, 0, 1, 1];
export function cropPoint(x, y, box) {
  if (!box.width || !box.height) return null;
  return [Math.max(0, Math.min(1, (x-box.left)/box.width)), Math.max(0, Math.min(1, (y-box.top)/box.height))];
}
export function selectedCrop(a, b) {
  return [Math.min(a[0], b[0]), Math.min(a[1], b[1]), Math.max(a[0], b[0]), Math.max(a[1], b[1])].map(v => Math.round(v*10000)/10000);
}

export default function HilObservationCrop({environment, crops = {}, size = 128, disabled, onChange}) {
  const [camera, setCamera] = useState('head');
  const [refresh, setRefresh] = useState(0);
  const [frame, setFrame] = useState(null);
  const [status, setStatus] = useState('');
  const [selection, setSelection] = useState(null);
  const drag = useRef(null);
  useEffect(() => {
    let alive = true, url;
    const controller = new AbortController();
    setFrame(null); setSelection(null); drag.current = null;
    if (!['isaac_sim','mujoco'].includes(environment)) { setStatus('Select a simulator to load camera images.'); return undefined; }
    setStatus('Loading camera snapshot…');
    const timeout = setTimeout(() => controller.abort(), 5000);
    (async () => {
      try {
        const response = await fetch(`/api/hilserl/${environment === 'mujoco' ? 'mujoco' : 'isaac'}/frame?camera=${camera}&quality=high`, {signal:controller.signal, cache:'no-store'});
        if (!response.ok || !response.headers.get('content-type')?.startsWith('image/jpeg')) throw new Error('Camera unavailable. Start the simulator bridge, then refresh the image.');
        const blob = await response.blob();
        if (!alive) return;
        url = URL.createObjectURL(blob); setFrame({camera, environment, url});
        setStatus('Snapshot · drag on the full image to select a training crop. Refresh image to capture a newer frame.');
      } catch (e) { if (alive) setStatus(e.name === 'AbortError' ? 'Camera timed out. Refresh image to retry.' : e.message); }
      finally { clearTimeout(timeout); }
    })();
    return () => { alive = false; clearTimeout(timeout); controller.abort(); if (url) URL.revokeObjectURL(url); };
  }, [camera, environment, refresh]);
  const image = frame?.camera === camera && frame?.environment === environment ? frame.url : null;
  const saved = crops[camera] || FULL;
  const roi = selection || saved;
  const valid = roi.every(Number.isFinite) && roi[2] > roi[0] && roi[3] > roi[1];
  const cancel = () => { drag.current = null; setSelection(null); };
  const point = e => cropPoint(e.clientX,e.clientY,e.currentTarget.getBoundingClientRect());
  const commit = e => {
    if (!drag.current || e.pointerId !== drag.current.id) return;
    const end = point(e), value = end && selectedCrop(drag.current.start,end);
    if (!disabled && value && value[2]-value[0] >= .01 && value[3]-value[1] >= .01) onChange({...crops,[camera]:value});
    cancel();
  };
  return <fieldset disabled={disabled} className="hil-observation-crop">
    <legend>Observation crop · before resizing to {size}×{size}</legend>
    <div className="hil-actions"><label>Crop camera <select aria-label="Crop camera" value={camera} onChange={e => {cancel(); setCamera(e.target.value);}}>
      {CAMERAS.map(([value,label]) => <option key={value} value={value}>{label}</option>)}
    </select></label><button type="button" onClick={() => setRefresh(v => v+1)}>Refresh image</button>
      <button type="button" onClick={() => {cancel(); const next={...crops}; delete next[camera]; onChange(next);}}>Use full image</button></div>
    <p className="hil-help" role="status">{status}</p>
    {image && <div className="hil-crop-editor-grid"><div>
      <p>Full camera image · draw a new rectangle to change the crop</p>
      <div className="hil-crop-source"><img src={image} alt={`${camera} original camera`} draggable={false} />
        <svg aria-label={`${camera} training crop area`} viewBox="0 0 1 1" preserveAspectRatio="none"
          onPointerDown={e => { if (disabled || e.button !== 0) return; const start=point(e); if (!start) return; e.preventDefault(); drag.current={start,id:e.pointerId}; e.currentTarget.setPointerCapture?.(e.pointerId); }}
          onPointerMove={e => {if (drag.current && e.pointerId === drag.current.id) {const end=point(e); if (end) setSelection(selectedCrop(drag.current.start,end));}}}
          onPointerUp={commit} onPointerCancel={cancel} onLostPointerCapture={cancel}>
          {valid && <rect x={roi[0]} y={roi[1]} width={roi[2]-roi[0]} height={roi[3]-roi[1]} className="hil-crop-selection" />}
        </svg>
      </div></div><div><p>Training image preview · {size}×{size}</p>
        {valid && <svg className="hil-crop-result" aria-label="Resized training crop preview" viewBox={`${roi[0]} ${roi[1]} ${roi[2]-roi[0]} ${roi[3]-roi[1]}`} preserveAspectRatio="none">
          <image href={image} x="0" y="0" width="1" height="1" preserveAspectRatio="none" />
        </svg>}
      </div></div>}
    <details><summary>Advanced crop coordinates</summary><div className="hil-fields">
      {['Left','Top','Right','Bottom'].map((edge,i) => <label key={edge}>{edge} (0–1)<input type="number" min="0" max="1" step=".01" aria-label={`${camera} crop ${edge}`} value={saved[i]}
        onChange={e => {const next=[...saved]; next[i]=Number(e.target.value); onChange({...crops,[camera]:next});}} /></label>)}
    </div></details>
    <p className="hil-help">0,0 → 1,1 selects the entire image, not a center crop. The selected rectangle is resized to a square for both policy and classifier observations. Save configuration and start a new session to apply. Existing artifacts require matching crops.</p>
  </fieldset>;
}
