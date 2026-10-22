import React, {useEffect, useId, useRef, useState} from 'react';
import {createPortal} from 'react-dom';
import HilObservationCrop from './HilObservationCrop';

function CropDialog({environment, crops, size, onApply, onClose}) {
  const dialog = useRef(null);
  const heading = useId();
  const [edited, setEdited] = useState(() => Object.fromEntries(Object.entries(crops || {}).map(([camera, roi]) => [camera, [...roi]])));
  useEffect(() => {
    const element = dialog.current;
    element.showModal();
    return () => element.close();
  }, []);
  const valid = Object.values(edited).every(roi => Array.isArray(roi) && roi.length === 4 &&
    roi.every(value => Number.isFinite(value) && value >= 0 && value <= 1) && roi[0] < roi[2] && roi[1] < roi[3]);
  return createPortal(<dialog className="hil-crop-dialog hil-page" ref={dialog} aria-labelledby={heading}
    onCancel={event => {event.preventDefault(); onClose();}} onKeyDown={event => event.stopPropagation()}>
    <header className="hil-card-heading"><h2 id={heading}>Observation crop editor</h2>
      <button type="button" aria-label="Close crop editor" onClick={onClose}>Close</button></header>
    <HilObservationCrop environment={environment} crops={edited} size={size} onChange={setEdited} />
    {!valid && <p role="alert">Select a valid rectangle: Left must be less than Right, and Top less than Bottom, within 0–1.</p>}
    <p className="hil-help">Apply updates the experiment settings form. Save configuration to use the crop in a new session.</p>
    <footer className="hil-actions"><button type="button" onClick={onClose}>Cancel</button>
      <button type="button" className="hil-primary" disabled={!valid} onClick={() => {onApply(edited); onClose();}}>Apply crop</button></footer>
  </dialog>, document.body);
}

export default function HilObservationCropDialog({environment, crops, size, disabled, onChange}) {
  const [open, setOpen] = useState(false);
  return <div className="hil-crop-launcher">
    <button type="button" disabled={disabled} onClick={() => setOpen(true)}>Edit observation crop</button>
    <span className="hil-help">Select training image regions · {size}×{size}</span>
    {open && <CropDialog environment={environment} crops={crops} size={size} onApply={onChange} onClose={() => setOpen(false)} />}
  </div>;
}
