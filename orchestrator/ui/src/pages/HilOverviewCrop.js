import React, { useId, useRef, useState } from 'react';

export const FULL_VIEW = { x: 0, y: 0, width: 4, height: 3 };

// Map screen coordinates through SVG's aspect-preserving letterbox into source space.
export function overviewPoint(clientX, clientY, rect, view) {
  const scale = Math.min(rect.width / view.width, rect.height / view.height);
  if (!Number.isFinite(scale) || scale <= 0) return null;
  const x = (clientX - rect.left - (rect.width - view.width * scale) / 2) / scale;
  const y = (clientY - rect.top - (rect.height - view.height * scale) / 2) / scale;
  return { x: view.x + Math.max(0, Math.min(view.width, x)), y: view.y + Math.max(0, Math.min(view.height, y)) };
}
export function cropRectangle(a, b) {
  return { x: Math.min(a.x, b.x), y: Math.min(a.y, b.y), width: Math.abs(a.x - b.x), height: Math.abs(a.y - b.y) };
}

export default function HilOverviewCrop({ frame, label, simulationLabel = 'Isaac' }) {
  const clipId = useId();
  const [view, setView] = useState(FULL_VIEW);
  const [selection, setSelection] = useState(null);
  const drag = useRef(null);
  const point = event => overviewPoint(event.clientX, event.clientY, event.currentTarget.getBoundingClientRect(), view);
  const cancel = () => { drag.current = null; setSelection(null); };
  const finish = event => {
    if (!drag.current || event.pointerId !== drag.current.id) return;
    const end = point(event);
    const rectangle = end && cropRectangle(drag.current.start, end);
    // Reject clicks and tiny selections; cap zoom at 20× on either source axis.
    if (rectangle && rectangle.width >= Math.max(.2, view.width * .03) && rectangle.height >= Math.max(.15, view.height * .03)) setView(rectangle);
    cancel();
  };
  return <>
    <div className="hil-overview-tools"><button onClick={() => { cancel(); setView(FULL_VIEW); }}>Reset crop</button>
      <span>{Math.max(4 / view.width, 3 / view.height).toFixed(1)}× · Drag an area to zoom</span></div>
    <div className="hil-video-frame hil-overview-frame">
      <svg aria-label="Overview crop area" viewBox={`${view.x} ${view.y} ${view.width} ${view.height}`} preserveAspectRatio="xMidYMid meet"
        onPointerDown={event => {
          if (event.button !== 0 || drag.current) return;
          const start = point(event);
          if (!start) return;
          event.preventDefault();
          event.currentTarget.setPointerCapture?.(event.pointerId);
          drag.current = { start, id: event.pointerId };
          setSelection(null);
        }}
        onPointerMove={event => {
          if (!drag.current || event.pointerId !== drag.current.id) return;
          const end = point(event);
          if (end) setSelection(cropRectangle(drag.current.start, end));
        }} onPointerUp={finish} onPointerCancel={cancel} onLostPointerCapture={cancel}>
        <defs><clipPath id={clipId}><rect {...view} /></clipPath></defs>
        <foreignObject clipPath={`url(#${clipId})`} x="0" y="0" width="4" height="3"><img src={frame} alt={`Live ${simulationLabel} simulation · ${label}`} draggable={false} /></foreignObject>
        {selection && <rect {...selection} className="hil-crop-selection" />}
      </svg>
    </div>
  </>;
}
