import { useEffect, useState } from 'react';
import { sendHilAction } from '../utils/hilserlApi';

// Manual deltas use TCP coordinates; the Actor maps them into the policy frame.
const keys = { KeyW: [0, 1], KeyS: [0, -1], KeyA: [1, 1], KeyD: [1, -1],
  KeyR: [2, 1], KeyF: [2, -1], KeyQ: [3, 1], KeyE: [3, -1],
  ArrowUp: [4, 1], ArrowDown: [4, -1], ArrowLeft: [5, 1], ArrowRight: [5, -1],
 };
const editing = (target) => target?.isContentEditable || /^(INPUT|TEXTAREA|SELECT)$/.test(target?.tagName || '');

export default function useHilKeyboard(session, enabled, onError) {
  const [pressedKeys, setPressedKeys] = useState([]);
  const id = session?.id;
  const revision = session?.revision;
  const dimensions = ['isaac_sim', 'mujoco'].includes(session?.config.environment) ? 6 : 3;
  useEffect(() => {
    setPressedKeys([]);
    if (!enabled) return undefined;
    const held = new Set();
    let alive = true, pending = false, wasMoving = false, sendStop = false;
    const send = async () => {
      if (pending) return;
      const value = Array(dimensions).fill(0);
      if (alive) held.forEach(code => { const [axis, sign] = keys[code]; if (axis < dimensions) value[axis] += sign; });
      const moving = value.some(Boolean);
      if (!moving && !wasMoving && !sendStop) return;
      pending = true; wasMoving = moving; sendStop = false;
      try { await sendHilAction({ id, revision }, value.map(v => Math.max(-1, Math.min(1, v)))); }
      catch (error) { held.clear(); if (alive) { setPressedKeys([]); onError(error.message); } }
      finally { pending = false; if (sendStop) send(); }
    };
    const stop = () => { held.clear(); if (alive) setPressedKeys([]); if (wasMoving || pending) { sendStop = true; send(); } };
    const down = event => {
      if (editing(event.target) || event.ctrlKey || event.metaKey || event.altKey) { stop(); return; }
      if (event.code === 'Space' && session?.mode === 'classifier') return;
      if (event.code === 'Space' || event.code === 'Escape') { event.preventDefault(); stop(); return; }
      if (event.repeat) return;
      if (!keys[event.code] || keys[event.code][0] >= dimensions) return;
      event.preventDefault(); held.add(event.code); setPressedKeys([...held]); send();
    };
    const up = event => { if (held.delete(event.code)) { setPressedKeys([...held]); event.preventDefault(); sendStop = true; send(); } };
    const focus = event => { if (editing(event.target)) stop(); };
    const visibility = () => { if (document.hidden) stop(); };
    window.addEventListener('keydown', down); window.addEventListener('keyup', up);
    window.addEventListener('blur', stop); document.addEventListener('focusin', focus); document.addEventListener('visibilitychange', visibility);
    const timer = setInterval(send, 100);
    return () => {
      alive = false; stop(); clearInterval(timer);
      window.removeEventListener('keydown', down); window.removeEventListener('keyup', up);
      window.removeEventListener('blur', stop); document.removeEventListener('focusin', focus); document.removeEventListener('visibilitychange', visibility);
    };
  }, [id, revision, dimensions, enabled, onError, session?.mode]);
  return enabled ? pressedKeys : [];
}
