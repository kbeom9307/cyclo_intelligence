import { useEffect, useRef, useState } from 'react';
import { sendHilClassifierLabel } from '../utils/hilserlApi';

// A short server lease prevents a stuck success label after browser disconnect.
export default function useHilCaptureKeyboard(session, enabled, onError, toggle = false) {
  const [successActive, setSuccessActive] = useState(false);
  const latest = useRef();
  latest.current = { session, onError };
  const id = session?.id;
  const revision = session?.revision;
  const classifier = session?.mode === 'classifier';
  useEffect(() => {
    setSuccessActive(false);
    if (!enabled || !classifier) return undefined;
    let held = false, alive = true, pending = false, releasePending = false;
    const send = async () => {
      if (pending) return;
      pending = true;
      const success = alive && held;
      releasePending = false;
      try { await sendHilClassifierLabel({ id, revision }, success); }
      catch (error) { held = false; if (alive) { setSuccessActive(false); latest.current.onError(error.message); } }
      finally { pending = false; if (releasePending) send(); }
    };
    const release = () => { if (held || pending) { held = false; if (alive) setSuccessActive(false); releasePending = true; send(); } };
    const editing = target => target?.isContentEditable || /^(INPUT|TEXTAREA|SELECT)$/.test(target?.tagName || '');
    const down = event => {
      if (editing(event.target) || event.ctrlKey || event.metaKey || event.altKey) { release(); return; }
      if (event.code === 'Escape') { release(); return; }
      if (event.code !== 'Space') return;
      event.preventDefault();
      if (event.repeat) return;
      if (toggle && held) { release(); return; }
      if (!held) { held = true; setSuccessActive(true); if (pending) releasePending = true; else send(); }
    };
    const up = event => { if (event.code === 'Space') { event.preventDefault(); if (!toggle) release(); } };
    const focus = event => { if (editing(event.target)) release(); };
    const visibility = () => { if (document.hidden) release(); };
    window.addEventListener('keydown', down); window.addEventListener('keyup', up);
    window.addEventListener('blur', release); document.addEventListener('focusin', focus); document.addEventListener('visibilitychange', visibility);
    const timer = setInterval(() => { if (held) send(); }, 100);
    return () => {
      alive = false; release(); clearInterval(timer);
      window.removeEventListener('keydown', down); window.removeEventListener('keyup', up);
      window.removeEventListener('blur', release); document.removeEventListener('focusin', focus); document.removeEventListener('visibilitychange', visibility);
    };
  }, [id, revision, enabled, classifier, toggle]);
  return enabled && classifier && successActive;
}
