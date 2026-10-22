import React, { useEffect, useRef, useState } from 'react';

const EMPTY_METRICS = {};

export function episodeResult(row, config) {
  if (!row.terminal && !row.truncated) return null;
  const info = row.environment_info || {};
  const success = row.label_only ? row.reward > 0 : config.reward_source === 'classifier'
    ? info.classifier_success === true : typeof info.is_success === 'boolean'
      ? info.is_success : Boolean(row.terminal && row.reward > 0);
  return {episode: row.episode, outcome: success ? 'Success' : row.truncated ? 'Truncated' : 'Failure',
    success, steps: info.episode_step ?? null, ended_at: row.timestamp ?? null,
    source: row.label_only ? 'Human label' : config.reward_source === 'classifier' ? 'Classifier' : 'Environment'};
}

export function accumulateEpisode(previous, row, config) {
  const value = previous?.episode === row.episode ? {...previous} : {
    episode: row.episode, return: 0, steps: 0, intervention_count: 0, intervention_steps: 0, wasHuman: false, started_at: row.timestamp ?? null,
  };
  value.return += Number(row.reward) || 0;
  if (!row.label_only) {
    value.steps += 1;
    if (row.human) {
      value.intervention_steps += 1;
      if (!value.wasHuman) value.intervention_count += 1;
    }
    value.wasHuman = Boolean(row.human);
  }
  const terminal = episodeResult(row, config);
  return {...value, ...(terminal || {}), steps: value.steps,
    recorded_seconds: row.timestamp != null && value.started_at != null ? Math.max(0, row.timestamp - value.started_at) : value.recorded_seconds ?? null};
}

const number = value => Number.isFinite(value) ? value.toLocaleString(undefined, {maximumFractionDigits: 4}) : '—';
function MetricTiles({items}) {
  return <dl className="hil-role-metrics">{items.map(([label, value]) => <div key={label}><dt>{label}</dt><dd>{value}</dd></div>)}</dl>;
}
function MetricChart({title, rows, metric, color = '#38bdf8', xLabel = 'Learner step'}) {
  const values = rows.map(row => ({x: row.x, y: row[metric]})).filter(p => Number.isFinite(p.y));
  const min = Math.min(...values.map(p => p.y)), max = Math.max(...values.map(p => p.y));
  const first = values[0]?.x ?? 0, last = values.at(-1)?.x ?? 0;
  const points = values.map(p => `${12 + 276 * (p.x - first) / Math.max(1, last - first)},${72 - 60 * (p.y - min) / (max - min || 1)}`).join(' ');
  return <figure className="hil-training-chart"><figcaption>{title}<strong>{number(values.at(-1)?.y)}</strong></figcaption>
    {values.length ? <><svg viewBox="0 0 300 84" role="img" aria-label={`${title}: ${values.length} observations`}><line x1="12" x2="288" y1="72" y2="72" stroke="currentColor" opacity=".2"/><polyline fill="none" stroke={color} strokeWidth="2" points={points}/>{values.length === 1 && <circle cx="12" cy="72" r="3" fill={color}/>}</svg>
      <small>Range {number(min)} – {number(max)} · {xLabel} {first} – {last}</small></> : <p className="hil-help">No measurements yet</p>}
  </figure>;
}

export function elapsedTimes(session, now) {
  let started = null, activeSince = null, active = 0, ended = null;
  for (const event of session.events || []) {
    const at = Date.parse(event.at) / 1000;
    if (!Number.isFinite(at)) continue;
    if (event.action === 'start' && started === null) started = at;
    if (['start', 'resume'].includes(event.action) && activeSince === null) activeSince = at;
    if (['pause', 'stop', 'success', 'failure', 'truncate', 'collection_boundary'].includes(event.action)) {
      if (activeSince !== null) active += Math.max(0, at - activeSince);
      activeSince = null;
    }
    if (event.action === 'stop') ended = at;
  }
  const end = ended ?? (session.state === 'stopped' ? (session.runtime?.heartbeat ?? now) : now);
  if (activeSince !== null) active += Math.max(0, end - activeSince);
  return {wall: started === null ? null : Math.max(0, end - started), active: started === null ? (session.resume?.active_seconds || null) : active + (session.resume?.active_seconds || 0)};
}

const duration = seconds => seconds == null ? '—' : `${Math.floor(seconds / 3600)}h ${Math.floor(seconds % 3600 / 60)}m ${Math.floor(seconds % 60)}s`;
const percent = value => value == null ? '—' : `${(100 * value).toFixed(1)}%`;

// Read only newly appended bytes, in bounded chunks. This also works for runtimes
// already running before the UI deployment and requires no API/runtime restart.
export function useEpisodeHistory(session) {
  const [result, setResult] = useState({episodes: [], current: null, loading: true, error: null});
  const config = useRef(session.config); config.current = session.config;
  useEffect(() => {
    let cancelled = false, timer, offset = 0, pending = '', current = null;
    let decoder = new TextDecoder();
    const controller = new AbortController();
    const episodes = new Map();
    setResult({episodes: [], current: null, loading: true, error: null});
    if (!/^[a-zA-Z0-9_-]+$/.test(session.id)) return () => controller.abort();
    const poll = async () => {
      try {
        let more = true;
        while (more && !cancelled) {
          const response = await fetch(`/files/workspace/hilserl/${encodeURIComponent(session.id)}/transitions.jsonl`, {
            headers: {Range: `bytes=${offset}-${offset + 1048575}`}, cache: 'no-store', signal: controller.signal,
          });
          if (response.status === 404) { more = false; break; }
          if (response.status === 416) {
            const total = Number(response.headers.get('Content-Range')?.split('/')[1]);
            if (Number.isFinite(total) && total < offset) {
              offset = 0; pending = ''; current = null; decoder = new TextDecoder(); episodes.clear(); continue;
            }
            more = false; break;
          }
          if (response.status !== 206) throw new Error(`Episode log range request failed (${response.status})`);
          const range = /^bytes (\d+)-(\d+)\/(\d+)$/.exec(response.headers.get('Content-Range') || '');
          if (!range || Number(range[1]) !== offset) throw new Error('Unexpected episode log range');
          const bytes = await response.arrayBuffer();
          const lines = (pending + decoder.decode(bytes, {stream: true})).split('\n');
          pending = lines.pop();
          for (const line of lines) {
            if (!line.trim()) continue;
            const row = JSON.parse(line);
            current = accumulateEpisode(current, row, config.current);
            if (row.terminal || row.truncated) episodes.set(current.episode, current);
          }
          while (episodes.size > 100) episodes.delete(episodes.keys().next().value);
          offset = Number(range[2]) + 1; more = offset < Number(range[3]);
          if (!cancelled) setResult({episodes: [...episodes.values()], current, loading: more, error: null});
        }
        if (!cancelled) setResult({episodes: [...episodes.values()], current, loading: false, error: null});
      } catch (error) {
        if (!cancelled) setResult({episodes: [...episodes.values()], current, loading: false, error: error.message});
      }
      if (!cancelled) timer = setTimeout(poll, 2000);
    };
    poll();
    return () => { cancelled = true; clearTimeout(timer); controller.abort(); };
  }, [session.id]);
  return result;
}

export default function HilTrainingMonitor({session}) {
  const [now, setNow] = useState(Date.now() / 1000);
  const [windowSize, setWindowSize] = useState(10);
  const [series, setSeries] = useState([]);
  const history = useEpisodeHistory(session);
  const observation = useRef(null);
  const observedSince = useRef(Date.now() / 1000);
  const [rates, setRates] = useState({actor: null, learner: null, lastLearnerUpdate: null});
  const runtime = session.runtime || {}, metrics = runtime.metrics || session.metrics || EMPTY_METRICS;
  const learner = runtime.compute?.learner || {};
  useEffect(() => { const timer = setInterval(() => setNow(Date.now() / 1000), 1000); return () => clearInterval(timer); }, []);
  useEffect(() => {
    const value = {id: session.id, time: runtime.heartbeat, actor: metrics.environment_steps, learner: metrics.learner_steps};
    const previous = observation.current;
    if (previous?.id === value.id && value.time > previous.time) {
      const dt = value.time - previous.time;
      setRates(old => ({actor: Math.max(0, (value.actor - previous.actor) / dt),
        learner: Math.max(0, (value.learner - previous.learner) / dt),
        lastLearnerUpdate: value.learner > previous.learner ? value.time : old.lastLearnerUpdate}));
    } else if (previous?.id !== value.id) {
      observedSince.current = Date.now() / 1000;
      setRates({actor: null, learner: null, lastLearnerUpdate: null});
    }
    observation.current = value;
    if (Number.isFinite(value.learner)) setSeries(old => {
      const base = previous?.id === value.id ? old : [];
      if (base.at(-1)?.x === value.learner) return base;
      return [...base, {x: value.learner, ...metrics}].slice(-180);
    });
  }, [session.id, runtime.heartbeat, metrics.environment_steps, metrics.learner_steps, metrics]);
  const elapsed = elapsedTimes(session, now);
  const recent = history.episodes.slice(-windowSize), last = history.episodes.at(-1);
  const recentSuccesses = recent.filter(item => item.success).length;
  const completed = metrics.completed_episodes ?? 0;
  const successes = metrics.success_rate == null ? null : Math.round(metrics.success_rate * completed);
  const stale = runtime.heartbeat != null && now - runtime.heartbeat > 10;
  const state = session.runtime_error ? 'Error' : session.state === 'created' ? 'Not started' : session.state === 'stopped' ? 'Stopped'
    : session.state === 'paused' ? 'Paused' : !runtime.heartbeat ? 'Initializing' : stale ? 'Heartbeat delayed' : null;
  const actorState = state || (runtime.last_action_source === 'human' ? 'Human intervention' : runtime.initializing ? 'Resetting episode' : 'Running policy');
  const learnerState = state || (metrics.replay_ready === false ? 'Waiting for replay / demo data'
    : now - (rates.lastLearnerUpdate ?? observedSince.current) > 15 ? 'No recent update observed' : rates.lastLearnerUpdate == null ? 'Observing updates' : 'Updating');
  const rate = value => Number.isFinite(value) ? `${value.toFixed(1)}/s` : '—';
  return <section className="hil-training-monitor hil-card" aria-label="Training monitor">
    <h2>Training monitor</h2>
    <div className="hil-metrics">{[
      ['Elapsed since Start', duration(elapsed.wall)], ['Running time (excluding pauses)', duration(elapsed.active)],
      ['Current episode', runtime.episode ?? session.episode], ['Completed episodes', completed],
      ['Cumulative success rate', `${percent(metrics.success_rate)}${successes == null ? '' : ` (${successes}/${completed})`}`],
      ['Last completed episode', history.loading ? 'Loading history…' : last ? `#${last.episode} · ${last.outcome}` : '—'],
    ].map(([label, value]) => <div key={label}><span>{label}</span><strong>{value}</strong></div>)}</div>
    <div className="hil-role-grid">
      <section className="hil-role-panel"><header><h3>Actor</h3><span className="hil-badge">{actorState}</span></header>
        <p>{runtime.compute?.actor_device || session.config.actor_device} · Environment interaction</p>
        <MetricTiles items={[
          ['Environment steps', number(metrics.environment_steps)], ['Sampling rate', rate(rates.actor)],
          ['Latest logged episode return', number(history.current?.return)], ['Last episode return', number(last?.return)],
          ['Intervention steps / total', `${metrics.human_steps ?? '—'} / ${metrics.environment_steps ?? '—'}`],
          ['Intervention ratio', metrics.environment_steps ? percent(metrics.human_steps / metrics.environment_steps) : '—'],
          ['Episode interventions', number(history.current?.intervention_count)], ['Episode human steps', number(history.current?.intervention_steps)],
          ['Policy version', number(metrics.policy_version)], ['Action source', runtime.last_action_source || '—'],
          ['Sample actions (ms)', number(metrics.sample_actions_seconds == null ? null : metrics.sample_actions_seconds * 1000)],
          ['Step environment (ms)', number(metrics.step_env_seconds == null ? null : metrics.step_env_seconds * 1000)],
          ['Actor work cycle (ms)', number(metrics.actor_cycle_seconds == null ? null : metrics.actor_cycle_seconds * 1000)],
        ]}/>
        <MetricChart title="Episode return" rows={history.episodes.map(e => ({x:e.episode, value:e.return}))} metric="value" xLabel="Episode"/>
      </section>
      <section className="hil-role-panel"><header><h3>Learner</h3><span className="hil-badge">{learnerState}</span></header>
        <p>{learner.placement || session.config.learner_placement} · {learner.learner_memory?.device || session.config.learner_device} · SAC updates</p>
        <MetricTiles items={[
          ['Learner updates', number(metrics.learner_steps)], ['Update rate', rate(rates.learner)], ['Critic updates', number(metrics.critic_steps)],
          ['Online replay', number(metrics.replay_size)], ['Demo / intervention replay', number(metrics.demo_replay_size)], ['Queued transitions', number(learner.pending_transitions)],
          ['Critic loss', number(metrics.critic_loss)], ['Actor loss', number(metrics.actor_loss)], ['Temperature α', number(metrics.temperature)],
          ['Entropy', number(metrics.entropy)], ['Temperature loss', number(metrics.temperature_loss)],
          ['Predicted Q', number(metrics.predicted_qs)], ['Target Q', number(metrics.target_qs)], ['Batch reward mean', number(metrics.rewards)],
          ['Sample replay (ms)', number(metrics.sample_replay_seconds == null ? null : metrics.sample_replay_seconds * 1000)],
          ['Critic-only training (ms)', number(metrics.train_critics_seconds == null ? null : metrics.train_critics_seconds * 1000)],
          ['Joint training (ms)', number(metrics.train_seconds == null ? null : metrics.train_seconds * 1000)],
        ]}/>
      </section>
    </div>
    <div className="hil-training-charts">{[['Critic loss','critic_loss'],['Actor loss','actor_loss'],['Temperature α','temperature'],['Entropy','entropy']].map(([title, key]) => <MetricChart key={key} title={title} rows={series} metric={key} color="#a78bfa"/>)}</div>
    <p className="hil-help">Learner curves show up to 180 telemetry samples observed since opening this page. New entropy/Q/timing telemetry appears in newly started runtimes; older workers show — for unavailable values. Timers show the latest measured cycle, rather than upstream's averaged timer logs. Episode statistics are reconstructed from the saved transition log.</p>
    <label>Recent episode window <select aria-label="Recent episode window" value={windowSize} onChange={e => setWindowSize(Number(e.target.value))}>{[5,10,20,50,100].map(n => <option key={n} value={n}>{n} episodes</option>)}</select></label>
    <p role="status">Recent success rate: {history.loading ? 'Loading history…' : recent.length ? `${percent(recentSuccesses / recent.length)} (${recentSuccesses}/${recent.length} completed episodes)` : '—'}</p>
    <p className="hil-help">Success uses the configured reward source ({session.config.reward_source}) or a human success label. Truncated episodes count as non-success; the unfinished episode is excluded. History follows flushed runtime logs and can lag briefly. Rates are observed between telemetry updates, not independent health checks.</p>
    {history.error && <p role="alert">Episode history unavailable: {history.error}</p>}
    <table><thead><tr><th>Episode</th><th>Result</th><th>Steps</th><th>Return</th><th>Interventions / steps</th><th>Recorded span</th><th>Decision source</th></tr></thead><tbody>
      {[...recent].reverse().map(item => <tr key={item.episode}><td>{item.episode}</td><td>{item.outcome}</td><td>{item.steps ?? '—'}</td><td>{number(item.return)}</td><td>{item.intervention_count ?? '—'} / {item.intervention_steps ?? '—'}</td><td>{duration(item.recorded_seconds)}</td><td>{item.source}</td></tr>)}
    </tbody></table>
  </section>;
}
