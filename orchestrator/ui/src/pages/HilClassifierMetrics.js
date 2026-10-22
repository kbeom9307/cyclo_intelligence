import React from 'react';
import { ResponsiveContainer, LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ReferenceLine } from 'recharts';

const number = value => Number.isFinite(value) ? value.toFixed(4) : '—';
const percent = value => Number.isFinite(value) ? `${(value * 100).toFixed(1)}%` : '—';
const count = value => Number.isFinite(value) ? value : '—';
export default function HilClassifierMetrics({ job }) {
  const history = job.history?.length ? job.history : job.epoch != null ? [job] : [];
  const rows = history.map(item => ({...item, f1: Number.isFinite(item.precision) && Number.isFinite(item.recall)
    ? (item.precision + item.recall ? 2 * item.precision * item.recall / (item.precision + item.recall) : 0) : null}));
  if (!rows.length) return <p>Learning graphs and metrics will appear after the first epoch.</p>;
  return <section className="hil-classifier-metrics" aria-label="Classifier learning metrics">
    <p>Epoch {job.epoch ?? rows[rows.length-1].epoch} / {job.epochs ?? '—'} · Probability threshold {job.threshold ?? '—'} · Saved checkpoint epoch {job.best_epoch ?? '—'}</p>
    <p>Checkpoint selection: {job.checkpoint_selection === 'validation_f0_5_precision_loss_v1'
      ? 'Highest validation F0.5 → highest precision → lowest validation loss'
      : 'Highest validation balanced accuracy → lowest validation loss (previous criterion)'}</p>
    <div className="hil-learning-charts">
      {[['Loss', [['loss','Training loss','#2563eb'],['validation_loss','Validation loss','#ea580c']]],
        ['Validation performance', [['train_balanced_accuracy','Training balanced accuracy','#64748b'],['balanced_accuracy','Validation balanced accuracy','#2563eb'],['precision','Precision','#16a34a'],['recall','Recall','#ea580c'],['f1','F1','#9333ea'],['f0_5','F0.5','#db2777']]]].map(([title, series]) => <figure key={title} aria-label={`${title} graph`}>
        <figcaption>{title}</figcaption>
        <ResponsiveContainer width="100%" height={260}>
          <LineChart data={rows} margin={{top:10,right:16,left:0,bottom:10}}>
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis dataKey="epoch" type="number" domain={['dataMin','dataMax']} allowDecimals={false} label={{value:'Epoch',position:'insideBottom',offset:-5}} />
            <YAxis domain={title==='Loss' ? [0,'auto'] : [0,1]} tickFormatter={title==='Loss' ? undefined : percent} width={62} />
            <Tooltip labelFormatter={epoch=>`Epoch ${epoch}`} formatter={title==='Loss' ? number : percent} />
            <Legend />
            {job.best_epoch != null && <ReferenceLine x={job.best_epoch} stroke="#94a3b8" strokeDasharray="4 4" />}
            {series.map(([key,name,color]) => <Line key={key} type="linear" dataKey={key} name={name} stroke={color} dot={rows.length===1} isAnimationActive={false} connectNulls={false} />)}
          </LineChart>
        </ResponsiveContainer>
      </figure>)}
    </div>
    <p className="hil-help">Training loss uses balanced sampled batches; validation loss uses held-out images. Precision, recall, F1, F0.5 and confusion counts use the displayed threshold. Dashed vertical lines mark the saved checkpoint epoch.</p>
    <div className="hil-data-table-wrap hil-metrics-table"><table className="hil-data-table" aria-label="Epoch performance metrics">
      <thead><tr>{['Epoch','Train loss','Validation loss','Train balanced accuracy','Validation balanced accuracy','Precision','Recall','F1','F0.5','TP','TN','FP','FN'].map(label=><th key={label}>{label}</th>)}</tr></thead>
      <tbody>{rows.map(row=><tr key={row.epoch} className={row.epoch===job.best_epoch?'hil-best-epoch':''}>
        <th scope="row">{row.epoch}{row.epoch===job.best_epoch?' · Saved':''}</th>
        <td>{number(row.loss)}</td><td>{number(row.validation_loss)}</td><td>{percent(row.train_balanced_accuracy)}</td><td>{percent(row.balanced_accuracy)}</td>
        <td>{percent(row.precision)}</td><td>{percent(row.recall)}</td><td>{percent(row.f1)}</td><td>{percent(row.f0_5)}</td>
        <td>{count(row.tp)}</td><td>{count(row.tn)}</td><td>{count(row.fp)}</td><td>{count(row.fn)}</td>
      </tr>)}</tbody>
    </table></div>
  </section>;
}
