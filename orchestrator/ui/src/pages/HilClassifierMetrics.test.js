import { render, screen, within } from '@testing-library/react';
import HilClassifierMetrics from './HilClassifierMetrics';
jest.mock('recharts',()=>{
  const React=require('react');
  return {...jest.requireActual('recharts'),ResponsiveContainer:({children})=>React.cloneElement(children,{width:600,height:260})};
});
test('graphs and table reflect epoch history and identify checkpoint rather than final epoch',()=>{
  render(<HilClassifierMetrics job={{epoch:2,epochs:20,threshold:.9,best_epoch:1,history:[
    {epoch:1,loss:.4,validation_loss:.5,train_balanced_accuracy:.9,balanced_accuracy:.8,precision:.75,recall:.6,tp:6,tn:9,fp:2,fn:4},
    {epoch:2,loss:.3,validation_loss:.7,train_balanced_accuracy:.95,balanced_accuracy:.65,precision:0,recall:0,tp:0,tn:10,fp:0,fn:10}
  ]}} />);
  expect(screen.getByRole('figure',{name:'Loss graph'})).toBeInTheDocument();
  expect(screen.getByRole('figure',{name:'Validation performance graph'})).toBeInTheDocument();
  const rows=within(screen.getByRole('table',{name:'Epoch performance metrics'})).getAllByRole('row');
  expect(rows).toHaveLength(3);
  expect(rows[1]).toHaveTextContent('1 · Saved');
  expect(rows[1]).toHaveTextContent('66.7%');
  expect(rows[2]).toHaveTextContent('0.0%');
});
test('does not invent history or metrics before the first epoch',()=>{
  const {rerender}=render(<HilClassifierMetrics job={{state:'queued'}} />);
  expect(screen.queryByRole('table')).toBeNull();
  rerender(<HilClassifierMetrics job={{epoch:20,validation_loss:.2}} />);
  expect(within(screen.getByRole('table')).getAllByRole('row')).toHaveLength(2);
  expect(screen.getAllByText('—').length).toBeGreaterThan(0);
});
test('shows F0.5 selection and the recorded score for new jobs',()=>{
  render(<HilClassifierMetrics job={{epoch:1,best_epoch:1,checkpoint_selection:'validation_f0_5_precision_loss_v1',history:[{epoch:1,f0_5:.923,precision:1,recall:.7}]}} />);
  expect(screen.getByText(/Highest validation F0.5/)).toBeInTheDocument();
  expect(within(screen.getByRole('table')).getByText('92.3%')).toBeInTheDocument();
});
