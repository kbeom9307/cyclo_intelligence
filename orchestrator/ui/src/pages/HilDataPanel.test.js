import { render, screen, fireEvent, waitFor } from '@testing-library/react';
import { HilCollectionControls, HilDataLibrary } from './HilDataPanel';
import { trainHilClassifier } from '../utils/hilserlApi';
jest.mock('../utils/hilserlApi');

test('classifier capture buttons record explicit image labels', () => {
  const command=jest.fn();
  render(<HilCollectionControls busy={false} command={command} session={{id:'session',mode:'classifier',state:'paused',runtime:{collection:{positive_samples:2,negative_samples:3}}}} />);
  expect(screen.getByText(/Success images: 2/)).toBeInTheDocument();
  fireEvent.click(screen.getByRole('button',{name:'Capture success image'}));
  expect(command).toHaveBeenCalledWith('capture_success');
  fireEvent.click(screen.getByRole('button',{name:'Capture failure image'}));
  expect(command).toHaveBeenCalledWith('capture_failure');
});

test('demo save requires successful paused episode and pending transitions', () => {
  const command=jest.fn(); const session={mode:'demo',state:'paused',episode_outcome:null,runtime:{collection:{pending_demo_steps:5}}};
  const {rerender}=render(<HilCollectionControls session={session} command={command} />);
  expect(screen.getByRole('button',{name:'Save successful demo'})).toBeDisabled();
  rerender(<HilCollectionControls session={{...session,episode_outcome:'success'}} command={command} />);
  fireEvent.click(screen.getByRole('button',{name:'Save successful demo'}));
  expect(command).toHaveBeenCalledWith('save_demo');
});

test('selected stopped dataset launches classifier training', async () => {
  trainHilClassifier.mockResolvedValue({id:'job'});
  const refresh=jest.fn();
  const workspace={sessions:[{id:'collection',state:'stopped'}],data_library:[{session_id:'collection',kind:'classifier',task:{arm:'right',target_part:'peg_round_8'},positive:10,negative:10,episodes:3}],classifier_jobs:[]};
  render(<HilDataLibrary workspace={workspace} busy={false} run={fn=>fn()} refresh={refresh} applyArtifact={jest.fn()} />);
  expect(screen.getByRole('button',{name:'Train classifier'})).toBeDisabled();
  fireEvent.click(screen.getByRole('checkbox'));
  fireEvent.click(screen.getByRole('button',{name:'Train classifier'}));
  await waitFor(()=>expect(trainHilClassifier).toHaveBeenCalledWith(['collection'],20));
  expect(refresh).toHaveBeenCalled();
});

test('dataset and checkpoint deletion require explicit confirmation', async () => {
  const { deleteHilDataset, deleteHilClassifier } = require('../utils/hilserlApi');
  deleteHilDataset.mockResolvedValue({}); deleteHilClassifier.mockResolvedValue({});
  const refresh=jest.fn();
  const workspace={sessions:[{id:'collection',state:'stopped'}],data_library:[{session_id:'collection',kind:'classifier',task:{arm:'right',target_part:'peg_round_8'},positive:10,negative:10,episodes:3}],classifier_jobs:[{id:'job',state:'failed',error:'bad data'}]};
  render(<HilDataLibrary workspace={workspace} busy={false} run={fn=>fn()} refresh={refresh} applyArtifact={jest.fn()} />);
  fireEvent.click(screen.getByRole('button',{name:'Delete dataset'}));
  expect(deleteHilDataset).not.toHaveBeenCalled();
  fireEvent.click(screen.getByRole('button',{name:'Confirm delete'}));
  await waitFor(()=>expect(deleteHilDataset).toHaveBeenCalledWith('collection','classifier'));
  fireEvent.click(screen.getByRole('button',{name:'Delete classifier checkpoint / job'}));
  fireEvent.click(screen.getAllByRole('button',{name:'Confirm delete'})[1]);
  await waitFor(()=>expect(deleteHilClassifier).toHaveBeenCalledWith('job'));
});

test('task and camera filters prevent training with hidden or active datasets', async () => {
  trainHilClassifier.mockClear(); trainHilClassifier.mockResolvedValue({id:'job'});
  const task={environment:'isaac_sim',robot_type:'ffw_sg2_rev1',arm:'right',target_part:'peg_round_8'};
  const entry=(id,target,cameras)=>({session_id:id,kind:'classifier',task:{...task,target_part:target},cameras,positive:20,negative:30,episodes:2});
  const workspace={sessions:['a','b','c','live'].map(id=>({id,state:id==='live'?'running':'stopped'})),data_library:[entry('a','peg_round_8',['head']),entry('b','peg_round_12',['head']),entry('c','peg_round_8',['right_wrist']),entry('live','peg_round_8',['head'])],classifier_jobs:[]};
  const props={workspace,busy:false,run:fn=>fn(),refresh:jest.fn(),applyArtifact:jest.fn()};
  const {rerender}=render(<HilDataLibrary {...props} />);
  expect(screen.getByRole('checkbox',{name:'Select dataset live'})).toBeDisabled();
  fireEvent.click(screen.getByRole('button',{name:'Select all datasets'}));
  fireEvent.click(screen.getByRole('button',{name:'Train classifier'}));
  await waitFor(()=>expect(trainHilClassifier).toHaveBeenLastCalledWith(['a'],20));
  fireEvent.change(screen.getByLabelText('Dataset cameras'),{target:{value:'right_wrist'}});
  expect(screen.getByRole('button',{name:'Train classifier'})).toBeDisabled();
  expect(screen.queryByRole('checkbox',{name:'Select dataset a'})).not.toBeInTheDocument();
  fireEvent.click(screen.getByRole('checkbox',{name:'Select dataset c'}));
  const nextTask=screen.getByRole('option',{name:/peg_round_12/}).value;
  fireEvent.change(screen.getByLabelText('Dataset task'),{target:{value:nextTask}});
  expect(screen.getByRole('button',{name:'Train classifier'})).toBeDisabled();
  fireEvent.click(screen.getByRole('checkbox',{name:'Select dataset b'}));
  fireEvent.click(screen.getByRole('button',{name:'Train classifier'}));
  await waitFor(()=>expect(trainHilClassifier).toHaveBeenLastCalledWith(['b'],20));
  rerender(<HilDataLibrary {...props} workspace={{...workspace,data_library:workspace.data_library.filter(item=>item.session_id!=='b')}} />);
  expect(screen.getByRole('button',{name:'Train classifier'})).toBeDisabled();
});
