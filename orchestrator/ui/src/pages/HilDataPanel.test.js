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

test('combines same-task demos across collection sources and keeps episode identity', async () => {
  const {combineHilDemos,deleteHilDemoEpisodes}=require('../utils/hilserlApi');
  combineHilDemos.mockResolvedValue({path:'/workspace/hilserl/demo_exports/combined.npz',cameras:['head']});
  deleteHilDemoEpisodes.mockClear(); deleteHilDemoEpisodes.mockResolvedValue({});
  const applyArtifact=jest.fn();
  const task={environment:'isaac_sim',arm:'right',target_part:'peg_round_8'};
  const workspace={sessions:['first','second','live'].map(id=>({id,state:id==='live'?'running':'stopped'})),data_library:['first','second','live'].map(session_id=>({session_id,kind:'demo',task,cameras:['head'],files:[{episode:1,steps:12,name:'episode-00001.npz'}]}))};
  render(<HilDataLibrary workspace={workspace} busy={false} run={fn=>fn()} refresh={jest.fn()} applyArtifact={applyArtifact} />);
  expect(screen.getAllByRole('button',{name:'Select all demos'})).toHaveLength(1);
  fireEvent.click(screen.getByRole('button',{name:'Select all demos'}));
  expect(screen.getByRole('checkbox',{name:'Select demo live episode 1'})).toBeChecked();
  fireEvent.click(screen.getByRole('checkbox',{name:'Select demo live episode 1'}));
  fireEvent.click(screen.getByRole('button',{name:'Apply demos to training settings'}));
  await waitFor(()=>expect(combineHilDemos).toHaveBeenCalledWith([{session_id:'first',episodes:[1]},{session_id:'second',episodes:[1]}]));
  await waitFor(()=>expect(applyArtifact).toHaveBeenCalledWith('demonstration_path','/workspace/hilserl/demo_exports/combined.npz',['head'],expect.objectContaining({path:'/workspace/hilserl/demo_exports/combined.npz'})));
  fireEvent.click(screen.getByRole('button',{name:'Delete selected demos'}));
  fireEvent.click(screen.getByRole('button',{name:'Confirm delete'}));
  await waitFor(()=>expect(deleteHilDemoEpisodes).toHaveBeenCalledWith('first',[1]));
  await waitFor(()=>expect(deleteHilDemoEpisodes).toHaveBeenCalledWith('second',[1]));
});

test('deletes only checked demo episodes after confirmation', async () => {
  const {deleteHilDemoEpisodes}=require('../utils/hilserlApi');
  deleteHilDemoEpisodes.mockClear(); deleteHilDemoEpisodes.mockResolvedValue({});
  const workspace={sessions:[{id:'demo',state:'stopped'}],data_library:[{session_id:'demo',kind:'demo',task:{arm:'right'},episodes:2,steps:24,files:[1,2].map(episode=>({episode,steps:12,name:`episode-0000${episode}.npz`}))}]};
  const refresh=jest.fn();
  render(<HilDataLibrary workspace={workspace} busy={false} run={fn=>fn()} refresh={refresh} />);
  expect(screen.getByRole('button',{name:'Delete selected demos'})).toBeDisabled();
  fireEvent.click(screen.getByRole('checkbox',{name:'Select demo demo episode 2'}));
  fireEvent.click(screen.getByRole('button',{name:'Delete selected demos'}));
  expect(deleteHilDemoEpisodes).not.toHaveBeenCalled();
  fireEvent.click(screen.getByRole('button',{name:'Confirm delete'}));
  await waitFor(()=>expect(deleteHilDemoEpisodes).toHaveBeenCalledWith('demo',[2]));
  await waitFor(()=>expect(refresh).toHaveBeenCalled());
  expect(screen.getByRole('checkbox',{name:'Select demo demo episode 1'})).not.toBeChecked();
});

test('classifier task library trains and deletes selected collections together', async () => {
  const {deleteHilDataset}=require('../utils/hilserlApi');deleteHilDataset.mockClear();deleteHilDataset.mockResolvedValue({});
  trainHilClassifier.mockClear();trainHilClassifier.mockResolvedValue({});
  const task={arm:'right',target_part:'peg_round_8'};
  const workspace={sessions:['a','b'].map(id=>({id,state:'stopped'})),data_library:['a','b'].map(session_id=>({session_id,kind:'classifier',task,positive:20,negative:30,episodes:2}))};
  render(<HilDataLibrary workspace={workspace} busy={false} run={fn=>fn()} refresh={jest.fn()} />);
  expect(screen.getByText(/Task classifier library: 40 success \/ 60 failure/)).toBeInTheDocument();
  fireEvent.click(screen.getByRole('button',{name:'Select all datasets'}));
  fireEvent.click(screen.getByRole('button',{name:'Train classifier'}));
  await waitFor(()=>expect(trainHilClassifier).toHaveBeenCalledWith(['a','b'],20));
  fireEvent.click(screen.getByRole('button',{name:'Delete selected classifier data'}));
  fireEvent.click(screen.getByRole('button',{name:'Confirm delete'}));
  await waitFor(()=>expect(deleteHilDataset).toHaveBeenCalledWith('a','classifier'));
  await waitFor(()=>expect(deleteHilDataset).toHaveBeenCalledWith('b','classifier'));
});

test('copies a generated training path and exposes it when clipboard is unavailable', async () => {
  const {combineHilDemos}=require('../utils/hilserlApi');combineHilDemos.mockClear();
  combineHilDemos.mockResolvedValue({path:'/workspace/hilserl/demo_exports/copy.npz',cameras:['head'],steps:12});
  const writeText=jest.fn().mockResolvedValue();
  Object.defineProperty(navigator,'clipboard',{value:{writeText},configurable:true});
  const workspace={sessions:[{id:'demo',state:'stopped'}],data_library:[{session_id:'demo',kind:'demo',task:{arm:'right'},files:[{episode:1,steps:12,name:'episode-00001.npz'}]}]};
  render(<HilDataLibrary workspace={workspace} busy={false} run={fn=>fn()} refresh={jest.fn()} />);
  fireEvent.click(screen.getByRole('button',{name:'Select all demos'}));
  fireEvent.click(screen.getByRole('button',{name:'Copy selected demo path'}));
  await waitFor(()=>expect(writeText).toHaveBeenCalledWith('/workspace/hilserl/demo_exports/copy.npz'));
  expect(screen.getByLabelText('Training demo path')).toHaveValue('/workspace/hilserl/demo_exports/copy.npz');
  writeText.mockRejectedValueOnce(new Error('Denied'));
  fireEvent.click(screen.getByRole('button',{name:'Copy selected demo path'}));
  expect(await screen.findByText(/Clipboard unavailable/)).toBeInTheDocument();
  expect(combineHilDemos).toHaveBeenCalledTimes(1);
  delete navigator.clipboard;
});
