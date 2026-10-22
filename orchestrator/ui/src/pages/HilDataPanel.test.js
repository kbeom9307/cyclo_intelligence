import { render, screen, fireEvent, waitFor } from '@testing-library/react';
import { HilCollectionControls, HilDataLibrary, HilDemoData } from './HilDataPanel';
import { trainHilClassifier } from '../utils/hilserlApi';
jest.mock('../utils/hilserlApi');

test('classifier displays progress before the first epoch and measured startup timings', () => {
  const workspace={sessions:[],data_library:[],classifier_jobs:[{
    id:'warmup',state:'training',device:'cuda:0',phase:'Preparing first GPU batch',
    phase_completed:0,phase_total:15,current_epoch:1,phase_started_at:Date.now()/1000,
    loading_seconds:2.4,initialization_seconds:3.1,
  }]};
  render(<HilDataLibrary workspace={workspace} run={fn=>fn()} refresh={jest.fn()} />);
  expect(screen.getByText(/Epoch 1 · Preparing first GPU batch · 0 \/ 15/)).toBeInTheDocument();
  expect(screen.getByRole('progressbar',{name:'Classifier phase progress'})).toHaveAttribute('value','0');
  expect(screen.getByText(/Images loaded in 2.4 s · Model initialized in 3.1 s/)).toBeInTheDocument();
  expect(screen.getByRole('button',{name:'Stop training'})).toBeEnabled();
});

test('running classifier can be stopped and its saved checkpoint remains usable', async () => {
  const {stopHilClassifier} = require('../utils/hilserlApi');
  stopHilClassifier.mockResolvedValue({state:'stopped'});
  const refresh=jest.fn(), applyArtifact=jest.fn();
  const job={id:'gpu-job',state:'training',device:'cuda:0',learning_backend:'hil_serl'};
  const workspace={sessions:[],data_library:[],classifier_jobs:[job]};
  const props={workspace,busy:false,run:fn=>fn(),refresh,applyArtifact};
  const {rerender}=render(<HilDataLibrary {...props} />);
  expect(screen.getByText(/ResNet-10.*GPU cuda:0/)).toBeInTheDocument();
  expect(screen.queryByRole('button',{name:'Use classifier'})).not.toBeInTheDocument();
  fireEvent.click(screen.getByRole('button',{name:'Stop training'}));
  await waitFor(()=>expect(stopHilClassifier).toHaveBeenCalledWith('gpu-job'));
  await waitFor(()=>expect(refresh).toHaveBeenCalled());
  rerender(<HilDataLibrary {...props} workspace={{...workspace,classifier_jobs:[{...job,state:'stopped',checkpoint_path:'/saved/classifier.npz',cameras:['head']}]}} />);
  expect(screen.queryByRole('button',{name:'Stop training'})).not.toBeInTheDocument();
  expect(screen.getByText(/The best saved checkpoint is available/)).toBeInTheDocument();
  fireEvent.click(screen.getByRole('button',{name:'Use classifier'}));
  expect(applyArtifact).toHaveBeenCalledWith('classifier_path','/saved/classifier.npz',['head'],expect.objectContaining({checkpoint_path:'/saved/classifier.npz'}));
  rerender(<HilDataLibrary {...props} workspace={{...workspace,classifier_jobs:[{...job,state:'stopped'}]}} />);
  expect(screen.queryByRole('button',{name:'Use classifier'})).not.toBeInTheDocument();
  expect(screen.getByText(/No completed checkpoint was saved/)).toBeInTheDocument();
});

test('classifier shows the initial intervention wait and blocks early manual capture', () => {
  const session = {mode:'classifier', state:'running', runtime:{collection:{waiting_for_intervention:true}}};
  const {rerender} = render(<HilCollectionControls session={session} command={jest.fn()} />);
  expect(screen.getByText(/Waiting for first intervention/)).toBeInTheDocument();
  expect(screen.queryByText(/Recording each step/)).not.toBeInTheDocument();
  expect(screen.getByText(/Space labels success but does not start collection/)).toBeInTheDocument();
  rerender(<HilCollectionControls session={{...session, state:'paused'}} command={jest.fn()} />);
  expect(screen.getByRole('button', {name:'Capture success image'})).toBeDisabled();
  expect(screen.getByRole('button', {name:'Capture failure image'})).toBeDisabled();
  rerender(<HilCollectionControls session={{...session, runtime:{collection:{waiting_for_intervention:false}}}} command={jest.fn()} />);
  expect(screen.queryByText(/Waiting for first intervention/)).not.toBeInTheDocument();
  expect(screen.getByText(/Recording each step/)).toBeInTheDocument();
});

test('classifier keeps success capture and explains skipped failures after step limit', () => {
  const session = {mode:'classifier',state:'running',runtime:{collection:{
    success_only:true,failure_step_limit:1000,skipped_negative_samples:12,
  }}};
  const {rerender} = render(<HilCollectionControls session={session} command={jest.fn()} />);
  expect(screen.getByText(/Recording success only/)).toBeInTheDocument();
  expect(screen.getByText(/Skipped failure steps: 12/)).toBeInTheDocument();
  rerender(<HilCollectionControls session={{...session,state:'paused'}} command={jest.fn()} />);
  expect(screen.getByRole('button',{name:'Capture failure image'})).toBeDisabled();
  expect(screen.getByRole('button',{name:'Capture success image'})).toBeEnabled();
});

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

test('demo shows intervention wait until the first movement starts recording', () => {
  const session = {mode:'demo', state:'running', runtime:{collection:{
    waiting_for_intervention:true, pending_demo_steps:0, pending_demo_classifier_samples:0,
  }}};
  const {rerender} = render(<HilCollectionControls session={session} command={jest.fn()} />);
  expect(screen.getByText(/Waiting for first intervention · No demonstration steps recorded/)).toBeInTheDocument();
  expect(screen.getByRole('button',{name:'Discard pending demo'})).toBeDisabled();
  rerender(<HilCollectionControls session={{...session, runtime:{collection:{waiting_for_intervention:false, pending_demo_steps:1}}}} command={jest.fn()} />);
  expect(screen.queryByText(/No demonstration steps recorded/)).not.toBeInTheDocument();
  expect(screen.getByText(/Pending steps: 1/)).toBeInTheDocument();
});

test.each([
  ['Discard only', 'discard_demo'],
  ['Discard & add classifier failures', 'discard_demo_as_failure'],
])('demo discard waits for the explicit %s choice', (label, action) => {
  const command = jest.fn();
  const session = {id:'demo', episode:2, mode:'demo', state:'paused', episode_outcome:'success',
    config:{classifier_cameras:['right_wrist']},
    runtime:{collection:{pending_demo_steps:5, pending_demo_classifier_samples:5}}};
  render(<HilCollectionControls session={session} command={command} />);
  fireEvent.click(screen.getByRole('button',{name:'Discard pending demo'}));
  expect(command).not.toHaveBeenCalled();
  expect(screen.getByText(/Adds 5 post-action images/)).toHaveTextContent('Classifier cameras: right_wrist');
  fireEvent.click(screen.getByRole('button',{name:'Cancel'}));
  expect(command).not.toHaveBeenCalled();
  fireEvent.click(screen.getByRole('button',{name:'Discard pending demo'}));
  fireEvent.click(screen.getByRole('button',{name:label}));
  expect(command).toHaveBeenCalledTimes(1);
  expect(command).toHaveBeenCalledWith(action);
});

test('older demo runtime can discard but cannot relabel unavailable classifier views', () => {
  const command = jest.fn();
  const session = {id:'old', mode:'demo', state:'paused', runtime:{collection:{pending_demo_steps:3}}};
  const {rerender} = render(<HilCollectionControls session={session} command={command} />);
  fireEvent.click(screen.getByRole('button',{name:'Discard pending demo'}));
  expect(screen.getByRole('button',{name:'Discard & add classifier failures'})).toBeDisabled();
  expect(screen.getByRole('button',{name:'Discard only'})).toBeEnabled();
  rerender(<HilCollectionControls session={{...session, runtime_pending:true}} command={command} />);
  expect(screen.getByRole('button',{name:'Discard only'})).toBeDisabled();
  rerender(<HilCollectionControls session={{...session, id:'new'}} command={command} />);
  expect(screen.queryByRole('group',{name:'Discard demo options'})).not.toBeInTheDocument();
  expect(command).not.toHaveBeenCalled();
});

test('selected stopped dataset launches classifier training', async () => {
  trainHilClassifier.mockResolvedValue({id:'job'});
  const refresh=jest.fn();
  const workspace={sessions:[{id:'collection',state:'stopped'}],data_library:[{session_id:'collection',kind:'classifier',task:{arm:'right',target_part:'peg_round_8'},positive:10,negative:10,episodes:3}],classifier_jobs:[]};
  render(<HilDataLibrary workspace={workspace} busy={false} run={fn=>fn()} refresh={refresh} applyArtifact={jest.fn()} />);
  expect(screen.queryByRole('region', {name:'Demo data management'})).not.toBeInTheDocument();
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

test('environment name and camera filters prevent training with hidden or active datasets', async () => {
  trainHilClassifier.mockClear(); trainHilClassifier.mockResolvedValue({id:'job'});
  const task={environment:'isaac_sim',robot_type:'ffw_sg2_rev1',arm:'right',target_part:'peg_round_8'};
  const entry=(id,name,cameras)=>({session_id:id,kind:'classifier',environment_name:name,task,cameras,positive:20,negative:30,episodes:2});
  const workspace={sessions:['a','b','c','live'].map(id=>({id,state:'stopped',dataset_finalizing:id==='live'})),data_library:[entry('a','NIST board A',['head']),entry('b','NIST board B',['head']),entry('c','NIST board A',['right_wrist']),entry('live','NIST board A',['head'])],classifier_jobs:[]};
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
  const nextEnvironment=screen.getByRole('option',{name:'NIST board B'}).value;
  fireEvent.change(screen.getByLabelText('Dataset environment'),{target:{value:nextEnvironment}});
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
  render(<HilDemoData workspace={workspace} task={task} busy={false} run={fn=>fn()} refresh={jest.fn()} applyArtifact={applyArtifact} />);
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
  render(<HilDemoData workspace={workspace} task={workspace.data_library[0].task} busy={false} run={fn=>fn()} refresh={refresh} />);
  expect(screen.getByRole('button',{name:'Delete selected demos'})).toBeDisabled();
  fireEvent.click(screen.getByRole('checkbox',{name:'Select demo demo episode 2'}));
  fireEvent.click(screen.getByRole('button',{name:'Delete selected demos'}));
  expect(deleteHilDemoEpisodes).not.toHaveBeenCalled();
  fireEvent.click(screen.getByRole('button',{name:'Confirm delete'}));
  await waitFor(()=>expect(deleteHilDemoEpisodes).toHaveBeenCalledWith('demo',[2]));
  await waitFor(()=>expect(refresh).toHaveBeenCalled());
  expect(screen.getByRole('checkbox',{name:'Select demo demo episode 1'})).not.toBeChecked();
});

test('classifier environment library trains and deletes selected collections together', async () => {
  const {deleteHilDataset}=require('../utils/hilserlApi');deleteHilDataset.mockClear();deleteHilDataset.mockResolvedValue({});
  trainHilClassifier.mockClear();trainHilClassifier.mockResolvedValue({});
  const task={arm:'right',target_part:'peg_round_8'};
  const workspace={sessions:['a','b'].map(id=>({id,state:'stopped'})),data_library:['a','b'].map(session_id=>({session_id,kind:'classifier',task,positive:20,negative:30,episodes:2}))};
  render(<HilDataLibrary workspace={workspace} busy={false} run={fn=>fn()} refresh={jest.fn()} />);
  expect(screen.getByText(/Environment classifier library: 40 success \/ 60 failure/)).toBeInTheDocument();
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
  render(<HilDemoData workspace={workspace} task={workspace.data_library[0].task} busy={false} run={fn=>fn()} refresh={jest.fn()} />);
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


test('environment library filters checkpoints and prevents incompatible collections sharing a name from training', () => {
  const task={environment:'mujoco',robot_type:'ffw_sg2_rev1',arm:'right',target_part:'peg_round_8'};
  const workspace={sessions:['a','b'].map(id=>({id,state:'stopped'})),
    data_library:[
      {session_id:'a',kind:'classifier',environment_name:'NIST board',task,positive:20,negative:30,episodes:1,files:[{episode:1,name:'episode-00001.npz',positive:20,negative:30}]},
      {session_id:'b',kind:'classifier',environment_name:'NIST board',task:{...task,environment:'isaac_sim'},positive:20,negative:30,episodes:1},
    ],classifier_jobs:[
      {id:'job-current',state:'failed',environment_names:['NIST board']},
      {id:'job-hidden',state:'failed',environment_names:['Other board']},
    ]};
  render(<HilDataLibrary workspace={workspace} run={fn=>fn()} refresh={jest.fn()} />);
  expect(screen.queryByLabelText('Dataset task')).not.toBeInTheDocument();
  expect(screen.getByText(/episode-00001.npz/)).toBeInTheDocument();
  expect(screen.getByText(/Classifier job-curr/)).toBeInTheDocument();
  expect(screen.queryByText(/Classifier job-hidd/)).not.toBeInTheDocument();
  fireEvent.click(screen.getByRole('button',{name:'Select all datasets'}));
  expect(screen.getByRole('button',{name:'Train classifier'})).toBeDisabled();
  expect(screen.getByRole('alert')).toHaveTextContent('different simulators');
  fireEvent.click(screen.getByRole('checkbox',{name:'Select dataset b'}));
  expect(screen.getByRole('button',{name:'Train classifier'})).toBeEnabled();
});
