import { render, screen, fireEvent, waitFor } from '@testing-library/react';
import HilSerlPage from './HilSerlPage';
import { getHilWorkspace, saveHilConfig, createHilSession, sendHilCommand, sendHilAction, checkHilLearner } from '../utils/hilserlApi';

jest.mock('../utils/hilserlApi');
jest.mock('./HilIsaacView', () => () => <div>Isaac live camera panel</div>);
const config = {
  name: 'Peg insertion', robot_type: 'ffw_bg2_rev4', environment: 'isaac_sim', board: 'nist_atb1',
  task: 'peg_insertion', arm: 'right', control_hz: 10, max_episode_steps: 200,
  translation_limit_m: 0.005, rotation_limit_rad: 0.05, success_depth_m: 0.02,
  success_tolerance_m: 0.002, success_hold_s: 0.5, reward_source: 'environment',
  camera_topic: '/camera/image', joint_state_topic: '/joint_states', demonstration_path: '', checkpoint_path: '',
  seed: 0, batch_size: 256, replay_capacity: 100000, learning_rate: 0.0003, discount: 0.99,
};
const session = { id: 'test', revision: 0, mode: 'preview', state: 'created', config,
  config_revision: 0, control_source: 'policy', episode: 1, episode_outcome: null,
  metrics: {}, events: [], created_at: '2026-10-16T00:00:00Z' };
let workspace;
beforeEach(() => {
  jest.resetAllMocks();
  workspace = { config, revision: 0, sessions: [], readiness: { can_preview: true, can_train: false,
    checks: [{ id: 'actor', label: 'Actor', ready: false, detail: 'Not connected' }] } };
  getHilWorkspace.mockImplementation(async () => workspace);
});

test('shows disconnected runtime and never offers real training', async () => {
  render(<HilSerlPage />);
  expect(await screen.findByText('Runtime readiness')).toBeInTheDocument();
  expect(screen.getByRole('button', { name: /Create Isaac Sim training/ })).toBeDisabled();
  expect(screen.getByText(/Actor inference and SAC learning use the selected GPU devices/)).toBeInTheDocument();
  expect(screen.getByRole('button', { name: 'Create preview session' })).toBeEnabled();
});

test('restores upstream model, batch, replay and update settings without starting a session', async () => {
  workspace.config = { ...config, learning_backend: 'legacy_torch', pixel_batch_size: 32, pixel_replay_capacity: 2048 };
  render(<HilSerlPage />);
  fireEvent.click(await screen.findByRole('button', { name: 'Restore upstream HIL-SERL settings' }));
  expect(screen.getByLabelText('Learning engine')).toHaveValue('hil_serl');
  expect(screen.getByLabelText('Pixel batch size')).toHaveValue(256);
  expect(screen.getByLabelText('Pixel replay capacity per buffer')).toHaveValue(200000);
  expect(screen.getByLabelText('Discount factor')).toHaveValue(.97);
  expect(screen.getByLabelText('Critic updates per actor update')).toHaveValue(2);
  expect(screen.getByLabelText('head crop Left')).toHaveValue(0);
  expect(createHilSession).not.toHaveBeenCalled();
});

test('saves config before allowing a preview with changed settings', async () => {
  saveHilConfig.mockImplementation(async (value) => {
    workspace = { ...workspace, config: value, revision: 1 };
    return { config: value, revision: 1 };
  });
  render(<HilSerlPage />);
  fireEvent.change(await screen.findByLabelText('Experiment name'), { target: { value: 'NIST run' } });
  expect(screen.getByRole('button', { name: 'Create preview session' })).toBeDisabled();
  fireEvent.click(screen.getByRole('button', { name: 'Save configuration' }));
  await waitFor(() => expect(saveHilConfig).toHaveBeenCalledWith(expect.objectContaining({ name: 'NIST run' }), 0));
  await waitFor(() => expect(screen.getByRole('button', { name: 'Create preview session' })).toBeEnabled());
});

test('starts preview with automatic keyboard intervention and no takeover toggle', async () => {
  createHilSession.mockImplementation(async () => {
    workspace = { ...workspace, sessions: [session] }; return session;
  });
  sendHilCommand.mockImplementation(async (current, action) => {
    const updated = { ...current, revision: current.revision + 1,
      state: 'running', control_source: action === 'take_over' ? 'human' : 'policy' };
    workspace = { ...workspace, sessions: [updated] }; return updated;
  });
  render(<HilSerlPage />);
  fireEvent.click(await screen.findByRole('button', { name: 'Create preview session' }));
  fireEvent.click(await screen.findByRole('button', { name: 'Start preview' }));
  await waitFor(() => expect(screen.getByRole('button', { name: '+X pulse' })).toBeEnabled());
  expect(screen.queryByRole('button', { name: /Take over/ })).not.toBeInTheDocument();
  fireEvent.keyDown(window, { code: 'KeyW' });
  await waitFor(() => expect(sendHilAction).toHaveBeenCalledWith(expect.objectContaining({ revision: 1 }), [1,0,0,0,0,0]));
  fireEvent.keyUp(window, { code: 'KeyW' });
  await waitFor(() => expect(sendHilAction).toHaveBeenLastCalledWith(expect.anything(), [0,0,0,0,0,0]));
  expect(sendHilCommand).toHaveBeenCalledTimes(1);
});

test('displays backend conflicts and preserves unsaved fields', async () => {
  saveHilConfig.mockRejectedValue(new Error('Configuration changed. Reload before saving.'));
  render(<HilSerlPage />);
  fireEvent.change(await screen.findByLabelText('Experiment name'), { target: { value: 'My draft' } });
  fireEvent.click(screen.getByRole('button', { name: 'Save configuration' }));
  expect(await screen.findByRole('alert')).toHaveTextContent('Configuration changed');
  expect(screen.getByLabelText('Experiment name')).toHaveValue('My draft');
});

test('failed initial load can be retried', async () => {
  getHilWorkspace.mockRejectedValueOnce(new Error('Service unavailable'));
  render(<HilSerlPage />);
  fireEvent.click(await screen.findByRole('button', { name: 'Retry connection' }));
  expect(await screen.findByLabelText('Experiment name')).toHaveValue('Peg insertion');
});


test('creates a real mock training session when readiness passes', async () => {
  workspace = { ...workspace, config: { ...config, environment: 'mock' }, readiness: { ...workspace.readiness, can_train: true } };
  createHilSession.mockImplementation(async () => {
    const item = { ...session, mode: 'train', config: workspace.config };
    workspace = { ...workspace, sessions: [item] }; return item;
  });
  render(<HilSerlPage />);
  fireEvent.click(await screen.findByRole('button', { name: 'Create Mock training session' }));
  expect(await screen.findByRole('button', { name: 'Start Mock runtime' })).toBeEnabled();
  expect(createHilSession).toHaveBeenCalledWith(0, 'train');
});

test('manual control sends bounded XYZ pulses while policy owns control', async () => {
  workspace = { ...workspace, sessions: [{ ...session, mode: 'train', config: { ...config, environment: 'mock' }, state: 'running', control_source: 'policy' }] };
  sendHilAction.mockResolvedValue({ accepted: true });
  render(<HilSerlPage />);
  fireEvent.click(await screen.findByRole('button', { name: 'Session & history' }));
  fireEvent.click(screen.getByRole('button', { name: '+X pulse' }));
  await waitFor(() => expect(sendHilAction).toHaveBeenCalledWith(expect.objectContaining({ control_source: 'policy' }), [1, 0, 0]));
});


test('applies the connected Isaac robot and step rate settings before saving', async () => {
  render(<HilSerlPage />);
  fireEvent.click(await screen.findByRole('button', { name: 'Use Isaac NIST settings' }));
  expect(screen.getByLabelText('Robot model')).toHaveValue('ffw_sg2_rev1');
  expect(screen.getByLabelText('RL step frequency (Hz)')).toHaveValue(20);
  expect(screen.getByRole('button', { name: 'Save configuration' })).toBeEnabled();
});

test('Isaac intervention sends six actions with no gripper control', async () => {
  workspace = { ...workspace, sessions: [{ ...session, mode: 'train', state: 'running', control_source: 'human' }] };
  sendHilAction.mockResolvedValue({ accepted: true });
  render(<HilSerlPage />);
  fireEvent.click(await screen.findByRole('button', { name: 'Session & history' }));
  expect(screen.queryByRole('button', { name: 'Close gripper pulse' })).toBeNull();
  fireEvent.click(screen.getByRole('button', { name: '+X pulse' }));
  await waitFor(() => expect(sendHilAction).toHaveBeenCalledWith(expect.anything(), [1, 0, 0, 0, 0, 0]));
});


test('Isaac preview shows manual controls and a full-width camera before the session title', async () => {
  workspace.sessions = [{ ...session, state: 'running', control_source: 'human' }];
  render(<HilSerlPage />);
  fireEvent.click(await screen.findByRole('button', { name: 'Session & history' }));
  const camera = screen.getByText('Isaac live camera panel');
  const title = screen.getByRole('heading', { name: 'Peg insertion' });
  expect(camera.closest('aside')).toBeNull();
  expect(camera.compareDocumentPosition(title) & Node.DOCUMENT_POSITION_FOLLOWING).toBeTruthy();
  expect(screen.getByRole('button', { name: '+X pulse' })).toBeEnabled();
  expect(screen.queryByRole('button', { name: 'Close gripper pulse' })).toBeNull();
});


test('progress uses current episode steps rather than cumulative training steps', async () => {
  workspace.sessions = [{ ...session, state: 'paused', runtime: { episode_step: 37 }, metrics: { environment_steps: 437 } }];
  render(<HilSerlPage />);
  fireEvent.click(await screen.findByRole('button', { name: 'Session & history' }));
  const progress = screen.getByRole('progressbar', { name: 'Episode steps' });
  expect(progress).toHaveAttribute('value', '37');
  expect(progress).toHaveAttribute('max', '200');
});

test('sends truncate for a paused unfinished episode', async () => {
  workspace.sessions = [{ ...session, state: 'paused' }];
  sendHilCommand.mockResolvedValue({});
  render(<HilSerlPage />);
  fireEvent.click(await screen.findByRole('button', { name: 'Session & history' }));
  fireEvent.click(screen.getByRole('button', { name: 'Truncated' }));
  await waitFor(() => expect(sendHilCommand).toHaveBeenCalledWith(expect.objectContaining({state:'paused'}), 'truncate'));
});

test('starts a fresh classifier collection after the previous collection stops', async () => {
  const stopped = { ...session, mode: 'classifier', state: 'stopped' };
  const fresh = { ...session, id: 'new-collection', mode: 'classifier' };
  workspace.sessions = [stopped];
  workspace.readiness.can_classifier = true;
  createHilSession.mockImplementation(async () => {
    workspace.sessions = [fresh, stopped];
    return fresh;
  });
  sendHilCommand.mockImplementation(async (current) => {
    workspace.sessions = [{ ...current, state: 'running' }, stopped];
  });
  render(<HilSerlPage />);
  fireEvent.click(await screen.findByRole('button', { name: 'Session & history' }));
  expect(screen.getByRole('button', { name: 'Start Isaac Sim runtime' })).toBeEnabled();
  fireEvent.click(screen.getByRole('button', { name: 'Start Isaac Sim runtime' }));
  await waitFor(() => expect(sendHilCommand).toHaveBeenCalledWith(fresh, 'start'));
  expect(createHilSession).toHaveBeenCalledWith(0, 'classifier');
  await waitFor(() => expect(screen.getByRole('button', { name: 'Start Isaac Sim runtime' })).toBeDisabled());
  expect(workspace.sessions[1]).toBe(stopped);
});

test.each(['active', 'unready'])('does not restart classifier collection when %s', async (reason) => {
  workspace.sessions = [{ ...session, mode: 'classifier', state: 'stopped' }];
  workspace.readiness.can_classifier = reason !== 'unready';
  if (reason === 'active') workspace.sessions.push({ ...session, id: 'other', state: 'running' });
  render(<HilSerlPage />);
  fireEvent.click(await screen.findByRole('button', { name: 'Session & history' }));
  expect(screen.getByRole('button', { name: 'Start Isaac Sim runtime' })).toBeDisabled();
  expect(createHilSession).not.toHaveBeenCalled();
});

test('can retry Start on the newly created collection if runtime startup fails', async () => {
  const stopped = { ...session, mode: 'classifier', state: 'stopped' };
  const fresh = { ...session, id: 'retry-collection', mode: 'classifier' };
  workspace.sessions = [stopped];
  workspace.readiness.can_classifier = true;
  createHilSession.mockImplementation(async () => {
    workspace.sessions = [fresh, stopped];
    return fresh;
  });
  sendHilCommand.mockRejectedValueOnce(new Error('Simulator unavailable')).mockResolvedValue({});
  render(<HilSerlPage />);
  fireEvent.click(await screen.findByRole('button', { name: 'Session & history' }));
  fireEvent.click(screen.getByRole('button', { name: 'Start Isaac Sim runtime' }));
  expect(await screen.findByRole('alert')).toHaveTextContent('Simulator unavailable');
  fireEvent.click(screen.getByRole('button', { name: 'Start Isaac Sim runtime' }));
  await waitFor(() => expect(sendHilCommand).toHaveBeenCalledTimes(2));
  expect(sendHilCommand).toHaveBeenLastCalledWith(fresh, 'start');
  expect(createHilSession).toHaveBeenCalledTimes(1);
});

test('demo collection screen includes task-scoped saved data management', async () => {
  workspace.sessions = [{...session, mode:'demo', state:'paused', episode_outcome:'success', runtime:{collection:{saved_demo_episodes:1,pending_demo_steps:0}}}];
  workspace.data_library = [{session_id:'test',kind:'demo',task:config,episodes:1,steps:12,files:[{name:'episode-00001.npz',episode:1,steps:12}]}];
  render(<HilSerlPage />);
  fireEvent.click(await screen.findByRole('button',{name:'Session & history'}));
  expect(screen.getByRole('region',{name:'Demo data management'})).toHaveTextContent('1 episodes / 12 steps');
  expect(screen.getByText(/Success pauses the episode for review/)).toBeInTheDocument();
  expect(screen.getByRole('button',{name:'Save successful demo'})).toBeDisabled();
  expect(screen.getByRole('button',{name:'Delete selected demos'})).toBeDisabled();
  expect(screen.getByRole('button',{name:'Reset episode'})).toBeEnabled();
});

test('stops demo, combines two sources, saves settings and creates training without starting it', async () => {
  const {combineHilDemos}=require('../utils/hilserlApi');
  const order=[];
  let stoppedPolls=0;
  getHilWorkspace.mockImplementation(async () => {
    if (order.includes('stop') && ++stoppedPolls >= 2) workspace.sessions=workspace.sessions.map(item=>item.id==='test'?{...item,runtime_pending:false,runtime:{artifacts_saved:true}}:item);
    return workspace;
  });
  workspace.sessions=[{...session,mode:'demo',state:'paused',runtime:{collection:{pending_demo_steps:4}}},{...session,id:'second',mode:'demo',state:'stopped'}];
  workspace.data_library=['test','second'].map(session_id=>({session_id,kind:'demo',task:config,files:[{episode:1,steps:1250,name:'episode-00001.npz'}]}));
  sendHilCommand.mockImplementation(async (item,action)=>{
    order.push(action);
    workspace.sessions=workspace.sessions.map(value=>value.id===item.id?{...value,state:'stopped',runtime_pending:true,runtime:{artifacts_saved:false}}:value);
  });
  combineHilDemos.mockImplementation(async selections=>{
    order.push('combine');
    expect(workspace.sessions.every(item=>item.state==='stopped')).toBe(true);
    expect(workspace.sessions[0].runtime.artifacts_saved).toBe(true);
    return {path:'/workspace/hilserl/demo_exports/chosen.npz',cameras:['head'],steps:2500,episodes:2,task:{environment:'isaac_sim',robot_type:'ffw_sg2_rev1',arm:'right',target_part:'peg_round_8'}};
  });
  saveHilConfig.mockImplementation(async config=>{
    order.push('save');workspace={...workspace,config,revision:1};return {config,revision:1};
  });
  createHilSession.mockImplementation(async (revision,mode)=>{
    order.push('create');const item={...session,id:'training',mode,config:workspace.config};workspace.sessions=[item,...workspace.sessions];return item;
  });
  render(<HilSerlPage />);
  fireEvent.click(await screen.findByRole('button',{name:'Session & history'}));
  fireEvent.click(screen.getByRole('button',{name:'Select all demos'}));
  expect(screen.getByText('Selected 2 demos from 2 sources · 2500 steps')).toBeInTheDocument();
  fireEvent.click(screen.getByRole('button',{name:'Apply demos to training settings'}));
  await waitFor(()=>expect(createHilSession).toHaveBeenCalledWith(1,'train'));
  expect(order).toEqual(['stop','combine','save','create']);
  expect(combineHilDemos).toHaveBeenCalledWith([{session_id:'test',episodes:[1]},{session_id:'second',episodes:[1]}]);
  expect(saveHilConfig).toHaveBeenCalledWith(expect.objectContaining({demonstration_path:'/workspace/hilserl/demo_exports/chosen.npz',observation_cameras:['head'],pixel_replay_capacity:2500}),0);
  expect(sendHilCommand).toHaveBeenCalledTimes(1);
  expect(await screen.findByText(/Training session created with 2 demos from 2 sources/)).toBeInTheDocument();
  expect(screen.getByRole('button',{name:'Start Isaac Sim runtime'})).toBeEnabled();
});

test('failed demo stop does not combine, save settings or create training', async () => {
  const {combineHilDemos}=require('../utils/hilserlApi');
  workspace.sessions=[{...session,mode:'demo',state:'paused'}];
  workspace.data_library=[{session_id:'test',kind:'demo',task:config,files:[{episode:1,steps:12,name:'episode-00001.npz'}]}];
  sendHilCommand.mockRejectedValue(new Error('Stop failed'));
  render(<HilSerlPage />);
  fireEvent.click(await screen.findByRole('button',{name:'Session & history'}));
  fireEvent.click(screen.getByRole('button',{name:'Select all demos'}));
  fireEvent.click(screen.getByRole('button',{name:'Apply demos to training settings'}));
  expect(await screen.findByRole('alert')).toHaveTextContent('Stop failed');
  expect(combineHilDemos).not.toHaveBeenCalled();
  expect(saveHilConfig).not.toHaveBeenCalled();
  expect(createHilSession).not.toHaveBeenCalled();
});


test('configures external GPU learner and checks the selected endpoint', async () => {
  checkHilLearner.mockResolvedValue({connected: true, busy: false, gpus: [{device: 'cuda:1', name: 'Remote GPU'}]});
  saveHilConfig.mockImplementation(async value => { workspace={...workspace,config:value,revision:1}; return workspace; });
  render(<HilSerlPage />);
  fireEvent.change(await screen.findByLabelText('Actor device'), {target:{value:'cuda:0'}});
  fireEvent.change(screen.getByLabelText('Learner location'), {target:{value:'remote'}});
  fireEvent.change(screen.getByLabelText('Learner device'), {target:{value:'cuda:1'}});
  fireEvent.change(screen.getByLabelText('External learner URL'), {target:{value:'https://learner.example.com'}});
  fireEvent.change(screen.getByLabelText('Local learner token file'), {target:{value:'/workspace/hilserl/learner-token'}});
  fireEvent.click(screen.getByRole('button', {name:'Test learner connection'}));
  expect(await screen.findByText(/Learner connected · Remote GPU/)).toBeInTheDocument();
  expect(checkHilLearner).toHaveBeenCalledWith(expect.objectContaining({learner_placement:'remote',learner_device:'cuda:1',actor_device:'cuda:0'}));
  fireEvent.click(screen.getByRole('button', {name:'Save configuration'}));
  await waitFor(()=>expect(saveHilConfig).toHaveBeenCalledWith(expect.objectContaining({learner_url:'https://learner.example.com'}),0));
});
