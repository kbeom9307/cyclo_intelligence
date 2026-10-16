import { render, screen, fireEvent, waitFor } from '@testing-library/react';
import HilSerlPage from './HilSerlPage';
import { getHilWorkspace, saveHilConfig, createHilSession, sendHilCommand, sendHilAction } from '../utils/hilserlApi';

jest.mock('../utils/hilserlApi');
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
  expect(screen.getByRole('button', { name: /Create Mock training/ })).toBeDisabled();
  expect(screen.getByText(/Mock runs a CPU SAC learner/)).toBeInTheDocument();
  expect(screen.getByRole('button', { name: 'Create preview session' })).toBeEnabled();
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

test('creates preview, starts it and records human takeover', async () => {
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
  const takeover = await screen.findByRole('button', { name: 'Take over (preview)' });
  await waitFor(() => expect(takeover).toBeEnabled());
  fireEvent.click(takeover);
  expect(await screen.findByRole('button', { name: 'Release to policy' })).toBeEnabled();
  expect(sendHilCommand).toHaveBeenLastCalledWith(expect.objectContaining({ revision: 1 }), 'take_over');
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

test('manual control sends bounded XYZ pulses only after takeover', async () => {
  workspace = { ...workspace, sessions: [{ ...session, mode: 'train', state: 'running', control_source: 'human' }] };
  sendHilAction.mockResolvedValue({ accepted: true });
  render(<HilSerlPage />);
  fireEvent.click(await screen.findByRole('button', { name: 'Session & history' }));
  fireEvent.click(screen.getByRole('button', { name: '+X pulse' }));
  await waitFor(() => expect(sendHilAction).toHaveBeenCalledWith(expect.objectContaining({ control_source: 'human' }), [1, 0, 0]));
});
