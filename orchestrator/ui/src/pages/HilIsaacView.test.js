import { render, screen, fireEvent, waitFor } from '@testing-library/react';
import HilIsaacView from './HilIsaacView';

const originalFetch = global.fetch;
beforeEach(() => {
  localStorage.clear();
  global.fetch = jest.fn().mockResolvedValue({ ok: true, headers: new Headers({ 'content-type': 'image/jpeg', 'X-Frame-Age-Ms': '25' }), blob: async () => new Blob(['jpeg']) });
  URL.createObjectURL = jest.fn().mockReturnValue('blob:test-frame');
  URL.revokeObjectURL = jest.fn();
});
afterEach(() => { global.fetch = originalFetch; });

test('shows live camera frames and pause does not send simulation commands', async () => {
  const { unmount } = render(<HilIsaacView />);
  expect(await screen.findAllByAltText(/Live Isaac simulation/)).toHaveLength(4);
  expect(screen.getAllByText(/frame age 25 ms/)).toHaveLength(4);
  screen.getAllByRole('button', { name: 'Pause video' }).forEach(button => fireEvent.click(button));
  expect(screen.queryAllByAltText(/Live Isaac simulation/)).toHaveLength(0);
  expect(screen.getAllByRole('button', { name: 'Resume video' })).toHaveLength(4);
  expect(global.fetch.mock.calls.every(([url]) => url.startsWith('/api/hilserl/isaac/frame?camera='))).toBe(true);
  unmount();
  expect(URL.revokeObjectURL).toHaveBeenCalledWith('blob:test-frame');
});

test('displays camera outage instead of a misleading static live image', async () => {
  global.fetch.mockResolvedValue({ ok: false });
  render(<HilIsaacView />);
  await waitFor(() => expect(screen.getAllByText(/Camera unavailable/).length).toBeGreaterThan(0));
  expect(screen.queryAllByAltText(/Live Isaac simulation/)).toHaveLength(0);
});

test('selects and persists quality independently for each camera', async () => {
  render(<HilIsaacView />);
  const selectors = screen.getAllByRole('combobox');
  fireEvent.change(selectors[0], { target: { value: 'high' } });
  await waitFor(() => expect(global.fetch.mock.calls.some(([url]) => url.includes('camera=spectator&quality=high'))).toBe(true));
  expect(localStorage.getItem('hil-camera-quality-spectator')).toBe('high');
  expect(selectors[1]).toHaveValue('standard');
});

test('offers native ultra resolution only for Overview', async () => {
  render(<HilIsaacView />);
  const selectors = screen.getAllByRole('combobox');
  expect(screen.getAllByRole('option', { name: /Ultra/ })).toHaveLength(1);
  fireEvent.change(selectors[0], { target: { value: 'ultra' } });
  await waitFor(() => expect(global.fetch.mock.calls.some(([url]) => url.includes('camera=spectator&quality=ultra'))).toBe(true));
  expect(localStorage.getItem('hil-camera-quality-spectator')).toBe('ultra');
  expect(selectors[1]).toHaveValue('standard');
});

test('closing a camera releases its view and persists until reopened', async () => {
  const {unmount}=render(<HilIsaacView />);
  await screen.findAllByAltText(/Live Isaac simulation/);
  fireEvent.click(screen.getByRole('button',{name:'Close Overview'}));
  expect(screen.queryByRole('region',{name:'Overview live view'})).toBeNull();
  expect(screen.getByRole('button',{name:'Open Overview'})).toBeInTheDocument();
  expect(JSON.parse(localStorage.getItem('hil-closed-cameras'))).toContain('spectator');
  expect(URL.revokeObjectURL).toHaveBeenCalled();
  unmount(); global.fetch.mockClear();
  render(<HilIsaacView />);
  await screen.findAllByAltText(/Live Isaac simulation/);
  expect(global.fetch.mock.calls.some(([url])=>url.includes('camera=spectator'))).toBe(false);
  fireEvent.click(screen.getByRole('button',{name:'Open Overview'}));
  await waitFor(()=>expect(global.fetch.mock.calls.some(([url])=>url.includes('camera=spectator'))).toBe(true));
});

test('MuJoCo renders all four views through its own bridge and switches cleanly', async () => {
  const {rerender} = render(<HilIsaacView environment="mujoco" />);
  expect(await screen.findAllByAltText(/Live MuJoCo simulation/)).toHaveLength(4);
  expect(global.fetch.mock.calls.every(([url]) => url.startsWith('/api/hilserl/mujoco/frame?'))).toBe(true);
  global.fetch.mockClear();
  rerender(<HilIsaacView environment="isaac_sim" />);
  await screen.findAllByAltText(/Live Isaac simulation/);
  expect(global.fetch.mock.calls.every(([url]) => url.startsWith('/api/hilserl/isaac/frame?'))).toBe(true);
});

test('resizes individual camera views, persists sizes and restores automatic layout', async () => {
  const {unmount, container} = render(<HilIsaacView environment="mujoco" />);
  await screen.findAllByAltText(/Live MuJoCo simulation/);
  const before = global.fetch.mock.calls.length;
  fireEvent.change(screen.getByLabelText('Overview view width'), {target:{value:'800'}});
  fireEvent.change(screen.getByLabelText('Overview view height'), {target:{value:'600'}});
  expect(container.querySelector('.hil-camera-grid').style.gridTemplateColumns).toBe('800px minmax(240px, 1fr) minmax(240px, 1fr) minmax(240px, 1fr)');
  expect(screen.getByRole('region', {name:'Overview live view'}).style.getPropertyValue('--hil-camera-height')).toBe('600px');
  expect(screen.getByRole('region', {name:'Head camera live view'}).style.getPropertyValue('--hil-camera-height')).toBe('');
  expect(global.fetch.mock.calls.length).toBe(before); // Resize never changes capture resolution or restarts requests.
  unmount();
  render(<HilIsaacView environment="mujoco" />);
  await screen.findAllByAltText(/Live MuJoCo simulation/);
  fireEvent.click(screen.getByLabelText('Individual camera sizes'));
  expect(screen.getByLabelText('Overview view width')).toHaveValue('800');
  expect(screen.getByLabelText('Overview view height')).toHaveValue('600');
  fireEvent.click(screen.getAllByRole('button', {name:'Reset view size'})[0]);
  expect(screen.getByRole('region', {name:'Overview live view'}).style.getPropertyValue('--hil-camera-height')).toBe('');
  expect(JSON.parse(localStorage.getItem('hil-camera-sizes')).spectator).toEqual({});
});

test('invalid stored display dimensions fall back to automatic sizing', async () => {
  localStorage.setItem('hil-camera-sizes', JSON.stringify({head:{width:-10,height:99999}}));
  const {container} = render(<HilIsaacView />);
  await screen.findAllByAltText(/Live Isaac simulation/);
  expect(container.querySelector('.hil-camera-grid').style.gridTemplateColumns).not.toContain('-10');
  expect(screen.getByRole('region', {name:'Head camera live view'}).style.getPropertyValue('--hil-camera-height')).toBe('320px');
});

test.each(['classifier','demo','train'])('shows saved session crops for %s and allows full-frame comparison', async mode => {
  const sessionConfig={learning_backend:'hil_serl',observation_mode:'pixels',image_size:256,
    camera_crops:{head:[.2,.1,.8,.9]},observation_cameras:['head'],classifier_cameras:['left_wrist'],reward_source:'classifier'};
  render(<HilIsaacView environment="mujoco" sessionId="saved" sessionMode={mode} sessionConfig={sessionConfig} />);
  expect(await screen.findByRole('img',{name:'Head camera session crop · 256×256'})).toBeInTheDocument();
  expect(screen.getAllByRole('button',{name:'Show full image'})).toHaveLength(3);
  expect(screen.getByAltText('Live MuJoCo simulation · Overview')).toBeInTheDocument();
  expect(screen.getByText(/Session crop · 256×256 · Classifier input/)).toBeInTheDocument();
  fireEvent.click(screen.getAllByRole('button',{name:'Show full image'})[0]);
  expect(screen.getByAltText('Live MuJoCo simulation · Head camera')).toBeInTheDocument();
  fireEvent.click(screen.getByRole('button',{name:'Show session crop'}));
  expect(screen.getByRole('img',{name:'Head camera session crop · 256×256'})).toBeInTheDocument();
});


test('uniform layout ignores old per-camera sizes and changes all viewport heights together', async () => {
  localStorage.setItem('hil-camera-sizes',JSON.stringify({spectator:{width:1200,height:800},head:{width:240,height:180}}));
  const {container}=render(<HilIsaacView sessionConfig={{learning_backend:'hil_serl',observation_mode:'pixels'}} sessionMode="train" />);
  await screen.findByRole('img',{name:'Head camera session crop · 128×128'});
  const grid=container.querySelector('.hil-camera-grid');
  expect(grid).toHaveClass('hil-camera-grid-uniform');
  expect(grid.style.gridTemplateColumns).toBe(Array(4).fill('minmax(240px, 1fr)').join(' '));
  fireEvent.change(screen.getByLabelText('All camera viewport heights'),{target:{value:'400'}});
  expect(grid.style.getPropertyValue('--hil-uniform-height')).toBe('400px');
  for (const card of container.querySelectorAll('.hil-live-view')) expect(card.style.getPropertyValue('--hil-camera-height')).toBe('400px');
  fireEvent.click(screen.getAllByRole('button',{name:'Show full image'})[0]);
  expect(grid).toHaveClass('hil-camera-grid-uniform');
});
