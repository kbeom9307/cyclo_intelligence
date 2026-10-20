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
