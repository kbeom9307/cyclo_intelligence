import { render, fireEvent, waitFor, screen } from '@testing-library/react';
import { HilPressedKeys } from './HilControlTelemetry';
import useHilKeyboard from './useHilKeyboard';
import { sendHilAction } from '../utils/hilserlApi';
jest.mock('../utils/hilserlApi');
const session = { id: 'test', revision: 2, config: { environment: 'isaac_sim' } };
const error = jest.fn();
function Harness({ enabled = true }) { const pressed = useHilKeyboard(session, enabled, error); return <><input aria-label="Edit" /><HilPressedKeys pressed={pressed} enabled={enabled} isaac /></>; }
beforeEach(() => { jest.clearAllMocks(); sendHilAction.mockResolvedValue({}); });
test('keys command the selected axes and release sends hold', async () => {
  render(<Harness />);
  fireEvent.keyDown(window, { code: 'KeyW' });
  await waitFor(() => expect(sendHilAction).toHaveBeenCalledWith({ id: 'test', revision: 2 }, [1,0,0,0,0,0]));
  fireEvent.keyUp(window, { code: 'KeyW' });
  await waitFor(() => expect(sendHilAction).toHaveBeenLastCalledWith(expect.anything(), [0,0,0,0,0,0]));
});
test('editing or unavailable takeover does not move; blur clears held keys', async () => {
  const { rerender } = render(<Harness enabled={false} />);
  fireEvent.keyDown(window, { code: 'KeyW' }); expect(sendHilAction).not.toHaveBeenCalled();
  rerender(<Harness />);
  fireEvent.keyDown(screen.getByLabelText('Edit'), { code: 'KeyW' }); expect(sendHilAction).not.toHaveBeenCalled();
  fireEvent.keyDown(window, { code: 'ArrowUp' });
  await waitFor(() => expect(sendHilAction).toHaveBeenLastCalledWith(expect.anything(), [0,0,0,0,1,0]));
  fireEvent.blur(window);
  await waitFor(() => expect(sendHilAction).toHaveBeenLastCalledWith(expect.anything(), [0,0,0,0,0,0]));
});

test('visualizes simultaneous keys and clears each released key and blur', () => {
  render(<Harness />);
  fireEvent.keyDown(window, { code: 'KeyW' });
  fireEvent.keyDown(window, { code: 'KeyA' });
  expect(screen.getByLabelText('W pressed')).toHaveAttribute('data-pressed', 'true');
  expect(screen.getByLabelText('A pressed')).toHaveAttribute('data-pressed', 'true');
  fireEvent.keyUp(window, { code: 'KeyW' });
  expect(screen.getByLabelText('W released')).toHaveAttribute('data-pressed', 'false');
  expect(screen.getByLabelText('A pressed')).toBeInTheDocument();
  fireEvent.blur(window);
  expect(screen.getByText('No keys pressed')).toBeInTheDocument();
});

test('fixed gripper ignores Z/X keyboard input', () => {
  render(<Harness />);
  fireEvent.keyDown(window, {code:'KeyZ'});
  fireEvent.keyDown(window, {code:'KeyX'});
  expect(sendHilAction).not.toHaveBeenCalled();
  expect(screen.queryByText('Gripper')).not.toBeInTheDocument();
});
