import { render, fireEvent, screen, waitFor, act } from '@testing-library/react';
import useHilCaptureKeyboard from './useHilCaptureKeyboard';
import { sendHilClassifierLabel } from '../utils/hilserlApi';
jest.mock('../utils/hilserlApi');
const onError = jest.fn();
function Harness({ enabled = true, mode = 'classifier', toggle = false }) {
  useHilCaptureKeyboard({ id: 's', revision: 2, mode }, enabled, onError, toggle);
  return <input aria-label="Edit" />;
}
beforeEach(() => { jest.clearAllMocks(); sendHilClassifierLabel.mockResolvedValue({}); });
test('holds success until key release and ignores key repeat', async () => {
  render(<Harness />);
  fireEvent.keyDown(window, { code: 'Space' });
  fireEvent.keyDown(window, { code: 'Space', repeat: true });
  await waitFor(() => expect(sendHilClassifierLabel).toHaveBeenCalledTimes(1));
  expect(sendHilClassifierLabel).toHaveBeenLastCalledWith({ id: 's', revision: 2 }, true);
  fireEvent.keyUp(window, { code: 'Space' });
  await waitFor(() => expect(sendHilClassifierLabel).toHaveBeenLastCalledWith({ id: 's', revision: 2 }, false));
});
test('editing, paused sessions and other modes never send success', () => {
  const { rerender } = render(<Harness />);
  fireEvent.keyDown(screen.getByLabelText('Edit'), { code: 'Space' });
  fireEvent.keyDown(window, { code: 'Space', ctrlKey: true });
  rerender(<Harness enabled={false} />); fireEvent.keyDown(window, { code: 'Space' });
  rerender(<Harness mode="preview" />); fireEvent.keyDown(window, { code: 'Space' });
  expect(sendHilClassifierLabel).not.toHaveBeenCalled();
});
test('release waits for an in-flight success request, preserving order', async () => {
  let resolve;
  sendHilClassifierLabel.mockImplementationOnce(() => new Promise(done => { resolve = done; }));
  render(<Harness />);
  fireEvent.keyDown(window, { code: 'Space' });
  fireEvent.blur(window);
  expect(sendHilClassifierLabel).toHaveBeenCalledTimes(1);
  await act(async () => resolve({}));
  expect(sendHilClassifierLabel).toHaveBeenLastCalledWith(expect.anything(), false);
});
test('refreshes the success lease while held and releases on unmount', async () => {
  jest.useFakeTimers();
  const { unmount } = render(<Harness />);
  fireEvent.keyDown(window, {code:'Space'});
  await act(async () => {});
  await act(async () => { jest.advanceTimersByTime(100); });
  expect(sendHilClassifierLabel).toHaveBeenCalledTimes(2);
  unmount();
  expect(sendHilClassifierLabel).toHaveBeenLastCalledWith(expect.anything(), false);
  jest.useRealTimers();
});

test('toggle holds success after keyup, ignores repeat, and next press releases', async () => {
  render(<Harness toggle />);
  fireEvent.keyDown(window, {code:'Space'});
  await act(async () => {});
  fireEvent.keyUp(window, {code:'Space'});
  fireEvent.keyDown(window, {code:'Space',repeat:true});
  expect(sendHilClassifierLabel).toHaveBeenCalledTimes(1);
  expect(sendHilClassifierLabel).toHaveBeenLastCalledWith(expect.anything(),true);
  fireEvent.keyDown(window, {code:'Space'});
  await waitFor(()=>expect(sendHilClassifierLabel).toHaveBeenLastCalledWith(expect.anything(),false));
});

test('toggle clears on mode change and blur', async () => {
  const {rerender}=render(<Harness toggle />);
  fireEvent.keyDown(window,{code:'Space'});
  await act(async () => {});
  rerender(<Harness toggle={false} />);
  await waitFor(()=>expect(sendHilClassifierLabel).toHaveBeenLastCalledWith(expect.anything(),false));
  rerender(<Harness toggle />);
  fireEvent.keyDown(window,{code:'Space'});
  await act(async () => {});
  fireEvent.blur(window);
  await waitFor(()=>expect(sendHilClassifierLabel).toHaveBeenLastCalledWith(expect.anything(),false));
});
