import { combineHilDemos, getHilWorkspace } from './hilserlApi';

beforeEach(() => {
  jest.useFakeTimers();
  global.fetch = jest.fn();
});
afterEach(() => { jest.useRealTimers(); jest.restoreAllMocks(); });

test('demo export waits beyond normal request timeout and preserves all selections', async () => {
  let finish;
  fetch.mockImplementation(() => new Promise(resolve => { finish = resolve; }));
  const selections = [{session_id:'a',episodes:[1,2]}, {session_id:'b',episodes:[3]}];
  const pending = combineHilDemos(selections);
  const options = fetch.mock.calls[0][1];
  jest.advanceTimersByTime(20000);
  expect(options.signal.aborted).toBe(false);
  expect(JSON.parse(options.body)).toEqual({selections});
  finish({ok:true,json:async () => ({path:'/combined.npz'})});
  await expect(pending).resolves.toEqual({path:'/combined.npz'});
  expect(jest.getTimerCount()).toBe(0);
});

test.each([[getHilWorkspace,15000],[() => combineHilDemos([]),120000]])('requests still have a bounded timeout', async (call, timeout) => {
  fetch.mockImplementation((url, {signal}) => new Promise((resolve,reject) => {
    signal.addEventListener('abort', () => reject(Object.assign(new Error(), {name:'AbortError'})));
  }));
  const pending = call();
  const checked = expect(pending).rejects.toThrow('Request timed out');
  jest.advanceTimersByTime(timeout);
  await checked;
});
