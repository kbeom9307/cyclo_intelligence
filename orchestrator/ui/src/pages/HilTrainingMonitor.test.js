import React from 'react';
import {render, screen, waitFor, fireEvent} from '@testing-library/react';
import HilTrainingMonitor, {episodeResult, elapsedTimes, accumulateEpisode} from './HilTrainingMonitor';
import {TextDecoder, TextEncoder} from 'util';

const session = {id:'test', mode:'train', state:'running', episode:4,
  config:{reward_source:'classifier',actor_device:'cuda:0',learner_device:'cuda:0',learner_placement:'local'},
  events:[{action:'start',at:'2026-10-21T00:00:00Z'}],
  runtime:{heartbeat:Date.now()/1000, episode:4, metrics:{completed_episodes:3,success_rate:2/3,environment_steps:12,learner_steps:5,replay_ready:true}, compute:{learner:{placement:'local'}}}};

beforeEach(() => {global.TextDecoder=TextDecoder;global.fetch=jest.fn();});
afterEach(() => {delete global.fetch;});

test('episode outcomes distinguish classifier, human labels, failure and truncation', () => {
  expect(episodeResult({terminal:false,truncated:false},session.config)).toBeNull();
  expect(episodeResult({terminal:true,environment_info:{is_success:true,classifier_success:false}},session.config).outcome).toBe('Failure');
  expect(episodeResult({terminal:true,label_only:true,reward:1},session.config).outcome).toBe('Success');
  expect(episodeResult({truncated:true,reward:0},session.config).outcome).toBe('Truncated');
  expect(episodeResult({terminal:true,environment_info:{is_success:true}},{reward_source:'environment'}).success).toBe(true);
});

test('elapsed time excludes pauses and freezes after Stop', () => {
  const at = n => new Date(n*1000).toISOString();
  const events = [{action:'start',at:at(10)},{action:'pause',at:at(30)},{action:'resume',at:at(50)},{action:'stop',at:at(90)}];
  expect(elapsedTimes({events,state:'stopped'},200)).toEqual({wall:80,active:60});
  expect(elapsedTimes({events:[],state:'created'},200)).toEqual({wall:null,active:null});
});

test('reads partial log lines across ranges and displays recent success rate without counting unfinished episodes', async () => {
  const rows=[{episode:1,terminal:true,environment_info:{classifier_success:true,episode_step:10}},
    {episode:2,truncated:true,label_only:true,reward:0},
    {episode:3,terminal:true,environment_info:{classifier_success:true,episode_step:15}},
    {episode:4,terminal:false,truncated:false}];
  const text=rows.map(r=>JSON.stringify(r)).join('\n')+'\n';
  const split=43, bytes=new TextEncoder().encode(text);
  global.fetch.mockImplementation(async (_url, options) => {
    const start=Number(options.headers.Range.match(/bytes=(\d+)/)[1]);
    const end=start===0?split:bytes.length;
    const part=bytes.slice(start,end);
    return {status:206,headers:{get:()=>`bytes ${start}-${end-1}/${bytes.length}`},arrayBuffer:async()=>part.buffer};
  });
  render(<HilTrainingMonitor session={session}/>);
  await waitFor(()=>expect(screen.getByText('Recent success rate: 66.7% (2/3 completed episodes)')).toBeInTheDocument());
  expect(screen.getByText('#3 · Success')).toBeInTheDocument();
  expect(screen.getByText('Truncated')).toBeInTheDocument();
  expect(global.fetch).toHaveBeenCalledTimes(2);
  expect(global.fetch.mock.calls[1][1].headers.Range).toMatch(/^bytes=43-/);
  fireEvent.change(screen.getByLabelText('Recent episode window'),{target:{value:'5'}});
  expect(screen.getByText('Recent success rate: 66.7% (2/3 completed episodes)')).toBeInTheDocument();
});

test('missing history does not invent outcomes', async () => {
  global.fetch.mockResolvedValue({status:404});
  render(<HilTrainingMonitor session={session}/>);
  await waitFor(()=>expect(screen.getByText('Recent success rate: —')).toBeInTheDocument());
  expect(screen.queryByText(/#3 ·/)).not.toBeInTheDocument();
});

test('shows learner warmup and then observed update rates', async () => {
  global.fetch.mockResolvedValue({status:404});
  const first={...session,runtime:{...session.runtime,metrics:{...session.runtime.metrics,replay_ready:false}}};
  const view=render(<HilTrainingMonitor session={first}/>);
  expect(screen.getByText('Waiting for replay / demo data')).toBeInTheDocument();
  const second={...session,runtime:{...session.runtime,heartbeat:session.runtime.heartbeat+2,
    metrics:{...session.runtime.metrics,environment_steps:20,learner_steps:9}}};
  view.rerender(<HilTrainingMonitor session={second}/>);
  await waitFor(()=>expect(screen.getByText('Updating')).toBeInTheDocument());
  expect(screen.getByText('4.0/s')).toBeInTheDocument();
  expect(screen.getByText('2.0/s')).toBeInTheDocument();
});


test('episode return and intervention segments follow upstream episode statistics', () => {
  let stats = null;
  for (const [i,human] of [false,true,true,false,true].entries()) {
    stats = accumulateEpisode(stats,{episode:1,reward:i===4?1:0,human,timestamp:100+i,
      terminal:i===4,environment_info:{classifier_success:i===4}},session.config);
  }
  expect(stats).toMatchObject({episode:1,return:1,steps:5,intervention_count:2,intervention_steps:3,recorded_seconds:4,outcome:'Success'});
  stats=accumulateEpisode(stats,{episode:2,reward:0,human:false,timestamp:110},session.config);
  expect(stats).toMatchObject({episode:2,return:0,steps:1,intervention_count:0,intervention_steps:0});
  stats=accumulateEpisode(stats,{episode:2,reward:1,human:true,label_only:true,terminal:true},session.config);
  expect(stats).toMatchObject({return:1,steps:1,intervention_count:0,intervention_steps:0,outcome:'Success'});
});
