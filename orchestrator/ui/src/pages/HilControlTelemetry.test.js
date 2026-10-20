import { render, screen, within } from '@testing-library/react';
import { HilActionTelemetry, HilGpuInventory, HilComputeTelemetry } from './HilControlTelemetry';

test('separates policy proposal from selected human command and shows preview disabled', () => {
  render(<HilActionTelemetry axes={['X','Y']} session={{mode:'preview',state:'paused',metrics:{},runtime:{episode_step:7,last_policy_action:[0,0],last_action:[.5,-.25],last_action_source:'human',last_reward:-.1}}} />);
  expect(screen.getByText(/Preview: policy disabled/)).toBeInTheDocument();
  const rows = screen.getAllByRole('row');
  expect(within(rows[1]).getByText('0.000')).toBeInTheDocument();
  expect(within(rows[1]).getByText('0.500')).toBeInTheDocument();
  expect(within(rows[2]).getByText('-0.250')).toBeInTheDocument();
  expect(screen.getByText(/Source: human/)).toBeInTheDocument();
});

test('shows live classifier logit, probability and hold count during demo', () => {
  render(<HilActionTelemetry axes={['X']} session={{mode:'demo', state:'running', metrics:{},
    config:{reward_source:'classifier',classifier_threshold:.9,classifier_hold_steps:3,classifier_cameras:['head']},
    runtime:{classifier_enabled:true,classifier_logit:4.2,classifier_probability:.985,classifier_hold_steps:2}}} />);
  const panel=screen.getByRole('region',{name:'Live success classifier'});
  expect(panel).toHaveTextContent('Logit: 4.200');
  expect(panel).toHaveTextContent('Success probability: 0.985');
  expect(panel).toHaveTextContent('Probability threshold: 0.900');
  expect(panel).toHaveTextContent('Consecutive success steps: 2 / 3');
});


test('separates used and free VRAM and identifies shared process readings', () => {
  render(<><HilGpuInventory hardware={{gpus:[{device:'cuda:0',name:'GPU',total_mib:8192,used_mib:7168,free_mib:1024}]}} />
    <HilComputeTelemetry runtime={{compute:{actor_device:'cuda:0',shared_process:true,actor_memory:{allocated_mib:32,reserved_mib:64,free_mib:512},learner:{placement:'local',learner_memory:{device:'cuda:0',allocated_mib:32,reserved_mib:64,free_mib:512}}}}} /></>);
  expect(screen.getByText(/Total 8.00 GiB · Used 7.00 GiB · Free 1.00 GiB/)).toBeInTheDocument();
  expect(screen.getByText(/these memory readings overlap; do not add them/)).toBeInTheDocument();
});
