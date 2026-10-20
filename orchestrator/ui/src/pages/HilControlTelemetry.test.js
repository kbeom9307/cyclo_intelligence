import { render, screen, within } from '@testing-library/react';
import { HilActionTelemetry } from './HilControlTelemetry';

test('separates policy proposal from selected human command and shows preview disabled', () => {
  render(<HilActionTelemetry axes={['X','Y']} session={{mode:'preview',state:'paused',metrics:{},runtime:{episode_step:7,last_policy_action:[0,0],last_action:[.5,-.25],last_action_source:'human',last_reward:-.1}}} />);
  expect(screen.getByText(/Preview: policy disabled/)).toBeInTheDocument();
  const rows = screen.getAllByRole('row');
  expect(within(rows[1]).getByText('0.000')).toBeInTheDocument();
  expect(within(rows[1]).getByText('0.500')).toBeInTheDocument();
  expect(within(rows[2]).getByText('-0.250')).toBeInTheDocument();
  expect(screen.getByText(/Source: human/)).toBeInTheDocument();
});
