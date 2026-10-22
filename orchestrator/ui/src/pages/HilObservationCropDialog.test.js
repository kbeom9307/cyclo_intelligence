import {render, screen, fireEvent} from '@testing-library/react';
import HilObservationCropDialog from './HilObservationCropDialog';

beforeEach(() => {
  HTMLDialogElement.prototype.showModal = function () { this.setAttribute('open',''); };
  HTMLDialogElement.prototype.close = function () { this.removeAttribute('open'); };
  global.fetch=jest.fn().mockResolvedValue({ok:false});
});

test('editor mounts only on demand, cancel discards, apply updates crops without submitting', async () => {
  const onChange=jest.fn(), onSubmit=jest.fn(e=>e.preventDefault());
  render(<form onSubmit={onSubmit}><HilObservationCropDialog environment="mujoco" size={256} crops={{}} onChange={onChange} /></form>);
  expect(screen.queryByRole('dialog')).not.toBeInTheDocument();
  expect(fetch).not.toHaveBeenCalled();
  fireEvent.click(screen.getByRole('button',{name:'Edit observation crop'}));
  expect(await screen.findByText(/Camera unavailable/)).toBeInTheDocument();
  expect(screen.getByRole('dialog',{name:'Observation crop editor'})).toBeInTheDocument();
  fireEvent.change(screen.getByLabelText('head crop Left'),{target:{value:'.2'}});
  fireEvent.click(screen.getByRole('button',{name:'Cancel'}));
  expect(onChange).not.toHaveBeenCalled();
  expect(screen.queryByRole('dialog')).not.toBeInTheDocument();
  fireEvent.click(screen.getByRole('button',{name:'Edit observation crop'}));
  await screen.findByText(/Camera unavailable/);
  expect(screen.getByLabelText('head crop Left')).toHaveValue(0);
  fireEvent.change(screen.getByLabelText('head crop Left'),{target:{value:'.3'}});
  fireEvent.click(screen.getByRole('button',{name:'Apply crop'}));
  expect(onChange).toHaveBeenCalledWith({head:[.3,0,1,1]});
  expect(onSubmit).not.toHaveBeenCalled();
  expect(screen.queryByRole('dialog')).not.toBeInTheDocument();
});

test('invalid crops cannot apply and Escape cancellation discards edits', async () => {
  const onChange=jest.fn();
  render(<HilObservationCropDialog environment="mujoco" size={128} onChange={onChange} />);
  fireEvent.click(screen.getByRole('button',{name:'Edit observation crop'}));
  await screen.findByText(/Camera unavailable/);
  fireEvent.change(screen.getByLabelText('head crop Left'),{target:{value:'1'}});
  expect(screen.getByRole('button',{name:'Apply crop'})).toBeDisabled();
  fireEvent(screen.getByRole('dialog'),new Event('cancel',{bubbles:false,cancelable:true}));
  expect(screen.queryByRole('dialog')).not.toBeInTheDocument();
  expect(onChange).not.toHaveBeenCalled();
});
