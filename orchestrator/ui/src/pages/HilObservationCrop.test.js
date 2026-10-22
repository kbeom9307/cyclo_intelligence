import {render, screen, fireEvent} from '@testing-library/react';
import HilObservationCrop, {cropPoint, selectedCrop} from './HilObservationCrop';

const oldFetch=global.fetch, oldPointer=window.PointerEvent;
beforeEach(() => {
  window.PointerEvent=MouseEvent;
  global.fetch=jest.fn().mockResolvedValue({ok:true,headers:new Headers({'content-type':'image/jpeg'}),blob:async()=>new Blob(['frame'])});
  URL.createObjectURL=jest.fn().mockReturnValue('blob:crop'); URL.revokeObjectURL=jest.fn();
});
afterEach(() => {global.fetch=oldFetch; window.PointerEvent=oldPointer;});

test('normalizes source coordinates in both directions and clamps the crop', () => {
  expect(cropPoint(210,120,{left:10,top:20,width:400,height:200})).toEqual([.5,.5]);
  expect(cropPoint(-10,900,{left:10,top:20,width:400,height:200})).toEqual([0,1]);
  expect(selectedCrop([.8,.9],[.2,.1])).toEqual([.2,.1,.8,.9]);
});

test('drag edits the selected camera only, preview follows props, and reset restores full frame', async () => {
  const onChange=jest.fn();
  const crops={right_wrist:[.1,.2,.7,.8]};
  const props={environment:'mujoco',size:256,crops,onChange};
  const {rerender,unmount}=render(<HilObservationCrop {...props} />);
  const area=await screen.findByLabelText('head training crop area');
  area.getBoundingClientRect=()=>({left:0,top:0,width:400,height:200});
  fireEvent.pointerDown(area,{button:0,clientX:300,clientY:150});
  fireEvent.pointerMove(area,{clientX:100,clientY:50});
  expect(screen.getByLabelText('Resized training crop preview')).toHaveAttribute('viewBox','0.25 0.25 0.5 0.5');
  fireEvent.pointerUp(area,{clientX:100,clientY:50});
  expect(onChange).toHaveBeenCalledWith({...crops,head:[.25,.25,.75,.75]});
  rerender(<HilObservationCrop {...props} crops={{...crops,head:[.25,.25,.75,.75]}} />);
  expect(screen.getByText(/before resizing to 256×256/)).toBeInTheDocument();
  fireEvent.click(screen.getByRole('button',{name:'Use full image'}));
  expect(onChange).toHaveBeenLastCalledWith(crops);
  fireEvent.change(screen.getByLabelText('Crop camera'),{target:{value:'right_wrist'}});
  await screen.findByLabelText('right_wrist training crop area');
  expect(screen.getByLabelText('right_wrist crop Left')).toHaveValue(.1);
  expect(fetch.mock.calls.every(([url])=>url.includes('/mujoco/frame?'))).toBe(true);
  unmount(); expect(URL.revokeObjectURL).toHaveBeenCalled();
});

test('cancel and tiny clicks do not save a crop, unavailable cameras remain editable numerically', async () => {
  const onChange=jest.fn();
  render(<HilObservationCrop environment="isaac_sim" onChange={onChange} />);
  const area=await screen.findByLabelText('head training crop area');
  area.getBoundingClientRect=()=>({left:0,top:0,width:400,height:200});
  fireEvent.pointerDown(area,{button:0,clientX:100,clientY:100});
  fireEvent.pointerCancel(area);
  fireEvent.pointerUp(area,{clientX:300,clientY:150});
  fireEvent.pointerDown(area,{button:0,clientX:100,clientY:100});
  fireEvent.pointerUp(area,{clientX:100,clientY:100});
  expect(onChange).not.toHaveBeenCalled();
  fetch.mockResolvedValue({ok:false});
  fireEvent.click(screen.getByRole('button',{name:'Refresh image'}));
  expect(await screen.findByText(/Camera unavailable/)).toBeInTheDocument();
  expect(screen.queryByLabelText('head training crop area')).toBeNull();
  fireEvent.change(screen.getByLabelText('head crop Left'),{target:{value:'.2'}});
  expect(onChange).toHaveBeenCalledWith({head:[.2,0,1,1]});
});
