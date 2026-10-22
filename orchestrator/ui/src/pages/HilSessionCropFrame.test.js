import {render, screen, fireEvent} from '@testing-library/react';
import HilSessionCropFrame from './HilSessionCropFrame';

test('crops the session ROI with pixel boundaries then resizes to the training dimensions', () => {
  const drawImage=jest.fn();
  const context=jest.spyOn(HTMLCanvasElement.prototype,'getContext').mockReturnValue({drawImage});
  const {container}=render(<HilSessionCropFrame frame="frame.jpg" crop={[.25,.2,.75,.8]} size={256} label="Head" />);
  const image=container.querySelector('img');
  Object.defineProperties(image,{naturalWidth:{value:640},naturalHeight:{value:360}});
  fireEvent.load(image);
  expect(drawImage).toHaveBeenCalledWith(image,160,72,320,216,0,0,256,256);
  expect(screen.getByRole('img',{name:'Head session crop · 256×256'})).toHaveAttribute('width','256');
  context.mockRestore();
});
