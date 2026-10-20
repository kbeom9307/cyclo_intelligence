import { render, screen, fireEvent } from '@testing-library/react';
import HilOverviewCrop, { overviewPoint, cropRectangle, FULL_VIEW } from './HilOverviewCrop';

test('maps letterboxed fullscreen coordinates and clamps to the visible crop', () => {
  const box = { left: 10, top: 20, width: 800, height: 400 };
  expect(overviewPoint(410, 220, box, FULL_VIEW)).toEqual({ x: 2, y: 1.5 });
  expect(overviewPoint(-100, 900, box, FULL_VIEW)).toEqual({ x: 0, y: 3 });
  expect(overviewPoint(410, 220, box, { x: 1, y: 1, width: 2, height: 1 })).toEqual({ x: 2, y: 1.5 });
  expect(cropRectangle({ x: 3, y: 2 }, { x: 1, y: 1 })).toEqual({ x: 1, y: 1, width: 2, height: 1 });
});

test('drag zoom persists on fresh frames, cancellation is harmless and reset restores full view', () => {
  const original = window.PointerEvent;
  window.PointerEvent = MouseEvent;
  const { rerender } = render(<HilOverviewCrop frame="first.jpg" label="Overview" />);
  const svg = screen.getByLabelText('Overview crop area');
  svg.getBoundingClientRect = () => ({ left: 0, top: 0, width: 400, height: 300 });
  fireEvent.pointerDown(svg, { button: 0, clientX: 300, clientY: 200 });
  fireEvent.pointerMove(svg, { clientX: 100, clientY: 100 });
  fireEvent.pointerUp(svg, { clientX: 100, clientY: 100 });
  expect(svg).toHaveAttribute('viewBox', '1 1 2 1');
  rerender(<HilOverviewCrop frame="next.jpg" label="Overview" />);
  expect(svg).toHaveAttribute('viewBox', '1 1 2 1');
  fireEvent.pointerDown(svg, { button: 0, clientX: 100, clientY: 100 });
  fireEvent.pointerCancel(svg);
  fireEvent.pointerUp(svg, { clientX: 300, clientY: 200 });
  expect(svg).toHaveAttribute('viewBox', '1 1 2 1');
  fireEvent.click(screen.getByText('Reset crop'));
  expect(svg).toHaveAttribute('viewBox', '0 0 4 3');
  window.PointerEvent = original;
});
