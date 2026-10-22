import React, {useRef, useState} from 'react';

/** Crop the live JPEG preview with the immutable session ROI; never alter observations. */
export default function HilSessionCropFrame({frame, crop = [0,0,1,1], size, label}) {
  const canvas = useRef(null);
  const [error, setError] = useState('');
  const draw = event => {
    const image = event.currentTarget;
    const width = image.naturalWidth, height = image.naturalHeight;
    if (!width || !height) return;
    try {
      const context = canvas.current.getContext('2d');
      if (!context) throw new Error('Canvas unavailable');
      const left = Math.floor(crop[0]*width), top = Math.floor(crop[1]*height);
      const right = Math.max(left+1,Math.floor(crop[2]*width));
      const bottom = Math.max(top+1,Math.floor(crop[3]*height));
      context.drawImage(image,left,top,right-left,bottom-top,0,0,size,size);
      setError('');
    } catch { setError('Crop preview unavailable. Switch to Full image.'); }
  };
  return <div className="hil-video-frame hil-session-crop-frame">
    <canvas ref={canvas} width={size} height={size} role="img" aria-label={`${label} session crop · ${size}×${size}`} />
    <img key={`${frame}-${crop.join(',')}-${size}`} src={frame} alt="" hidden onLoad={draw} onError={() => setError('Camera image could not be decoded.')} />
    {error && <p role="status">{error}</p>}
  </div>;
}
