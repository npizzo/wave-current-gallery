import math
import numpy as np
def check_scalar_maps(scalar,maps):
    by={round(r['time_s'],7):r for r in scalar};errors=[]
    keys=[f'{region}_{suffix}' for region in ('full','interior','stokes') for suffix in
          ([f'Z_{part}' for part in ('Q','eddy','profile')]+[f'P_{part}_{term}' for part in ('Q','eddy','profile') for term in ('A0','C','W','D','F')])]
    for key in keys:
        pairs=[];bounds=[]
        for r in maps:
            s=by.get(round(r['time_s'],7))
            if s is None:raise ValueError('Scalar budget lacks a saved map time')
            pairs.append((r[key],s[key]))
            if '_P_' in key:
                region,_,part,term=key.split('_')
                bounds.append(max(math.sqrt(2*max(x[f'{region}_Z_{part}'],0))*x[f'{region}_source_rms_{term}'] for x in (r,s)))
        p=np.array(pairs);error=float(np.max(abs(p[:,0]-p[:,1])));scale=float(np.max(abs(p)))
        natural=max(bounds,default=scale)
        tolerance=1e-9*scale+128*np.finfo(float).eps*natural
        if not np.isfinite(p).all() or not math.isfinite(natural) or error>max(tolerance,1e-300):
            raise ValueError(f'Scalar/map mismatch {key}: error={error}, tolerance={tolerance}')
        errors.append(error/max(scale,natural,1e-300))
    return max(errors)
