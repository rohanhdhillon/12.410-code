# Requires: astrometry, astropy, numpy, scipy
from pathlib import Path
import json, logging
import numpy as np
from astropy.io import fits
from scipy.ndimage import gaussian_filter, uniform_filter, maximum_filter
from scipy.spatial import cKDTree
import astrometry

logging.basicConfig(level=logging.INFO)
root=Path(__file__).resolve().parent
p=next((root/'data/big_void_galaxy/2026-10-06').glob('*00000099.fit'))
a=fits.getdata(p)
b=a.reshape(a.shape[0]//4,4,a.shape[1]//4,4).mean(axis=(1,3))
s=gaussian_filter(b,1)-uniform_filter(b,31)
sigma=1.4826*np.median(np.abs(s-np.median(s)))
y,x=np.where((s==maximum_filter(s,11)) & (s>10*sigma))
order=np.argsort(s[y,x])[::-1]
stars=[]
for k in order:
 xx,yy=x[k],y[k]
 if not (10<xx<b.shape[1]-10 and 10<yy<b.shape[0]-10):continue
 # Exclude the extended galaxy from the reference star list.
 if 1400<xx<1650 and 1000<yy<1150:continue
 stamp=b[yy-4:yy+5,xx-4:xx+5]
 weights=np.maximum(stamp-np.median(np.concatenate([stamp[0],stamp[-1],stamp[:,0],stamp[:,-1]])),0)
 gy,gx=np.indices(stamp.shape)
 cx=xx-4+(gx*weights).sum()/weights.sum()
 cy=yy-4+(gy*weights).sum()/weights.sum()
 # Convert the binned centroid to full-resolution FITS (one-based) pixels.
 stars.append([4*(cx+.5)+.5,4*(cy+.5)+.5])
 if len(stars)==250:break
stars=np.array(stars)
np.save('/tmp/initial_stars.npy',stars)
print('Extracted',len(stars),'stars',flush=True)
cache=Path('/tmp/initial-astrometry-indexes/5200/index-5206-11.fits')
if not cache.exists():
    import urllib.request
    cache.parent.mkdir(parents=True, exist_ok=True)
    urllib.request.urlretrieve('https://portal.nersc.gov/project/cosmo/temp/dstn/index-5200/LITE/index-5206-11.fits', cache)
files=[cache]
with astrometry.Solver(files) as solver:
 sol=solver.solve(stars=stars,size_hint=astrometry.SizeHint(.17,.23),position_hint=astrometry.PositionHint(267.364233,70.143664,1),solution_parameters=astrometry.SolutionParameters(positional_noise_pixels=3,sip_order=2,logodds_callback=lambda odds:astrometry.Action.STOP if odds[0]>100 else astrometry.Action.CONTINUE))
 if not sol.has_match():raise RuntimeError('No astrometric match')
 match=sol.best_match()
 wcs=match.astropy_wcs()
 out=root/'ngc6503_g.wcs.fits'
 h=wcs.to_header(relax=True)
 h['IMAGEW']=a.shape[1];h['IMAGEH']=a.shape[0]
 fits.PrimaryHDU(header=h).writeto(out,overwrite=True)
 result={'source_image':str(p.relative_to(root)),'solver':'astrometry 4.3.0 (Astrometry.net)','index':str(match.index_path.name),'n_extracted':len(stars),'logodds':match.logodds,'center_ra_deg':match.center_ra_deg,'center_dec_deg':match.center_dec_deg,'scale_arcsec_per_pixel':match.scale_arcsec_per_pixel,'reference_stars':[[t.ra_deg,t.dec_deg] for t in match.stars]}
 predicted=wcs.all_world2pix(result['reference_stars'],1)
 distances,_=cKDTree(stars).query(predicted)
 residuals=distances[distances<5]*match.scale_arcsec_per_pixel
 result.update(n_matched=len(residuals), median_residual_arcsec=float(np.median(residuals)), rms_residual_arcsec=float(np.sqrt(np.mean(residuals**2))))
 (root/'ngc6503_g.astrometry.json').write_text(json.dumps(result,indent=2)+'\n')
 print(json.dumps({k:v for k,v in result.items() if k!='reference_stars'},indent=2),flush=True)
