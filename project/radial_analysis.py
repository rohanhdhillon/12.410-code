"""Residual sky, common elliptical annuli, and foreground-extinction diagnostics.

Rates are ADU/s on the common WCS grid. Missing calibrations stay instrumental.
Original images and the earlier foreground-review products are never modified.
"""
from pathlib import Path
import json
import warnings
import numpy as np
from astropy.table import Table
from astropy.io import fits
from scipy.ndimage import binary_dilation
from scipy.spatial import cKDTree
import photometric_analysis as pa


def ellipse_radius(shape, g):
    y, x = np.indices(shape, dtype=np.float32)
    t = np.deg2rad(g['major_axis_angle_deg'])
    x -= g['center_xy'][0]; y -= g['center_xy'][1]
    return np.hypot(x*np.cos(t)+y*np.sin(t),
                    (-x*np.sin(t)+y*np.cos(t))/g['axis_ratio'])


def source_masks(frame, table, gcfg, cfg, output):
    """Keep strong Gaia evidence in crowded regions; defer other crowded groups."""
    shape = frame['data'].shape
    red = np.zeros(shape, bool); review = np.zeros(shape, bool)
    chosen = np.asarray(table['auto_mask'], bool).copy()
    points = np.column_stack([table['x'], table['y']])
    reason = np.full(len(table), 'review candidate', dtype='U90')
    if len(table):
        ids = np.flatnonzero(chosen)
        if len(ids):
            neighbors = cKDTree(points[ids]).query_ball_point(
                points[ids], cfg['crowded_group_radius_arcsec']/frame['scale'])
            for j, nearby in zip(ids, neighbors):
                strong_gaia = (str(table['id'][j]).startswith('gaia:')
                               and table['classification'][j] == 'foreground_star')
                manual = str(table['id'][j]) in cfg['manual_accept_ids']
                inside = ellipse_radius_at(points[j], gcfg) <= gcfg['outer_radius_pixels']
                crowded = len(nearby) >= cfg['crowded_group_minimum']
                chosen[j] = not (inside and crowded and not strong_gaia and not manual)
                reason[j] = ('secure foreground/background evidence' if chosen[j]
                             else 'crowded on-galaxy group without strong Gaia evidence; deferred')
        for j, row in enumerate(table):
            region, weight = pa.aperture_weights(shape, row['x'], row['y'],
                                                 row['mask_radius_arcsec']/frame['scale'])
            if region is not None:
                (red if chosen[j] else review)[region] |= weight > 0
    audit = table.copy(); audit['radial_masked'] = chosen; audit['radial_reason'] = reason
    output = Path(output); output.mkdir(parents=True, exist_ok=True)
    audit.write(output/'source_mask_audit.ecsv', overwrite=True)
    return red, review


def ellipse_radius_at(xy, g):
    dx, dy = np.asarray(xy)-g['center_xy']; t = np.deg2rad(g['major_axis_angle_deg'])
    return np.hypot(dx*np.cos(t)+dy*np.sin(t), (-dx*np.sin(t)+dy*np.cos(t))/g['axis_ratio'])


def _plane(x, y, shape):
    ny, nx = shape
    return np.array([1., (x-(nx-1)/2)/(nx/2), (y-(ny-1)/2)/(ny/2)])


def _robust_plane(X, values, minimum):
    keep = np.isfinite(values)
    for _ in range(8):
        if keep.sum() < minimum or np.linalg.matrix_rank(X[keep]) < 3:
            raise ValueError('Too few spatially distributed off-galaxy sky boxes for a plane.')
        beta = np.linalg.lstsq(X[keep], values[keep], rcond=None)[0]
        residual = values-X@beta; scatter = pa.mad(residual[keep])
        if not np.isfinite(scatter) or scatter == 0: break
        new = keep & (np.abs(residual) <= 3.5*scatter)
        if np.array_equal(new, keep): break
        keep = new
    beta = np.linalg.lstsq(X[keep], values[keep], rcond=None)[0]
    return beta, keep


def fit_sky(frames, source_mask, gcfg, cfg, output):
    """Fit residual planes to clipped off-galaxy boxes; resample spatial blocks jointly."""
    import sep
    output = Path(output); output.mkdir(parents=True, exist_ok=True)
    bands = list('gri'); shape = frames['r']['data'].shape; ny, nx = shape
    rad = ellipse_radius(shape, gcfg)
    usable = np.ones(shape, bool)
    for frame in frames.values(): usable &= ~frame['mask']
    usable &= ~source_mask
    # Use one stable reference for segmentation. Re-detecting on the union of
    # three increasingly perforated masks creates spurious mask-edge detections.
    frame = frames[cfg.get('sky_detection_band','r')]
    image = np.ascontiguousarray(np.where(usable, frame['data'], 0), dtype=np.float32)
    background = sep.Background(image, mask=~usable, bw=128, bh=128)
    _, seg = sep.extract(image-background.back(), cfg.get('sky_detection_sigma',5.),
                         err=background.rms(), mask=~usable, minarea=5, segmentation_map=True)
    usable &= ~binary_dilation(seg > 0, iterations=cfg['sky_source_dilation_pixels'])
    rows = []; side = cfg['sky_box_pixels']
    for y in range(0, ny-side+1, side):
        for x in range(0, nx-side+1, side):
            sl = np.s_[y:y+side, x:x+side]
            good = usable[sl] & (rad[sl] > gcfg['sky_exclusion_pixels'])
            if good.mean() < cfg['sky_box_valid_fraction']: continue
            vals = []
            for b in bands:
                a = frames[b]['data'][sl][good]; keep = np.ones(len(a), bool)
                for _ in range(4):
                    center = np.median(a[keep]); scatter = pa.mad(a[keep])
                    if scatter == 0: break
                    keep &= np.abs(a-center) <= 3*scatter
                vals.append(float(np.mean(a[keep])))
            # Use the actual mean coordinates of the admitted pixels.
            yy, xx = np.nonzero(good)
            rows.append((x+xx.mean(), y+yy.mean(), rad[y+side//2, x+side//2], *vals))
    boxes = Table(rows=rows, names=['x','y','elliptical_radius_pixels','g','r','i'])
    if len(boxes) < cfg['minimum_sky_boxes']:
        raise ValueError(f'Only {len(boxes)} sky boxes; inspect sky exclusion/coverage before photometry.')
    X = np.array([_plane(x,y,shape) for x,y in zip(boxes['x'],boxes['y'])])
    Y = np.column_stack([boxes[b] for b in bands]); coefficients = []; kept = []
    for j, b in enumerate(bands):
        beta, keep = _robust_plane(X, Y[:,j], cfg['minimum_sky_boxes'])
        coefficients.append(beta); kept.append(keep)
        boxes[b+'_used'] = keep; boxes[b+'_residual'] = Y[:,j]-X@beta
    coefficients = np.array(coefficients); kept = np.array(kept)
    block = cfg['sky_bootstrap_block_pixels']
    group = (np.asarray(boxes['x'])//block + (1+nx//block)*(np.asarray(boxes['y'])//block)).astype(int)
    groups = np.unique(group)
    if len(groups) < 6: raise ValueError('Fewer than six independent sky blocks; sky uncertainty is not constrained.')
    rng = np.random.default_rng(cfg['random_seed']); draws = []
    for _ in range(cfg['sky_bootstrap_draws']):
        ids = np.concatenate([np.flatnonzero(group == k) for k in rng.choice(groups,len(groups))])
        result = []
        for j in range(3):
            use = ids[kept[j,ids]]
            if len(use)<6 or np.linalg.matrix_rank(X[use])<3: break
            result.extend(np.linalg.lstsq(X[use],Y[use,j],rcond=None)[0])
        if len(result)==9: draws.append(result)
    if len(draws) < cfg['sky_bootstrap_draws']//2: raise ValueError('Too few valid spatial sky bootstrap fits.')
    covariance = np.cov(np.array(draws), rowvar=False)
    alternatives = []
    for factor in cfg['sky_exclusion_sensitivity_factors']:
        eligible = np.asarray(boxes['elliptical_radius_pixels']) > factor*gcfg['sky_exclusion_pixels']
        if eligible.sum() < cfg['minimum_sky_boxes']: continue
        alternatives.append([_robust_plane(X[eligible],Y[eligible,j],cfg['minimum_sky_boxes'])[0]
                             for j in range(3)])
    result = dict(coefficients=coefficients, covariance=covariance,
                  alternatives=np.asarray(alternatives), boxes=boxes,
                  residual_floor=np.array([pa.mad(boxes[b+'_residual'][kept[j]]) for j,b in enumerate(bands)]))
    boxes.write(output/'sky_boxes.ecsv',overwrite=True)
    np.savez(output/'sky_model.npz', **{k:v for k,v in result.items() if k!='boxes'})
    return result


def annulus_weights(shape, gcfg, inner, outer, subpixels):
    from photutils.aperture import EllipticalAperture, EllipticalAnnulus
    q = gcfg['axis_ratio']; t = np.deg2rad(gcfg['major_axis_angle_deg'])
    aperture = (EllipticalAperture(gcfg['center_xy'],outer,q*outer,theta=t) if inner==0
                else EllipticalAnnulus(gcfg['center_xy'],inner,outer,q*outer,b_in=q*inner,theta=t))
    mask = aperture.to_mask(method='subpixel',subpixels=subpixels)
    large, small = mask.get_overlap_slices(shape)
    return large, None if large is None else mask.data[small]


def measure_annuli(frames, sky, red, review, gcfg, cfg, edges, error_paths):
    """Mean intensity on identical admitted pixels in all bands; never zero-fill holes."""
    shape = frames['r']['data'].shape; scale = frames['r']['scale']; n = len(edges)-1
    common = ~red.copy()
    for frame in frames.values(): common &= ~frame['mask']
    area = np.zeros(n); full = np.pi*gcfg['axis_ratio']*np.diff(np.asarray(edges)**2)
    basis = np.zeros((n,3)); mean = np.full((3,n),np.nan)
    random = np.zeros((3,n)); flat = np.zeros((3,n)); review_delta = np.full((3,n),np.nan)
    for k,(lo,hi) in enumerate(zip(edges[:-1],edges[1:])):
        sl,w = annulus_weights(shape,gcfg,lo,hi,cfg['annulus_subpixels'])
        if sl is None: continue
        w = w*common[sl]; area[k] = w.sum()
        if area[k]==0: continue
        yy,xx = np.indices(w.shape); xx += sl[1].start; yy += sl[0].start
        basis[k] = [1.,np.sum(w*(xx-(shape[1]-1)/2)/(shape[1]/2))/area[k],
                    np.sum(w*(yy-(shape[0]-1)/2)/(shape[0]/2))/area[k]]
    for j,b in enumerate('gri'):
        frame = frames[b]
        # Separate random detector noise from the shared-master bound.
        with fits.open(error_paths[b],memmap=False) as h:
            rand = h['VARRAND'].data if 'VARRAND' in h else h['VAR'].data
            cal = h['VARCAL'].data if 'VARCAL' in h else np.zeros(shape,np.float32)
            for k,(lo,hi) in enumerate(zip(edges[:-1],edges[1:])):
                if area[k]==0: continue
                sl,w = annulus_weights(shape,gcfg,lo,hi,cfg['annulus_subpixels'])
                w = w*common[sl]/area[k]
                data = np.where(common[sl],frame['data'][sl],0.)
                mean[j,k] = np.sum(w*data)-basis[k]@sky['coefficients'][j]
                if frame['psf_matched']:
                    random[j,k] = np.sum(w*np.sqrt(np.where(common[sl],frame['variance'][sl],0)))**2
                else:
                    random[j,k] = np.sum(w*w*np.where(common[sl],rand[sl],0))*frame.get('correlation_factor',1.)
                    flat[j,k] = np.sum(w*np.sqrt(np.where(common[sl],cal[sl],0)))
                wr = w*~review[sl]
                if wr.sum() > cfg['minimum_annulus_fraction']:
                    wr /= wr.sum()
                    # Sensitivity only: orange candidates may be genuine galaxy light.
                    d = np.sum(wr*data)-np.sum(w*data)
                    review_delta[j,k] = d
    # Flatten band-major: index = band*n_annuli + annulus.
    D = np.zeros((3*n,9))
    for j in range(3): D[j*n:(j+1)*n,j*3:(j+1)*3] = basis
    stat = np.diag(random.ravel()) + D@sky['covariance']@D.T
    sky_bound = np.zeros_like(mean)
    if len(sky['alternatives']):
        for j in range(3):
            sky_bound[j] = np.max(np.abs(basis@(sky['alternatives'][:,j]-sky['coefficients'][j]).T),axis=1)
    # Box residuals and mask-choice shifts are coherent sensitivity bounds, not N^-1/2 errors.
    sky_bound = np.maximum(sky_bound,sky['residual_floor'][:,None])
    total = stat.copy()
    for j in range(3):
        sl = slice(j*n,(j+1)*n)
        total[sl,sl] += np.outer(flat[j],flat[j])+np.outer(sky_bound[j],sky_bound[j])
    valid = area/full >= cfg['minimum_annulus_fraction']
    mean[:,~valid] = np.nan
    return dict(edges=np.asarray(edges),mean=mean,area=area,full_area=full,
                valid=valid,covariance=total,stat_covariance=stat,flat_bound=flat,
                sky_bound=sky_bound,review_sensitivity=review_delta,scale=scale)


def _half_light(profile, band=1):
    usable = profile['valid'] & np.isfinite(profile['mean'][band])
    # Do not bridge gaps or extrapolate across unmeasured outer annuli.
    stop = np.flatnonzero(~usable)[0] if (~usable).any() else len(usable)
    flux = profile['mean'][band,:stop]*profile['full_area'][:stop]
    curve = np.r_[0.,np.cumsum(flux)]; radii = profile['edges'][:stop+1]
    if len(curve)<5 or curve[-1]<=0: raise ValueError('No usable positive reference curve of growth for R_e.')
    k = np.flatnonzero(curve>=curve[-1]/2)[0]
    re = np.interp(curve[-1]/2,curve[k-1:k+1],radii[k-1:k+1])
    tail = (curve[-1]-curve[max(0,len(curve)-6)])/curve[-1]
    return float(re),radii,curve,float(tail),stop


def reference_growth(profile, cfg):
    """Use direct growth where possible; explicitly model missing reference rings."""
    good = profile['valid'] & np.isfinite(profile['mean'][1])
    if good.all():
        return profile, 'observed annular means', 0.
    from scipy.optimize import least_squares
    n = len(good); mid = (profile['edges'][:-1]+profile['edges'][1:])/2
    y = profile['mean'][1]
    variance = np.diag(profile['stat_covariance'])[n:2*n]
    fit = good & (y>3*np.sqrt(np.maximum(variance,0)))
    if fit.sum()<8:
        raise ValueError('Too few reliable r annuli for either direct growth or an exponential reference model.')
    V = profile['covariance'][n:2*n,n:2*n][np.ix_(fit,fit)]
    eig,vec = np.linalg.eigh(V)
    whiten = (vec/np.sqrt(np.maximum(eig,max(eig.max()*1e-8,1e-20)))).T
    slope,intercept = np.polyfit(mid[fit],np.log(y[fit]),1)
    h0 = np.clip(-1/min(slope,-1e-10),np.diff(profile['edges']).min(),mid[-1])
    def model(p, r): return np.exp(p[0]-r/np.exp(p[1]))
    solution = least_squares(lambda p:whiten@(y[fit]-model(p,mid[fit])),
                             [float(np.clip(intercept,-40,40)),np.log(h0)],
                             bounds=([-50,np.log(np.diff(profile['edges']).min()/2)],
                                     [50,np.log(profile['edges'][-1]*3)]))
    if not solution.success: raise ValueError('Exponential reference model did not converge.')
    h = np.exp(solution.x[1]); I0 = np.exp(solution.x[0]); edges = profile['edges']
    # Integrate the model over each projected annulus, rather than its midpoint.
    def primitive(r): return -(r+h)*h*np.exp(-r/h)
    model_mean = 2*I0*np.diff(primitive(edges))/np.diff(edges**2)
    filled = dict(profile,mean=profile['mean'].copy(),valid=np.ones(n,bool))
    filled['mean'][1,~good] = model_mean[~good]
    inferred = np.sum(model_mean[~good]*profile['full_area'][~good])
    total = np.sum(filled['mean'][1]*profile['full_area'])
    return filled, 'exponential disk infill for missing r reference annuli; science bins remain masked', float(inferred/total)


def determine_geometry(frames, sky, red, review, gcfg, cfg, error_paths, fraction, output):
    fine_edges = np.linspace(0,gcfg['outer_radius_pixels'],cfg['growth_annuli']+1)
    growth = measure_annuli(frames,sky,red,review,gcfg,cfg,fine_edges,error_paths)
    reference,method,inferred = reference_growth(growth,cfg)
    re,radii,curve,tail,stop = _half_light(reference)
    n = len(fine_edges)-1; rng = np.random.default_rng(cfg['random_seed']); draws = []
    covariance = growth['covariance'][n:2*n,n:2*n]
    eig,vec = np.linalg.eigh(covariance); root = vec*np.sqrt(np.maximum(eig,0))
    for _ in range(cfg['radius_uncertainty_draws']):
        sample = dict(growth,mean=growth['mean'].copy())
        sample['mean'][1] += root@rng.normal(size=n)
        try: draws.append(_half_light(reference_growth(sample,cfg)[0])[0])
        except ValueError: pass
    re_error = float(np.std(draws,ddof=1)) if len(draws)>10 else np.nan
    scale = frames['r']['scale']; distance = gcfg['distance_mpc']; ds = gcfg['distance_error_mpc']
    kpc_per_pixel = 1000*distance*scale/206264.806247
    result = dict(gcfg,re_pixels=re,re_error_pixels=re_error,re_arcsec=re*scale,
                  re_kpc=re*kpc_per_pixel,
                  re_error_kpc=float(np.hypot(re_error*kpc_per_pixel,re*kpc_per_pixel*ds/distance)),
                  kpc_per_pixel=kpc_per_pixel,bin_width_pixels=fraction*re,
                  fraction_of_half_light_radius=fraction,reference_band='r',
                  reference_method=method,modelled_reference_light_fraction=inferred,
                  measured_outer_pixels=float(radii[-1]),outer_light_fraction=tail,
                  radius_definition='half of measured r light within the reported outer ellipse',
                  aperture_limited=bool(tail>cfg['growth_tail_fraction']))
    if result['aperture_limited']:
        warnings.warn(f"R_e is aperture-limited: outer growth rings contribute {tail:.1%} of measured light.")
    if inferred:
        warnings.warn(f"Reference R_e needs exponential infill for {inferred:.1%} of its light; its uncertainty is conditional on that model.")
    edges = np.arange(0,radii[-1],fraction*re)
    if radii[-1]-edges[-1] > .5*fraction*re: edges = np.r_[edges,radii[-1]]
    result['edges'] = edges.tolist()
    output = Path(output); output.mkdir(parents=True,exist_ok=True)
    (output/'geometry.json').write_text(json.dumps(result,indent=2)+'\n')
    Table({'semimajor_pixels':radii,'cumulative_r_ADU_s':curve}).write(output/'curve_of_growth.ecsv',overwrite=True)
    return result,growth


def magnitude_profiles(profile, solutions, cfg):
    """Solve only calibration equations whose nonzero color dependencies are available."""
    n = profile['mean'].shape[1]; I = profile['mean']/profile['scale']**2
    raw = np.where(I>0,-2.5*np.log10(np.maximum(I,1e-300)),np.nan)
    J = np.zeros((3*n,3*n))
    for j in range(3):
        index = np.arange(j*n,(j+1)*n)
        J[index,index] = np.divide(-2.5/np.log(10),profile['mean'][j],
                                  out=np.zeros(n),where=profile['mean'][j]>0)
    raw_cov = J@profile['covariance']@J.T
    active = set(solutions)&set('gri')
    while True:
        allowed = {b for b in active if solutions[b]['color_term']==0 or set(solutions[b]['color_pair'])<=active}
        if allowed==active: break
        active = allowed
    calibrated = np.full((3,n),np.nan); cov = np.full((3*n,3*n),np.nan)
    status = {b:'instrumental: zero point or color dependency unavailable' for b in 'gri'}
    active = [b for b in 'gri' if b in active]
    if active:
        A = np.eye(len(active)); offset = []; color_vectors = []
        for j,b in enumerate(active):
            sol = solutions[b]; c = sol['color_term']; v = np.zeros(len(active))
            if c:
                v[active.index(sol['color_pair'][0])] += 1
                v[active.index(sol['color_pair'][1])] -= 1
            A[j] -= c*v; color_vectors.append(v)
            offset.append(sol['zp']-c*sol['pivot'])
        if np.linalg.cond(A) <= cfg['maximum_calibration_condition']:
            inv = np.linalg.inv(A); inds = ['gri'.index(b) for b in active]
            calibrated[inds] = inv@(np.nan_to_num(raw[inds])+np.asarray(offset)[:,None])
            for j, idx in enumerate(inds):
                needed = np.abs(inv[j]) > 1e-12
                calibrated[idx,~np.isfinite(raw[np.asarray(inds)[needed]]).all(axis=0)] = np.nan
            transform = np.zeros((3*n,3*n)); cal_jac = np.zeros((3*n,2*len(active)))
            cal_cov = np.zeros((2*len(active),2*len(active)))
            for i,b in enumerate(active):
                sol = solutions[b]; cal_cov[2*i:2*i+2,2*i:2*i+2] = sol['covariance']
                color = color_vectors[i]@calibrated[inds]-sol['pivot'] if sol['color_term'] else np.zeros(n)
                for j,bb in enumerate(active):
                    out = 'gri'.index(bb)*n+np.arange(n)
                    transform[out,'gri'.index(b)*n+np.arange(n)] = inv[j,i]
                    cal_jac[out,2*i] = inv[j,i]; cal_jac[out,2*i+1] = inv[j,i]*color
                status[b] = ('provisional Sloan: large color response' if abs(sol['color_term'])>.5 else 'Sloan calibrated')
            computed = transform@raw_cov@transform.T+np.nan_to_num(cal_jac)@cal_cov@np.nan_to_num(cal_jac).T
            index = np.concatenate([np.arange('gri'.index(b)*n,('gri'.index(b)+1)*n) for b in active])
            cov[np.ix_(index,index)] = computed[np.ix_(index,index)]
        else:
            for b in active: status[b] = 'instrumental: ill-conditioned color transformation'
    return dict(instrumental=raw,instrumental_covariance=raw_cov,magnitudes=calibrated,
                covariance=cov,status=status)


def foreground_extinction(frame, gcfg, cfg, cache):
    """Sample off-galaxy SFD E(B-V); avoid the target's own far-IR emission."""
    import astropy.units as u
    from astroquery.ipac.irsa.irsa_dust import IrsaDust
    cache = Path(cache); cache.parent.mkdir(parents=True,exist_ok=True)
    center = frame['wcs'].pixel_to_world(*gcfg['center_xy'])
    key = dict(ra=float(center.ra.deg),dec=float(center.dec.deg),
               sample_radius_deg=gcfg['dust_sample_radius_deg'],samples=cfg['dust_samples'])
    samples = None
    if cache.exists():
        saved = json.loads(cache.read_text())
        if saved.get('query')==key: samples = saved['samples']
    if samples is None:
        samples = []
        for angle in np.linspace(0,360,cfg['dust_samples'],endpoint=False):
            pos = center.directional_offset_by(angle*u.deg,gcfg['dust_sample_radius_deg']*u.deg)
            table = IrsaDust.get_query_table(pos,section='ebv',radius=2*u.deg,
                                             timeout=cfg['dust_timeout_seconds'])
            value = float(pa.column(table,'ext SFD ref')[0])
            if not np.isfinite(value) or value<0: raise ValueError('Invalid SFD reddening response.')
            samples.append(dict(ra=float(pos.ra.deg),dec=float(pos.dec.deg),ebv_sfd=value))
    values = np.array([s['ebv_sfd'] for s in samples]); ebv = float(np.median(values))
    sigma = float(max(pa.mad(values),cfg['dust_ebv_floor'],cfg['dust_fractional_error']*ebv))
    R = np.array([cfg['extinction_coefficients'][b] for b in 'gri'])
    extinction = R*ebv
    cov = np.outer(R,R)*sigma**2+np.diag((extinction*cfg['extinction_law_fractional_error'])**2)
    result = dict(query=key,samples=samples,ebv_sfd=ebv,ebv_error_assumed=sigma,
                  extinction=extinction.tolist(),covariance=cov.tolist(),
                  source='IRSA SFD off-galaxy samples; SF11 R_V=3.1 SDSS coefficients per original SFD E(B-V)',
                  uncertainty='spatial MAD or assumed fractional/floor uncertainty, whichever is largest')
    cache.write_text(json.dumps(result,indent=2)+'\n'); return result


def save_profiles(profile, photometry, geometry, output, extinction=None):
    """Save tabular values and full cross-band/cross-annulus covariance."""
    output = Path(output); output.mkdir(parents=True,exist_ok=True)
    n = profile['mean'].shape[1]; mid = (profile['edges'][1:]+profile['edges'][:-1])/2
    A = np.zeros(3) if extinction is None else np.asarray(extinction['extinction'])
    ext_cov = np.zeros((3,3)) if extinction is None else np.asarray(extinction['covariance'])
    delta_cov = np.kron(ext_cov,np.ones((n,n)))
    rows = []; magnitude = photometry['magnitudes']-A[:,None]
    instrumental = photometry['instrumental']-A[:,None]
    covariance = photometry['covariance']+delta_cov
    raw_cov = photometry['instrumental_covariance']+delta_cov
    for j,b in enumerate('gri'):
        for k in range(n):
            idx = j*n+k; sigma = np.sqrt(max(profile['covariance'][idx,idx],0))
            rows.append((b,k,profile['edges'][k],profile['edges'][k+1],mid[k]*profile['scale'],
                         mid[k]*geometry['kpc_per_pixel'],mid[k]*geometry['kpc_per_pixel']*geometry['distance_error_mpc']/geometry['distance_mpc'],
                         profile['area'][k]/profile['full_area'][k],profile['mean'][j,k]/profile['scale']**2,
                         np.sqrt(max(profile['stat_covariance'][idx,idx],0))/profile['scale']**2,
                         sigma/profile['scale']**2,profile['flat_bound'][j,k]/profile['scale']**2,
                         profile['sky_bound'][j,k]/profile['scale']**2,profile['review_sensitivity'][j,k]/profile['scale']**2,
                         instrumental[j,k],np.sqrt(max(raw_cov[idx,idx],0)),magnitude[j,k],
                         np.sqrt(covariance[idx,idx]),photometry['status'][b],A[j],
                         bool(profile['valid'][k] and profile['mean'][j,k]>3*sigma)))
    names=['band','annulus','inner_pixels','outer_pixels','radius_arcsec','radius_kpc','radius_kpc_distance_error',
           'valid_fraction','intensity_ADU_s_arcsec2','random_and_sky_error','total_error_bound',
           'shared_master_bound','sky_model_bound','orange_mask_sensitivity','instrumental_mu','instrumental_mu_error',
           'mu','mu_error','calibration_status','A_MW','above_3sigma']
    table = Table(rows=rows,names=names)
    table['intensity_upper_3sigma'] = np.maximum(table['intensity_ADU_s_arcsec2'],0)+3*table['total_error_bound']
    table.meta.update(surface_brightness='mag/arcsec2; instrumental_mu has arbitrary zero point',
                      extinction_applied=extinction is not None,
                      extinction_columns='correction applies to magnitude columns; intensity columns retain observed values',
                      error_note='total_error_bound includes conservative coherent master/sky bounds; orange sensitivity separate',
                      covariance_order='band-major g annuli, r annuli, i annuli',
                      extinction_note='MW only; nominal SDSS coefficients remain provisional for unverified passbands')
    table.write(output/'surface_photometry.ecsv',overwrite=True)
    np.savez(output/'covariance.npz',flux=profile['covariance'],flux_stat=profile['stat_covariance'],
             magnitudes=covariance,instrumental=raw_cov,extinction=delta_cov)
    return table


def plot_profiles(table, covariance_path, galaxy, output, corrected=False):
    import matplotlib.pyplot as plt
    cov = np.load(covariance_path); n = len(table)//3
    has_cal = np.isfinite(table['mu']).any()
    has_raw = any(not np.isfinite(table['mu'][table['band']==b]).any() for b in 'gri')
    count = 2 if has_cal and has_raw else 1
    fig,axes = plt.subplots(count+2,1,figsize=(10,3.1*(count+2)),sharex=True,layout='constrained')
    brightness = {True:axes[0],False:axes[1] if count==2 else axes[0]}
    color_axes = axes[count:]
    for j,(b,color) in enumerate(zip('gri',['tab:green','tab:red','tab:purple'])):
        t = table[table['band']==b]; calibrated = np.isfinite(t['mu']).any()
        value = np.asarray(t['mu'] if calibrated else t['instrumental_mu'])
        error = np.asarray(t['mu_error'] if calibrated else t['instrumental_mu_error'])
        good = np.asarray(t['above_3sigma']) & np.isfinite(value)
        ax = brightness[calibrated]
        ax.errorbar(t['radius_kpc'][good],value[good],yerr=error[good],fmt='.-',label=b,color=color)
        # Faint/nonpositive fluxes have bounds, not logarithmic detections.
        upper = np.maximum(np.asarray(t['intensity_ADU_s_arcsec2']),0)+3*np.asarray(t['total_error_bound'])
        limit = -2.5*np.log10(np.maximum(upper,1e-300))-np.asarray(t['A_MW'])
        offset = np.asarray(t['mu']-t['instrumental_mu'])
        const_offset = offset[np.isfinite(offset)]
        can_limit = not calibrated or (len(const_offset)>0 and np.std(const_offset)<.01)
        if calibrated and len(const_offset): limit += np.median(const_offset)
        nondetect = ~np.asarray(t['above_3sigma']) & np.isfinite(t['intensity_ADU_s_arcsec2'])
        if can_limit and nondetect.any():
            ax.scatter(t['radius_kpc'][nondetect],limit[nondetect],marker='v',facecolors='none',
                       edgecolors=color,s=32,label=b+' 3σ faint limits')
    for calibrated in ([True,False] if count==2 else [bool(has_cal)]):
        ax = brightness[calibrated]; ax.invert_yaxis()
        ax.set_ylabel('Surface brightness (mag/arcsec²)')
        ax.set_title('Calibrated Sloan profiles' if calibrated else 'Instrumental profiles — arbitrary zero points')
        ax.legend()
    for ax,(a,b) in zip(color_axes,[('g','r'),('r','i')]):
        ta,tb = table[table['band']==a],table[table['band']==b]
        calibrated = np.isfinite(ta['mu']).any() and np.isfinite(tb['mu']).any()
        col = 'mu' if calibrated else 'instrumental_mu'; V = cov['magnitudes' if calibrated else 'instrumental']
        ia,ib = 'gri'.index(a)*n+np.arange(n),'gri'.index(b)*n+np.arange(n)
        error = np.sqrt(np.maximum(V[ia,ia]+V[ib,ib]-2*V[ia,ib],0))
        good = np.asarray(ta['above_3sigma']) & np.asarray(tb['above_3sigma'])
        value = np.asarray(ta[col]-tb[col]); good &= np.isfinite(value)
        ax.errorbar(ta['radius_kpc'][good],value[good],yerr=error[good],fmt='.-',
                    label=f'{a}−{b}'+(' calibrated' if calibrated else ' instrumental / provisional'))
        if not calibrated:
            ua = np.maximum(np.asarray(ta['intensity_ADU_s_arcsec2']),0)+3*np.asarray(ta['total_error_bound'])
            ub = np.maximum(np.asarray(tb['intensity_ADU_s_arcsec2']),0)+3*np.asarray(tb['total_error_bound'])
            la = -2.5*np.log10(np.maximum(ua,1e-300))-np.asarray(ta['A_MW'])
            lb = -2.5*np.log10(np.maximum(ub,1e-300))-np.asarray(tb['A_MW'])
            lower = ~np.asarray(ta['above_3sigma']) & np.asarray(tb['above_3sigma']) & np.isfinite(ta['intensity_ADU_s_arcsec2'])
            upper = np.asarray(ta['above_3sigma']) & ~np.asarray(tb['above_3sigma']) & np.isfinite(tb['intensity_ADU_s_arcsec2'])
            if lower.any(): ax.scatter(ta['radius_kpc'][lower],(la-np.asarray(tb[col]))[lower],marker='^',facecolors='none',edgecolors='gray',label='3σ lower bounds')
            if upper.any(): ax.scatter(ta['radius_kpc'][upper],(np.asarray(ta[col])-lb)[upper],marker='v',facecolors='none',edgecolors='gray',label='3σ upper bounds')
        ax.set_ylabel('Color (mag)'); ax.legend()
        if not good.any(): ax.text(.98,.05,'No joint >3σ detections',transform=ax.transAxes,ha='right',fontsize=9)
    fig.suptitle(galaxy+(' — Milky Way extinction corrected' if corrected else ' — observed'))
    axes[-1].set_xlabel('Elliptical semimajor radius (kpc)')
    for ax in axes: ax.grid(alpha=.2)
    fig.savefig(output,dpi=160); return fig
