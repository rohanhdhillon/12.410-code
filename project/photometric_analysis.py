"""PSF diagnostics, Sloan calibration, and candidate contamination masks.

All expensive/image/network work is called explicitly from pipeline_one.ipynb.
Catalog morphology is a candidate classification, not proof of galaxy membership.
"""
from pathlib import Path
import json
import warnings
import numpy as np
from astropy.io import fits
from astropy.table import Table, vstack, unique
from astropy.wcs import WCS
from astropy.wcs.utils import proj_plane_pixel_scales
from astropy.coordinates import SkyCoord
from astropy.time import Time
import astropy.units as u
from scipy.ndimage import shift, gaussian_filter
from scipy.signal import fftconvolve
from scipy.spatial import cKDTree


def mad(a):
    a = np.asarray(a); a = a[np.isfinite(a)]
    return float(1.4826*np.median(np.abs(a-np.median(a)))) if a.size else np.nan


def column(table, *names):
    lookup = {k.lower(): k for k in table.colnames}
    for name in names:
        if name.lower() in lookup:
            c = table[lookup[name.lower()]]
            return np.asarray(np.ma.filled(np.ma.asarray(c, dtype=float), np.nan))
    raise ValueError(f"Missing expected catalog column: {names}; available: {table.colnames}")


def geometry(frame, config):
    x0, y0 = config["center_xy"]
    angle = np.deg2rad(config["major_axis_angle_deg"])
    return np.array([x0, y0]), np.array([np.cos(angle), np.sin(angle)])


def radius_arcsec(x, y, frame, config):
    center, _ = geometry(frame, config)
    return np.hypot(np.asarray(x)-center[0], np.asarray(y)-center[1])*frame["scale"]


def galaxy_mask(shape, config, scale=1.0):
    y, x = np.indices(shape, dtype=float)
    dx, dy = x-config["center_xy"][0], y-config["center_xy"][1]
    a = np.deg2rad(config["major_axis_angle_deg"])
    major = dx*np.cos(a)+dy*np.sin(a)
    minor = -dx*np.sin(a)+dy*np.cos(a)
    return ((major/(scale*config["mask_semiaxes_pixels"][0]))**2
            +(minor/(scale*config["mask_semiaxes_pixels"][1]))**2) <= 1


def load_frames(paths, error_paths, cfg):
    frames = {}
    for galaxy, bands in paths.items():
        frames[galaxy] = {}
        reference = None
        for band, path in bands.items():
            path = Path(path)
            if not path.exists():
                raise FileNotFoundError(f"Missing analysis input: {path}; run its preparation section first.")
            with fits.open(path, memmap=False) as h:
                data = h["SCI"].data.astype(np.float64)
                header = h[0].header.copy(); wcs = WCS(header)
                if "VAR" in h:
                    variance = h["VAR"].data.astype(float)
                else:
                    with fits.open(error_paths[galaxy][band], memmap=False) as e:
                        variance = e["VAR"].data.astype(float)
                mask = ~np.isfinite(data) | ~np.isfinite(variance) | (variance < 0)
                if "ANALYSIS_MASK" in h:mask |= h["ANALYSIS_MASK"].data.astype(bool)
                if "NVALID" in h:
                    accepted = h["NVALID"].data
                    # IQR rejection is not a hole in the coadd. Test input coverage
                    # before clipping; finite variance still gates usable SCI pixels.
                    coverage = accepted.astype(float)
                    if "NREJECT" in h:coverage += h["NREJECT"].data
                    mask |= (coverage < cfg["coverage_fraction"]*np.nanmax(coverage)) | (accepted < 2)
                bkgvar = h["BKGVAR"].data.astype(float) if "BKGVAR" in h else None
                bkgcov = h["BKG_COV"].data.copy() if "BKG_COV" in h else None
                bkg_systematic = h["BKG_SYS"].data.astype(float) if "BKG_SYS" in h else None
            scale = float(np.sqrt(np.prod(proj_plane_pixel_scales(wcs)))*3600)
            frame = dict(path=path, data=data, variance=variance, mask=mask, header=header,
                         wcs=wcs, scale=scale, bkgvar=bkgvar, bkgcov=bkgcov,
                         bkg_systematic=bkg_systematic, psf_matched=False)
            if reference is not None:
                if data.shape != reference["data"].shape:
                    raise ValueError(f"{galaxy}: bands must share an image shape.")
                ny, nx = data.shape
                x, y = np.array([nx*.2, nx*.8]), np.array([ny*.2, ny*.8])
                if np.max(wcs.pixel_to_world(x,y).separation(reference["wcs"].pixel_to_world(x,y)).arcsec) > cfg["grid_tolerance_arcsec"]:
                    raise ValueError(f"{galaxy}: analysis bands do not share a sky grid.")
            else:reference = frame
            frames[galaxy][band] = frame
    return frames


def detect(frame, cfg):
    import sep
    image = np.ascontiguousarray(np.where(frame["mask"], 0, frame["data"]), dtype=np.float32)
    background = sep.Background(image, mask=frame["mask"], bw=cfg["detection_mesh"], bh=cfg["detection_mesh"])
    objects = sep.extract(image-background.back(), cfg["detection_sigma"], err=background.rms(),
                          mask=frame["mask"], minarea=cfg["detection_minarea"])
    return objects


def aperture_weights(shape, x, y, radius, inner=None):
    """Exact fractional circular weights, restricted to a small cutout."""
    from photutils.aperture import CircularAperture, CircularAnnulus
    aperture = CircularAperture((x,y), radius) if inner is None else CircularAnnulus((x,y), inner, radius)
    am = aperture.to_mask(method="exact")
    large, small = am.get_overlap_slices(shape)
    if large is None:return None, None
    return large, am.data[small]


def photometry(frame, x, y, radius_arcsec, cfg):
    """Local-sky aperture flux; no band-dependent galaxy apertures are chosen here."""
    rp = radius_arcsec/frame["scale"]
    outer = cfg["annulus_outer_factor"]*rp; inner = cfg["annulus_inner_factor"]*rp
    ny,nx=frame["data"].shape
    if x-outer<0 or y-outer<0 or x+outer>=nx or y+outer>=ny:return None
    region, weights = aperture_weights(frame["data"].shape, x, y, outer)
    if region is None:return None
    # Build aperture and annulus on the same bounding box for correct covariance weights.
    yy, xx = np.indices(frame["data"][region].shape)
    gx, gy = xx+region[1].start, yy+region[0].start
    local_x, local_y = x-region[1].start, y-region[0].start
    from photutils.aperture import CircularAperture, CircularAnnulus
    ap = CircularAperture((local_x,local_y), rp).to_mask(method="exact").to_image(xx.shape)
    ann = CircularAnnulus((local_x,local_y), inner, outer).to_mask(method="exact").to_image(xx.shape)
    good = ~frame["mask"][region]
    if np.sum(ap*good)/np.sum(ap) < cfg["aperture_valid_fraction"]:
        return None
    sky_good = good & (ann > 0)
    pixels = frame["data"][region]
    for _ in range(cfg["sky_clip_iterations"]):
        values = pixels[sky_good]
        if values.size < cfg["min_sky_pixels"]:return None
        center, scatter = np.median(values), mad(values)
        if scatter <= 0:break
        sky_good &= np.abs(pixels-center) <= cfg["sky_clip_sigma"]*scatter
    aw = ap*good; sw = ann*sky_good
    if sw.sum() < cfg["min_sky_pixels"]:return None
    # A clipped weighted mean keeps the sky estimator and its linear weights consistent.
    effective = aw-aw.sum()*sw/sw.sum()
    flux = float(np.sum(effective*np.where(good,pixels,0)))
    variance = frame["variance"][region].copy()
    if frame["bkgvar"] is not None:
        variance = np.maximum(variance-frame["bkgvar"][region],0)
    diag = float(np.sum(effective**2*np.where(good,variance,0)))
    # Resampling correlations are bounded/estimated separately; the diagonal is a floor.
    variance_flux = diag*frame.get("correlation_factor",cfg["aperture_noise_correlation_factor"])
    if frame["psf_matched"]:
        # Cauchy bound also covers correlations between convolved output pixels.
        variance_flux = float(np.sum(np.abs(effective)*np.sqrt(np.where(good,variance,0)))**2)
    if frame["bkgcov"] is not None:
        ny,nx = frame["data"].shape
        X=(gx-(nx-1)/2)/(nx/2);Y=(gy-(ny-1)/2)/(ny/2)
        v=np.array([effective.sum(),np.sum(effective*X),np.sum(effective*Y)])
        variance_flux += float(v@frame["bkgcov"]@v)
    systematic = 0.0
    if frame["bkg_systematic"] is not None:
        systematic = float(np.sum(np.abs(effective)*np.nan_to_num(frame["bkg_systematic"][region])))
    return dict(flux=flux, variance=variance_flux, systematic_bound=systematic,
                sky=float(np.sum(sw*np.where(sky_good,pixels,0))/sw.sum()), radius=radius_arcsec)


def characterize_psfs(frames, cfg, output, *, strict=False):
    """Measure empirical PSFs; omit unavailable bands, or raise with strict=True."""
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    results={};diagnostics={};failures=[]
    for galaxy,bands in frames.items():
        results[galaxy]={};gcfg=cfg["galaxies"][galaxy]
        for band,frame in bands.items():
            objects=detect(frame,cfg);ny,nx=frame["data"].shape
            radius_pix=int(np.ceil(gcfg["psf_stamp_radius_arcsec"]/frame["scale"]))
            positions=np.column_stack([objects["x"],objects["y"]])
            counts={"detected":len(objects), "stamp_radius_pixels":radius_pix,
                    "input":str(frame.get("path","in-memory"))}
            diagnostics[f"{galaxy}_{band}"]=counts
            if len(objects)<cfg["min_psf_stars"]:
                counts["accepted"]=0
                failures.append(f"{galaxy} {band}: too few detected sources ({len(objects)})")
                continue
            distance=cKDTree(positions).query(positions,k=2)[0][:,1]
            protected=galaxy_mask((ny,nx),gcfg,cfg["psf_galaxy_exclusion_scale"])
            good=(objects["a"]/np.maximum(objects["b"],.01)<cfg["psf_axis_ratio"])
            good &= (objects["a"]>cfg["psf_min_sigma_pixels"]) & (objects["flag"]==0)
            good &= distance > cfg["psf_isolation_factor"]*radius_pix
            good &= (objects["x"]>radius_pix)&(objects["x"]<nx-radius_pix-1)&(objects["y"]>radius_pix)&(objects["y"]<ny-radius_pix-1)
            good &= ~protected[np.clip(objects["y"].astype(int),0,ny-1),np.clip(objects["x"].astype(int),0,nx-1)]
            peak_limit=cfg["maximum_star_peak_rate"][galaxy][band]
            if peak_limit is not None:good &= objects["peak"]<peak_limit
            indices=np.flatnonzero(good);indices=indices[np.argsort(objects["flux"][indices])[::-1]]
            radii=np.linspace(cfg["growth_min_pixels"],cfg["growth_outer_fraction"]*radius_pix,cfg["growth_samples"])
            counts.update(candidates=len(indices), rejected_mask=0, rejected_snr=0,
                          rejected_plateau=0, rejected_overshoot=0)
            stamps=[];curves=[];rows=[]
            sy,sx=np.indices((2*radius_pix+1,)*2);rr=np.hypot(sx-radius_pix,sy-radius_pix)
            for j in indices:
                x,y=float(objects[j]["x"]),float(objects[j]["y"]);ix,iy=int(round(x)),int(round(y))
                region=np.s_[iy-radius_pix:iy+radius_pix+1,ix-radius_pix:ix+radius_pix+1]
                a=frame["data"][region].copy();bad=frame["mask"][region]
                if bad.mean()>1-cfg["psf_valid_fraction"] or bad[radius_pix-2:radius_pix+3,radius_pix-2:radius_pix+3].any():
                    counts["rejected_mask"]+=1
                    continue
                sky=(rr>cfg["psf_sky_inner_fraction"]*radius_pix)&~bad
                sky_level=np.median(a[sky]);a[bad]=sky_level
                # One interpolation places the measured centroid at the stamp center.
                a=shift(a-sky_level, (iy-y, ix-x),order=1,mode="constant",cval=0)
                growth=[]
                for radius in radii:
                    region_local,weight=aperture_weights(a.shape,radius_pix,radius_pix,radius)
                    growth.append(np.sum(a[region_local]*weight))
                growth=np.asarray(growth);total=float(growth[-1])
                variance=float(np.nansum(frame["variance"][region][rr<=radii[-1]]))
                if total<=0 or total/np.sqrt(max(variance,1e-30))<cfg["psf_min_snr"]:
                    counts["rejected_snr"]+=1
                    continue
                plateau=abs(growth[-1]-growth[-cfg["plateau_samples"]])/total
                if plateau>cfg["growth_plateau_tolerance"]:
                    counts["rejected_plateau"]+=1
                    continue
                curve=growth/total
                if np.max(curve)>1+cfg["growth_overshoot_tolerance"]:
                    counts["rejected_overshoot"]+=1
                    continue
                enclosed=np.maximum.accumulate(curve)
                aper=float(np.interp(cfg["encircled_fraction"],enclosed,radii)*frame["scale"])
                stamps.append(a/total);curves.append(curve)
                rows.append((x,y,radius_arcsec(x,y,frame,gcfg),
                             2.35482*np.sqrt(objects[j]["a"]*objects[j]["b"])*frame["scale"],aper))
                if len(stamps)>=cfg["max_psf_stars"]:break
            counts["accepted"]=len(stamps)
            counts["unexamined"]=len(indices)-len(stamps)-sum(
                value for key,value in counts.items() if key.startswith("rejected_"))
            print(f"{galaxy} {band}: PSF selection {counts}",flush=True)
            if len(stamps)<cfg["min_psf_stars"]:
                failures.append(f"{galaxy} {band}: only {len(stamps)} isolated, plateaued PSF stars "
                                f"(candidates={len(indices)}, mask={counts['rejected_mask']}, "
                                f"S/N={counts['rejected_snr']}, plateau={counts['rejected_plateau']}, "
                                f"overshoot={counts['rejected_overshoot']})")
                continue
            stamps=np.asarray(stamps);curves=np.asarray(curves);rows=np.asarray(rows)
            psf=np.median(stamps,axis=0);psf[rr>radii[-1]]=0;psf/=psf.sum()
            zones=[]
            edges=np.asarray(gcfg["psf_radial_edges_arcsec"])
            for low,high in zip(edges[:-1],edges[1:]):
                selected=(rows[:,2]>=low)&(rows[:,2]<high)
                borrowed=selected.sum()<cfg["min_psf_stars_per_zone"]
                chosen=np.ones(len(rows),bool) if borrowed else selected
                curve=np.median(curves[chosen],axis=0)
                radius=float(np.interp(cfg["encircled_fraction"],np.maximum.accumulate(curve),radii)*frame["scale"])
                zones.append(dict(low=float(low),high=float(high),stars=int(selected.sum()),
                                  radius_arcsec=radius,borrowed_global=bool(borrowed),curve=curve.tolist()))
            result=dict(example_stamps=stamps[:cfg['preview_stars']],psf=psf,radii_arcsec=radii*frame["scale"],curve=np.median(curves,axis=0),
                        aperture_arcsec=float(np.median(rows[:,4])),fwhm_arcsec=float(np.median(rows[:,3])),
                        zones=zones,stars=Table(rows,names=['x','y','galactocentric_arcsec','fwhm_arcsec','aperture_arcsec']))
            results[galaxy][band]=result
            result['stars'].write(output/f'{galaxy}_{band}_psf_stars.ecsv',overwrite=True)
            np.savez(output/f'{galaxy}_{band}_psf.npz',psf=psf,radii_arcsec=result['radii_arcsec'],curve=result['curve'])
    diagnostic_path=output/"selection_diagnostics.json"
    diagnostic_path.write_text(json.dumps(diagnostics,indent=2)+"\n")
    if failures:
        message=("PSF unavailable for some bands:\n"+"\n".join(failures)
                 +f"\nSelection counts for every band: {diagnostic_path}")
        if strict:raise ValueError(message)
        warnings.warn(message)
    # Missing bands are omitted; callers must not substitute another band's EE curve.
    return results


def common_aperture(psfs, galaxy, radius):
    """A common angular aperture from the widest observed stellar EE radius in each zone."""
    values=[]
    for result in psfs.get(galaxy,{}).values():
        zones=result['zones']
        zone=next((z for z in zones if z['low']<=radius<z['high']),None)
        values.append(result['aperture_arcsec'] if zone is None else zone['radius_arcsec'])
    if not values:raise ValueError(f"{galaxy}: no measured PSF aperture is available.")
    return float(max(values))


def match_psfs(frames, psfs, cfg, output):
    """Optional global homogenization; never overwrite original master lights.

    Growth curves remain the aperture diagnostic. A spatially varying PSF cannot
    be corrected faithfully by this single kernel: inspect the radial diagnostics.
    """
    if not cfg['apply_psf_matching']:return frames
    try:
        from photutils.psf_matching import make_kernel as create_matching_kernel, SplitCosineBellWindow
    except ImportError:
        try:
            from photutils.psf_matching import create_matching_kernel, SplitCosineBellWindow
        except ImportError:
            from photutils.psf.matching import create_matching_kernel, SplitCosineBellWindow
    output=Path(output);output.mkdir(parents=True,exist_ok=True);matched={}
    for galaxy,bands in frames.items():
        missing=set(bands)-set(psfs.get(galaxy,{}))
        if missing:
            warnings.warn(f"{galaxy}: skipping PSF matching for all bands; missing PSFs: {sorted(missing)}.")
            matched[galaxy]=dict(bands)
            continue
        broadest=max(psfs[galaxy],key=lambda b:psfs[galaxy][b]['aperture_arcsec'])
        target=gaussian_filter(psfs[galaxy][broadest]['psf'],cfg['target_blur_pixels'])
        target/=target.sum();matched[galaxy]={}
        for band,frame in bands.items():
            kernel=create_matching_kernel(psfs[galaxy][band]['psf'],target,
                window=SplitCosineBellWindow(alpha=cfg['kernel_alpha'],beta=cfg['kernel_beta']))
            kernel/=kernel.sum()
            if not np.isfinite(kernel).all() or np.sum(np.abs(kernel))>cfg['kernel_max_l1']:
                raise ValueError(f'{galaxy} {band}: unstable PSF kernel; leave matching disabled and inspect stars.')
            predicted=fftconvolve(psfs[galaxy][band]['psf'],kernel,mode='same')
            residual=np.sum(np.abs(predicted-target))
            if residual>cfg['kernel_max_residual']:
                raise ValueError(f'{galaxy} {band}: PSF matching residual {residual:.3f} exceeds tolerance.')
            valid=~frame['mask'];den=fftconvolve(valid.astype(float),kernel,mode='same')
            fraction=fftconvolve(valid.astype(float),np.abs(kernel),mode='same')/np.abs(kernel).sum()
            good=(fraction>=cfg['matched_valid_fraction'])&(np.abs(den)>cfg['kernel_min_denominator'])
            data=fftconvolve(np.where(valid,frame['data'],0),kernel,mode='same')/np.where(good,den,np.nan)
            # A bound, rather than an independence assumption for resampled pixels.
            sigma=fftconvolve(np.sqrt(np.where(valid,frame['variance'],0)),np.abs(kernel),mode='same')
            variance=(sigma/np.where(good,np.abs(den),np.nan))**2
            new=dict(frame,data=data,variance=variance,mask=~good,psf_matched=True,bkgvar=None,bkgcov=None)
            if frame['bkg_systematic'] is not None:
                new['bkg_systematic']=fftconvolve(np.nan_to_num(frame['bkg_systematic']),np.abs(kernel),mode='same')/np.where(good,np.abs(den),np.nan)
            header=frame['header'].copy();header['PSFMATCH']=True;header['VARBOUND']=True
            fits.HDUList([fits.PrimaryHDU(header=header),fits.ImageHDU(data.astype('f4'),name='SCI'),
                fits.ImageHDU(variance.astype('f4'),name='VAR_BOUND'),fits.ImageHDU((~good).astype('u1'),name='MASK'),
                fits.ImageHDU(kernel.astype('f4'),name='KERNEL')]).writeto(output/f'{galaxy}_{band}_matched.fit',overwrite=True)
            matched[galaxy][band]=new
    return matched


def image_cone(frame):
    ny,nx=frame['data'].shape
    center=frame['wcs'].pixel_to_world((nx-1)/2,(ny-1)/2)
    corners=frame['wcs'].pixel_to_world([0,nx-1,0,nx-1],[0,0,ny-1,ny-1])
    return center,float(np.max(center.separation(corners).deg))


def sdss_catalog(frame, cfg):
    """Native SDSS PSF magnitudes, observed rather than dereddened.

    Tile the footprint in SQL; query_region's small cone limit is unsuitable
    for M33. Primary detections prevent repeated overlap observations.
    """
    from astroquery.sdss import SDSS
    SDSS.TIMEOUT=cfg['catalog_timeout_seconds']
    center,radius=image_cone(frame)
    step=cfg['sdss_tile_arcmin']/60
    offsets=np.arange(-radius,radius+step,step)
    tiles=[]
    for north in offsets:
        for east in offsets:
            position=center.spherical_offsets_by(east*u.deg,north*u.deg)
            cone=step*np.sqrt(2)/2*60+cfg['catalog_cone_margin_arcmin']
            query=f"""SELECT p.objID, p.ra, p.dec, p.type, p.clean,
              p.psfMag_g AS g, p.psfMag_r AS r, p.psfMag_i AS i,
              p.psfMagErr_g AS eg, p.psfMagErr_r AS er, p.psfMagErr_i AS ei,
              s.z, s.zWarning, s.class AS specclass
              FROM dbo.fGetNearbyObjEq({position.ra.deg},{position.dec.deg},{cone}) n
              JOIN PhotoObj p ON n.objID=p.objID
              LEFT JOIN SpecObj s ON s.bestObjID=p.objID
              WHERE p.mode=1"""
            table=SDSS.query_sql(query,data_release=cfg['sdss_release'],cache=True)
            if table is not None and len(table):tiles.append(table)
    if not tiles:raise ValueError('No SDSS coverage/reference objects in this footprint.')
    table=unique(vstack(tiles,metadata_conflicts='silent'),keys='objID')
    x,y=frame['wcs'].world_to_pixel(SkyCoord(column(table,'ra'),column(table,'dec'),unit='deg'))
    ny,nx=frame['data'].shape
    return table[(x>=0)&(x<nx)&(y>=0)&(y<ny)]


def reference_catalog(frame, cfg, cache):
    """Prefer SDSS; a clearly labeled APASS Sloan g'r'i' fallback is opt-in."""
    cache=Path(cache);cache.parent.mkdir(parents=True,exist_ok=True)
    if cache.exists():
        table=Table.read(cache);return table,str(table.meta['catalog'])
    try:
        table=sdss_catalog(frame,cfg)
        count=np.sum((column(table,'type')==6)&(column(table,'clean')==1))
        if count<cfg['minimum_catalog_stars']:raise ValueError(f'Only {count} SDSS point-source standards.')
        name=f"SDSS DR{cfg['sdss_release']} native ugriz PSF photometry"
    except Exception as exc:
        if not cfg['allow_apass_fallback']:raise
        warnings.warn(f'SDSS unavailable/insufficient ({exc}); trying the enabled APASS Sloan fallback.',stacklevel=2)
        from astroquery.vizier import Vizier
        from requests.exceptions import RequestException
        center,radius=image_cone(frame)
        servers=cfg.get('vizier_servers',('vizier.cfa.harvard.edu','vizier.idia.ac.za','vizier.cds.unistra.fr'))
        errors=[]
        for server in servers:
            try:
                tables=Vizier(columns=['**'],row_limit=-1,timeout=cfg['catalog_timeout_seconds'],
                              vizier_server=server).query_region(
                    center,radius=radius*u.deg,catalog='II/336/apass9')
            except RequestException as error:
                errors.append(f'{server}: {error}')
                warnings.warn(f'APASS query failed at {server}; trying another VizieR mirror.',stacklevel=2)
                continue
            if not tables or not len(tables[0]):
                raise ValueError(f'No APASS references returned by {server}; no magnitude calibration can be inferred.')
            raw=tables[0]
            break
        else:
            raise RuntimeError('APASS unavailable from all configured VizieR mirrors: '+'; '.join(errors))
        table=Table()
        table['ra']=column(raw,'RAJ2000');table['dec']=column(raw,'DEJ2000')
        table['objID']=np.arange(len(raw));table['type']=np.full(len(raw),6);table['clean']=np.ones(len(raw),int)
        for band in 'gri':
            table[band]=column(raw,f"{band}'mag")
            table['e'+band]=column(raw,f"e_{band}'mag")
        name="APASS DR9 published Sloan g'r'i' (coarser resolution; inspect blending)"
        table.meta['vizier_server']=server
    table.meta['catalog']=name;table.write(cache,overwrite=True)
    return table,name


def fit_color_response(xx, yy, err, cfg):
    """Fit a stellar zero point/color term, testing leverage through slope precision."""
    xx=np.asarray(xx,float);yy=np.asarray(yy,float);err=np.asarray(err,float)
    pivot=float(np.median(xx));span=float(np.percentile(xx,90)-np.percentile(xx,10))
    can_color=(len(xx)>=cfg['minimum_color_stars']) and np.ptp(xx)>0 and cfg['fit_color_terms']
    def solve(use_color):
        design=np.column_stack([np.ones(len(xx)),xx-pivot]) if use_color else np.ones((len(xx),1))
        keep=np.ones(len(xx),bool)
        for _ in range(cfg['zp_clip_iterations']):
            w=1/err[keep]**2;normal=design[keep].T@(w[:,None]*design[keep])
            covariance=np.linalg.inv(normal);beta=covariance@(design[keep].T@(w*yy[keep]))
            residual=yy-design@beta
            new=keep&(np.abs(residual)<=cfg['zp_clip_sigma']*np.maximum(err,mad(residual[keep])))
            if np.array_equal(keep,new):break
            if new.sum()<max(cfg['minimum_calibration_stars'],design.shape[1]+2):break
            keep=new
        # Refit once using exactly the reported accepted rows.
        w=1/err[keep]**2;covariance=np.linalg.inv(design[keep].T@(w[:,None]*design[keep]))
        beta=covariance@(design[keep].T@(w*yy[keep]));residual=yy-design@beta
        chi2=float(np.sum((residual[keep]/err[keep])**2));dof=keep.sum()-design.shape[1]
        covariance*=max(1,chi2/max(dof,1))
        return beta,covariance,keep,residual
    beta,cov,keep,residual=solve(can_color)
    status='zero point only: color fit disabled or too few distinct-color standards'
    if len(beta)==2:
        slope_error=float(np.sqrt(cov[1,1]))
        if slope_error>cfg.get('maximum_color_term_error',0.1):
            beta,cov,keep,residual=solve(False)
            status='zero point only: color slope insufficiently constrained'
        elif abs(beta[1])<cfg['color_term_significance']*slope_error:
            beta,cov,keep,residual=solve(False)
            status='zero point only: color slope not significant'
        else:
            preferred=cfg.get('preferred_color_span_mag',cfg.get('minimum_color_span_mag',0.5))
            status='color term measured' if span>=preferred else 'color term measured over a narrow color range'
    return beta,cov,keep,residual,pivot,span,status


def fit_zero_points(frames, psfs, catalogs, cfg, output):
    """Robust catalog fit: m_Sloan = -2.5 log10(F_total ADU/s) + ZP + C(color-pivot)."""
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    solutions={};measurements={};diagnostics={}
    for galaxy,bands in frames.items():
        solutions[galaxy]={};measurements[galaxy]={};cat=catalogs[galaxy]
        coordinates=SkyCoord(column(cat,'ra'),column(cat,'dec'),unit='deg')
        for band,frame in bands.items():
            counts={'catalog_objects':len(cat)};diagnostics[f'{galaxy}_{band}']=counts
            if band not in psfs.get(galaxy,{}):
                counts['reason']='no measured PSF aperture correction'
                warnings.warn(f"{galaxy} {band}: skipping ZP fit; no measured stellar aperture correction.")
                continue
            x,y=frame['wcs'].world_to_pixel(coordinates);ny,nx=frame['data'].shape
            protect=galaxy_mask((ny,nx),cfg['galaxies'][galaxy],cfg['calibration_galaxy_exclusion_scale'])
            objs=detect(frame,cfg);positions=np.column_stack([objs['x'],objs['y']])
            tree=cKDTree(positions);distance,nearest=tree.query(np.column_stack([x,y]))
            neighbors=tree.query(positions,k=2)[0][:,1]
            first,second=cfg['color_pairs'][band];color=column(cat,first)-column(cat,second)
            catalog_mag=column(cat,band);catalog_error=column(cat,'e'+band)
            accepted=(column(cat,'type')==6)&(column(cat,'clean')==1)
            accepted &= np.isfinite(catalog_mag)&np.isfinite(catalog_error)&np.isfinite(color)
            accepted &= (catalog_error>0)&(catalog_error<cfg['max_catalog_error'])
            accepted &= (catalog_mag>cfg['catalog_magnitude_range'][0])&(catalog_mag<cfg['catalog_magnitude_range'][1])
            accepted &= distance*frame['scale']<cfg['catalog_match_arcsec']
            accepted &= (x>=0)&(x<nx)&(y>=0)&(y<ny)
            counts['catalog_and_match_pass']=int(accepted.sum())
            counts.update(rejected_peak=0,rejected_galaxy_or_flag=0,rejected_isolation=0,
                          rejected_aperture=0,rejected_flux_error=0)
            rows=[]
            for j in np.flatnonzero(accepted):
                k=nearest[j];sx,sy=float(objs[k]['x']),float(objs[k]['y'])
                peak_limit=cfg['maximum_star_peak_rate'][galaxy][band]
                if peak_limit is not None and objs[k]['peak']>=peak_limit:
                    counts["rejected_peak"]+=1
                    continue
                if protect[int(sy),int(sx)] or objs[k]['flag']!=0:
                    counts["rejected_galaxy_or_flag"]+=1
                    continue
                rad=common_aperture(psfs,galaxy,radius_arcsec(sx,sy,frame,cfg['galaxies'][galaxy]))
                if neighbors[k]*frame['scale']<cfg['calibration_isolation_factor']*rad:
                    counts["rejected_isolation"]+=1
                    continue
                measured=photometry(frame,sx,sy,rad,cfg)
                if measured is None or measured['flux']<=0:
                    counts["rejected_aperture"]+=1
                    continue
                fractional=np.sqrt(measured['variance'])/measured['flux']
                if fractional>cfg['max_instrumental_fractional_error']:
                    counts["rejected_flux_error"]+=1
                    continue
                # Reference total magnitudes need the measured stellar aperture correction.
                ee=float(np.interp(rad,psfs[galaxy][band]['radii_arcsec'],psfs[galaxy][band]['curve']))
                inst=-2.5*np.log10(measured['flux']/ee)
                error=np.hypot(catalog_error[j],2.5/np.log(10)*fractional)
                error=np.hypot(error,cfg['aperture_correction_floor_mag'])
                rows.append((int(j),sx,sy,rad,ee,inst,catalog_mag[j],color[j],error))
            counts['measured_standards']=len(rows)
            if len(rows)<cfg['minimum_calibration_stars']:
                counts['reason']=f"only {len(rows)}/{cfg['minimum_calibration_stars']} required standards"
                warnings.warn(f'{galaxy} {band}: skipping ZP fit; only {len(rows)} clean standards.')
                continue
            table=Table(rows=rows,names=['catalog_row','x','y','radius_arcsec','encircled_fraction','m_inst','m_catalog','color','sigma'])
            xx=np.asarray(table['color']);yy=np.asarray(table['m_catalog']-table['m_inst']);err=np.asarray(table['sigma'])
            beta,cov,keep,residual,pivot,span,color_status=fit_color_response(xx,yy,err,cfg)
            covariance=np.zeros((2,2));covariance[:len(beta),:len(beta)]=cov
            coefficient=float(beta[1]) if len(beta)==2 else 0.
            if abs(coefficient)>0.5:
                warnings.warn(f'{galaxy} {band}: large measured color term {coefficient:.3f}; '
                              'check filter/passband assignments, catalog matches and blending before trusting calibrated colors.')
            solution=dict(zp=float(beta[0]),color_term=coefficient,pivot=pivot,covariance=covariance.tolist(),
                color_pair=[first,second],stars=int(keep.sum()),scatter_mag=mad(residual[keep]),
                color_fit_status=color_status,color_span_mag=span,
                color_term_error=float(np.sqrt(covariance[1,1])),
                color_range=[float(xx[keep].min()),float(xx[keep].max())],catalog=str(cat.meta['catalog']),
                aperture_noise_status='conservative bound' if frame['psf_matched'] else 'blank-aperture inflation at stellar scale; coherent calibration systematics excluded',
                provisional_weak_g_flat=(galaxy=='m33' and band=='g'))
            counts['reason']='calibrated';counts['fitted_standards']=int(keep.sum())
            table['accepted']=keep;table['m_calibrated']=table['m_inst']+beta[0]+coefficient*(xx-pivot)
            table['residual_mag']=table['m_catalog']-table['m_calibrated']
            table.write(output/f'{galaxy}_{band}_calibration.ecsv',overwrite=True)
            solutions[galaxy][band]=solution;measurements[galaxy][band]=table
    (output/'selection_diagnostics.json').write_text(json.dumps(diagnostics,indent=2)+'\n')
    (output/'sloan_zero_points.json').write_text(json.dumps(solutions,indent=2))
    return solutions,measurements


def sloan_magnitudes(flux, flux_covariance, solutions, calibration_covariance=None):
    """Calibrate net COMMON-aperture g/r/i fluxes, including coupled color terms.

    Fluxes are extended-source aperture fluxes in ADU/s: do not apply a stellar
    aperture correction to a galaxy region. flux_covariance must include sky and
    aperture correlations. Calibration covariance order is Zg,Cg,Zr,Cr,Zi,Ci.
    Catalog and passband systematics are separate from these formal errors.
    """
    missing=set('gri')-set(solutions)
    if missing:raise ValueError(f"Cannot calibrate joint g/r/i magnitudes; missing zero points: {sorted(missing)}. Raw aperture flux analysis remains available.")
    flux=np.asarray(flux,float);V=np.asarray(flux_covariance,float)
    if flux.shape!=(3,) or V.shape!=(3,3):raise ValueError('Supply g,r,i fluxes and a 3x3 covariance.')
    if np.any(flux<=0):raise ValueError('Non-positive flux: report a flux/upper limit, not a logarithmic magnitude.')
    color_matrix=np.zeros((3,3));Z=[];C=[];pivot=[]
    for j,b in enumerate('gri'):
        sol=solutions[b];Z.append(sol['zp']);C.append(sol['color_term']);pivot.append(sol['pivot'])
        a,c=sol['color_pair'];color_matrix[j,'gri'.index(a)]=1;color_matrix[j,'gri'.index(c)]=-1
    A=np.eye(3)-np.diag(C)@color_matrix;inv=np.linalg.inv(A)
    magnitude=inv@(-2.5*np.log10(flux)+Z-np.asarray(C)*pivot)
    jac_flux=inv@np.diag(-2.5/np.log(10)/flux)
    if calibration_covariance is None:
        calibration_covariance=np.zeros((6,6))
        for j,b in enumerate('gri'):calibration_covariance[2*j:2*j+2,2*j:2*j+2]=solutions[b]['covariance']
    jac_cal=np.zeros((3,6))
    for j in range(3):
        jac_cal[:,2*j]=inv[:,j]
        jac_cal[:,2*j+1]=inv[:,j]*(color_matrix[j]@magnitude-pivot[j])
    covariance=jac_flux@V@jac_flux.T+jac_cal@calibration_covariance@jac_cal.T
    for j,b in enumerate('gri'):
        low,high=solutions[b]['color_range']
        if not low<=color_matrix[j]@magnitude<=high:
            warnings.warn(f'{b}: aperture color lies outside the calibration-star range; color correction is extrapolated.')
    return magnitude,covariance


def gaia_catalog(frame,cfg,cache):
    from astroquery.gaia import Gaia
    cache=Path(cache)
    if cache.exists():return Table.read(cache)
    center,radius=image_cone(frame)
    query=f"""SELECT source_id,ra,dec,ref_epoch,pmra,pmdec,pmra_error,pmdec_error,
    pmra_pmdec_corr,parallax,parallax_error,ruwe,phot_g_mean_mag
    FROM gaiadr3.gaia_source WHERE 1=CONTAINS(POINT('ICRS',ra,dec),
    CIRCLE('ICRS',{center.ra.deg},{center.dec.deg},{radius}))"""
    table=Gaia.launch_job_async(query,dump_to_file=False).get_results()
    cache.parent.mkdir(parents=True,exist_ok=True);table.write(cache,overwrite=True)
    return table


def contaminants(frame, psf, reference, gaia, galaxy, cfg, output):
    """Secure masks plus a separate review list. No HII knots are auto-deleted."""
    gcfg=cfg['galaxies'][galaxy];center,axis=geometry(frame,gcfg);ny,nx=frame['data'].shape
    rows=[]
    star_radius=(psf['aperture_arcsec']*cfg['contaminant_mask_radius_factor'] if psf is not None
                 else gcfg.get('fallback_star_mask_radius_arcsec',cfg['extended_mask_radius_arcsec']))
    if psf is None:
        warnings.warn(f"{galaxy}: no r PSF; using a fixed {star_radius:g} arcsec star-mask radius; sharpness tests skipped.")
    def add(ra,dec,label,evidence,secure,source_id,radius=None):
        x,y=frame['wcs'].world_to_pixel(SkyCoord(ra,dec,unit='deg'));x=float(x);y=float(y)
        if not (0<=x<nx and 0<=y<ny):return
        delta=(np.array([x,y])-center)*frame['scale']
        along=float(delta@axis);across=float(delta@np.array([-axis[1],axis[0]]))
        inline=abs(along)<=gcfg['corridor_half_length_arcsec'] and abs(across)<=gcfg['corridor_half_width_arcsec']
        r=star_radius if radius is None else radius
        key=str(source_id)
        if key in cfg['manual_accept_ids']:secure=True
        if key in cfg['manual_reject_ids']:secure=False
        rows.append((key,ra,dec,x,y,along,across,inline,label,evidence,secure,r))
    ra=column(gaia,'ra');dec=column(gaia,'dec');pmra=column(gaia,'pmra');pmdec=column(gaia,'pmdec')
    # Move Gaia astrometry from its reference epoch to the exposure date when available.
    finite=np.isfinite(pmra)&np.isfinite(pmdec)
    date=frame['header'].get('DATE-OBS')
    if date and finite.any():
        position=SkyCoord(ra=ra[finite]*u.deg,dec=dec[finite]*u.deg,
            pm_ra_cosdec=pmra[finite]*u.mas/u.yr,pm_dec=pmdec[finite]*u.mas/u.yr,
            obstime=Time(column(gaia,'ref_epoch')[finite],format='jyear'))
        moved=position.apply_space_motion(new_obstime=Time(date));ra[finite]=moved.ra.deg;dec[finite]=moved.dec.deg
    pe=column(gaia,'parallax_error');par=column(gaia,'parallax')
    ea=column(gaia,'pmra_error');ed=column(gaia,'pmdec_error');corr=column(gaia,'pmra_pmdec_corr')
    # Joint proper-motion significance accounts for the Gaia covariance.
    rho=np.clip(corr,-.999,.999)
    with np.errstate(divide='ignore',invalid='ignore'):
        significance=np.sqrt(np.maximum(0,((pmra/ea)**2+(pmdec/ed)**2-2*rho*pmra*pmdec/(ea*ed))/(1-rho**2)))
        foreground=((par/pe)>=cfg['foreground_significance'])|(significance>=cfg['foreground_significance'])
    foreground &= column(gaia,'ruwe')<cfg['maximum_gaia_ruwe']
    for j in range(len(gaia)):
        add(ra[j],dec[j],'foreground_star' if foreground[j] else 'stellar_candidate',
            'Gaia significant parallax/proper motion' if foreground[j] else 'Gaia detection; membership uncertain',
            bool(foreground[j]),f"gaia:{gaia['source_id'][j]}")
    # Only a reliable extragalactic spectrum is a secure background-galaxy label.
    for j in range(len(reference)):
        if column(reference,'type')[j]!=3:continue
        secure=False
        if 'z' in reference.colnames:
            z=column(reference,'z')[j];zw=column(reference,'zWarning')[j]
            spectral_class=reference['specclass'][j]
            if isinstance(spectral_class,bytes):spectral_class=spectral_class.decode()
            secure=np.isfinite(z) and z>cfg['background_minimum_redshift'] and zw==0 and str(spectral_class).strip().upper()=='GALAXY'
        add(float(reference['ra'][j]),float(reference['dec'][j]),'background_galaxy' if secure else 'extended_candidate',
            'SDSS reliable extragalactic redshift' if secure else 'Catalog extension; could be a galaxy/HII region/blend',
            secure,f"sdss:{reference['objID'][j]}",cfg['extended_mask_radius_arcsec'])
    objects=detect(frame,cfg)
    for j,obj in enumerate(objects):
        width=2.35482*np.sqrt(obj['a']*obj['b'])*frame['scale']
        sharp=psf is not None and width<cfg['artifact_width_fraction']*psf['fwhm_arcsec']
        elongated=obj['a']/max(obj['b'],.01)>cfg['artifact_axis_ratio']
        if not (sharp or elongated):continue
        sky=frame['wcs'].pixel_to_world(obj['x'],obj['y'])
        add(sky.ra.deg,sky.dec.deg,'artifact_candidate','Unusually sharp/elongated image source; inspect other bands',
            False,f'image:{galaxy}:{j}')
    names=['id','ra','dec','x','y','major_axis_arcsec','cross_axis_arcsec','in_corridor','classification','evidence','auto_mask','mask_radius_arcsec']
    table=Table(rows=rows,names=names) if rows else Table(names=names,dtype=['U64','f8','f8','f8','f8','f8','f8','bool','U40','U100','bool','f8'])
    safe=frame['mask'].copy();review=np.zeros((ny,nx),bool)
    for row in table:
        # Retain classifications across the field; highlight the major-axis corridor.
        slices,weight=aperture_weights((ny,nx),row['x'],row['y'],row['mask_radius_arcsec']/frame['scale'])
        if slices is None:continue
        if row['auto_mask']:safe[slices] |= weight>0
        else:review[slices] |= weight>0
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    table.meta['star_mask_radius_source']='measured PSF' if psf is not None else 'configured fixed radius'
    table.meta['star_mask_radius_arcsec']=star_radius
    table.write(output/f'{galaxy}_contaminants.ecsv',overwrite=True)
    header=frame['header'].copy();header['MASKNOTE']='1 means excluded; candidates require review'
    header['PSFAVAIL']=psf is not None;header['STARMASK']=star_radius
    fits.HDUList([fits.PrimaryHDU(header=header),fits.ImageHDU(safe.astype('u1'),name='SAFE_MASK'),
        fits.ImageHDU(review.astype('u1'),name='REVIEW_MASK')]).writeto(output/f'{galaxy}_contamination_masks.fit',overwrite=True)
    return table,safe,review


def estimate_aperture_noise(frames,psfs,cfg):
    """Blank-aperture variance inflation at the measured stellar aperture scale.

    This estimates resampling/local-sky correlations empirically. It is not a
    measurement of coherent flat-field error or a galaxy-scale sky uncertainty.
    """
    rng=np.random.default_rng(cfg['random_seed']);rows=[]
    for galaxy,bands in frames.items():
        available=psfs.get(galaxy,{})
        if not available:
            warnings.warn(f"{galaxy}: skipping stellar-scale blank-aperture noise estimate; no PSF aperture is measured.")
            continue
        for band,frame in bands.items():
            radius=max(p['aperture_arcsec'] for p in available.values())
            objects=detect(frame,cfg);tree=cKDTree(np.column_stack([objects['x'],objects['y']]))
            protect=galaxy_mask(frame['data'].shape,cfg['galaxies'][galaxy],cfg['calibration_galaxy_exclusion_scale'])
            ny,nx=frame['data'].shape;margin=cfg['annulus_outer_factor']*radius/frame['scale']
            values=[];expected=[]
            for _ in range(cfg['blank_aperture_attempts']):
                x=rng.uniform(margin,nx-margin);y=rng.uniform(margin,ny-margin)
                if protect[int(y),int(x)]:continue
                if tree.query([x,y])[0]*frame['scale']<cfg['blank_source_clearance_factor']*radius:continue
                result=photometry(frame,x,y,radius,cfg)
                if result is None or result['variance']<=0:continue
                values.append(result['flux']);expected.append(result['variance'])
                if len(values)>=cfg['blank_apertures']:break
            if len(values)<cfg['minimum_blank_apertures']:
                warnings.warn(f'{galaxy} {band}: only {len(values)} usable blank apertures; correlation factor remains unmeasured.')
                rows.append((galaxy,band,radius,len(values),np.nan))
                continue
            factor=max(1.,mad(values)**2/np.median(expected))
            frame['correlation_factor']=factor
            rows.append((galaxy,band,radius,len(values),factor))
    return Table(rows=rows,names=['galaxy','band','radius_arcsec','blank_apertures','variance_inflation'])
