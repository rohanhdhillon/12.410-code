"""WCS registration and IQR stacks for pipeline_one.ipynb.

Variance products are linearized, diagonal pixel variances. Shared calibration
errors use a conservative correlation bound; image resampling also correlates
neighboring output pixels, so aperture errors need blank-aperture checks.
"""
from contextlib import ExitStack
from pathlib import Path
import hashlib
import gzip
import json
import warnings

import numpy as np
from astropy.io import fits
from astropy.wcs import WCS
from astropy.wcs.utils import fit_wcs_from_points, pixel_to_pixel, proj_plane_pixel_area
from astropy.coordinates import SkyCoord
import astropy.units as u
from scipy.ndimage import uniform_filter
from scipy.spatial import cKDTree


# I/O controls are set from the first notebook cell; compression remains lossless.
FITS_GZIP_LEVEL = 1
FITS_TILE_ROWS = 16


def sigma(a):
    a = np.asarray(a); a = a[np.isfinite(a)]
    return float(1.4826 * np.median(np.abs(a - np.median(a))))


def load(path):
    return fits.getdata(path, memmap=False).astype(np.float32)


def slices(shape, rows):
    for y in range(0, shape[0], rows):
        yield slice(y, min(y + rows, shape[0]))


def write_images(path, images, header=None):
    """Lossless GZIP_2, including float arrays: quantization is disabled."""
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    hdus = [fits.PrimaryHDU(header=header)]
    for name, data in images.items():
        extension = fits.CompImageHDU(np.asarray(data), name=name,
                    compression_type="GZIP_2", quantize_level=0.0,
                    tile_shape=(min(FITS_TILE_ROWS, data.shape[0]), data.shape[1]))
        if header is not None and "BUNIT" in header:
            unit = header["BUNIT"]
            extension.header["BUNIT"] = f"({unit})^2" if name.startswith("VAR") else unit
        if name in {"NVALID", "NREJECT"}:
            extension.header["BUNIT"] = "count"
        hdus.append(extension)
    temporary = path.with_name(path.name + ".tmp")
    # Astropy 8's Python codec defaults to gzip level 9. Select level 1 only
    # during this serial write; restore the codec afterward. Pixel values are unchanged.
    from astropy.io.fits.hdu.compressed import _codecs
    original_compressor = _codecs.gzip_compress
    try:
        _codecs.gzip_compress = lambda data: gzip.compress(data, compresslevel=FITS_GZIP_LEVEL, mtime=0)
        fits.HDUList(hdus).writeto(temporary, overwrite=True)
    finally:
        _codecs.gzip_compress = original_compressor
    temporary.replace(path)


def signature(paths, parameters):
    info = [(str(p), Path(p).stat().st_size, Path(p).stat().st_mtime_ns) for p in paths]
    return hashlib.sha256(json.dumps([info, parameters], sort_keys=True, default=str).encode()).hexdigest()[:24]


def calibration_parameters(cfg):
    # A different output footprint does not change detector calibration/noise maps.
    return {k: v for k, v in cfg.items() if k != "grid_bounds"}


def grid_signature(cfg):
    return hashlib.sha256(json.dumps(cfg.get("grid_bounds", {}), sort_keys=True).encode()).hexdigest()[:16]


def cached(path, key, reuse):
    return reuse and Path(path).exists() and fits.getheader(path).get("RUNKEY") == key


def extract_stars(path, cfg):
    """Extract centroids in original, zero-based detector coordinates."""
    import sep
    a = load(path)
    factor = max(1, int(np.ceil(max(a.shape) / cfg["detection_size"])))
    height, width = (v // factor for v in a.shape)
    a = a[:height*factor, :width*factor].reshape(height, factor, width, factor).mean((1, 3))
    a = np.ascontiguousarray(a, dtype=np.float32)
    background = sep.Background(a)
    objects = sep.extract(a - background.back(), cfg["detection_sigma"], err=background.rms())
    good = ((objects["a"] < cfg["star_max_radius"]) & (objects["b"] > 0.5)
            & (objects["a"] < cfg["star_axis_ratio"] * objects["b"])
            & (objects["x"] > 10) & (objects["y"] > 10)
            & (objects["x"] < width-10) & (objects["y"] < height-10))
    objects = objects[good]
    objects = objects[np.argsort(objects["flux"])[::-1]][:cfg["max_stars"]]
    stars = (np.column_stack([objects["x"], objects["y"]]) + 0.5) * factor - 0.5
    if len(stars) < cfg["min_matches"]:
        raise ValueError(f"Too few stellar sources in {path}: {len(stars)}")
    return stars


def solve_anchor(path, output, cfg):
    """Local Astrometry.net solve; pointing/scale headers are search hints only."""
    import astrometry
    from astropy_healpix import HEALPix
    import urllib.request
    output = Path(output)
    key = signature([path], {k: cfg[k] for k in ("detection_size", "detection_sigma", "astrometry_scale")})
    if cached(output, key, cfg["reuse"]):
        return WCS(fits.getheader(output)), extract_stars(path, cfg)
    header = fits.getheader(path)
    center = SkyCoord(header["OBJCTRA"], header["OBJCTDEC"], unit=(u.hourangle, u.deg))
    hp = HEALPix(nside=2, order="nested")
    nested = int(hp.lonlat_to_healpix(center.ra, center.dec))
    face, local = divmod(nested, 4)
    # Astrometry.net uses face*nside^2 + x*nside + y, rather than nested x+2*y.
    tile = face*4 + (local & 1)*2 + (local >> 1)
    index = output.parent / f"index-520{cfg['astrometry_scale']}-{tile:02d}.fits"
    if not index.exists():
        url = ("https://portal.nersc.gov/project/cosmo/temp/dstn/index-5200/LITE/" + index.name)
        print(f"Downloading astrometric index {index.name}", flush=True)
        temporary = index.with_suffix(".download")
        urllib.request.urlretrieve(url, temporary)
        temporary.replace(index)
    stars = extract_stars(path, cfg)
    scale = 206.265 * float(header["XPIXSZ"]) / float(header["FOCALLEN"])
    with astrometry.Solver([index]) as solver:
        solution = solver.solve(stars=(stars + 1).tolist(),
            size_hint=astrometry.SizeHint(scale * 0.7, scale * 1.3),
            position_hint=astrometry.PositionHint(center.ra.deg, center.dec.deg, cfg["search_radius_deg"]),
            solution_parameters=astrometry.SolutionParameters(
                sip_order=cfg["sip_degree"], positional_noise_pixels=2,
                logodds_callback=lambda odds: astrometry.Action.STOP if odds[0] > cfg["solve_logodds"] else astrometry.Action.CONTINUE))
        if not solution.has_match():
            raise RuntimeError(f"No measured WCS solution for {path}; do not substitute pointing headers.")
        match = solution.best_match()
        wcs = match.astropy_wcs()
        h = wcs.to_header(relax=True)
        h["RUNKEY"], h["LOGODDS"] = key, match.logodds
        h["SOURCE"] = path.name
        h.add_history("Measured local Astrometry.net solution; coordinates use FITS origin 1.")
        fits.PrimaryHDU(header=h).writeto(output, overwrite=True)
    return wcs, stars


def transfer_wcs(path, stars, anchor_stars, anchor_wcs, cfg):
    import astroalign
    transform, (source, target) = astroalign.find_transform(stars, anchor_stars,
                                            max_control_points=cfg["control_points"])
    # Find additional one-to-one correspondences after the robust asterism match.
    distance, index = cKDTree(anchor_stars).query(transform(stars))
    tolerance = cfg["match_radius_pixels"]
    good = distance < tolerance
    source = stars[good]; target = anchor_stars[index[good]]
    _, unique = np.unique(index[good], return_index=True)
    source, target = source[unique], target[unique]
    if len(source) < cfg["min_matches"]:
        raise ValueError(f"Only {len(source)} matches for {path}")
    sky = anchor_wcs.pixel_to_world(target[:, 0], target[:, 1])
    projection = anchor_wcs.deepcopy()
    projection.sip = None
    projection.wcs.ctype = [v.replace("-SIP", "") for v in projection.wcs.ctype]
    # fit_wcs_from_points accepts the zero-based centroids used by pixel_to_world.
    for _ in range(cfg["wcs_clip_iterations"]):
        fit = fit_wcs_from_points((source[:, 0], source[:, 1]), sky,
                                 projection=projection, sip_degree=cfg["sip_degree"])
        predicted = np.column_stack(fit.world_to_pixel(sky))
        residual = np.linalg.norm(predicted - source, axis=1)
        keep = residual <= max(cfg["wcs_clip_floor"], cfg["wcs_clip_sigma"] * sigma(residual))
        if keep.all():
            break
        source, sky = source[keep], sky[keep]
        if len(source) < cfg["min_matches"]:
            raise ValueError(f"Insufficient WCS inliers for {path}")
    fit = fit_wcs_from_points((source[:, 0], source[:, 1]), sky,
                             projection=projection, sip_degree=cfg["sip_degree"])
    predicted = np.column_stack(fit.world_to_pixel(sky))
    residual = np.linalg.norm(predicted-source, axis=1)
    rms = float(np.sqrt(np.mean(residual**2)))
    if not np.isfinite(rms) or rms > cfg["max_wcs_rms"]:
        raise ValueError(f"WCS RMS {rms:.3f} pixels exceeds limit for {path}")
    return fit, rms, len(source)


def numeric_gain(headers, cfg):
    values = []
    for header in headers:
        for key in cfg["gain_header_keys"]:
            value = header.get(key)
            if isinstance(value, (float, int)) and np.isfinite(value) and value > 0:
                values.append(float(value)); break
    if values and not np.allclose(values, np.median(values), rtol=cfg["gain_agreement"]):
        raise ValueError("Conflicting numerical FITS gains; set a detector-specific override.")
    return float(np.median(values)) if values else None


def calibration_models(setups, root, cfg):
    """Estimate temporal noise and linearized uncertainty of the measured masters."""
    models = {}
    for name, setup in setups.items():
        paths = setup["bias_paths"] + setup["flat_paths"]
        key = signature(paths + [setup["output_dir"] / "master_bias.fit"]
                        + list(setup["flat_files"].values()), calibration_parameters(cfg))
        folder = root / "uncertainty" / name; folder.mkdir(parents=True, exist_ok=True)
        metadata = folder / "model.json"
        if cfg["reuse"] and metadata.exists():
            saved = json.loads(metadata.read_text())
            if saved["key"] == key and all(Path(p).exists() for p in saved["products"]):
                models[name] = saved
                print(f"{name}: reusing calibration uncertainty maps", flush=True)
                continue
        shape = setup["shape"]
        bias = load(setup["output_dir"] / "master_bias.fit")
        bias_std = np.empty(shape, np.float32)
        with ExitStack() as stack:
            hdus = [stack.enter_context(fits.open(p, memmap=False)) for p in setup["bias_paths"]]
            for rows in slices(shape, cfg["chunk_rows"]):
                cube = np.stack([h[0].section[rows, :].astype(np.float32) for h in hdus])
                bias_std[rows] = np.std(cube, axis=0, ddof=1)
        rn = float(np.median(bias_std[bias_std > 0]))
        vb = (np.pi / (2 * len(setup["bias_paths"]))) * bias_std**2
        bh = fits.Header(dict(RUNKEY=key, NCOMBINE=len(setup["bias_paths"]),
                              RNADU=rn, BIASERR=float(np.median(bias_std)), MEDAPPRO=True))
        bias_file = folder / "bias_uncertainty.fit"
        write_images(bias_file, {"STD": bias_std, "VAR": vb.astype(np.float32)}, bh)
        del bias_std, vb
        # Search the full headers first; mode names like 'High Gain' are not measurements.
        headers = [fits.getheader(p) for p in paths + setup["science_paths"]]
        gain = numeric_gain(headers, cfg)
        gain_source = "HEADER" if gain is not None else "PAIRED_FLATS"
        if cfg["gain_overrides"].get(name) is not None:
            gain = float(cfg["gain_overrides"][name]); gain_source = "OVERRIDE"
        estimates = []
        if gain is None:
            for band, flat_paths in setup["flat_groups"].items():
                if band in setup["low_signal_flats"]:
                    continue
                # Use many independent pixels and disjoint pairs, with an illumination correction.
                for p1, p2 in zip(flat_paths[::2], flat_paths[1::2]):
                    a, b = load(p1)-bias, load(p2)-bias
                    h, w = shape; size = min(cfg["gain_patch_size"], h-2, w-2)
                    sl = np.s_[h//2-size//2:h//2+size//2, w//2-size//2:w//2+size//2]
                    a, b = a[sl].copy(), b[sl].copy()
                    ratio = float(np.median(a) / np.median(b))
                    diff = a-ratio*b
                    highpass = diff-uniform_filter(diff, cfg["gain_highpass_size"])
                    variance = sigma(highpass)**2 / (1-1/cfg["gain_highpass_size"]**2)
                    denominator = variance - (1+ratio**2)*rn**2
                    estimate = float((np.median(a)+ratio**2*np.median(b)) / denominator) if denominator > 0 else np.nan
                    if np.isfinite(estimate) and cfg["gain_bounds"][0] < estimate < cfg["gain_bounds"][1]:
                        estimates.append(estimate)
            if not estimates:
                raise ValueError(f"Cannot estimate numerical gain for {name}; supply gain_overrides at the top.")
            gain = float(np.median(estimates))
        if not np.isfinite(gain) or gain <= 0:
            raise ValueError(f"Invalid gain for {name}: {gain}")
        model = dict(key=key, gain=gain, gain_source=gain_source,
                     gain_estimates=estimates, gain_scatter=sigma(estimates) if estimates else None,
                     read_noise_adu=rn, read_noise_electrons=rn*gain,
                     bias_file=str(bias_file), flats={}, products=[str(bias_file)])
        for band, flat_paths in setup["flat_groups"].items():
            flat = load(setup["flat_files"][band]); scales, samples = [], []
            for path in flat_paths:
                raw = load(path)
                valid = np.isfinite(raw) & (raw < cfg["saturation_adu"])
                raw -= bias; valid &= raw > 0
                scales.append(float(np.median(raw[valid])))
                samples.append(raw[::cfg["normalization_stride"], ::cfg["normalization_stride"]].copy()/scales[-1])
            # Recover the normalization applied when the existing master was written.
            sample = np.stack(samples)
            sample_median = np.median(sample, axis=0)
            sample_flat = flat[::cfg["normalization_stride"], ::cfg["normalization_stride"]]
            good = np.isfinite(sample_flat) & (sample_flat > 0)
            normalization = float(np.median(sample_median[good] / sample_flat[good]))
            alpha = float(np.median(1/np.asarray(scales)) / normalization)
            vf = np.empty(shape, np.float32)
            with ExitStack() as stack:
                hdus = [stack.enter_context(fits.open(p, memmap=False)) for p in flat_paths]
                for rows in slices(shape, cfg["chunk_rows"]):
                    strips = []
                    for h, scale in zip(hdus, scales):
                        raw = h[0].section[rows, :].astype(np.float32)
                        raw[(~np.isfinite(raw)) | (raw >= cfg["saturation_adu"])] = np.nan
                        strips.append((raw-bias[rows])/scale/normalization)
                    cube = np.stack(strips)
                    vf[rows] = np.pi/(2*len(flat_paths)) * np.nanvar(cube, axis=0, ddof=1)
            vf[~np.isfinite(flat)] = np.nan
            # PDF diagnostic: scatter of 500 aperture means, in unbinned-pixel sizes.
            size = max(1, int(round(cfg["flat_aperture_unbinned"]["e24" if name.startswith("e24") else "refractor"]
                                   / setup["reference"]["XBINNING"])))
            rng = np.random.default_rng(cfg["random_seed"])
            means = []
            for _ in range(cfg["flat_apertures"]):
                y = int(rng.integers(0, shape[0]-size+1)); x = int(rng.integers(0, shape[1]-size+1))
                patch = flat[y:y+size, x:x+size]
                means.append(float(np.mean(patch)) if np.isfinite(patch).all() else np.nan)
            aperture_variance = float(np.nanvar(means, ddof=1))
            safe = ''.join(c for c in band if c.isalnum())
            flat_file = folder / f"flat_{safe}_uncertainty.fit"
            fh = fits.Header(dict(RUNKEY=key, FILTER=band, NCOMBINE=len(flat_paths),
                                  FLATLOW=band in setup["low_signal_flats"], BIASALPH=alpha,
                                  APVAR=aperture_variance, APNUM=cfg["flat_apertures"], APSIZE=size))
            fh.add_history("VAR is repeat-flat variance of a median, conditional on the master bias.")
            fh.add_history("APVAR is spatial aperture scatter, not treated as random measurement noise.")
            write_images(flat_file, {"VAR": vf}, fh)
            model["flats"][band] = dict(file=str(flat_file), alpha=alpha, normalization=normalization,
                                          aperture_variance=aperture_variance)
            model["products"].append(str(flat_file))
            del flat, vf
        metadata.write_text(json.dumps(model, indent=2)+"\n")
        models[name] = model
        print(f"{name}: gain={gain:.4g} e-/ADU ({gain_source}), read noise={rn:.3g} ADU", flush=True)
    return models


def calibrate_exposures(setups, models, root, cfg):
    records = []
    for name, setup in setups.items():
        bias = load(setup["output_dir"] / "master_bias.fit")
        bad = fits.getdata(setup["output_dir"] / "bad_pixel_mask.fit", memmap=False).astype(bool)
        model = models[name]
        vb = fits.getdata(model["bias_file"], extname="VAR", memmap=False)
        for band, flat_path in setup["flat_files"].items():
            flat = load(flat_path)
            fm = model["flats"][band]; vf = fits.getdata(fm["file"], extname="VAR", memmap=False)
            for source in setup["science_paths"]:
                h = fits.getheader(source)
                if h["FILTER"] != band:
                    continue
                weak = band in setup["low_signal_flats"]
                if weak and not cfg["allow_weak_m33_g"]:
                    continue
                key = signature([source, flat_path, setup["output_dir"] / "master_bias.fit", fm["file"], model["bias_file"]], calibration_parameters(cfg))
                output = root / "calibrated" / name / f"{source.stem}.fit"
                record = dict(source=str(source), calibrated=str(output), setup=name,
                              galaxy="ngc6503" if name.startswith("e24") else "m33",
                              band=band.replace("'", ""), weak_flat=weak, key=key)
                registered_cache = root / "registered" / name / f"{source.stem}.fit"
                already_registered = (cfg["reuse"] and registered_cache.exists()
                                      and fits.getheader(registered_cache).get("CALKEY") == key
                                      and fits.getheader(registered_cache).get("GRIDHASH") == grid_signature(cfg))
                if not already_registered and not cached(output, key, cfg["reuse"]):
                    raw = load(source); signal = raw-bias; t = float(h["EXPTIME"])
                    invalid = bad | ~np.isfinite(flat) | (flat <= 0) | (raw >= cfg["saturation_adu"])
                    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
                        data = signal / flat / t
                        vi = (np.maximum(signal, 0)/model["gain"] + model["read_noise_adu"]**2) / (flat*t)**2
                        # Include the same bias appearing in both light-bias and flat-bias.
                        db = (-1/flat + signal*fm["alpha"]/flat**2)/t
                        df = -signal/(flat**2*t)
                        cs = np.sqrt(db**2*vb + df**2*vf)
                    invalid |= ~np.isfinite(data) | ~np.isfinite(vi) | ~np.isfinite(cs)
                    for a in (data, vi, cs):a[invalid] = np.nan
                    h["BUNIT"], h["RUNKEY"], h["BIASCOR"], h["FLATCOR"] = "adu/s", key, True, True
                    h["DARKCOR"], h["MASKCOR"], h["FLATLOW"] = False, True, weak
                    h["GAIN"], h["GAINSRC"], h["RNADU"] = model["gain"], model["gain_source"], model["read_noise_adu"]
                    h["RDMISS"] = setup["missing_readout"]
                    h["VARAPPRO"], h["NORMCOV"] = True, False
                    h.add_history("Linearized median-master errors; normalization/gain-fit uncertainty not included.")
                    if weak:h.add_history("WARNING: weak M33 g flat explicitly used provisionally; inspect large flat errors.")
                    write_images(output, {"SCI": data.astype(np.float32), "VARIND": vi.astype(np.float32),
                                          "CALSIG": cs.astype(np.float32)}, h)
                    del raw, signal, data, vi, cs, db, df
                records.append(record)
                print(f"Calibrated {name}: {source.name}" + (" [WEAK g FLAT]" if weak else ""), flush=True)
    return records


def resample_noise(variance, calibration_sigma, source_wcs, target_wcs, shape, cfg):
    """Bilinear independent variance uses squared weights, not image interpolation."""
    result_var = np.full(shape, np.nan, np.float32)
    result_sigma = np.full(shape, np.nan, np.float32)
    area_ratio = abs(proj_plane_pixel_area(target_wcs) / proj_plane_pixel_area(source_wcs))
    for rows in slices(shape, cfg["chunk_rows"]):
        y, x = np.mgrid[rows.start:rows.stop, :shape[1]]
        sx, sy = pixel_to_pixel(target_wcs, source_wcs, x, y)
        finite = np.isfinite(sx) & np.isfinite(sy)
        sx = np.where(finite, sx, -1); sy = np.where(finite, sy, -1)
        ix, iy = np.floor(sx).astype(int), np.floor(sy).astype(int)
        dx, dy = sx-ix, sy-iy
        valid = finite & (ix >= 0) & (iy >= 0) & (ix+1 < variance.shape[1]) & (iy+1 < variance.shape[0])
        v = np.zeros(x.shape); c = np.zeros(x.shape)
        for ox, oy, weight in ((0, 0, (1-dx)*(1-dy)), (1, 0, dx*(1-dy)),
                               (0, 1, (1-dx)*dy), (1, 1, dx*dy)):
            xx = np.clip(ix+ox, 0, variance.shape[1]-1)
            yy = np.clip(iy+oy, 0, variance.shape[0]-1)
            a, b = variance[yy, xx], calibration_sigma[yy, xx]
            valid &= np.isfinite(a) & np.isfinite(b)
            v += weight**2*np.nan_to_num(a, nan=0)
            # Worst-case correlation within each master and across interpolation neighbors.
            c += np.abs(weight)*np.nan_to_num(b, nan=0)
        result_var[rows] = np.where(valid, v*area_ratio**2, np.nan)
        result_sigma[rows] = np.where(valid, c*area_ratio, np.nan)
    return result_var, result_sigma


def register_exposures(records, root, project_dir, cfg, keep_calibrated=False):
    from reproject import reproject_interp
    anchors = {}
    wcs_folder = root / "wcs"; wcs_folder.mkdir(parents=True, exist_ok=True)
    for galaxy in sorted({r["galaxy"] for r in records}):
        subset = [r for r in records if r["galaxy"] == galaxy]
        if galaxy == "ngc6503":
            metadata = json.loads((project_dir / "ngc6503_g.astrometry.json").read_text())
            source = project_dir / metadata["source_image"]
            anchor_wcs = WCS(fits.getheader(project_dir / "ngc6503_g.wcs.fits"))
            stars = extract_stars(source, cfg)
        else:
            # r 1x1 has the finest sampling and avoids the weak g flat for the anchor solve.
            source = Path(next(r["source"] for r in subset if r["setup"] == "refractor_1x1" and r["band"] == "r"))
            anchor_wcs, stars = solve_anchor(source, wcs_folder / "m33_anchor.fits", cfg)
        h = fits.getheader(source)
        shape = (h["NAXIS2"], h["NAXIS1"])
        grid_wcs = anchor_wcs
        bounds = cfg.get("grid_bounds", {}).get(galaxy)
        if bounds is not None:
            ymin, ymax, xmin, xmax = bounds
            if not (0 <= ymin < ymax <= shape[0] and 0 <= xmin < xmax <= shape[1]):
                raise ValueError(f"Invalid output grid bounds for {galaxy}: {bounds}")
            grid_wcs = anchor_wcs[ymin:ymax, xmin:xmax]
            shape = (ymax-ymin, xmax-xmin)
        # All bands of a galaxy share this one reference sky grid and pixel area.
        anchors[galaxy] = (anchor_wcs, stars, grid_wcs, shape)
        fits.PrimaryHDU(header=grid_wcs.to_header(relax=True)).writeto(wcs_folder / f"{galaxy}_grid.fits", overwrite=True)
    registered = []
    for record in records:
        source = Path(record["source"]); anchor_wcs, anchor_stars, grid_wcs, shape = anchors[record["galaxy"]]
        output = root / "registered" / record["setup"] / f"{source.stem}.fit"
        key = signature([source, project_dir / "ngc6503_g.wcs.fits"]
                        + ([wcs_folder / "m33_anchor.fits"] if record["galaxy"] == "m33" else []),
                        dict(cfg, calibration_key=record["key"]))
        if not cached(output, key, cfg["reuse"]):
            stars = extract_stars(source, cfg)
            source_wcs, rms, matches = transfer_wcs(source, stars, anchor_stars, anchor_wcs, cfg)
            with fits.open(record["calibrated"], memmap=False) as hdus:
                h = hdus[0].header.copy()
                data, variance, cs = [hdus[k].data for k in ("SCI", "VARIND", "CALSIG")]
                factor = abs(proj_plane_pixel_area(grid_wcs) / proj_plane_pixel_area(source_wcs))
                aligned, footprint = reproject_interp((data, source_wcs), grid_wcs, shape_out=shape,
                    order="bilinear", block_size=(cfg["chunk_rows"], shape[1]), parallel=False)
                aligned = aligned.astype(np.float32)*factor
                vi, c = resample_noise(variance, cs, source_wcs, grid_wcs, shape, cfg)
                good = (footprint > 0) & np.isfinite(aligned) & np.isfinite(vi) & np.isfinite(c)
                for a in (aligned, vi, c):a[~good] = np.nan
                h.update(grid_wcs.to_header(relax=True))
                h["RUNKEY"], h["WCSRMS"], h["WCSMATCH"] = key, rms, matches
                h["CALKEY"] = record["key"]
                h["GRIDHASH"] = grid_signature(cfg)
                bounds = cfg.get("grid_bounds", {}).get(record["galaxy"])
                h["GRIDY0"], h["GRIDX0"] = (bounds[0], bounds[2]) if bounds else (0, 0)
                h["GRIDNY"], h["GRIDNX"] = shape
                h["REGISTER"], h["INTERP"], h["PIXCOV"] = True, "bilinear", True
                h["CALBOUND"] = True
                h.add_history("Common WCS reference grid; ADU/s scaled by output/input projected pixel area.")
                h.add_history("VARIND uses squared interpolation weights; CALSIG is a shared-error upper bound.")
                h.add_history("Neighboring output pixels are correlated; no full spatial covariance stored.")
                write_images(output, {"SCI": aligned, "VARIND": vi, "CALSIG": c}, h)
            sh = source_wcs.to_header(relax=True); sh["SOURCE"] = source.name
            sh["WCSRMS"], sh["WCSMATCH"] = rms, matches
            fits.PrimaryHDU(header=sh).writeto(wcs_folder / f"{source.stem}.fits", overwrite=True)
            del aligned, variance, cs, vi, c, data, footprint, good
        h = fits.getheader(output)
        if not keep_calibrated:
            Path(record["calibrated"]).unlink(missing_ok=True)
        registered.append(dict(record, registered=str(output), rms=h["WCSRMS"], matches=h["WCSMATCH"]))
        print(f"Registered {source.name}: {h['WCSMATCH']} stars, RMS {h['WCSRMS']:.3f} native pixels", flush=True)
    (root / "registration.json").write_text(json.dumps(registered, indent=2)+"\n")
    return registered


def iqr_selection(cube, multiplier):
    valid = np.isfinite(cube)
    # Vectorize NumPy's default linear percentile rule for the small exposure axis.
    # nanpercentile otherwise dispatches a Python function for every sky pixel.
    ordered = np.sort(np.where(valid, cube, np.nan), axis=0)
    available = valid.sum(axis=0)
    quantiles = []
    for fraction in (0.25, 0.5, 0.75):
        rank = np.maximum(available-1, 0)*fraction
        lower, upper = np.floor(rank).astype(int), np.ceil(rank).astype(int)
        low = np.take_along_axis(ordered, lower[None, ...], axis=0)[0]
        high = np.take_along_axis(ordered, upper[None, ...], axis=0)[0]
        value = low+(high-low)*(rank-lower)
        quantiles.append(np.where(available > 0, value, np.nan))
    q1, median, q3 = quantiles
    # Strict > means equal-valued pixels remain valid, including an IQR of zero.
    keep = valid & (np.abs(cube-median) <= multiplier*(q3-q1))
    n = keep.sum(axis=0)
    total = np.sum(np.where(keep, cube, 0), axis=0, dtype=np.float64)
    mean = np.divide(total, n, out=np.full(n.shape, np.nan), where=n > 0)
    scatter = np.sum(np.where(keep, (cube-mean)**2, 0), axis=0, dtype=np.float64)
    variance = np.divide(scatter, n-1, out=np.full(n.shape, np.nan), where=n > 1)
    return keep, mean, np.sqrt(variance), n, valid.sum(axis=0)-n


def compact_registered_noise(paths):
    # Per-frame error components have already been summed with the accepted stack
    # weights. Retain registered SCI and WCS, while freeing the large redundant maps.
    for path in paths:
        if fits.getheader(path).get("NOISESUM", False):
            continue
        with fits.open(path, memmap=False) as h:
            header = h[0].header.copy()
            data = h["SCI"].data
            header["NOISESUM"] = True
            header.add_history("Per-exposure noise maps aggregated in the matching stack VARIND/CALSIG.")
            write_images(path, {"SCI": data}, header)


def stack_exposures(records, root, cfg, keep_registered_noise=False):
    """PDF median +/- 2.5 IQR selection, then an unweighted arithmetic mean."""
    from collections import defaultdict
    groups = defaultdict(list)
    for r in records:groups[(r["galaxy"], r["band"])].append(r)
    results = {}
    for (galaxy, band), group in sorted(groups.items()):
        paths = [Path(r["registered"]) for r in group]
        # Compaction changes storage, not the registered science values/provenance.
        input_keys = [(str(p), fits.getheader(p).get("RUNKEY", signature([p], cfg))) for p in paths]
        key = hashlib.sha256(json.dumps([input_keys, cfg], sort_keys=True).encode()).hexdigest()[:24]
        output = root / "stacks" / f"{galaxy}_{band}_stack.fit"
        if cached(output, key, cfg["reuse"]):
            if not keep_registered_noise:
                compact_registered_noise(paths)
            results[(galaxy, band)] = output
            print(f"Reusing {output.name}", flush=True); continue
        with ExitStack() as stack:
            hdus = [stack.enter_context(fits.open(p, memmap=False)) for p in paths]
            shape = hdus[0]["SCI"].shape
            mean, std = [np.full(shape, np.nan, np.float32) for _ in range(2)]
            vi, cs = [np.zeros(shape, np.float32) for _ in range(2)]
            count, rejected = [np.zeros(shape, np.uint16) for _ in range(2)]
            # A per-input compressed rejection mask is written stripwise below via memmaps.
            scratch = root / "scratch"; scratch.mkdir(exist_ok=True)
            masks = [np.memmap(scratch / f"{galaxy}_{band}_{j}.mask", mode="w+", dtype=np.uint8, shape=shape)
                     for j in range(len(paths))]
            for rows in slices(shape, cfg["chunk_rows"]):
                cube = np.stack([h["SCI"].section[rows, :] for h in hdus])
                keep, m, s, n, nr = iqr_selection(cube, cfg["iqr_multiplier"])
                vind = np.stack([h["VARIND"].section[rows, :] for h in hdus])
                calsig = np.stack([h["CALSIG"].section[rows, :] for h in hdus])
                summed = np.sum(np.where(keep, vind, 0), axis=0, dtype=np.float64)
                bound = np.sum(np.where(keep, calsig, 0), axis=0, dtype=np.float64)
                vi[rows] = np.divide(summed, n**2, out=np.full(m.shape, np.nan), where=n > 0)
                cs[rows] = np.divide(bound, n, out=np.full(m.shape, np.nan), where=n > 0)
                mean[rows], std[rows], count[rows], rejected[rows] = m, s, n, nr
                for mask, values, accepted in zip(masks, cube, keep):
                    # 0=kept, 1=IQR rejected, 2=no coverage/bad detector pixel.
                    mask[rows] = np.where(~np.isfinite(values), 2, np.where(accepted, 0, 1))
            header = hdus[0][0].header.copy()
            header["RUNKEY"], header["NCOMBINE"], header["IQRCLIP"] = key, len(paths), cfg["iqr_multiplier"]
            header["COMBMETH"], header["FLATLOW"] = "IQR_MEAN", any(r["weak_flat"] for r in group)
            header["FILTER"] = band
            header.add_history("Reject outside median +/- IQRCLIP*IQR; mean and sample STD of survivors.")
            for path in paths:header.add_history(f"Stack input: {path.name}")
            write_images(output, {"SCI": mean, "STD": std, "NVALID": count, "NREJECT": rejected,
                                  "VARIND": vi, "CALSIG": cs}, header)
            for j, mask in enumerate(masks):
                mh = fits.Header(dict(SOURCE=paths[j].name, MASKCODE="0=kept 1=IQR 2=invalid"))
                write_images(root / "rejection_masks" / galaxy / f"{paths[j].stem}.fit", {"MASK": mask}, mh)
                mask.flush()
                filename = Path(mask.filename); del mask; filename.unlink()
            del masks, mean, std, vi, cs, count, rejected
        if not keep_registered_noise:
            compact_registered_noise(paths)
        results[(galaxy, band)] = output
        print(f"Stacked {galaxy} {band}: {len(paths)} exposures", flush=True)
    return results


def propagate_stack_errors(stacks, root, cfg):
    results = {}
    for (galaxy, band), path in stacks.items():
        output = root / "stacks" / f"{galaxy}_{band}_uncertainty.fit"
        key = signature([path], cfg)
        if not cached(output, key, cfg["reuse"]):
            with fits.open(path, memmap=False) as h:
                n, std = h["NVALID"].data, h["STD"].data
                independent = h["VARIND"].data
                empirical = np.divide(std**2, n, out=np.full(n.shape, np.nan, np.float32), where=n > 1)
                # Use the larger noise estimate; do not add the same random noise twice.
                random = np.fmax(independent, empirical)
                shared_bound = h["CALSIG"].data**2
                total = random + shared_bound
                header = h[0].header.copy(); header["RUNKEY"] = key
                header["ERRMETH"], header["CALBOUND"], header["PIXCOV"] = "MAXIND+CALBOUND", True, True
                header["NORMCOV"], header["GAINCOV"], header["ASTRCOV"] = False, False, False
                header.add_history("VAR=max(sum(input independent variances)/N^2, STD^2/N)+shared calibration bound.")
                header.add_history("Shared masters never assumed independent between exposures.")
                header.add_history("Linearized diagonal errors only; aperture errors require covariance/blank apertures.")
                header.add_history("Gain-fit, global flat normalization, and astrometric-fit uncertainties omitted.")
                write_images(output, {"ERROR": np.sqrt(total).astype(np.float32), "VAR": total,
                                      "VARRAND": random, "VARCAL": shared_bound}, header)
        results[(galaxy, band)] = output
        print(f"Uncertainty {galaxy} {band}" + (" [PROVISIONAL: weak g flat]" if fits.getheader(path)["FLATLOW"] else ""), flush=True)
    return results
