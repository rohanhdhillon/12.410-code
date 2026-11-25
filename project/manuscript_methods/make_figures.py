"""Render the Methods figures and print diagnostics from the saved reduction.

Run with the pipeline's Python environment. Reads, but never rewrites, FITS or
analysis products. The output directory defaults to figures beside this file.
"""

import argparse
from collections import Counter
import json
from pathlib import Path
import sys
import warnings

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import AsinhNorm, BoundaryNorm, ListedColormap, Normalize
from matplotlib.patches import Ellipse
from matplotlib import patheffects
from matplotlib.ticker import AutoMinorLocator, MaxNLocator
import numpy as np
from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.table import Table
from astropy.wcs import FITSFixedWarning, WCS
from astropy.wcs.utils import proj_plane_pixel_area
from scipy.ndimage import binary_dilation
from scipy.spatial import cKDTree


PROJECT = Path(__file__).resolve().parents[1]
GALAXIES = {'ngc6503': 'NGC 6503', 'm33': 'M33'}
BANDS = {'ngc6503': 'gri', 'm33': 'ugr'}
BLUE, ORANGE, TEAL = '#176a9c', '#ba501e', '#157b66'
COLORS = dict(u='#674080', g=TEAL, r=ORANGE, i='#663c17')
SEED = 6503

plt.rcParams.update({
    'font.family': 'STIXGeneral', 'mathtext.fontset': 'stix',
    'font.size': 9, 'axes.labelsize': 9, 'axes.titlesize': 10,
    'xtick.labelsize': 8, 'ytick.labelsize': 8,
    'axes.linewidth': 0.65, 'xtick.direction': 'in', 'ytick.direction': 'in',
    'legend.fontsize': 8, 'legend.frameon': False, 'pdf.fonttype': 42,
    'savefig.dpi': 360, 'figure.facecolor': 'white',
})
warnings.filterwarnings('ignore', category=FITSFixedWarning)


def source_path(project, path):
    """Resolve a path recorded by the pipeline on the machine that wrote it."""
    path = Path(path)
    if path.is_file():
        return path
    if not path.is_absolute():
        return project / path
    i = len(path.parts) - 1 - path.parts[::-1].index('project')
    return project.joinpath(*path.parts[i + 1:])


def stack_path(project, galaxy, band, suffix='stack'):
    prefix = 'm33_ugr' if galaxy == 'm33' else galaxy
    return project / 'reduction' / 'stacks' / f'{prefix}_{band}_{suffix}.fit'


def analysis_root(project):
    return project / 'reduction' / 'analysis_ugr'


def scale_arcsec(wcs):
    return np.sqrt(abs(proj_plane_pixel_area(wcs))) * 3600


def read_science(path):
    with fits.open(path, memmap=False) as hdus:
        return hdus['SCI'].data, WCS(hdus[0].header).celestial


def record_wcs(project, record):
    suffix = '_ugr' if record['galaxy'] == 'm33' else ''
    path = project / 'reduction' / 'wcs' / (Path(record['source']).stem + suffix + '.fits')
    return WCS(fits.getheader(path)).celestial


def save(fig, output, name):
    fig.savefig(output / f'{name}.pdf', bbox_inches='tight', pad_inches=0.035)
    plt.close(fig)


def panel(ax, letter, name):
    ax.set_title(f'({letter})  {name}', loc='left', pad=5)


def ticks(ax):
    ax.tick_params(which='both', top=True, right=True)
    ax.xaxis.set_minor_locator(AutoMinorLocator())
    ax.yaxis.set_minor_locator(AutoMinorLocator())


def show_field(ax, image, norm=None, cmap='gray', step=2, average=1):
    ny, nx = image.shape
    if average > 1:
        blocks = np.ma.masked_invalid(image).reshape(ny // average, average, nx // average, average)
        display = blocks.mean(axis=(1, 3))
    else:
        display = image[::step, ::step]
    ax.imshow(display, origin='lower', cmap=cmap, norm=norm,
              interpolation='nearest', extent=(-0.5, nx - 0.5, -0.5, ny - 0.5))
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_facecolor('0.88')


def galaxy_stretch(image):
    low, high = np.nanpercentile(image[::3, ::3], [22., 99.3])
    return AsinhNorm(linear_width=(high - low) / 15, vmin=low, vmax=high)


def field_key(ax, image, wcs, bar_arcmin, *, compass=False, light=False):
    ny, nx = image.shape
    x, y = .06 * nx, .08 * ny
    length = bar_arcmin * 60 / scale_arcsec(wcs)
    color = '#252525' if light else 'white'
    outline = 'white' if light else '#1f1f1f'
    ax.plot((x, x + length), (y, y), color=color, lw=1.25,
            path_effects=[patheffects.withStroke(linewidth=2.4, foreground=outline)])
    ax.text(x + length / 2, y + .035 * ny, rf"${bar_arcmin:g}^\prime$",
            color=color, ha='center', va='bottom',
            path_effects=[patheffects.withStroke(linewidth=1.6, foreground=outline)])
    if compass:
        x, y = .87 * nx, .17 * ny
        center = wcs.pixel_to_world(x, y)
        for angle, label in [(0, 'N'), (90, 'E')]:
            endpoint = center.directional_offset_by(angle * u.deg, .075 * nx * scale_arcsec(wcs) * u.arcsec)
            px, py = wcs.world_to_pixel(endpoint)
            ax.annotate('', (px, py), (x, y), arrowprops=dict(arrowstyle='->', color='white', lw=.9))
            ax.text(x + 1.27 * (px - x), y + 1.27 * (py - y), label, color='white', ha='center', va='center')


def raw_aperture(image, x, y, pixel_scale):
    radius = 18 / pixel_scale
    outer = 2.5 * radius
    ny, nx = image.shape
    if x - outer < 1 or y - outer < 1 or x + outer >= nx - 1 or y + outer >= ny - 1:
        return np.nan
    region, aperture = pa.aperture_weights(image.shape, x, y, radius)
    sky_region, annulus = pa.aperture_weights(image.shape, x, y, outer, inner=1.5 * radius)
    pixels = image[region]
    if np.any(pixels[aperture > 0] >= 60_000):
        return np.nan
    values = image[sky_region][annulus > 0]
    weights = annulus[annulus > 0]
    for _ in range(3):
        center, scatter = np.median(values), pa.mad(values)
        if scatter == 0:
            break
        keep = np.abs(values - center) <= 3 * scatter
        values, weights = values[keep], weights[keep]
    sky = np.average(values, weights=weights)
    return np.sum(aperture * (pixels - sky))


def filter_audit(project, records):
    cat = Table.read(analysis_root(project) / 'm33_sloan_reference.ecsv')
    mags = {b: pa.column(cat, b) for b in 'ugri'}
    wcs = WCS(fits.getheader(project / 'reduction' / 'wcs' / 'm33_ugr_grid.fits')).celestial
    sky = SkyCoord(cat['ra'], cat['dec'], unit='deg')
    x, y = wcs.world_to_pixel(sky)
    geometry = pa.galaxy_mask((3008, 3008), dict(center_xy=(1655.5, 1367.5),
                              major_axis_angle_deg=156.8, mask_semiaxes_pixels=(1300, 900)), 1.05)
    on_disk = geometry[np.clip(y.astype(int), 0, 3007), np.clip(x.astype(int), 0, 3007)]
    selected = (cat['type'] == 6) & (cat['clean'] == 1) & (mags['r'] > 14) & (mags['r'] < 18) & ~on_disk
    for band in 'ugri':
        err = pa.column(cat, 'e' + band)
        selected &= np.isfinite(mags[band]) & (err > 0) & (err < (.12 if band == 'u' else .05))
    neighbors = (mags['r'] > 0) & (mags['r'] < 19.5)
    positions = np.column_stack((x, y))
    distance = cKDTree(positions[neighbors]).query(positions[selected], k=2)[0][:, 1]
    indices = np.flatnonzero(selected)[distance * scale_arcsec(wcs) > 35]
    rates = {}
    for band in BANDS['m33']:
        exposures = [r for r in records if (r['galaxy'], r['band']) == ('m33', band)]
        fluxes = []
        for record in exposures:
            wcs = record_wcs(project, record)
            image, header = fits.getdata(source_path(project, record['source']), header=True, memmap=False)
            sx, sy = wcs.world_to_pixel(sky[indices])
            rate = [raw_aperture(image, px, py, scale_arcsec(wcs)) / header['EXPTIME']
                    for px, py in zip(sx, sy)]
            fluxes.append(rate)
        rates[band] = np.array(fluxes)
    print('Filter audit candidates after isolation:', len(indices))
    for nominal, band in zip('gri', BANDS['m33']):
        rate = rates[band]
        snr = rate.mean(axis=0) / (rate.std(axis=0, ddof=1) / np.sqrt(rate.shape[0]))
        complete = np.isfinite(rate).all(axis=0) & (snr >= 8)
        rows = indices[complete]
        catalog = {b: value[rows] for b, value in mags.items()}
        median_rate = np.median(rate[:, complete], axis=0)
        if np.any(median_rate <= 0):
            raise ValueError('The filter audit requires positive median rates.')
        precise = snr[complete] >= np.quantile(snr[complete], .75)
        offsets = np.array([-2.5 * np.log10(median_rate) - m for m in catalog.values()])
        typical = np.median(np.abs(offsets - np.median(offsets, axis=1, keepdims=True)), axis=0)
        reference = int(np.argmin(np.where(precise, typical, np.inf)))
        color = catalog['g'] - catalog['r']
        examples = [reference]
        for quantile in (.1, .35, .65, .9):
            for j in np.argsort(np.abs(color - np.quantile(color, quantile))):
                if int(j) not in examples:
                    examples.append(int(j))
                    break
        ratio_series = rate[:, complete] / rate[:, complete][:, reference, None]
        measured = median_rate / median_rate[reference]
        predictions = {b: 10**(-.4 * (m - m[reference])) for b, m in catalog.items()}
        widths = {b: float(pa.mad(-2.5 * np.log10(measured / f))) for b, f in predictions.items()}
        repeat = np.std(ratio_series, axis=0, ddof=1)
        print(f'FITS {nominal}: {len(rows)} stars; SDSS g-r range', (color.min(), color.max()),
              'MAD (mag) vs catalog u/g/r/i:', widths)
        for j in examples:
            row = cat[rows[j]]
            print(f"  {row['objID']}  {measured[j]:.3f}  {repeat[j]:.3f}  "
                  f"{predictions[nominal][j]:.3f}  {predictions[band][j]:.3f}  {color[j]:.3f}")


def registration_figure(project, records, output):
    red = project / 'reduction'
    metadata = json.loads((project / 'ngc6503_g.astrometry.json').read_text())
    anchor_files = {
        'ngc6503': (project / 'ngc6503_g.wcs.fits', project / metadata['source_image']),
        'm33': (red / 'wcs' / 'm33_anchor.fits', source_path(project, next(
            r['source'] for r in records if (r['galaxy'], r['band']) == ('m33', 'g')))),
    }
    config = dict(detection_size=2500, detection_sigma=5., star_max_radius=6.,
                  star_axis_ratio=2.5, max_stars=300, min_matches=12)
    fig, axes = plt.subplots(1, 2, figsize=(6.85, 3.46), layout='constrained')
    for ax, (galaxy, name), letter, bar in zip(axes, GALAXIES.items(), 'ab', (2, 15)):
        wcs_path, anchor = anchor_files[galaxy]
        wcs = WCS(fits.getheader(wcs_path)).celestial
        refs = lr.extract_stars(anchor, config)
        record = next(r for r in records if (r['galaxy'], r['band']) == (galaxy, 'r'))
        sample = lr.extract_stars(source_path(project, record['source']), config)
        native = record_wcs(project, record)
        sky = native.pixel_to_world(*sample.T)
        distance, index = cKDTree(refs).query(np.column_stack(wcs.world_to_pixel(sky)))
        closest = np.argsort(distance)
        closest = closest[distance[closest] < 2.0]
        near, unique = np.unique(index[closest], return_index=True)
        matched_sample = sample[closest[unique]]
        registered = source_path(project, record['registered'])
        image, grid = read_science(registered)
        xx, yy = grid.world_to_pixel(wcs.pixel_to_world(*refs[near].T))
        ny, nx = image.shape
        inside = (xx > 0) & (xx < nx - 1) & (yy > 0) & (yy < ny - 1)
        coins = ax.scatter(xx[inside], yy[inside], s=19, facecolors='none',
                           edgecolors='#63daff', linewidths=.6, zorder=3)
        coins.set_clip_on(True)
        show_field(ax, image, galaxy_stretch(image))
        ax.set_anchor('N')
        field_key(ax, image, grid, bar, compass=True)
        panel(ax, letter, name)
        candidates = np.flatnonzero(inside & (xx > .08 * nx) & (xx < .92 * nx)
                                    & (yy > .16 * ny) & (yy < .9 * ny))
        for number, (x, y) in enumerate(((.18, .71), (.50, .27), (.78, .72)), start=1):
            best = candidates[np.argmin((xx[candidates] / nx - x)**2 + (yy[candidates] / ny - y)**2)]
            label = letter.upper() + str(number)
            ax.annotate(label, (xx[best], yy[best]), xytext=(4, 4), textcoords='offset points',
                        color='white', fontsize=8, weight='bold',
                        path_effects=[patheffects.withStroke(linewidth=1.6, foreground='black')])
            coord = wcs.pixel_to_world(*refs[near[best]]).icrs
            star_x, star_y = matched_sample[best]
            resid = np.linalg.norm(matched_sample[best] - np.asarray(native.world_to_pixel(coord)))
            print(f'{label}  {star_x:.2f}  {star_y:.2f}  {coord.ra.deg:.6f}  {coord.dec.deg:.6f}  {resid:.3f}')
            candidates = candidates[candidates != best]
        print('Registration example', name, Path(record['source']).name, 'matches', record['matches'],
              'RMS', record['rms'], 'displayed coincidences', inside.sum())
    save(fig, output, 'registration')


def read_observation_mask(project, galaxy, band):
    corrected = galaxy == 'm33' and band == 'g'
    path = stack_path(project, galaxy, band, 'background_corrected' if corrected else 'stack')
    with fits.open(path, memmap=False) as hdus:
        kept = hdus['NVALID'].data
        coverage = kept + hdus['NREJECT'].data
        bad = (kept < 2) | (coverage < .8 * coverage.max()) | ~np.isfinite(hdus['SCI'].data)
        if corrected:
            bad |= hdus['ANALYSIS_MASK'].data.astype(bool)
            variance = hdus['VAR'].data
        else:
            variance = fits.getdata(stack_path(project, galaxy, band, 'uncertainty'),
                                    extname='VAR', memmap=False)
        bad |= ~np.isfinite(variance) | (variance < 0)
    return bad


def source_exclusions(audit, shape, scale):
    """Flag pixels with positive circle overlap, as in the saved mask audit."""
    mask = np.zeros(shape, bool)
    ny, nx = shape
    for row in audit[audit['radial_masked']]:
        x, y, radius = row['x'], row['y'], row['mask_radius_arcsec'] / scale
        xmin, xmax = max(0, int(np.floor(x - radius))), min(nx, int(np.ceil(x + radius + 1)))
        ymin, ymax = max(0, int(np.floor(y - radius))), min(ny, int(np.ceil(y + radius + 1)))
        dx = np.maximum(0, np.abs(np.arange(xmin, xmax) - x) - .5)
        dy = np.maximum(0, np.abs(np.arange(ymin, ymax) - y) - .5)
        mask[ymin:ymax, xmin:xmax] |= dy[:, None]**2 + dx**2 < radius**2
    return mask


def sky_pixel_mask(image, invalid, excluded, geometry):
    import sep

    usable = ~(invalid | excluded)
    data = np.ascontiguousarray(np.where(usable, image, 0), dtype=np.float32)
    sky = sep.Background(data, mask=~usable, bw=128, bh=128)
    _, segments = sep.extract(data - sky.back(), 5., err=sky.rms(), minarea=5,
                              mask=~usable, segmentation_map=True)
    usable &= ~binary_dilation(segments > 0, iterations=2)
    return usable & (ra.ellipse_radius(image.shape, geometry) > geometry['sky_exclusion_pixels'])


def draw_ellipse(ax, geometry, radius, **kwargs):
    ax.add_patch(Ellipse(geometry['center_xy'], 2 * radius, 2 * radius * geometry['axis_ratio'],
                        angle=geometry['major_axis_angle_deg'], fill=False, **kwargs))


def aperture_contrast(image, invalid, source_mask, geometry, coefficients):
    print('Contrast in', geometry['bands'], 'r at 0.1--0.3 and 1.5--2.0 R_e:')
    ny, nx = image.shape
    for low, high in [(.1, .3), (1.5, 2.)]:
        lo, hi = np.asarray([low, high]) * geometry['re_pixels']
        region, w = ra.annulus_weights(image.shape, geometry, lo, hi, 5)
        good = ~(invalid[region] | source_mask[region])
        w *= good
        yy, xx = np.ogrid[region[0].start:region[0].stop, region[1].start:region[1].stop]
        a, b, c = coefficients
        sky = a + b * (xx - (nx - 1) / 2) / (nx / 2) + c * (yy - (ny - 1) / 2) / (ny / 2)
        area = w.sum()
        source = np.sum(w * np.where(good, image[region] - sky, 0)) / area
        level = np.sum(w * sky) / area
        print('  coverage', area / (np.pi * geometry['axis_ratio'] * (hi**2 - lo**2)),
              'counts, sky, net/sky', source, level, source / level)


def masks_figure(project, output):
    fig, axes = plt.subplots(2, 2, figsize=(6.85, 6.0), layout='constrained')
    eligible = {}
    for row, (galaxy, name) in enumerate(GALAXIES.items()):
        root = analysis_root(project) / 'radial' / galaxy
        geometry = json.loads((root / 'geometry.json').read_text())
        profile = Table.read(root / 'observed' / 'surface_photometry.ecsv')
        adopted = dict(geometry, center_xy=profile.meta.get('annulus_center_xy', geometry['center_xy']))
        image, wcs = read_science(stack_path(project, galaxy, 'r'))
        scale = scale_arcsec(wcs)
        base = source_exclusions(Table.read(root / 'source_mask_audit.ecsv'), image.shape, scale)
        final = (source_exclusions(Table.read(root / 'observed' / 'source_mask_audit.ecsv'), image.shape, scale)
                 if galaxy == 'm33' else base)
        invalid = np.logical_or.reduce([read_observation_mask(project, galaxy, b) for b in BANDS[galaxy]])
        eligible[galaxy] = sky_pixel_mask(image, invalid, base, geometry)
        boxes = Table.read(root / 'sky_boxes.ecsv')
        coefficients = np.load(root / 'sky_model.npz')['coefficients'][BANDS[galaxy].index('r')]
        aperture_contrast(image, invalid, final, adopted, coefficients)
        ax, sx = axes[row]; stretch = galaxy_stretch(image)
        for target in [ax, sx]:
            show_field(target, image, stretch)
        amber = ListedColormap(['#e58742'])
        # Render at full sampling before the PDF backend downsamples to its output DPI.
        ny, nx = image.shape
        ax.imshow(np.ma.masked_where(~final, final), interpolation='nearest', origin='lower',
                  cmap=amber, alpha=.82, extent=(-.5, nx - .5, -.5, ny - .5))
        for radius in [.5, 1, 1.5, 2]:
            draw_ellipse(ax, adopted, radius * geometry['re_pixels'], edgecolor='#74d7ed', lw=.65)
        draw_ellipse(ax, adopted, adopted['outer_radius_pixels'], edgecolor='white', lw=.65, ls='--')
        x, y = adopted['center_xy']
        if galaxy == 'ngc6503':
            ax.set(xlim=(x - 1210, x + 1210), ylim=(y - 830, y + 830))
        else:
            ax.set(xlim=(x - 1150, x + 1150), ylim=(y - 940, y + 940))
        panel(ax, 'ac'[row], f'{name} source exclusions')
        bar = 1 if galaxy == 'ngc6503' else 10
        left, right = ax.get_xlim(); bottom, top = ax.get_ylim()
        bar_x, bar_y = left + .06 * (right - left), bottom + .07 * (top - bottom)
        length = bar * 60 / scale
        ax.plot((bar_x, bar_x + length), (bar_y, bar_y), c='white', lw=1.2)
        ax.text(bar_x + length / 2, bar_y + .03 * (top - bottom), rf"${bar}^\prime$",
                ha='center', color='white')
        use = boxes['r_used']
        sx.scatter(boxes['x'][use], boxes['y'][use], c='#8de6ef', marker='s',
                   s=1.0 if row == 0 else 2.5, linewidths=0)
        draw_ellipse(sx, geometry, geometry['sky_exclusion_pixels'], edgecolor='white', lw=.75, ls='--')
        field_key(sx, image, wcs, 2 if row == 0 else 15)
        panel(sx, 'bd'[row], f'{name} sky samples')
    save(fig, output, 'masks_and_sky')
    return eligible


def stack_bootstrap(project, group, sampling, rng):
    selected = rng.choice(np.flatnonzero(sampling), 8192, replace=False)
    samples = []
    for record in group:
        registered = source_path(project, record['registered'])
        with fits.open(registered, memmap=False) as hdus:
            samples.append(hdus['SCI'].data.ravel()[selected])
    pixels = np.array(samples)
    if not np.isfinite(pixels).all():
        raise ValueError('Bootstrap samples require full input coverage.')
    n = len(group)
    medians, clipped = [], []
    for indices in rng.integers(0, n, size=(512, n)):
        sample = pixels[indices]
        medians.append(np.median(sample, axis=0))
        clipped.append(lr.iqr_selection(sample, 2.5)[1])
    return [float(np.median(np.std(draws, axis=0, ddof=1))) for draws in (medians, clipped)]


def rejection_figure(project, records, eligible, output):
    cmap = ListedColormap(['#efefec', '#afc7d1', '#52879f', '#204760', '#62284a'])
    norm = BoundaryNorm(np.arange(-.5, 5.5), cmap.N)
    fig, axes = plt.subplots(2, 3, figsize=(6.85, 4.8), layout='constrained')
    rng = np.random.default_rng(SEED)
    changes = []
    for row, (galaxy, name) in enumerate(GALAXIES.items()):
        for col, band in enumerate(BANDS[galaxy]):
            file = stack_path(project, galaxy, band)
            group = [r for r in records if (r['galaxy'], r['band']) == (galaxy, band)]
            with fits.open(file, memmap=False) as hdus:
                rejected = hdus['NREJECT'].data
                available = hdus['NVALID'].data + rejected
                wcs = WCS(hdus[0].header).celestial
                full = available == len(group)
                display = np.ma.masked_where(~full, np.minimum(rejected, 4))
                artist = show_rejections(axes[row, col], display, norm, cmap)
                field_key(axes[row, col], display, wcs, 2 if row == 0 else 15, light=True)
            panel(axes[row, col], 'abcdef'[3 * row + col], rf"{name}  ${band}^\prime$")
            error = fits.getdata(stack_path(project, galaxy, band, 'uncertainty'), extname='ERROR', memmap=False)
            median, iqr = stack_bootstrap(project, group, eligible[galaxy] & full, rng)
            change = 100 * (1 - iqr / median)
            changes.append(change)
            print(f'{name} {band}: finite rejected={100 * rejected.sum() / available.sum():.3f}% '
                  f'full-coverage pixels={100 * full.mean():.1f}%, coadd error={np.nanmedian(error):.6f}, '
                  f'median bootstrap={median:.6f}, IQR-mean bootstrap={iqr:.6f}, reduction={change:.2f}%')
    print(f'Mean reduction across bands: {np.mean(changes):.2f}%')
    bar = fig.colorbar(artist, ax=axes, orientation='horizontal', ticks=np.arange(5), shrink=.61, pad=.047, aspect=40)
    bar.set_ticklabels(['0', '1', '2', '3', r'$\geq 4$'])
    bar.set_label('Rejected exposures per pixel')
    save(fig, output, 'iqr_rejection')


def show_rejections(ax, display, norm, cmap):
    ny, nx = display.shape
    ax.set(xticks=[], yticks=[], facecolor='#222222')
    return ax.imshow(display[::2, ::2], origin='lower', interpolation='nearest',
                     extent=(-.5, nx - .5, -.5, ny - .5), norm=norm, cmap=cmap)


def gradient_figure(project, output):
    before, wcs = read_science(stack_path(project, 'm33', 'g'))
    path = stack_path(project, 'm33', 'g', 'background_corrected')
    after, _ = read_science(path)
    header = fits.getheader(path)
    a, b, c = (header[key] for key in ('BKG_A', 'BKG_BX', 'BKG_BY'))
    boxes = Table.read(path, hdu='SKY_BOXES')
    good = boxes['IN_FIT']
    x, y, rate = (boxes[key][good] for key in ('X_NORM', 'Y_NORM', 'MEAN'))
    norm = Normalize(-.36, .36)
    fig = plt.figure(figsize=(6.85, 2.78), layout='constrained')
    ax0, ax1, graph = fig.subplots(1, 3, width_ratios=(1., 1., 1.25))
    show_field(ax0, before - a, norm, average=4)
    show_field(ax1, after, norm, average=4)
    panel(ax0, 'a', 'Original, offset by $a$')
    panel(ax1, 'b', 'Plane subtracted')
    ny, nx = before.shape
    ax1.axvspan(nx - header['RIGHTMSK'] - .5, nx - .5, color='0.79', alpha=.72)
    for ax in (ax0, ax1):
        field_key(ax, before, wcs, 10)
    graph.scatter(x, rate - a - c * y, color=ORANGE, s=7, lw=0, alpha=.55, label='Before')
    graph.scatter(x, rate - a - b * x - c * y, color=BLUE, s=7, lw=0, alpha=.55, label='After')
    xx = np.array([x.min(), x.max()])
    graph.plot(xx, b * xx, c=ORANGE, lw=1.)
    graph.axhline(0, c=BLUE, lw=.8, ls='--')
    graph.set(xlabel='Normalized image coordinate $X$', ylabel=r'Sky rate (ADU s$^{-1}$)', xlim=(-1, .85))
    graph.legend(loc='upper right', markerscale=1.25, labelspacing=.22, borderpad=.2)
    ticks(graph)
    panel(graph, 'c', 'Accepted sky boxes')
    save(fig, output, 'm33_background')


def growth_figure(project, output):
    fig, axes = plt.subplots(1, 2, figsize=(6.85, 2.57), sharey=True, layout='constrained')
    for ax, (galaxy, name), letter in zip(axes, GALAXIES.items(), 'ab'):
        root = analysis_root(project) / 'radial' / galaxy
        geometry = json.loads((root / 'geometry.json').read_text())
        curve = Table.read(root / 'curve_of_growth.ecsv')
        radius = curve['semimajor_pixels'] * geometry['kpc_per_pixel']
        flux = curve['cumulative_r_ADU_s'] / curve['cumulative_r_ADU_s'][-1]
        ax.plot(radius, flux, c=TEAL, lw=1.4)
        re, error = geometry['re_kpc'], geometry['re_error_kpc']
        ax.axvspan(re - error, re + error, color=TEAL, alpha=.13, lw=0)
        ax.axhline(.5, color='0.48', ls=':', lw=.8)
        ax.axvline(re, color=TEAL, ls='--', lw=.85)
        ax.set(xlabel='Semimajor radius (kpc)', xlim=(0, radius[-1] * 1.025), ylim=(0, 1.04))
        ax.xaxis.set_major_locator(MaxNLocator(6, integer=True))
        ticks(ax)
        panel(ax, letter, name)
    axes[0].set_ylabel(r"Enclosed $r^\prime$ flux fraction")
    save(fig, output, 'reference_growth')


def psf_figure(project, output):
    fig, axes = plt.subplots(1, 2, figsize=(6.85, 2.57), sharey=True, layout='constrained')
    for ax, (galaxy, name), letter in zip(axes, GALAXIES.items(), 'ab'):
        for b, ls in zip(BANDS[galaxy], ('-', '--', '-.')):
            product = analysis_root(project) / 'psf_original' / f'{galaxy}_{b}_psf.npz'
            with np.load(product) as psf:
                ax.plot(psf['radii_arcsec'], psf['curve'], c=COLORS[b], label=rf"${b}^\prime$", ls=ls, lw=1.2)
        ax.axhline(.95, c='.45', linestyle=':', lw=.8)
        ax.set(xlabel='Aperture radius (arcsec)', ylim=(0, 1.045), xlim=(0, None))
        ax.legend(loc='lower right', handlelength=2.4, labelspacing=.25)
        ticks(ax)
        panel(ax, letter, name)
    axes[0].set_ylabel('Enclosed stellar flux fraction')
    save(fig, output, 'stellar_growth')


def main():
    global lr, pa, ra
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project', type=Path, default=PROJECT)
    parser.add_argument('--output', type=Path, default=Path(__file__).parent / 'figures')
    parser.add_argument('--filter-only', action='store_true', help='Print the raw-exposure filter audit only.')
    parser.add_argument('--skip-filter-audit', action='store_true', help='Render figures without the raw filter audit.')
    args = parser.parse_args()
    if not (args.project / 'light_reduction.py').is_file():
        parser.error('--project must point to the project directory with the reduction modules and saved products')
    sys.path.insert(0, str(args.project.resolve()))
    import light_reduction as lr
    import photometric_analysis as pa
    import radial_analysis as ra

    args.output.mkdir(parents=True, exist_ok=True)
    manifest = args.project / 'reduction' / 'registration_ugr.json'
    records = json.loads(manifest.read_text())
    if not args.skip_filter_audit:
        filter_audit(args.project, records)
    if not args.filter_only:
        registration_figure(args.project, records, args.output)
        gradient_figure(args.project, args.output)
        psf_figure(args.project, args.output)
        growth_figure(args.project, args.output)
        eligible = masks_figure(args.project, args.output)
        rejection_figure(args.project, records, eligible, args.output)


if __name__ == '__main__':
    main()
