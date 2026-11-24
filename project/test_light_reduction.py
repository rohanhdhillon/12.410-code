"""Run with: .venv/bin/python -m unittest discover -s project -p 'test_light_reduction.py'."""
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from astropy.io import fits
from astropy.wcs import WCS
from light_reduction import (iqr_selection, resample_noise, transfer_wcs,
                             write_images, stack_exposures, propagate_stack_errors,
                             install_external_flat, calibration_models, calibrate_exposures,
                             register_exposures)


def example_wcs():
    w = WCS(naxis=2)
    w.wcs.ctype = ['RA---TAN', 'DEC--TAN']
    w.wcs.crval, w.wcs.crpix = [23, 30], [10, 10]
    w.wcs.cdelt = [-.001, .001]
    return w


class ReductionTests(unittest.TestCase):
    def test_ugr_registration_keeps_raw_r_anchor_and_separate_products(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary); (root/'wcs').mkdir()
            wcs = example_wcs(); shape = (20,20)
            fits.writeto(root/'ngc6503_g.wcs.fits', np.ones(shape), wcs.to_header())
            fits.PrimaryHDU(header=wcs.to_header()).writeto(root/'wcs/m33_anchor.fits')
            (root/'registration.json').write_text('previous run manifest')
            records = []
            for raw, physical, setup in [('r','g','refractor_1x1'), ('i','r','refractor_3x3')]:
                source = root/f'raw_{raw}.fit'
                fits.writeto(source, np.ones(shape), wcs.to_header())
                calibrated = root/f'cal_{physical}.fit'
                header = wcs.to_header(); header['FILTER'] = physical
                header['RAWFILT'], header['BANDMAP'] = raw, True
                write_images(calibrated, dict(SCI=np.ones(shape), VARIND=np.ones(shape),
                                             CALSIG=np.full(shape,.1)), header)
                records.append(dict(source=str(source), calibrated=str(calibrated),
                                    setup=setup, reduction_setup=setup+'_ugr', galaxy='m33',
                                    product_prefix='m33_ugr', raw_band=raw, band=physical,
                                    key=physical, weak_flat=False))
            stars = np.zeros((12,2))
            cfg = dict(reuse=True, chunk_rows=4, grid_bounds={'m33':None})
            with patch('light_reduction.solve_anchor', return_value=(wcs,stars)) as anchor, \
                 patch('light_reduction.extract_stars', return_value=stars), \
                 patch('light_reduction.transfer_wcs', return_value=(wcs,.1,12)) as transfer:
                result = register_exposures(records,root,root,cfg,keep_calibrated=True)
                register_exposures(records,root,root,cfg,keep_calibrated=True)
                self.assertEqual(anchor.call_args.args[0], root/'raw_r.fit')
                self.assertEqual(transfer.call_count, 2)  # Second call reuses registered images.
            self.assertEqual((root/'registration.json').read_text(), 'previous run manifest')
            self.assertTrue((root/'registration_ugr.json').exists())
            self.assertTrue((root/'wcs/m33_ugr_grid.fits').exists())
            for record in result:
                self.assertIn('_ugr', Path(record['registered']).parent.name)
                self.assertEqual(fits.getheader(record['registered'])['FILTER'], record['band'])

    def test_external_flat_uncertainty_masks_and_cache(self):
        import warnings
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary); folder = root/'masters'; folder.mkdir()
            shape = (20, 20)
            reference = fits.Header(dict(TELESCOP='test', INSTRUME='test', XBINNING=3,
                YBINNING=3, XORGSUBF=0, YORGSUBF=0, **{'SET-TEMP': -15.}))
            header = reference.copy()
            header.update(dict(FILTER="g'", MASTER=True, BIASFILE='independent_bias.fit',
                               NORMTYPE='MEDIAN', NCOMBINE=10))
            source = root/'supplied.fits'
            flat = np.ones(shape, np.float32); flat[5, 5] = 2.
            fits.writeto(source, flat, header)
            fits.writeto(folder/'master_bias.fit', np.full(shape, 10., np.float32))
            fits.writeto(folder/'bad_pixel_mask.fit', np.zeros(shape, np.uint8))
            biases = []
            for j, value in enumerate((9., 11.)):
                path = root/f'bias{j}.fit'; fits.writeto(path, np.full(shape, value), reference)
                biases.append(path)
            science = root/'light.fit'; h = reference.copy()
            h.update(dict(FILTER='g', EXPTIME=2., EGAIN=1.))
            fits.writeto(science, np.full(shape, 110., np.float32), h)
            # Deliberately nonexistent raw g inputs: external variance must not read them.
            setup = dict(reference=reference, shape=shape, output_dir=folder,
                         flat_files={}, low_signal_flats={'g'}, missing_readout=True,
                         bias_paths=biases, flat_paths=[], science_paths=[science],
                         flat_groups={'g': [root/'missing_old_flat.fit']})
            installed = install_external_flat(setup, source, 'g')
            timestamp = installed.stat().st_mtime_ns
            install_external_flat(setup, source, 'g')
            self.assertEqual(installed.stat().st_mtime_ns, timestamp)
            np.testing.assert_array_equal(fits.getdata(installed), flat)
            self.assertNotIn('g', setup['low_signal_flats'])
            cfg = dict(reuse=True, chunk_rows=4, gain_header_keys=['EGAIN'],
                       gain_agreement=.05, gain_overrides={}, normalization_stride=2,
                       flat_aperture_unbinned={'refractor': 6}, random_seed=1,
                       flat_apertures=10, saturation_adu=65535, allow_weak_m33_g=False)
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                models = calibration_models({'refractor_3x3': setup}, root, cfg)
            model = models['refractor_3x3']; fm = model['flats']['g']
            self.assertEqual(fm['alpha'], 0.)
            vf = fits.getdata(fm['file'], extname='VAR')
            expected = np.pi/20*(5884.+2.)/5884.**2 + .01**2
            self.assertAlmostEqual(float(vf[1, 1]), expected, places=10)
            records = calibrate_exposures({'refractor_3x3': setup}, models, root, cfg)
            with fits.open(records[0]['calibrated']) as h:
                self.assertTrue(h[0].header['FLATPROV'])
                self.assertFalse(h[0].header['FLATLOW'])
                self.assertTrue(np.isnan(h['SCI'].data[:, 0]).all())
                self.assertTrue(np.isnan(h['SCI'].data[5, 5]))
                self.assertAlmostEqual(float(h['SCI'].data[1, 1]), 50.)
                vb = fits.getdata(model['bias_file'], extname='VAR')
                self.assertAlmostEqual(float(h['CALSIG'].data[1, 1]**2),
                                       float(vb[1, 1]/4 + 2500*vf[1, 1]), places=6)
            setup.update(band_map={'g':'u','r':'g','i':'r'}, reduction_suffix='_ugr', product_prefix='m33_ugr')
            mapped = calibrate_exposures({'refractor_3x3': setup}, models, root, cfg)
            self.assertEqual(mapped[0]['band'], 'u')
            self.assertEqual(mapped[0]['raw_band'], 'g')
            self.assertNotEqual(mapped[0]['key'], records[0]['key'])
            self.assertNotEqual(mapped[0]['calibrated'], records[0]['calibrated'])
            with fits.open(mapped[0]['calibrated']) as h:
                self.assertEqual(h[0].header['FILTER'], 'u')
                self.assertEqual(h[0].header['RAWFILT'], 'g')
                self.assertTrue(h[0].header['BANDMAP'])
            self.assertEqual(fits.getheader(science)['FILTER'], 'g')
            cached_models = calibration_models({'refractor_3x3': setup}, root, cfg)
            self.assertEqual(cached_models['refractor_3x3']['key'], model['key'])
            install_external_flat(setup, source, 'g', fractional_floor=.02)
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                revised = calibration_models({'refractor_3x3': setup}, root, cfg)
            self.assertNotEqual(revised['refractor_3x3']['key'], model['key'])

    def test_iqr_rejection_and_no_coverage(self):
        a = np.zeros((10, 2, 2), np.float32)
        a[-1, 0, 0] = 100
        a[:, 1, 1] = np.nan
        _, mean, _, count, rejected = iqr_selection(a, 2.5)
        self.assertEqual(rejected[0, 0], 1)
        self.assertEqual(count[0, 0], 9)
        self.assertEqual(mean[0, 0], 0)
        self.assertEqual(count[0, 1], 10)
        self.assertEqual(count[1, 1], 0)
        self.assertTrue(np.isnan(mean[1, 1]))

    def test_iqr_matches_numpy_linear_percentiles(self):
        import warnings
        rng = np.random.default_rng(21)
        for exposures in (1, 2, 3, 9, 10, 11):
            a = rng.normal(size=(exposures, 7, 9))
            a[rng.random(a.shape) < .2] = np.nan
            a[:, 0, 0] = np.nan
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                q1, median, q3 = np.nanpercentile(a, [25, 50, 75], axis=0)
            expected = np.isfinite(a) & (np.abs(a-median) <= 2.5*(q3-q1))
            keep, _, _, _, _ = iqr_selection(a, 2.5)
            np.testing.assert_array_equal(keep, expected)

    def test_squared_interpolation_weights(self):
        source = example_wcs()
        target = source.deepcopy()
        target.wcs.crpix = np.array(target.wcs.crpix)+.5
        variance, shared = resample_noise(np.ones((20, 20)), np.full((20, 20), 2.),
                                          source, target, (20, 20), {'chunk_rows': 4})
        np.testing.assert_allclose(variance[2:-2, 2:-2], .25, atol=1e-6)
        np.testing.assert_allclose(shared[2:-2, 2:-2], 2., atol=1e-6)

    def test_pixel_area_scaling_for_different_binnings(self):
        from reproject import reproject_interp
        from astropy.wcs.utils import proj_plane_pixel_area
        target = example_wcs()
        target.wcs.crpix = [31, 31]
        source = example_wcs()
        source.wcs.crpix = [11, 11]
        source.wcs.cdelt = np.array(source.wcs.cdelt)*3
        image, _ = reproject_interp((np.full((21, 21), 9.), source), target, shape_out=(61, 61))
        image *= abs(proj_plane_pixel_area(target)/proj_plane_pixel_area(source))
        np.testing.assert_allclose(image[5:-5, 5:-5], 1., atol=1e-6)

    def test_wcs_matches_numpy_pixel_origin(self):
        target = np.random.default_rng(1).uniform(10, 190, (70, 2))
        source = target+np.array([4.2, -3.3])
        w = example_wcs()
        cfg = dict(control_points=50, match_radius_pixels=6, min_matches=12,
                   wcs_clip_iterations=4, sip_degree=2, wcs_clip_floor=.5,
                   wcs_clip_sigma=4, max_wcs_rms=2)
        fitted, rms, _ = transfer_wcs('test', source, target, w, cfg)
        predicted = np.column_stack(fitted.world_to_pixel(w.pixel_to_world(target[:, 0], target[:, 1])))
        np.testing.assert_allclose(predicted, source, atol=1e-3)
        self.assertLess(rms, 1e-3)

    def test_lossless_fits_and_shared_calibration_error(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            a = np.random.default_rng(2).normal(size=(20, 20)).astype(np.float32)
            a[0, 0] = np.nan
            write_images(root/'lossless.fit', {'SCI': a})
            np.testing.assert_array_equal(fits.getdata(root/'lossless.fit', extname='SCI'), a)
            records = []
            for j in range(2):
                path = root/f'input{j}.fit'
                header = example_wcs().to_header()
                header['FLATLOW'], header['BUNIT'] = False, 'adu/s'
                header['RUNKEY'] = f'test-input-{j}'
                write_images(path, {'SCI': np.ones((20, 20), np.float32),
                    'VARIND': np.ones((20, 20), np.float32),
                    'CALSIG': np.full((20, 20), 2., np.float32)}, header)
                records.append(dict(galaxy='test', band='g', registered=str(path), weak_flat=False))
            cfg = dict(reuse=True, chunk_rows=4, iqr_multiplier=2.5)
            for r in records: r['product_prefix'] = 'test_ugr'
            stacks = stack_exposures(records, root, cfg)
            self.assertEqual(stacks[('test','g')].name, 'test_ugr_g_stack.fit')
            errors = propagate_stack_errors(stacks, root, cfg)
            self.assertEqual(errors[('test','g')].name, 'test_ugr_g_uncertainty.fit')
            # Reuse remains valid after per-frame noise maps are compacted.
            again = stack_exposures(records, root, cfg)
            self.assertEqual(again, stacks)
            self.assertTrue(fits.getheader(root/'input0.fit')['NOISESUM'])
            with fits.open(errors[('test', 'g')]) as h:
                np.testing.assert_allclose(h['VAR'].data, 4.5)
                np.testing.assert_allclose(h['VARCAL'].data, 4.)


if __name__ == '__main__':
    unittest.main()
