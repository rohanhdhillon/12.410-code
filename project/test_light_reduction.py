"""Run with: .venv/bin/python -m unittest discover -s project -p 'test_light_reduction.py'."""
from pathlib import Path
import tempfile
import unittest

import numpy as np
from astropy.io import fits
from astropy.wcs import WCS
from light_reduction import (iqr_selection, resample_noise, transfer_wcs,
                             write_images, stack_exposures, propagate_stack_errors)


def example_wcs():
    w = WCS(naxis=2)
    w.wcs.ctype = ['RA---TAN', 'DEC--TAN']
    w.wcs.crval, w.wcs.crpix = [23, 30], [10, 10]
    w.wcs.cdelt = [-.001, .001]
    return w


class ReductionTests(unittest.TestCase):
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
            stacks = stack_exposures(records, root, cfg)
            errors = propagate_stack_errors(stacks, root, cfg)
            # Reuse remains valid after per-frame noise maps are compacted.
            again = stack_exposures(records, root, cfg)
            self.assertEqual(again, stacks)
            self.assertTrue(fits.getheader(root/'input0.fit')['NOISESUM'])
            with fits.open(errors[('test', 'g')]) as h:
                np.testing.assert_allclose(h['VAR'].data, 4.5)
                np.testing.assert_allclose(h['VARCAL'].data, 4.)


if __name__ == '__main__':
    unittest.main()
