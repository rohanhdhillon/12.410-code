"""Selection checks for scoped catalog-error overrides."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import warnings

import numpy as np
from astropy.table import Table
from astropy.wcs import WCS
import photometric_analysis as pa


class SDSSQueryTests(unittest.TestCase):
    def setUp(self):
        self.cfg = dict(sdss_release=17, catalog_timeout_seconds=10, sdss_tile_arcmin=30.,
                        catalog_cone_margin_arcmin=.1)
        self.rows = [dict(objID=1237679579471347788+j, ra=30., dec=20., type=6, clean=1,
                          u=16., g=15., r=14.5, i=14.3, eu=.03, eg=.02, er=.02, ei=.02,
                          z=None, zWarning=None, specclass='') for j in range(2)]

    def response(self, payload):
        from requests import Response
        response = Response(); response.status_code = 200
        response._content = json.dumps(payload).encode()
        return response

    def json_table(self, rows=None):
        rows = self.rows if rows is None else rows
        payload = [dict(TableName='SqlQuery', Rows=[dict(query='metadata, not photometry')]),
                   dict(TableName='Table1', Rows=rows)]
        with patch('requests.get', return_value=self.response(payload)):
            return pa.sdss_query_json('SELECT photometry', self.cfg)

    def test_json_keeps_object_ids_exact_and_handles_null_spectra(self):
        table = self.json_table()
        self.assertEqual(table['objID'].dtype, np.dtype('uint64'))
        self.assertEqual(int(table['objID'][1])-int(table['objID'][0]), 1)
        self.assertTrue(np.isnan(table['z']).all())
        self.assertTrue(np.isnan(table['zWarning']).all())
        self.assertTrue((table['specclass'] == '').all())
        self.assertIsNone(self.json_table([]))

    def test_json_service_errors_are_not_treated_as_empty_catalogs(self):
        for payload in (dict(ErrorMessage='SQL failed'), [dict(TableName='SqlQuery', Rows=[])],
                        [dict(TableName='Table1', Rows=[dict(error='invalid SQL')])]):
            with self.subTest(payload=payload), patch('requests.get', return_value=self.response(payload)):
                with self.assertRaises(RuntimeError):
                    pa.sdss_query_json('SELECT photometry', self.cfg)
        response = self.response({}); response._content = b'<html>403 Forbidden</html>'
        with patch('requests.get', return_value=response):
            with self.assertRaisesRegex(RuntimeError, 'non-JSON'):
                pa.sdss_query_json('SELECT photometry', self.cfg)

    def test_csv_parse_failure_switches_remaining_tiles_to_json(self):
        from astropy.coordinates import SkyCoord
        from astropy.io.ascii import InconsistentTableError
        from astroquery.sdss import SDSS
        wcs = WCS(naxis=2); wcs.wcs.ctype = ['RA---TAN','DEC--TAN']
        wcs.wcs.crval = [30.,20.]; wcs.wcs.crpix = [10.,10.]; wcs.wcs.cdelt = [-.001,.001]
        frame = dict(data=np.zeros((20,20)), wcs=wcs)
        first = self.json_table(self.rows[:1]); second = self.json_table(self.rows[1:])
        for initial in (False, True):
            effects = [first, InconsistentTableError('HTML error page')] if initial else InconsistentTableError('HTML error page')
            with self.subTest(initial_csv_success=initial), \
                 patch.object(pa, 'image_cone', return_value=(SkyCoord(30.,20.,unit='deg'), .01)), \
                 patch.object(SDSS, 'query_sql', side_effect=effects) as csv, \
                 patch.object(pa, 'sdss_query_json', return_value=second if initial else self.json_table()) as api, \
                 warnings.catch_warnings():
                warnings.simplefilter('ignore')
                catalog = pa.sdss_catalog(frame, self.cfg)
                self.assertEqual(csv.call_count, 2 if initial else 1)
                self.assertEqual(api.call_count, 3 if initial else 4)
                self.assertEqual(len(catalog), 2)
                self.assertEqual(int(catalog['objID'][1])-int(catalog['objID'][0]), 1)
                self.assertEqual(catalog.meta['sdss_transport'], 'SkyServer JSON fallback')


class ColorResponseTests(unittest.TestCase):
    def setUp(self):
        self.cfg = dict(minimum_color_stars=20, fit_color_terms=True,
                        minimum_calibration_stars=8, zp_clip_iterations=4,
                        zp_clip_sigma=100., color_term_significance=1.5)

    def test_significance_threshold_for_both_slope_signs(self):
        x = np.linspace(-.1, .1, 24)
        error = np.full(len(x), .2)
        slope_error = .2 / np.sqrt(np.sum(x**2))
        self.assertGreater(slope_error, .1)  # Previously rejected regardless of significance.
        for sign in (-1, 1):
            for significance, detected in ((1.49, False), (1.51, True)):
                with self.subTest(sign=sign, significance=significance):
                    y = 25. + sign * significance * slope_error * x
                    beta, covariance, *_ = pa.fit_color_response(x, y, error, self.cfg)
                    self.assertEqual(len(beta), 2 if detected else 1)
                    if detected:
                        self.assertAlmostEqual(beta[1], sign * significance * slope_error)
                        self.assertAlmostEqual(np.sqrt(covariance[1, 1]), slope_error)

    def test_uncertain_measurement_has_less_influence(self):
        x = np.linspace(-1., 1., 25)
        y = 25. + .5*x
        y[-1] += 5.
        error = np.full(len(x), .02)
        error[-1] = 100.
        design = np.column_stack([np.ones(len(x)), x])
        expected = np.linalg.lstsq(design/error[:, None], y/error, rcond=None)[0]
        beta, *_ = pa.fit_color_response(x, y, error, self.cfg)
        np.testing.assert_allclose(beta, expected, atol=1e-10)
        self.assertAlmostEqual(beta[1], .5, places=5)
        # The zero-point-only fallback uses inverse-variance weights too.
        beta, *_ = pa.fit_color_response(x, y, error, dict(self.cfg, fit_color_terms=False))
        self.assertAlmostEqual(beta[0], np.average(y, weights=1/error**2))


class CatalogSelectionTests(unittest.TestCase):
    def test_ugr_catalog_refresh_and_no_apass_u_substitution(self):
        cfg = dict(minimum_catalog_stars=1, sdss_release=17, allow_apass_fallback=True)
        old = Table(dict(type=[6], clean=[1], g=[15.], r=[14.5], i=[14.3],
                         eg=[.01], er=[.01], ei=[.01]))
        old.meta['catalog'] = 'old gri cache'
        with tempfile.TemporaryDirectory() as temporary:
            cache = Path(temporary)/'catalog.ecsv'; old.write(cache)
            with patch.object(pa, 'sdss_catalog', side_effect=RuntimeError('offline')):
                with self.assertRaisesRegex(RuntimeError, 'APASS has no u band'):
                    pa.reference_catalog({}, cfg, cache, bands='ugr')
            updated = old.copy(); updated['u'] = [16.]; updated['eu'] = [.02]
            with patch.object(pa, 'sdss_catalog', return_value=updated) as query:
                table, _ = pa.reference_catalog({}, cfg, cache, bands='ugr')
                self.assertIn('u', table.colnames)
                pa.reference_catalog({}, cfg, cache, bands='ugr')
                self.assertEqual(query.call_count, 1)

    def test_notebook_contaminant_masks_preserve_calibration_stars(self):
        notebook = json.loads(Path(__file__).with_name('pipeline_one.ipynb').read_text())
        cell = next(c for c in notebook['cells']
                    if ''.join(c['source']).startswith('# 3a. Build classifications'))
        original_mask = np.zeros((5, 5), bool)
        frame = dict(mask=original_mask.copy())
        secure = original_mask.copy(); secure[2, 2] = True
        table = Table(dict(in_corridor=[True]))
        ns = dict(pa=pa, photometry_frames={'ngc6503': {'r': frame}},
                  photometry_psfs={}, sloan_catalogs={'ngc6503': table},
                  analysis_parameters={}, analysis_output_dir=Path('/unused'),
                  display=lambda value: None)
        with patch.object(pa, 'gaia_catalog', return_value=table), \
             patch.object(pa, 'contaminants', return_value=(table, secure, original_mask)):
            exec(compile(''.join(cell['source']), 'contaminant_cell', 'exec'), ns)
        np.testing.assert_array_equal(frame['mask'], original_mask)
        self.assertTrue(ns['secure_masks']['ngc6503'][2, 2])

    def test_error_override_is_limited_to_requested_galaxy_and_band(self):
        wcs = WCS(naxis=2)
        wcs.wcs.ctype = ['RA---TAN', 'DEC--TAN']
        wcs.wcs.crval = [30., 20.]
        wcs.wcs.cdelt = [-.001, .001]
        positions = np.array([(x, y) for x in (5., 15., 25., 35.) for y in (5., 20.)])
        ra, dec = wcs.pixel_to_world_values(*positions.T)
        cat = Table(dict(ra=ra, dec=dec, type=np.full(8, 6), clean=np.ones(8)))
        for b in 'gri':
            cat[b] = np.full(8, 15.)
            cat['e'+b] = np.full(8, .08)
        cat.meta['catalog'] = 'synthetic Sloan standards'
        objects = np.zeros(8, dtype=[('x', float), ('y', float), ('peak', float), ('flag', int)])
        objects['x'], objects['y'] = positions.T
        frame = dict(wcs=wcs, data=np.ones((40, 40)), scale=1., psf_matched=False, header={})
        galaxies = ('ngc6503', 'm33')
        frames = {g: {b: frame for b in 'gri'} for g in galaxies}
        psf = dict(radii_arcsec=[0., 1.], curve=[0., 1.])
        psfs = {g: {b: psf for b in 'gri'} for g in galaxies}
        cfg = dict(galaxies={g: dict(center_xy=(20., 20.), major_axis_angle_deg=0.) for g in galaxies}, calibration_galaxy_exclusion_scale=1.,
                   color_pairs=dict(g=('g', 'r'), r=('g', 'r'), i=('r', 'i')),
                   max_catalog_error=.05, catalog_error_overrides={'ngc6503': {'i': .10}},
                   catalog_magnitude_range=(12., 19.), catalog_match_arcsec=2.,
                   maximum_star_peak_rate={g: {b: None for b in 'gri'} for g in galaxies},
                   calibration_isolation_factor=3., max_instrumental_fractional_error=.1,
                   aperture_correction_floor_mag=.02, minimum_calibration_stars=8,
                   minimum_color_stars=20, fit_color_terms=True, zp_clip_iterations=4, zp_clip_sigma=3.)
        with tempfile.TemporaryDirectory() as temporary, \
             patch.object(pa, 'detect', return_value=objects), \
             patch.object(pa, 'galaxy_mask', return_value=np.zeros((40, 40), bool)), \
             patch.object(pa, 'common_aperture', return_value=1.), \
             patch.object(pa, 'photometry', return_value=dict(flux=100., variance=1.)), \
             warnings.catch_warnings():
            warnings.simplefilter('ignore')
            solutions, _ = pa.fit_zero_points(frames, psfs, {g: cat for g in galaxies}, cfg, temporary)
            self.assertEqual(set(solutions['ngc6503']), {'i'})
            self.assertEqual(solutions['m33'], {})
            self.assertEqual(solutions['ngc6503']['i']['stars'], 8)
            counts = json.loads((Path(temporary)/'selection_diagnostics.json').read_text())
            self.assertEqual(counts['ngc6503_i']['catalog_error_limit_mag'], .10)
            self.assertEqual(counts['m33_i']['catalog_error_limit_mag'], .05)
