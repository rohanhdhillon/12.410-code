"""Numerical checks for the radial measurement/covariance contracts."""
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
from astropy.io import fits
from astropy.table import Table
from astropy.wcs import WCS
import radial_analysis as ra


class RadialAnalysisTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(); self.path = Path(self.tmp.name)
        self.shape = (64,64)
        self.g = dict(center_xy=(31.5,31.5),axis_ratio=.7,major_axis_angle_deg=20.,
                      outer_radius_pixels=25.,distance_mpc=1.,distance_error_mpc=.05,
                      dust_sample_radius_deg=.5,kpc_per_pixel=1000/206264.806247)
        self.cfg = dict(annulus_subpixels=5,minimum_annulus_fraction=.6,
                        maximum_calibration_condition=100.,crowded_group_radius_arcsec=8.,
                        crowded_group_minimum=3,manual_accept_ids=[],dust_samples=4,
                        dust_timeout_seconds=1,dust_fractional_error=.16,dust_ebv_floor=.01,
                        extinction_law_fractional_error=0.,
                        extinction_coefficients=dict(g=3.303,r=2.285,i=1.698))
        self.wcs = WCS(naxis=2); self.wcs.wcs.ctype=['RA---TAN','DEC--TAN']
        self.wcs.wcs.cdelt=[-.001,.001]; self.wcs.wcs.crval=[30.,20.]
        yy,xx = np.indices(self.shape); self.sky_image=5+.3*(xx-31.5)/32-.1*(yy-31.5)/32
        self.frames={}; self.errors={}
        for b,signal in zip('gri',[10.,20.,30.]):
            self.frames[b]=dict(data=signal+self.sky_image,variance=np.ones(self.shape),
                                mask=np.zeros(self.shape,bool),scale=1.,psf_matched=False,
                                wcs=self.wcs)
            path=self.path/f'{b}.fit'; self.errors[b]=path
            fits.HDUList([fits.PrimaryHDU(),fits.ImageHDU(np.ones(self.shape),name='VARRAND'),
                          fits.ImageHDU(np.full(self.shape,.04),name='VARCAL')]).writeto(path)
        self.sky=dict(coefficients=np.tile([5.,.3,-.1],(3,1)),covariance=np.eye(9)*1e-6,
                      alternatives=np.empty((0,3,3)),residual_floor=np.zeros(3))

    def tearDown(self): self.tmp.cleanup()

    def profile(self):
        red=np.zeros(self.shape,bool); red[30:33,30:33]=True
        # A different bad pixel in one band must be excluded from every band.
        self.frames['g']['mask'][35,35]=True
        return ra.measure_annuli(self.frames,self.sky,red,np.zeros(self.shape,bool),
                                 self.g,self.cfg,[0.,10.,20.],self.errors)

    def test_masked_mean_and_coherent_bound(self):
        p=self.profile()
        np.testing.assert_allclose(p['mean'],np.array([[10,10],[20,20],[30,30]]),atol=1e-12)
        np.testing.assert_allclose(p['flat_bound'],.2,atol=1e-12)
        self.assertLess(p['area'][0],p['full_area'][0])
        self.assertTrue(np.linalg.eigvalsh(p['covariance']).min()>-1e-10)
        self.assertGreater(p['covariance'][0,1],0)  # shared systematics persist between annuli

    def test_partial_calibration_dependencies(self):
        p=self.profile()
        def solution(z,c,pair):
            return dict(zp=z,color_term=c,pivot=.4,color_pair=pair,covariance=[[.01,0],[0,0]])
        s=dict(g=solution(25,0,['g','r']),r=solution(24,0,['g','r']))
        m=ra.magnitude_profiles(p,s,self.cfg)
        np.testing.assert_allclose(m['magnitudes'][0],m['instrumental'][0]+25)
        self.assertTrue(np.isnan(m['magnitudes'][2]).all())
        missing=dict(r=solution(24,-1,['g','r']),i=solution(23,-1,['r','i']))
        self.assertTrue(np.isnan(ra.magnitude_profiles(p,missing,self.cfg)['magnitudes']).all())

    def test_plot_separates_calibrated_and_instrumental_scales(self):
        import matplotlib.pyplot as plt
        p=self.profile()
        solution=dict(zp=25.,color_term=0.,pivot=0.,color_pair=['g','r'],covariance=[[.01,0],[0,0]])
        m=ra.magnitude_profiles(p,{'g':solution,'r':solution},self.cfg)
        t=ra.save_profiles(p,m,self.g,self.path/'plot')
        figure=ra.plot_profiles(t,self.path/'plot/covariance.npz','test',self.path/'plot/profiles.png')
        self.assertEqual(len(figure.axes),4)
        self.assertIn('Calibrated',figure.axes[0].get_title())
        self.assertIn('Instrumental',figure.axes[1].get_title())
        plt.close(figure)

    def test_exponential_half_light_radius(self):
        edges=np.linspace(0,100,1001); h=10.
        cumulative=1-(1+edges/h)*np.exp(-edges/h)
        areas=np.pi*np.diff(edges**2)
        p=dict(edges=edges,mean=np.tile(np.diff(cumulative)/areas,(3,1)),
               full_area=areas,valid=np.ones(len(areas),bool))
        re,*_=ra._half_light(p)
        self.assertAlmostEqual(re/h,1.678,places=2)

    def test_residual_sky_plane_with_correlated_bands(self):
        shape=(256,256); y,x=np.indices(shape); X=(x-127.5)/128; Y=(y-127.5)/128
        rng=np.random.default_rng(12); common=rng.normal(0,.03,shape)
        geometry=dict(self.g,center_xy=(127.5,127.5),sky_exclusion_pixels=85.)
        r=ra.ellipse_radius(shape,geometry)
        galaxy=np.where(r<70,5*np.exp(-r/12),0)
        frames={b:dict(data=2+.3*X-.1*Y+galaxy+common+rng.normal(0,.01,shape),
                       mask=np.zeros(shape,bool)) for b in 'gri'}
        cfg=dict(self.cfg,sky_box_pixels=32,sky_box_valid_fraction=.45,
                 minimum_sky_boxes=12,sky_source_dilation_pixels=2,
                 sky_bootstrap_block_pixels=64,sky_bootstrap_draws=80,
                 sky_exclusion_sensitivity_factors=[1.05],random_seed=1)
        sky=ra.fit_sky(frames,np.zeros(shape,bool),geometry,cfg,self.path)
        np.testing.assert_allclose(sky['coefficients'],np.tile([2,.3,-.1],(3,1)),atol=.01)
        self.assertTrue(np.linalg.eigvalsh(sky['covariance']).min()>-1e-12)
        self.assertGreater(sky['covariance'][0,3],0)

    def test_reference_model_does_not_unmask_science_bins(self):
        edges=np.linspace(0,100,101); h=10.
        cumulative=1-(1+edges/h)*np.exp(-edges/h); areas=np.pi*np.diff(edges**2)
        intensity=np.diff(cumulative)/areas
        valid=np.ones(100,bool); valid[:5]=False
        means=np.tile(intensity,(3,1)); means[:,~valid]=np.nan
        V=np.eye(300)*1e-14
        profile=dict(edges=edges,mean=means,full_area=areas,valid=valid,
                     covariance=V,stat_covariance=V)
        filled,method,fraction=ra.reference_growth(profile,self.cfg)
        self.assertTrue(np.isnan(profile['mean'][1,:5]).all())
        self.assertFalse(profile['valid'][0])
        self.assertIn('exponential',method); self.assertGreater(fraction,0)
        self.assertAlmostEqual(ra._half_light(filled)[0]/h,1.678,places=2)

    def test_crowded_groups_need_strong_evidence(self):
        table=Table(dict(id=['sdss:1','sdss:2','sdss:3','gaia:4'],x=[30.,31.,32.,33.],
                         y=[31.,31.,31.,31.],auto_mask=[True]*4,
                         classification=['background_galaxy']*3+['foreground_star'],
                         mask_radius_arcsec=[1.]*4))
        red,review=ra.source_masks(self.frames['r'],table,self.g,self.cfg,self.path)
        audit=Table.read(self.path/'source_mask_audit.ecsv')
        np.testing.assert_array_equal(audit['radial_masked'],[False,False,False,True])
        self.assertTrue(red.any() and review.any())

    def test_foreground_cache_and_correlated_color_errors(self):
        from astroquery.ipac.irsa.irsa_dust import IrsaDust
        with patch.object(IrsaDust,'get_query_table',return_value=Table({'ext SFD ref':[.1]})) as query:
            dust=ra.foreground_extinction(self.frames['r'],self.g,self.cfg,self.path/'dust.json')
            ra.foreground_extinction(self.frames['r'],self.g,self.cfg,self.path/'dust.json')
            changed=dict(self.cfg,extinction_coefficients=dict(g=4.,r=2.,i=1.))
            updated=ra.foreground_extinction(self.frames['r'],self.g,changed,self.path/'dust.json')
            self.assertAlmostEqual(updated['extinction'][0],.4)
            self.assertEqual(query.call_count,4)
        np.testing.assert_allclose(dust['extinction'],[.3303,.2285,.1698])
        V=np.asarray(dust['covariance']); sigma=.016
        self.assertAlmostEqual(V[0,0]+V[1,1]-2*V[0,1],(3.303-2.285)**2*sigma**2)
        p=self.profile(); m=ra.magnitude_profiles(p,{},self.cfg)
        observed=ra.save_profiles(p,m,self.g,self.path/'observed')
        corrected=ra.save_profiles(p,m,self.g,self.path/'corrected',dust)
        np.testing.assert_allclose(observed['instrumental_mu'][:2]-corrected['instrumental_mu'][:2],.3303)


if __name__=='__main__': unittest.main()
