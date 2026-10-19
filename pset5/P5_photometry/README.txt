P5_photometry - aperture photometry frames for 12.410 Problem Set 5
===================================================================

The open cluster M34, through the r' filter on the Wallace Shed C14
(Pier 3), on the night of 2026 Oct 20-21.  Every frame is already
reduced: bias and dark subtracted and flat-fielded, stored as float32
in ADU.  The camera was run at unity gain (GAIN = 1.0 e-/ADU), so one
ADU is one detected photon.  The sky has NOT been subtracted.

    M34_r.fits         60 s, near the meridian
    seeing/            eight 30 s frames, 512 x 512 subframes of the
                       middle of the field, taken as M34 rose from the
                       northeast to the meridian
    M34_catalog.csv    reference stars on M34_r.fits: position (ra, dec
                       in degrees), r' magnitude and its uncertainty

Each header carries the exposure time, the airmass at mid-exposure,
the plate scale (PIXSCALE, 0.397 arcsec/pixel), the read noise, the
linearity limit (SATURATE, in ADU) and a WCS.
