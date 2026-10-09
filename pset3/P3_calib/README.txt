P3_calib - calibration frames for 12.410 Problem Set 3
======================================================

Every frame here came off the same camera: the CMOS imager on the Wallace
Shed C14 (Pier 3), binned 2x2, read out in HCG mode at gain slider 100 and
offset 30.  That is the camera that took the NEO frames in Problem Set 2.

    bias/           10 x 0 s, sensor at -10 C
    illuminated/    14 pairs of flat-panel exposures, lamp_NNa / lamp_NNb,
                    laddering from a few tens of ADU to near saturation,
                    sensor at -10 C
    dark/           2 frames each at 30, 60, 120, 300 and 600 s, sensor
                    at -10 C
    dark_temp/      600 s darks at -15, -5, 0 and +5 C

MANUFACTURER'S DATA SHEET
-------------------------

    gain                    1.20  e-/ADU
    read noise              4.50  e- rms      (median pixel)
    dark current            0.020 e-/s/pixel  (median pixel, at -10 C)
    full well              62000  e-
    ADC range              65535  ADU (16-bit)

These are nominal figures for the sensor model, not a calibration of this
particular camera, and two of them are quoted for the median pixel.  Deciding
whether you believe them, and which statistic you actually need, is the
problem set.
