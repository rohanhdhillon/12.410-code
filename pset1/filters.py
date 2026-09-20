"""
Filters available at MIT Wallace Astrophysical Observatory.
Source: web.mit.edu/wallace/instruments.html (telescopes/filter wheels),
cross-referenced with standard photometric-system definitions for
center_nm/width_nm:
- Sloan u'g'r'i'z' : Fukugita et al. 1996, AJ 111, 1748 (SDSS camera paper)
- Johnson-Cousins BVRI : Table 10.2, course textbook (Photometry.pdf),
values from Bessell 1992 / Bessell, Castelli & Plez 1998
- Halpha/OIII/SII : standard narrowband line wavelengths (air). Wallace's
instruments page does not list exact FWHM for these narrowband
filters, so a single typical commercial value is given in
width_nm, with the assumed range noted in 'notes'.
- Clear, BB (blue-blocking), L/R/G/B (LRGB) : these are not standardized
photometric bandpasses. center_nm/width_nm are approximate,
nominal descriptions of the passband, not calibrated filter-curve
parameters.
Telescope codes used in the 'telescopes' field:
Elliot24 = Elliot 24-inch (QHY461PH CMOS, main imaging port)
Wallace24 = original Wallace 24-inch, Ealing / SBIG STL-1001e
ShedC14 = the four 14-inch C14's in the Shed (Piers 2-4; identical filter sets)
Pier1 = Pier 1 90mm Astro-Tech refractor
NOTE ON NAMING: the Pier 1 LRGB tri-color set includes filters also called
"R" and "B", which would collide with the Cousins R and Johnson B keys
below. To keep dictionary keys unique, the LRGB filters are keyed as
'L_rgb', 'R_rgb', 'G_rgb', 'B_rgb'.
Access pattern: filters['filter_name']['property'], e.g.
filters['I']['center_nm'] -> 798
filters['I']['width_nm'] -> 149
filters['I']['telescopes'] -> ['Wallace24']
"""
filters = {
"Clear": {
"telescopes": ["Wallace24", "ShedC14"],
"center_nm": 675,
"width_nm": 650,
"notes": "Unfiltered / full detector response, not a bandpass.",
},
"BB": {
"telescopes": ["Elliot24"],
"center_nm": 750,
"width_nm": 500,
"notes": ("Blue-blocking long-pass filter; blocks below ~500 nm, "
"passes through to the detector's red cutoff (~1000-1100 nm)."),
},
"u'": {
"telescopes": ["Elliot24", "ShedC14", "Pier1"],
"center_nm": 350,
"width_nm": 60,
"notes": "Sloan/SDSS u'.",
},
"g'": {
"telescopes": ["Elliot24", "ShedC14", "Pier1"],
"center_nm": 480,
"width_nm": 140,
"notes": "Sloan/SDSS g'.",
},
"r'": {
"telescopes": ["Elliot24", "ShedC14", "Pier1"],
"center_nm": 625,
"width_nm": 140,
"notes": "Sloan/SDSS r'.",
},
"i'": {
"telescopes": ["Elliot24", "ShedC14", "Pier1"],
"center_nm": 770,
"width_nm": 150,
"notes": "Sloan/SDSS i'.",
},
"z'": {
"telescopes": ["Elliot24", "ShedC14", "Pier1"],
"center_nm": 910,
"width_nm": 120,
"notes": ("Sloan/SDSS z'; red edge is set by the CMOS/CCD QE cutoff "
"(~1000-1100 nm), not the filter itself."),
},
"B": {
"telescopes": ["Wallace24"],
"center_nm": 436,
"width_nm": 94,
"notes": "Johnson B (course textbook Table 10.2).",
},
"V": {
"telescopes": ["Wallace24"],
"center_nm": 545,
"width_nm": 88,
"notes": "Johnson V (course textbook Table 10.2).",
},
"R": {
"telescopes": ["Wallace24", "ShedC14"],
"center_nm": 641,
"width_nm": 138,
"notes": ("Assumed Cousins Rc (course textbook Table 10.2); "
"Wallace's page doesn't specify Johnson vs. Cousins."),
},
"I": {
"telescopes": ["Wallace24"],
"center_nm": 798,
"width_nm": 149,
"notes": ("Assumed Cousins Ic (course textbook Table 10.2); "
"Wallace's page doesn't specify Johnson vs. Cousins."),
},
"Halpha": {
"telescopes": ["Elliot24", "Pier1"],
"center_nm": 656.3,
"width_nm": 5,
"notes": ("Narrowband H-alpha; exact FWHM not specified on Wallace's "
"instruments page. width_nm=5 is a representative value "
"for a typical commercial 3-7 nm narrowband filter."),
},
"OIII": {
"telescopes": ["Pier1"],
"center_nm": 500.7,
"width_nm": 6,
"notes": ("Narrowband [OIII]; exact FWHM not specified on Wallace's "
"instruments page. width_nm=6 is a representative value "
"for a typical commercial 3-9 nm narrowband filter."),
},
"SII": {
"telescopes": ["Pier1"],
"center_nm": 671.6,
"width_nm": 5,
"notes": ("Narrowband [SII]; exact FWHM not specified on Wallace's "
"instruments page. width_nm=5 is a representative value "
"for a typical commercial 3-8 nm narrowband filter."),
},
"L_rgb": {
"telescopes": ["Pier1"],
"center_nm": 550,
"width_nm": 300,
"notes": ("Luminance filter from the LRGB set, ~400-700 nm. "
"Generic, not a standardized photometric system."),
},
"R_rgb": {
"telescopes": ["Pier1"],
"center_nm": 650,
"width_nm": 100,
"notes": ("Red filter from the LRGB tri-color set, ~600-700 nm. "
"Generic, not a standardized photometric system; "
"distinct from the Cousins 'R' filter above."),
},
"G_rgb": {
"telescopes": ["Pier1"],
"center_nm": 550,
"width_nm": 100,
"notes": ("Green filter from the LRGB tri-color set, ~500-600 nm. "
"Generic, not a standardized photometric system."),
},
"B_rgb": {
"telescopes": ["Pier1"],
"center_nm": 450,
"width_nm": 100,
"notes": ("Blue filter from the LRGB tri-color set, ~400-500 nm. "
"Generic, not a standardized photometric system; "
"distinct from the Johnson 'B' filter above."),
},
}
if __name__ == "__main__":
# quick self-check / usage example
    for name, props in filters.items():
        print(f"{name:10s} center={props['center_nm']:>6} nm "
            f"width={props['width_nm']:>5} nm "
            f"telescopes={props['telescopes']}")