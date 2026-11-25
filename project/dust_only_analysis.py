"""Measure Q and fit the fixed-screen dust-only model to the saved radial profiles.

Run from the repository root with .venv/bin/python project/dust_only_analysis.py.
This reads the Milky Way corrected, calibrated photometry and does not rerun
image reduction. It exports per-annulus measurements, the full covariance, fit
statistics, and vector figures to project/dust_only_results by default.

The fits use annuli with the pipeline's coverage cut and a >3 sigma flux
detection in each calibrated band. Their probabilities treat the pipeline's
linearized covariance as fixed, including fully coherent sky and calibration
bounds. The separate flux-envelope check allows those bounds to vary across
annuli and determines whether the allowed Q intervals have a common value.
"""

import argparse
import csv
from dataclasses import dataclass
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.patches import Ellipse
import numpy as np
from astropy.table import Table, vstack
from scipy.linalg import cholesky, solve_triangular
from scipy.optimize import brentq, minimize_scalar
from scipy.stats import chi2, norm

import photometric_analysis as pa
import radial_analysis as ra


PROJECT = Path(__file__).resolve().parent
GALAXIES = {"ngc6503": ("NGC 6503", "gri"), "m33": ("M33", "ugr")}
# Schlafly & Finkbeiner (2011), Table 6, Fitzpatrick (1999), R_V = 3.1.
# The common E(B-V)_SFD normalization cancels in the color-excess ratio.
EXTINCTION = {"u": 4.239, "g": 3.303, "r": 2.285, "i": 1.698}
M33_U_FLAT_REPEATABILITY = 0.00506


@dataclass
class Fit:
    parameters: np.ndarray
    covariance: np.ndarray
    chi_square: float
    dof: int
    p: float


@dataclass
class QProfile:
    galaxy: str
    bands: str
    alpha: float
    annuli: Table
    colors: np.ndarray  # Blue color for all selected annuli, then red color.
    color_covariance: np.ndarray
    q: np.ndarray
    q_covariance: np.ndarray
    pivot_kpc: float
    constant: Fit
    linear: Fit


@dataclass
class BoundCheck:
    systematic: tuple[np.ndarray, np.ndarray]
    expanded: tuple[np.ndarray, np.ndarray]
    summary: dict


def read_photometry(profile_dir, bands):
    table = Table.read(profile_dir / "surface_photometry.ecsv")
    n, remainder = divmod(len(table), len(bands))
    if remainder or not n or table.meta.get("bands") != list(bands):
        raise ValueError(f"Unexpected photometry band order in {profile_dir}.")
    expected_order = f"band-major: {','.join(bands)}"
    if table.meta.get("covariance_order") != expected_order:
        raise ValueError(f"Unexpected covariance order in {profile_dir}.")
    if not table.meta.get("extinction_applied"):
        raise ValueError(f"Milky Way corrected photometry is required in {profile_dir}.")
    radii = np.asarray(table["radius_kpc"][:n])
    if not np.isfinite(radii).all() or np.any(np.diff(radii) <= 0):
        raise ValueError(f"Annulus radii are invalid in {profile_dir}.")
    for j, band in enumerate(bands):
        rows = table[j * n:(j + 1) * n]
        if not (np.all(rows["band"] == band) and
                np.array_equal(rows["annulus"], table["annulus"][:n]) and
                np.array_equal(rows["radius_kpc"], radii)):
            raise ValueError(f"Photometry rows do not match covariance order in {profile_dir}.")
    return table


def m33_covariance_with_flat_repeatability(radial_dir, table, saved_stat, extinction_covariance,
                                          solutions):
    """Repropagate the adopted per-pixel u flat error using the saved annuli."""
    analysis_dir = radial_dir.parent.parent
    reduction = analysis_dir.parent
    stacks = reduction / "stacks"
    paths = {band: stacks / f"m33_ugr_{band}_stack.fit" for band in "ugr"}
    paths["g"] = stacks / "m33_ugr_g_background_corrected.fit"
    errors = {band: stacks / f"m33_ugr_{band}_uncertainty.fit" for band in "ugr"}
    frames = pa.load_frames({"m33": paths}, {"m33": errors},
                            {"coverage_fraction": 0.8, "grid_tolerance_arcsec": 0.05})["m33"]
    for row in Table.read(analysis_dir / "blank_aperture_noise.ecsv"):
        if row["galaxy"] == "m33" and np.isfinite(row["variance_inflation"]):
            frames[row["band"]]["correlation_factor"] = float(row["variance_inflation"])

    red = np.zeros(frames["r"]["data"].shape, dtype=bool)
    audit = Table.read(radial_dir / "observed/source_mask_audit.ecsv")
    for source in audit[audit["radial_masked"]]:
        region, weight = pa.aperture_weights(red.shape, source["x"], source["y"],
                                              source["mask_radius_arcsec"] / frames["r"]["scale"])
        if region is not None:
            red[region] |= weight > 0

    geometry = json.loads((radial_dir / "geometry.json").read_text())
    geometry["center_xy"] = table.meta["annulus_center_xy"]
    geometry["minimum_annulus_fraction"] = table.meta["minimum_annulus_fraction"]
    u_rows = table[table["band"] == "u"]
    edges = np.r_[u_rows["inner_pixels"], u_rows["outer_pixels"][-1]]
    with np.load(radial_dir / "sky_model.npz") as saved:
        sky = dict(saved)

    radial_config = {"minimum_annulus_fraction": 0.65, "annulus_subpixels": 5,
                     "maximum_calibration_condition": 100.}
    native_masters = dict(galaxy="m33", manifest=reduction / "registration_ugr.json",
                          masters_dir=PROJECT / "calibration",
                          flat_fractional_errors={"u": M33_U_FLAT_REPEATABILITY})
    profile = ra.measure_annuli(frames, sky, red, np.zeros_like(red), geometry, radial_config,
                                edges, errors, native_masters=native_masters)

    # Reuse the same measured fluxes, footprints, sky and Sloan solution as the notebook.
    np.testing.assert_allclose(profile["area"] / profile["full_area"], u_rows["valid_fraction"])
    np.testing.assert_allclose(profile["mean"].ravel() / profile["scale"]**2,
                               table["intensity_ADU_s_arcsec2"], rtol=2e-5, atol=1e-9)
    np.testing.assert_allclose(profile["stat_covariance"], saved_stat, rtol=2e-5, atol=1e-10)
    photometry = ra.magnitude_profiles(profile, solutions, radial_config)
    np.testing.assert_allclose(photometry["magnitudes"].ravel() - np.asarray(table["A_MW"]),
                               table["mu"], rtol=2e-5, atol=1e-7)
    flux_error = np.sqrt(np.diag(profile["covariance"])).reshape(profile["mean"].shape)
    detected = profile["valid"] & (profile["mean"] > 3 * flux_error)
    measured = profile["stat_covariance"] + profile["master_pixel_covariance"]
    return photometry["covariance"] + extinction_covariance, detected, (
        np.sqrt(np.diag(measured)) / profile["scale"]**2)


def fit_gls(design, q, factor):
    """Fit by whitening with the Cholesky factor of the supplied Q covariance."""
    white_design = solve_triangular(factor, design, lower=True)
    white_q = solve_triangular(factor, q, lower=True)
    orthogonal, triangular = np.linalg.qr(white_design, mode="reduced")
    parameters = solve_triangular(triangular, orthogonal.T @ white_q)
    inverse = solve_triangular(triangular, np.eye(design.shape[1]))
    residuals = white_q - white_design @ parameters
    statistic = float(residuals @ residuals)
    dof = len(q) - len(parameters)
    return Fit(parameters, inverse @ inverse.T, statistic, dof, float(chi2.sf(statistic, dof)))


def make_q_profile(galaxy, table, magnitude_covariance, detected):
    name, bands = GALAXIES[galaxy]
    n = len(table) // 3
    mu = np.asarray(table["mu"]).reshape(3, n)
    coverage = np.asarray(table["coverage_pass"]).reshape(3, n)
    selected = np.flatnonzero(np.all(coverage & detected & np.isfinite(mu), axis=0))
    if len(selected) < 3:
        raise ValueError(f"{name} has fewer than three jointly reliable annuli.")
    rows = np.concatenate([selected + j * n for j in range(3)])
    mag_covariance = magnitude_covariance[np.ix_(rows, rows)]
    if not np.isfinite(mag_covariance).all():
        raise ValueError(f"{name} has missing covariance for a selected annulus.")
    np.testing.assert_allclose(mag_covariance, mag_covariance.T, rtol=1e-10, atol=1e-12)

    coefficients = np.array([EXTINCTION[band] for band in bands])
    alpha = (coefficients[0] - coefficients[1]) / (coefficients[1] - coefficients[2])
    identity = np.eye(len(selected))
    to_colors = np.kron([[1, -1, 0], [0, 1, -1]], identity)
    to_q = np.hstack([identity, -alpha * identity])
    # The adopted dust vector must vanish in Q even for a different amplitude in each bin.
    np.testing.assert_allclose(to_q @ to_colors @ np.kron(coefficients[:, None], identity),
                               0., atol=1e-12)
    colors = (to_colors @ mu[:, selected].ravel()).reshape(2, -1)
    color_covariance = to_colors @ mag_covariance @ to_colors.T
    q = to_q @ colors.ravel()
    q_covariance = to_q @ color_covariance @ to_q.T
    q_covariance = (q_covariance + q_covariance.T) / 2
    factor = cholesky(q_covariance, lower=True)

    radii = np.asarray(table["radius_kpc"][selected])
    pivot = float(radii.mean())
    constant = fit_gls(np.ones((len(q), 1)), q, factor)
    linear = fit_gls(np.column_stack([np.ones(len(q)), radii - pivot]), q, factor)
    return QProfile(galaxy, bands, float(alpha), table[selected], colors, color_covariance,
                    q, q_covariance, pivot, constant, linear)


def assess_flux_bounds(profile, table, measured_sigma, solutions):
    """Transform flux bounds exactly, with a shared stellar color term.

    The bounds are sensitivity intervals. Overlap asks whether a constant Q
    is possible if the spatial pattern of these errors is unspecified.
    """
    a, b, c = profile.bands
    n = len(table) // 3
    use = np.isin(table["annulus"][:n], profile.annuli["annulus"])
    flux = np.asarray(table["intensity_ADU_s_arcsec2"]).reshape(3, n)[:, use]
    sigma = measured_sigma.reshape(3, n)[:, use]
    bounds = [np.asarray(table["sky_model_bound"]).reshape(3, n)[:, use]]
    if profile.galaxy == "ngc6503":
        bounds.append(np.asarray(table["shared_master_bound"]).reshape(3, n)[:, use])
    envelope = np.sum(bounds, axis=0)
    zero_a, zero_b, zero_c = (solutions[band]["zp"] for band in profile.bands)
    ext_a, ext_b, ext_c = np.asarray(table["A_MW"]).reshape(3, n)[:, 0]
    term = solutions[a]["color_term"]
    term_sigma = solutions[a]["color_term_error"]
    if solutions[b]["color_term"] or solutions[c]["color_term"] or (
            (term or term_sigma) and tuple(solutions[a]["color_pair"]) != (a, b)):
        raise ValueError("The flux-bound test requires a color term in the blue band only.")
    alpha = profile.alpha

    def q_weights(inverse_response):
        return -(2.5 / np.log(10)) * np.array(
            [inverse_response, -inverse_response - alpha, alpha])[:, None]

    def interval(bound_multiplier, measured_multiple, inverse_response):
        error = bound_multiplier * envelope + measured_multiple * sigma
        lo_flux, hi_flux = np.maximum(flux - error, 0), flux + error
        weights = q_weights(inverse_response)
        # 1 - C is positive throughout the tested calibration range.
        # The signs of the Q weights determine which flux corners set the bounds.
        pivot = solutions[a]["pivot"]
        offset = inverse_response * (zero_a - zero_b - pivot) + pivot
        offset -= alpha * (zero_b - zero_c) + (ext_a - ext_b) - alpha * (ext_b - ext_c)
        with np.errstate(divide="ignore"):
            low = offset + np.sum(weights * np.log(np.where(weights > 0, lo_flux, hi_flux)), axis=0)
            high = offset + np.sum(weights * np.log(np.where(weights > 0, hi_flux, lo_flux)), axis=0)
        return low, high

    def gap(limits):
        return float(limits[0].max() - limits[1].min())

    response = 1 / (1 - term)
    np.testing.assert_allclose(interval(0., 0., response)[0], profile.q, atol=1e-7)
    systematic = interval(1., 0., response)
    expanded = interval(1., 2., response)
    def gap_at_multiplier(multiplier):
        return gap(interval(multiplier, 0., response))

    positive_bounds = envelope > 0
    ceiling = max(1., np.max(flux[positive_bounds] / envelope[positive_bounds], initial=0.) + 1.)
    multiplier = 0.
    if gap_at_multiplier(0.) > 0.:
        multiplier = (brentq(gap_at_multiplier, 0., ceiling)
                      if gap_at_multiplier(ceiling) <= 0. else np.inf)

    best_gap = gap(expanded)
    if term_sigma:
        term_limits = term + np.array([-2., 2.]) * term_sigma
        if term_limits[1] >= 1.:
            raise ValueError("The color-term sensitivity range includes a singular calibration.")
        low, high = 1 / (1 - term_limits)

        def gap_at_response(x):
            return gap(interval(1., 2., x))

        # The gap is a maximum of affine functions of inverse response.
        minimum = minimize_scalar(gap_at_response, bounds=(low, high), method="bounded")
        best_gap = min(gap_at_response(low), gap_at_response(high), minimum.fun)

    # Sensitivity to radial coherence alone, with the marginal error bars fixed.
    bound_modes = (q_weights(response) * np.asarray(bounds) / flux).reshape(-1, flux.shape[1])
    coherent_covariance = bound_modes.T @ bound_modes
    diagonal_covariance = np.diag(coherent_covariance.diagonal())
    perturbed = profile.q_covariance + 0.01 * (diagonal_covariance - coherent_covariance)
    factor = cholesky(perturbed, lower=True)
    fit = fit_gls(np.ones((len(profile.q), 1)), profile.q, factor)
    summary = dict(
        galaxy=GALAXIES[profile.galaxy][0], annuli=len(profile.q),
        fully_coherent_constant_p=profile.constant.p,
        constant_p_with_1pct_of_bound_variance_annulus_specific=fit.p,
        systematic_only_common_q_low_mag=systematic[0].max(),
        systematic_only_common_q_high_mag=systematic[1].min(),
        systematic_only_gap_mag=gap(systematic),
        lowest_systematic_multiplier_for_overlap=multiplier,
        gap_with_2sigma_measured_per_band_mag=gap(expanded),
        smallest_gap_with_2sigma_measured_and_color_term_mag=best_gap,
        strongest_lower_limit_radius_kpc=profile.annuli["radius_kpc"][expanded[0].argmax()],
        strongest_upper_limit_radius_kpc=profile.annuli["radius_kpc"][expanded[1].argmin()],
    )
    return BoundCheck(systematic, expanded, summary)


def fit_summary(profile):
    constant, linear = profile.constant, profile.linear
    slope = linear.parameters[1]
    slope_error = np.sqrt(linear.covariance[1, 1])
    delta = max(0., constant.chi_square - linear.chi_square)
    return dict(
        galaxy=GALAXIES[profile.galaxy][0], bands=profile.bands, extinction_rv=3.1,
        alpha=profile.alpha, u_flat_repeatability_fraction=(
            M33_U_FLAT_REPEATABILITY if profile.galaxy == "m33" else ""),
        annuli=len(profile.q), radius_min_kpc=profile.annuli["radius_kpc"][0],
        radius_max_kpc=profile.annuli["radius_kpc"][-1], median_q_sigma_mag=(
            np.median(np.sqrt(np.diag(profile.q_covariance)))),
        constant_q_mag=constant.parameters[0], constant_q_sigma_mag=np.sqrt(constant.covariance[0, 0]),
        constant_chi_square=constant.chi_square, constant_dof=constant.dof, constant_p=constant.p,
        linear_pivot_kpc=profile.pivot_kpc, linear_q_at_pivot_mag=linear.parameters[0],
        linear_q_at_pivot_sigma_mag=np.sqrt(linear.covariance[0, 0]),
        slope_mag_per_kpc=slope, slope_sigma_mag_per_kpc=slope_error,
        slope_ci95_low_mag_per_kpc=slope - norm.ppf(0.975) * slope_error,
        slope_ci95_high_mag_per_kpc=slope + norm.ppf(0.975) * slope_error,
        linear_chi_square=linear.chi_square, linear_dof=linear.dof, linear_p=linear.p,
        delta_chi_square=delta, delta_p=chi2.sf(delta, 1),
    )


def save_data(profiles, checks, output):
    summaries = [fit_summary(profile) for profile in profiles]
    with (output / "q_fits.csv").open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(summaries[0]))
        writer.writeheader()
        writer.writerows(summaries)
    with (output / "q_covariance_audit.csv").open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(checks[0].summary))
        writer.writeheader()
        writer.writerows(check.summary for check in checks)

    combined = []
    covariance_data = {}
    for profile, check in zip(profiles, checks):
        table = profile.annuli["annulus", "radius_kpc", "radius_arcsec", "valid_fraction"].copy()
        table.meta.clear()
        table["galaxy"] = GALAXIES[profile.galaxy][0]
        a, b, c = profile.bands
        table["blue_color"] = f"{a}-{b}"
        table["red_color"] = f"{b}-{c}"
        table["blue_mag"], table["red_mag"] = profile.colors
        n = len(table)
        i = np.arange(n)
        color_cov = profile.color_covariance
        table["blue_sigma_mag"], table["red_sigma_mag"] = np.sqrt(np.diag(color_cov)).reshape(2, n)
        table["blue_red_cov_mag2"] = color_cov[i, n + i]
        table["q_mag"] = profile.q
        table["q_sigma_mag"] = np.sqrt(np.diag(profile.q_covariance))
        table["q_systematic_low_mag"], table["q_systematic_high_mag"] = check.systematic
        table["q_expanded_low_mag"], table["q_expanded_high_mag"] = check.expanded
        table["constant_q_mag"] = profile.constant.parameters[0]
        intercept, slope = profile.linear.parameters
        table["linear_q_mag"] = intercept + slope * (table["radius_kpc"] - profile.pivot_kpc)
        combined.append(table)

        # Matrices follow the same row order as this galaxy's CSV rows.
        prefix = profile.galaxy
        covariance_data[f"{prefix}_annulus"] = np.asarray(table["annulus"])
        covariance_data[f"{prefix}_q_cov_mag2"] = profile.q_covariance
        covariance_data[f"{prefix}_color_cov_mag2"] = color_cov
        covariance_data[f"{prefix}_linear_parameter_cov"] = profile.linear.covariance

        # Check the same-annulus propagation independently before exporting.
        q_variance = (table["blue_sigma_mag"]**2 + profile.alpha**2 * table["red_sigma_mag"]**2
                      - 2 * profile.alpha * table["blue_red_cov_mag2"])
        np.testing.assert_allclose(table["q_sigma_mag"]**2, q_variance)

    vstack(combined).write(output / "q_annuli.csv", format="ascii.csv", overwrite=True)
    np.savez_compressed(output / "q_covariance.npz", **covariance_data)
    return summaries


def plot_q_profiles(profiles, directory):
    fig, axes = plt.subplots(1, 2, figsize=(7.05, 3.1), layout="constrained")
    for ax, profile in zip(axes, profiles):
        radius = profile.annuli["radius_kpc"]
        x = np.array([radius[0], radius[-1]])
        intercept, slope = profile.linear.parameters
        line, = ax.plot(x, intercept + slope * (x - profile.pivot_kpc), color="#b34029",
                        lw=1.15, label="Linear GLS, coherent bounds")
        constant, = ax.plot(x, np.full(2, profile.constant.parameters[0]), color="#4f555b",
                            lw=1.05, ls="--", label="Constant GLS, coherent bounds")
        points = ax.errorbar(radius, profile.q, yerr=np.sqrt(np.diag(profile.q_covariance)),
                             fmt="o", ms=3, capsize=1.8, lw=0.7, color="#216589",
                             label=r"Annuli ($1\sigma$)")
        ax.set(xlabel="Elliptical semimajor radius (kpc)",
               ylabel=rf"$Q_{{\mathrm{{{profile.bands}}}}}$ (mag)")
        ax.set_title(GALAXIES[profile.galaxy][0], loc="left")
    fig.legend(handles=[points, constant, line], loc="outside lower center", ncols=3)
    fig.savefig(directory / "q_radial_profiles.pdf")
    plt.close(fig)


def plot_color_loci(profiles, directory):
    """Show the dust locus and 68% marginal color ellipses for four selected bins."""
    fig, axes = plt.subplots(1, 2, figsize=(7.05, 3.1), layout="constrained")
    limits = Normalize(0., max(profile.annuli["radius_kpc"][-1] for profile in profiles))
    cmap = plt.get_cmap("viridis")
    for ax, profile in zip(axes, profiles):
        blue, red = profile.colors
        radius = np.asarray(profile.annuli["radius_kpc"])
        x = np.array([red.min(), red.max()])
        ax.plot(x, profile.constant.parameters[0] + profile.alpha * x, color="#4f555b",
                ls="--", lw=1., label="Constant-Q dust locus")
        ax.scatter(red, blue, c=radius, cmap=cmap, norm=limits, s=15, edgecolor="white",
                   linewidth=0.25, zorder=3)
        for i in np.linspace(0, len(red) - 1, 4, dtype=int):
            xy = [len(red) + i, i]  # Plot red on x, blue on y.
            covariance = profile.color_covariance[np.ix_(xy, xy)]
            eigenvalues, eigenvectors = np.linalg.eigh(covariance)
            width, height = 2 * np.sqrt(eigenvalues[::-1] * chi2.ppf(0.68, 2))
            angle = np.rad2deg(np.arctan2(eigenvectors[1, 1], eigenvectors[0, 1]))
            ax.add_patch(Ellipse((red[i], blue[i]), width, height, angle=angle, fill=False,
                                 edgecolor=cmap(limits(radius[i])), lw=0.8, alpha=0.55))
        ax.autoscale_view()
        a, b, c = profile.bands
        ax.set(xlabel=rf"${b}^\prime-{c}^\prime$ (mag)",
               ylabel=rf"${a}^\prime-{b}^\prime$ (mag)")
        ax.set_title(GALAXIES[profile.galaxy][0], loc="left")
    fig.legend(*axes[0].get_legend_handles_labels(), loc="outside lower center")
    fig.colorbar(plt.cm.ScalarMappable(norm=limits, cmap=cmap), ax=axes, pad=0.025,
                 label="Radius (kpc)")
    fig.savefig(directory / "q_color_color.pdf")
    plt.close(fig)


def plot_flux_envelopes(profiles, checks, directory):
    fig, axes = plt.subplots(1, 2, figsize=(7.05, 3.25), layout="constrained")
    for ax, profile, check in zip(axes, profiles, checks):
        radius = profile.annuli["radius_kpc"]
        measured = ax.fill_between(radius, *check.expanded, color="#e0ecf3",
                                   label=r"Bounds plus $2\sigma$ per band")
        systematic = ax.fill_between(radius, *check.systematic, color="#8db9d2",
                                     label="Systematic bounds")
        points, = ax.plot(radius, profile.q, marker="o", ms=2.8, lw=0.85, color="#205978",
                          label="Measured Q")
        low, high = check.expanded[0].max(), check.expanded[1].min()
        ax.axhspan(min(low, high), max(low, high),
                   color="#28885d" if low <= high else "#a63f29", alpha=0.18, zorder=4)
        for limit in (low, high):
            ax.axhline(limit, color="#28885d" if low <= high else "#a63f29", ls="--", lw=0.7)
        label = "Common Q interval" if low <= high else f"Gap of {low - high:.2f} mag"
        ax.set_title(GALAXIES[profile.galaxy][0], loc="left")
        ax.set_title(label, loc="right", fontsize=8)
        ax.set(xlabel="Elliptical semimajor radius (kpc)",
               ylabel=rf"$Q_{{\mathrm{{{profile.bands}}}}}$ (mag)")
    fig.legend(handles=[points, systematic, measured], loc="outside lower center", ncols=3)
    fig.savefig(directory / "q_systematic_envelopes.pdf")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=PROJECT / "dust_only_results")
    args = parser.parse_args()
    radial = PROJECT / "reduction/analysis_ugr/radial"
    profiles = []
    checks = []
    for galaxy, (_, bands) in GALAXIES.items():
        radial_dir = radial / galaxy
        profile_dir = radial_dir / "MW_corrected"
        table = read_photometry(profile_dir, bands)
        solutions = json.loads(
            (radial_dir / "calibration/sloan_zero_points.json").read_text())[galaxy]
        with np.load(profile_dir / "covariance.npz") as saved:
            if galaxy == "m33":
                print(f"Propagating the adopted M33 u′ flat repeatability ({M33_U_FLAT_REPEATABILITY:.3%}).",
                      flush=True)
                covariance, detected, measured_sigma = m33_covariance_with_flat_repeatability(
                    radial_dir, table, saved["flux_stat"], saved["extinction"], solutions)
            else:
                covariance = saved["magnitudes"]
                detected = np.asarray(table["above_3sigma"]).reshape(3, -1)
                measured_sigma = np.asarray(table["random_and_sky_error"])
                np.testing.assert_allclose(np.diag(covariance), np.asarray(table["mu_error"])**2)
        profile = make_q_profile(galaxy, table, covariance, detected)
        profiles.append(profile)
        checks.append(assess_flux_bounds(profile, table, measured_sigma, solutions))

    pdf_dir = args.output / "pdf"
    pdf_dir.mkdir(parents=True, exist_ok=True)
    summaries = save_data(profiles, checks, args.output)
    plt.rcParams.update({"font.family": "STIXGeneral", "mathtext.fontset": "stix",
                         "font.size": 9, "axes.titlesize": 10.5, "legend.fontsize": 8,
                         "legend.frameon": False, "axes.linewidth": 0.65, "pdf.fonttype": 42})
    plot_q_profiles(profiles, pdf_dir)
    plot_color_loci(profiles, pdf_dir)
    plot_flux_envelopes(profiles, checks, pdf_dir)

    for result in summaries:
        print(f"{result['galaxy']}  N={result['annuli']}, alpha={result['alpha']:.6f}, "
              f"constant chi-square/dof={result['constant_chi_square']:.2f}/{result['constant_dof']} "
              f"(p={result['constant_p']:.3g}), "
              f"linear chi-square/dof={result['linear_chi_square']:.2f}/{result['linear_dof']} "
              f"(p={result['linear_p']:.3g}), "
              f"slope={result['slope_mag_per_kpc']:+.5f} ± {result['slope_sigma_mag_per_kpc']:.5f} mag/kpc.")
    print(f"Saved figures and data in {args.output}.")


if __name__ == "__main__":
    main()
