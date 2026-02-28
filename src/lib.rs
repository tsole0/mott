use pyo3::prelude::*;
use pyo3::exceptions::PyValueError;

// ─────────────────────────────────────────────
// BOND LENGTH FEATURES
// Input: list of (x,y,z) site coords for a single TM atom + its neighbors
// ─────────────────────────────────────────────

#[pyfunction]
fn bond_length_stats(
    site: (f64, f64, f64),
    neighbors: Vec<(f64, f64, f64)>,
) -> PyResult<(f64, f64, f64, f64)> {
    if neighbors.is_empty() {
        return Err(PyValueError::new_err("No neighbors provided"));
    }

    let lengths: Vec<f64> = neighbors
        .iter()
        .map(|&(nx, ny, nz)| {
            let dx = site.0 - nx;
            let dy = site.1 - ny;
            let dz = site.2 - nz;
            (dx * dx + dy * dy + dz * dz).sqrt()
        })
        .collect();

    let n = lengths.len() as f64;
    let mean = lengths.iter().sum::<f64>() / n;
    let min = lengths.iter().cloned().fold(f64::INFINITY, f64::min);
    let max = lengths.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let std = (lengths.iter().map(|l| (l - mean).powi(2)).sum::<f64>() / n).sqrt();

    Ok((mean, min, max, std))
}

// ─────────────────────────────────────────────
// OCTAHEDRAL DISTORTION INDEX
// Δd = (1/6) * Σ|d_i - d_mean| / d_mean
// Classic Mott/Jahn-Teller signal
// Input: exactly 6 bond lengths (octahedral coordination)
// ─────────────────────────────────────────────

#[pyfunction]
fn octahedral_distortion(bond_lengths: Vec<f64>) -> PyResult<f64> {
    if bond_lengths.len() != 6 {
        return Err(PyValueError::new_err(
            "Octahedral distortion requires exactly 6 bond lengths",
        ));
    }

    let mean = bond_lengths.iter().sum::<f64>() / 6.0;
    let delta = bond_lengths
        .iter()
        .map(|&d| (d - mean).abs() / mean)
        .sum::<f64>()
        / 6.0;

    Ok(delta)
}

// ─────────────────────────────────────────────
// COORDINATION NUMBER
// Simple cutoff-based CN for a single site
// Input: site coord, all other coords, cutoff radius in Angstroms
// ─────────────────────────────────────────────

#[pyfunction]
fn coordination_number(
    site: (f64, f64, f64),
    all_sites: Vec<(f64, f64, f64)>,
    cutoff: f64,
) -> PyResult<usize> {
    let cn = all_sites
        .iter()
        .filter(|&&(nx, ny, nz)| {
            let dx = site.0 - nx;
            let dy = site.1 - ny;
            let dz = site.2 - nz;
            let dist = (dx * dx + dy * dy + dz * dz).sqrt();
            dist > 1e-8 && dist <= cutoff  // exclude self
        })
        .count();

    Ok(cn)
}

// ─────────────────────────────────────────────
// HOPPING INTEGRAL PROXY
// t ≈ 1 / d² for each TM-O bond (tight binding approximation)
// Returns mean hopping integral across all bonds
// ─────────────────────────────────────────────

#[pyfunction]
fn hopping_integral(bond_lengths: Vec<f64>) -> PyResult<f64> {
    if bond_lengths.is_empty() {
        return Err(PyValueError::new_err("No bond lengths provided"));
    }

    let t_mean = bond_lengths
        .iter()
        .map(|&d| 1.0 / (d * d))
        .sum::<f64>()
        / bond_lengths.len() as f64;

    Ok(t_mean)
}

// ─────────────────────────────────────────────
// BANDWIDTH ESTIMATE
// W = 2 * z * t_mean
// z = coordination number, t_mean = mean hopping integral
// ─────────────────────────────────────────────

#[pyfunction]
fn bandwidth(coordination_num: usize, bond_lengths: Vec<f64>) -> PyResult<f64> {
    if bond_lengths.is_empty() {
        return Err(PyValueError::new_err("No bond lengths provided"));
    }

    let t_mean = bond_lengths
        .iter()
        .map(|&d| 1.0 / (d * d))
        .sum::<f64>()
        / bond_lengths.len() as f64;

    Ok(2.0 * coordination_num as f64 * t_mean)
}

// ─────────────────────────────────────────────
// U/W RATIO
// hubbard_u: U value in eV (from MP or literature)
// w: bandwidth computed above
// Mott criterion: U/W > 1
// ─────────────────────────────────────────────

#[pyfunction]
fn uw_ratio(hubbard_u: f64, w: f64) -> PyResult<f64> {
    if w <= 0.0 {
        return Err(PyValueError::new_err("Bandwidth W must be positive"));
    }
    Ok(hubbard_u / w)
}

// ─────────────────────────────────────────────
// BATCH VERSION — run all features for a list of materials
// Takes flat inputs to avoid overhead of many small Python->Rust calls
// Returns list of dicts (as Vec of tuples for PyO3)
// ─────────────────────────────────────────────

#[pyfunction]
fn compute_all_features(
    site: (f64, f64, f64),
    neighbors: Vec<(f64, f64, f64)>,
    hubbard_u: f64,
    cutoff: f64,
) -> PyResult<(f64, f64, f64, f64, f64, f64, f64, f64)> {
    // bond length stats
    let (bl_mean, bl_min, bl_max, bl_std) = bond_length_stats(site, neighbors.clone())?;

    // coordination number
    let cn = coordination_number(site, neighbors.clone(), cutoff)?;

    // bandwidth
    let bond_lengths: Vec<f64> = neighbors
        .iter()
        .map(|&(nx, ny, nz)| {
            let dx = site.0 - nx;
            let dy = site.1 - ny;
            let dz = site.2 - nz;
            (dx * dx + dy * dy + dz * dz).sqrt()
        })
        .collect();

    let w = bandwidth(cn, bond_lengths.clone())?;
    let uw = uw_ratio(hubbard_u, w)?;

    // distortion — only if octahedral (cn == 6)
    let distortion = if bond_lengths.len() == 6 {
        octahedral_distortion(bond_lengths)?
    } else {
        -1.0  // sentinel: not octahedral
    };

    // returns: (bl_mean, bl_min, bl_max, bl_std, cn as f64, W, U/W, distortion)
    Ok((bl_mean, bl_min, bl_max, bl_std, cn as f64, w, uw, distortion))
}

// ─────────────────────────────────────────────
// MODULE REGISTRATION
// ─────────────────────────────────────────────

#[pymodule]
fn mott(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(bond_length_stats, m)?)?;
    m.add_function(wrap_pyfunction!(octahedral_distortion, m)?)?;
    m.add_function(wrap_pyfunction!(coordination_number, m)?)?;
    m.add_function(wrap_pyfunction!(hopping_integral, m)?)?;
    m.add_function(wrap_pyfunction!(bandwidth, m)?)?;
    m.add_function(wrap_pyfunction!(uw_ratio, m)?)?;
    m.add_function(wrap_pyfunction!(compute_all_features, m)?)?;
    Ok(())
}
