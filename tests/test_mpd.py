from __future__ import annotations

from pathlib import Path

import pytest

np = pytest.importorskip("numpy")
pd = pytest.importorskip("pandas")
pytest.importorskip("plotly.graph_objects")
pytest.importorskip("scipy.signal")

from asaf.mpd import MPD, normalize


def test_mpd_requires_probability_or_lnp_columns() -> None:
    df = pd.DataFrame({"macrostate": [0, 1, 2], "value": [1.0, 2.0, 3.0]})
    with pytest.raises(ValueError):
        MPD(dataframe=df, temperature=300.0, fugacity=1000.0)

def test_mpd_requires_beta_mu_or_fugacity() -> None:
    df = pd.DataFrame({
        "macrostate": [0, 1, 2],
        "P_up": [0.1, 0.2, 0.3],
        "P_down": [0.2, 0.3, 0.4]
    })
    with pytest.raises(ValueError):
        MPD(dataframe=df, temperature=300.0)

def test_mpd_interpolates_missing_ln_probability() -> None:
    df = pd.DataFrame({
        "macrostate": [0, 2],
        "P_up": [0.1, 0.3],
        "P_down": [0.2, 0.4]
    })
    mpd = MPD(dataframe=df, temperature=300.0, fugacity=1000.0)
    assert list(mpd.dataframe()["macrostate"]) == [0, 1, 2]
    probabilities = np.exp(mpd.dataframe()["lnp"].to_numpy())
    assert probabilities.sum() == pytest.approx(1.0)

# def test_reweight_to_fugacity_updates_internal_state() -> None:
#     macrostate = np.array([0, 1, 2])
#     probabilities = np.array([0.2, 0.5, 0.3])
#     lnp = np.log(probabilities)
#     df = pd.DataFrame({"macrostate": macrostate, "lnp": lnp})
#
#     mpd = MPD(dataframe=df, temperature=300.0, fugacity=1000.0)
#     base_lnp = mpd.lnp().copy()
#     beta = mpd.beta
#     mu_initial = mpd.mu
#
#     new_fugacity = 2000.0
#     mu_new = MPD.fugacity_to_mu(new_fugacity, beta)
#     delta = beta * (mu_new - mu_initial)
#
#     expected = base_lnp.copy()
#     expected["lnp"] = normalize(expected["lnp"] + delta * expected["macrostate"])
#
#     mpd.reweight_to_fug(new_fugacity)
#     updated_lnp = mpd.lnp()
#
#     assert np.allclose(updated_lnp["lnp"], expected["lnp"])
#     assert mpd.fugacity == pytest.approx(new_fugacity)
#     assert mpd.mu == pytest.approx(mu_new)

def test_normalize_creates_unit_probability_distribution() -> None:
    values = np.array([-1.0, -2.0, -3.0])
    normalized = normalize(values)
    probabilities = np.exp(normalized)
    assert probabilities.sum() == pytest.approx(1.0)

def test_mpd_equilibrium_and_observables() -> None:
    """This test uses data from NIST SRWS
    (https://www.nist.gov/programs-projects/nist-standard-reference-simulation-website)
    (https://mmlapps.nist.gov/srs/LJ_PURE/eostmmc.htm)
    """
    data_path = Path(__file__).parent / "data" / "mpd_lj_120.csv"
    mpd = MPD(dataframe=pd.read_csv(data_path), temperature=144.3, beta_mu=-2.902929)
    equilibrium_fugacity, p_low, p_high = mpd.find_phase_equilibrium(return_probabilities=True)
    assert p_low == pytest.approx(0.5, rel=1e-4)
    assert p_high == pytest.approx(0.5, rel=1e-4)
    mpd.reweight_to_fug(equilibrium_fugacity, inplace=True)
    minima = mpd.minimums(order=50)
    assert len(minima) == 1
    assert minima["macrostate"].values == 168
    average_macrostate = mpd.average_macrostate_at_fugacity(equilibrium_fugacity)
    volume = 8.0**3
    density_low = average_macrostate[0] / volume
    density_high = average_macrostate[1] / volume
    assert density_low == pytest.approx(1.0030e-01, rel=1e-2)
    assert density_high == pytest.approx(5.6329e-01, rel=1e-2)
