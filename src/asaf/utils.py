"""Utility functions for the ASAF package."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from typing import Union

    from numpy.typing import ArrayLike

import numpy as np
import pandas as pd

from asaf.constants import _BOLTZMANN_CONSTANT


def quote(string: str) -> str:
    """Quotes a string with single quotes."""
    return f"'{string}'"


def interpolate_df(df: pd.DataFrame, based_on: str = "macrostate") -> pd.DataFrame:
    """Interpolates data in a dataframe.

    Parameters
    ----------
    df
        A pandas DataFrame containing data to be interpolated.
    based_on
        Column name in df containing values of the independent variable.
        Values must be real, finite, and in strictly increasing order.

    Returns
    -------
    df_interp
        A pandas DataFrame containing interpolated data.

    """
    columns = list(df)
    columns.remove(based_on)
    n_max = df[based_on].max()
    n_arranged = np.arange(0, n_max + 1, dtype=int)
    df_interp = pd.DataFrame({based_on: n_arranged})
    for col in columns:
        val_interp = np.interp(n_arranged, df[based_on], df[col])
        df_interp[col] = val_interp

    return df_interp


def calculate_lnp(prob_df: pd.DataFrame) -> pd.DataFrame:
    """Calculate the natural logarithm of the macrostate transition probability[1].

    Parameters
    ----------
    prob_df
        A pandas DataFrame containing transition probabilities.

    Returns
    -------
    lnp_df
        A pandas DataFrame containing natural logarithm of the macrostate transition probability.

    References
    ----------
    .. [1] Shen, V. K., & Errington, J. R. (2004). Metastability and Instability in the Lennard-Jones Fluid
       Investigated by Transition-Matrix Monte Carlo, In The Journal of Physical Chemistry B
       (Vol. 108, Issue 51, pp. 19595–19606). American Chemical Society (ACS). https://doi.org/10.1021/jp040218y

    """
    diff = prob_df["macrostate"].diff()
    if not (diff.iloc[1:] == 1).all():
        print("Interpolating data...")
        prob_df = interpolate_df(prob_df)

    dlnp = np.log(prob_df["P_up"].shift(1) / prob_df["P_down"])
    dlnp[0] = 0
    lnp_df = pd.DataFrame(
        data={
            "macrostate": prob_df["macrostate"],
            "lnp": normalize(dlnp.cumsum()),
        }
    )
    return lnp_df


def normalize(lnp: Union[pd.Series, np.ndarray]) -> Union[pd.Series, np.ndarray]:
    """
    Normalize the natural logarithm of the macrostate probability.

    Parameters
    ----------
    lnp
        Natural logarithm of the macrostate probability.

    Returns
    -------
    Normalized natural logarithm of the macrostate probability.
    """
    lnp_cp = lnp.copy()
    maxx = np.max(lnp_cp)
    lnp_cp -= np.log(sum(np.exp(lnp - maxx))) + maxx

    return lnp_cp


def beta_to_temperature(beta: ArrayLike) -> ArrayLike:
    """Convert beta (in J^-1) to temperature (in K) using Boltzmann constant."""
    temp = 1 / _BOLTZMANN_CONSTANT / beta
    return temp


def temperature_to_beta(temp: ArrayLike) -> ArrayLike:
    """Convert temperature (in K) to beta (in J^-1) using Boltzmann constant."""
    beta = 1 / _BOLTZMANN_CONSTANT / temp
    return beta


def fugacity_to_mu(fugacity: ArrayLike, beta: float) -> ArrayLike:
    """Convert fugacity (in Pa) to chemical potential (in J).

    Note: the volume scale for fugacity is Å^3.
    Ref: Equation 73 in Dubbeldam et al., Molecular Simulation, 2013, Vol. 39, Nos. 14–15, 1253–1292,
    https://dx.doi.org/10.1080/08927022.2013.819102
    """
    mu = np.log(fugacity * 1e-30 * beta) / beta  # J A^-3
    return mu


def mu_to_fugacity(mu: ArrayLike, beta: float) -> ArrayLike:
    """Convert chemical potential (in J) to fugacity (in Pa).

    Note: the volume scale for fugacity is Å^3.
    Ref: Equation 73 in Dubbeldam et al., Molecular Simulation, 2013, Vol. 39, Nos. 14–15, 1253–1292,
    https://dx.doi.org/10.1080/08927022.2013.819102
    """
    fug = np.exp(beta * mu) / beta / 1e-30  # Pa
    return fug
