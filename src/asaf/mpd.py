"""Module for handling Macrostate Probability Distribution (MPD) data."""
from __future__ import annotations

import json
from math import factorial
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from typing import Any, Dict, List, Optional, Tuple, Union

    from numpy.typing import ArrayLike

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from scipy.signal import argrelextrema

from asaf.constants import _AVOGADRO_CONSTANT, _BOLTZMANN_CONSTANT
from asaf.isotherm import Isotherm


# TODO: move static functions to utility module
class MPD:
    """Class for storing and processing macrostate probability distribution."""

    def __init__(
        self,
        dataframe: pd.DataFrame,
        temperature: float,
        beta_mu: Optional[float] = None,
        fugacity: Optional[float] = None,
        metadata: Optional[dict[str, Any]] = None,
        order: int = 50,
        tolerance: float = 10.0,
    ) -> None:
        """Initialize the MPD class.

        Parameters
        ----------
        dataframe
            a pandas dataframe with state specific data
        temperature
            temperature (in K) at which the simulation was performed
        beta_mu
            beta_mu (unitless) at which the simulation was performed. At least one of beta_mu
            or fugacity must be specified
        fugacity
            fugacity (in Pa) at which the simulation was performed. At least one of beta_mu
            or fugacity must be specified
        metadata
            a dictionary with the simulation metadata
        order
            how many points on each side use to find minimum in lnp
        tolerance
            used when checking the probability at lnp tail
        """
        self.temperature = temperature

        if (beta_mu is None) and (fugacity is None):
            raise ValueError("Must provide `beta_mu` or/and `fugacity`.")
        if beta_mu is None:
            self.fugacity = fugacity
            self._beta_mu = self.beta * self.mu
        else:
            self._beta_mu = beta_mu
            self.mu = beta_mu / self.beta

        lnp_headers = ["macrostate", "lnp"]
        prob_headers = ["macrostate", "P_up", "P_down"]

        have_lnp = set(lnp_headers).issubset(dataframe.columns)
        have_prob = set(prob_headers).issubset(dataframe.columns)

        if not (have_lnp or have_prob):
            all_required = set(lnp_headers) | set(prob_headers)
            missing = all_required - set(dataframe.columns)
            raise ValueError(f"Some of the columns names {missing} are missing.")

        self._dataframe = dataframe

        if not have_lnp:
            lnp_df = self.calculate_lnp(dataframe[prob_headers])
            merged = lnp_df.merge(
                dataframe, on="macrostate", how="left", suffixes=("", "_inp")
            )
            self._dataframe = merged

        self._metadata = metadata or {}
        self.order = order
        self.tolerance = tolerance

        self._system_size_prod = 1

        if "system_size" in self.metadata:
            self.system_size = self.metadata["system_size"]
        else:
            self.system_size = [1, 1, 1]

        self.check_tail(order, tolerance)

    @classmethod
    def from_csv(cls, file_name: str, **kwargs: object) -> MPD:
        """Read natural logarithm of macrostates probability or transition probabilities from a csv file."""
        df = pd.read_csv(file_name, **kwargs)

        metadata_filename = file_name.removesuffix(".csv") + ".metadata.json"
        with open(metadata_filename) as f:
            metadata = json.load(f)

        temperature = metadata.get("temperature")

        if temperature is None:
            raise ValueError("Metadata must contain 'temperature'.")

        beta_mu = metadata.get("beta_mu")
        fugacity = metadata.get("fugacity")

        if (beta_mu is None) and (fugacity is None):
            raise ValueError("Metadata must contain 'beta_mu' or/and 'fugacity'.")

        return cls(
            dataframe=df,
            temperature=temperature,
            beta_mu=beta_mu,
            fugacity=fugacity,
            metadata=metadata,
        )

    @staticmethod
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

    @staticmethod
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
            prob_df = MPD.interpolate_df(prob_df)

        dlnp = np.log(prob_df["P_up"].shift(1) / prob_df["P_down"])
        dlnp[0] = 0
        lnp_df = pd.DataFrame(
            data={
                "macrostate": prob_df["macrostate"],
                "lnp": normalize(dlnp.cumsum()),
            }
        )
        return lnp_df

    def dataframe(self) -> pd.DataFrame:
        """Return dataframe."""
        return self._dataframe

    @staticmethod
    def beta_to_temperature(beta: ArrayLike) -> ArrayLike:
        """Convert beta to temperature using Boltzmann constant."""
        temp = 1 / _BOLTZMANN_CONSTANT / beta
        return temp

    @staticmethod
    def temperature_to_beta(temp: ArrayLike) -> ArrayLike:
        """Convert temperature to beta using Boltzmann constant."""
        beta = 1 / _BOLTZMANN_CONSTANT / temp
        return beta

    @staticmethod
    def fugacity_to_mu(fugacity: ArrayLike, beta: float) -> ArrayLike:
        """Convert fugacity (in Pa) to chemical potential (in J A^-3)."""
        mu = np.log(fugacity * 1e-30 * beta) / beta  # J A^-3
        return mu

    @staticmethod
    def mu_to_fugacity(mu: ArrayLike, beta: float) -> ArrayLike:
        """Convert chemical potential (in J A^-3) to fugacity (in Pa)."""
        fug = np.exp(beta * mu) / beta / 1e-30  # Pa
        return fug

    @property
    def temperature(self) -> float:
        """Return the temperature (in K)."""
        return self._temperature

    @temperature.setter
    def temperature(self, temperature: float) -> None:
        self._temperature = temperature
        self._beta = self.temperature_to_beta(temperature)

    @property
    def beta(self) -> float:
        """Return the beta (in J^-1)."""
        return self._beta

    @beta.setter
    def beta(self, beta: float) -> None:
        self._beta = beta
        self._temperature = self.beta_to_temperature(beta)

    @property
    def fugacity(self) -> float:
        """Return the fugacity (in Pa)."""
        return self._fugacity

    @fugacity.setter
    def fugacity(self, fugacity: float) -> None:
        self._fugacity = fugacity
        self._mu = self.fugacity_to_mu(fugacity, self.beta)

    @property
    def mu(self) -> float:
        """Return the chemical potential (in J A^-3)."""
        return self._mu

    @mu.setter
    def mu(self, mu: float) -> None:
        self._mu = mu
        self._fugacity = self.mu_to_fugacity(mu, self.beta)

    @property
    def beta_mu(self) -> float:
        """Return the beta_mu (unitless)."""
        return self._mu * self._beta

    @property
    def system_size(self) -> List[int]:
        """Return the system size as a list of integers."""
        return self._system_size

    @system_size.setter
    def system_size(self, system_size: List[int]) -> None:
        self._system_size = system_size
        self._system_size_prod = np.prod(system_size)

    @property
    def metadata(self) -> Dict[str, Any]:
        """Return the metadata dictionary."""
        return self._metadata

    @metadata.setter
    def metadata(self, metadata: Dict[str, Any]):
        if metadata:
            for k, v in metadata.items():
                self._metadata[k] = v

    @property
    def order(self) -> int:
        """Return the order used to find minimum in lnp."""
        return self._order

    @order.setter
    def order(self, order: int) -> None:
        self._order = order

    @property
    def tolerance(self) -> float:
        """Return the tolerance used when checking the probability at lnp tail."""
        return self._tolerance

    @tolerance.setter
    def tolerance(self, tolerance: float) -> None:
        self._tolerance = tolerance

    @property
    def lnp(self) -> pd.DataFrame:
        """Return a dataframe with the natural logarithm of the macrostate probability."""
        return self._dataframe[["macrostate", "lnp"]]

    def check_tail(
        self, order: int, tolerance: float, lnp: Optional[pd.DataFrame] = None
    ) -> None:
        """Check the probability at the tail of the lnp distribution."""
        if lnp is None:
            lnp = self.lnp
        mins = self.minimums(lnp=lnp["lnp"], order=order)
        if len(mins) == 0:
            difference = lnp["lnp"].max() - lnp["lnp"].iloc[-1]
        else:
            minn = mins[mins.lnp == mins.lnp.min()].index[0]
            lnp_b = lnp[minn + 1 :].copy()
            difference = lnp_b["lnp"].max() - lnp_b["lnp"].iloc[-1]

        if difference < tolerance:
            print(
                f"WARNING! lnPi at N_max has a relative value higher ({difference:.1f}) than tolerance ({tolerance:.1f})."
            )
            print(
                "The results may be erroneous. Provide data for higher macrostate values."
            )

    def reweight(self, delta_beta_mu: float) -> pd.DataFrame:
        """Reweight the MPD to a new mu / fugacity value using `delta_beta_mu`."""
        lnp_rw = self.lnp.copy()
        lnp_rw["lnp"] += delta_beta_mu * lnp_rw["macrostate"]
        lnp_rw["lnp"] = normalize(lnp_rw["lnp"])
        self.check_tail(lnp=lnp_rw, order=self.order, tolerance=self.tolerance)

        return lnp_rw

    def reweight_to_fug(
        self, fugacity: float, inplace: bool = True
    ) -> None | pd.DataFrame:
        """Reweight the MPD to a new mu / fugacity value using desired fugacity."""
        beta_0 = self.beta
        mu_0 = self.mu
        mu = self.fugacity_to_mu(fugacity, beta_0)
        delta_beta_mu = beta_0 * (mu - mu_0)
        lnp_rw = self.reweight(delta_beta_mu)
        if inplace:
            self._dataframe["lnp"] = lnp_rw["lnp"]
            self.fugacity = fugacity
            return None
        else:
            return lnp_rw

    # TODO: optimize search method and test for edge cases
    def find_phase_equilibrium(
        self,
        tolerance: float = 1e-6,
        max_iterations: int = 100,
        return_probabilities: bool = False,
    ) -> Union[Tuple[float, float, float], float]:
        """Find the fugacity at which the two phases are in equilibrium.

        Arguments
        ---------
        tolerance
            Tolerance for the root finding algorithm.
        max_iterations
            Maximum number of iterations for the root finding algorithm.
        return_probabilities
            Whether to return the probabilities of the two phases at equilibrium.

        Returns
        -------
        float or Tuple[float, float, float]
            The fugacity at which the two phases are in equilibrium. If `return_probabilities`
            is True, also returns the probabilities of the two phases at equilibrium.
        """
        from scipy.optimize import newton
        from scipy.special import logsumexp

        def objective(beta_mu) -> float:
            delta_beta_mu = beta_mu - self.beta_mu
            lnp = self.reweight(delta_beta_mu)
            min_index = self.minimums(order=self._order, lnp=lnp["lnp"])
            if len(min_index) == 0:
                return float(lnp.lnp.values[0] - lnp.lnp.values[-1]) ** 2
            min_index = min_index[min_index.lnp == min_index.lnp.min()].index[0]
            logsum_low = logsumexp(lnp.lnp[:min_index])
            logsum_high = logsumexp(lnp.lnp[min_index:])
            return logsum_low - logsum_high

        equilibrium_beta_mu = newton(
            objective, self.beta_mu, tol=tolerance, maxiter=max_iterations
        )
        equilibrium_fugacity = self.mu_to_fugacity(
            equilibrium_beta_mu / self._beta, self._beta
        )

        equilibrium_lnp = self.reweight_to_fug(equilibrium_fugacity, inplace=False)
        equilibrium_min_index = self.minimums(order=self._order, lnp=equilibrium_lnp)
        equilibrium_min_index = equilibrium_min_index[
            equilibrium_min_index.lnp == equilibrium_min_index.lnp.min()
        ].index[0]
        equilibrium_logsum_low = logsumexp(equilibrium_lnp.lnp[:equilibrium_min_index])
        equilibrium_logsum_high = logsumexp(equilibrium_lnp.lnp[equilibrium_min_index:])
        equilibrium_p_low = float(np.exp(equilibrium_logsum_low))
        equilibrium_p_high = float(np.exp(equilibrium_logsum_high))

        if return_probabilities:
            return equilibrium_fugacity, equilibrium_p_low, equilibrium_p_high
        else:
            return equilibrium_fugacity

    def average_macrostate(
        self, lnp: Optional[pd.DataFrame] = None
    ) -> Union[float, Tuple[float, float]]:
        """Calculate the average macrostate from the MPD data."""
        if lnp is None:
            lnp = self.lnp
        return (np.exp(lnp["lnp"]) * lnp["macrostate"]).sum()

    def minimums(self, order: int, lnp: Optional[pd.Series] = None) -> pd.DataFrame:
        """Find the local minimums in the lnp data."""
        if lnp is None:
            lnp = self._dataframe["lnp"]
        min_loc = argrelextrema(lnp.values, np.less, order=order)[0]
        min_loc = min_loc[(10 < min_loc) & (min_loc < lnp.shape[0] - 10)]

        return self._dataframe.iloc[min_loc]

    def plot(
        self,
        fig: Optional[go.Figure] = None,
        name: Optional[str] = None,
        show: bool = True,
    ) -> None:
        """Plot the MPD data using plotly."""
        font = {"family": "Helvetica Neue", "size": 14, "color": "black"}

        axes = {
            "showline": True,
            "linewidth": 1,
            "linecolor": "black",
            "gridcolor": "lightgrey",
            "mirror": True,
            "zeroline": False,
            "ticks": "inside",
        }

        if fig is None:
            fig = go.Figure()

        fig.add_trace(
            go.Scatter(
                x=self.lnp["macrostate"],
                y=self.lnp["lnp"],
                mode="lines",
                name=name,
            )
        )

        xaxis_title = "Macrostate"
        yaxis_title = "lnΠ"

        fig.update_layout(
            font=font,
            xaxis=axes,
            xaxis_title=xaxis_title,
            yaxis=axes,
            yaxis_title=yaxis_title,
            plot_bgcolor="white",
            width=700,
            height=500,
            margin=dict(l=30, r=30, t=30, b=30),
        )

        if show:
            fig.show()

    def free_energy_at_fugacity(self, fug: float) -> pd.DataFrame:
        """Calculate the free energy profile at a given fugacity."""
        beta_0 = self._beta
        mu_0 = self._mu
        mu = self.fugacity_to_mu(fug, beta_0)
        delta_beta_mu = beta_0 * (mu - mu_0)
        lnp_rw = self.reweight(delta_beta_mu)
        free_en = (
            -0.001
            * _BOLTZMANN_CONSTANT
            * _AVOGADRO_CONSTANT
            * self._temperature
            * lnp_rw["lnp"]
        )
        free_en -= free_en.min()
        free_energy = pd.DataFrame(
            {"macrostate": lnp_rw["macrostate"].copy(), "free_energy_kJ/mol": free_en}
        )

        return free_energy

    def average_macrostate_at_fugacity(
        self, fug: float, order: Optional[int] = None
    ) -> List[float]:
        """Calculate the average macrostate at a given fugacity."""
        beta_0 = self._beta
        mu_0 = self._mu
        mu = self.fugacity_to_mu(fug, beta_0)
        delta_beta_mu = beta_0 * (mu - mu_0)
        lnp_rw = self.reweight(delta_beta_mu)
        if order is None:
            order = self.order
        mins = self.minimums(lnp=lnp_rw["lnp"], order=order)

        if len(mins) == 0:
            return [self.average_macrostate(lnp_rw) / self._system_size_prod]
        else:
            minn = mins[mins.lnp == mins.lnp.min()].index[0]
            lnp_a = lnp_rw[:minn].copy()
            lnp_b = lnp_rw[minn + 1 :].copy()

            p_a = np.exp(lnp_a["lnp"]).sum()
            p_b = np.exp(lnp_b["lnp"]).sum()

            lnp_a["lnp"] = normalize(lnp_a["lnp"])
            lnp_b["lnp"] = normalize(lnp_b["lnp"])

            if p_a > p_b:
                return [
                    self.average_macrostate(lnp_a) / self._system_size_prod,
                    self.average_macrostate(lnp_b) / self._system_size_prod,
                ]
            else:
                return [
                    self.average_macrostate(lnp_b) / self._system_size_prod,
                    self.average_macrostate(lnp_a) / self._system_size_prod,
                ]

    def calculate_isotherm(
        self,
        fugacity: ArrayLike,
        saturation_fugacity: Optional[float] = None,
        pressure: Optional[ArrayLike] = None,
        order: Optional[int] = None,
        return_dataframe: bool = True,
    ) -> Union[pd.DataFrame | Isotherm]:
        """Calculate the adsorption isotherm.

        Parameters
        ----------
        fugacity
            Array of fugacities.
        saturation_fugacity
            Saturation pressure to calculate the pressure in relative scale (p/p0).
        pressure
            Array of pressures corresponding to the fugacities.
        order
            How many points on each side use to find minimum in lnp.
        return_dataframe
            Whether to return the adsorption isotherm as a dataframe or Isotherm instance.

        Returns
        -------
        pd.DataFrame or Isotherm
            DataFrame containing the adsorption isotherm or Isotherm instance if return_dataframe is False.

        Args:
            pressure:
        """
        from asaf import Isotherm
        stable_phase = []
        metastable_gas = []
        metastable_liq = []

        if order is None:
            order = self.order

        for fug in fugacity:
            uptake = self.average_macrostate_at_fugacity(fug, order=order)
            if len(uptake) > 1:
                if uptake[0] > uptake[1]:
                    stable_phase.append([fug, uptake[0]])
                    metastable_gas.append([fug, uptake[1]])
                else:
                    stable_phase.append([fug, uptake[0]])
                    metastable_liq.append([fug, uptake[1]])
            else:
                stable_phase.append([fug, uptake[0]])

        isotherm = pd.DataFrame(stable_phase, columns=["fugacity", "uptake"])

        if len(metastable_gas) > 0:
            iso_metastable_gas = pd.DataFrame(
                metastable_gas, columns=["fugacity", "metastable_gas"]
            )

            isotherm = pd.merge(
                isotherm, iso_metastable_gas, on="fugacity", how="outer"
            )

        if len(metastable_liq) > 0:
            iso_metastable_liq = pd.DataFrame(
                metastable_liq, columns=["fugacity", "metastable_liq"]
            )

            isotherm = pd.merge(
                isotherm, iso_metastable_liq, on="fugacity", how="outer"
            )

        if saturation_fugacity is not None:
            isotherm.insert(1, "f/f0", isotherm["fugacity"] / saturation_fugacity)

        if pressure is not None:
            isotherm.insert(1, "pressure", np.array(pressure))

        if return_dataframe:
            return isotherm
        else:
            return Isotherm(
                data=isotherm,
                saturation_fugacity=saturation_fugacity,
                metadata=self.metadata,
            )

    def extrapolate(
        self,
        temperature: float,
        energy: Optional[pd.DataFrame | pd.Series] = None,
        terms: int = 1,
    ) -> "MPD":
        """Extrapolates the MPD to a new temperature.

        Parameters
        ----------
        temperature
            Temperature to which to extrapolate MPD.
        energy
            Energy fluctuation data. If None ASAF will look for data in prob_df.
        terms
            Number of Taylor series terms used for extrapolation.

        Returns
        -------
        MPD
            Extrapolated MPD.
        """
        if energy is None:
            if "term_1" in self._dataframe.columns:
                energy = self._dataframe[["macrostate", "term_1"]].copy()
            else:
                raise ValueError("Energy related data is missing.")

        beta = self.temperature_to_beta(temperature)
        delta_beta = beta - self.beta
        lnp_extrapolated = self.lnp.copy()
        lnp_extrapolated["lnp"] += (
            self.mu * lnp_extrapolated["macrostate"] - energy["term_1"]
        ) * delta_beta
        lnp_extrapolated["lnp"] = normalize(lnp_extrapolated["lnp"])

        for i in range(2, terms + 1):
            lnp_extrapolated["lnp"] += (
                1 / factorial(i) * energy[f"term_{i}"] * np.power(delta_beta, i)
            )
            lnp_extrapolated["lnp"] = normalize(lnp_extrapolated["lnp"])

        return MPD(
            dataframe=lnp_extrapolated,
            temperature=temperature,
            fugacity=self.mu_to_fugacity(self.mu, beta),
            metadata=self.metadata,
        )


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
