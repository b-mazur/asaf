from __future__ import annotations
import pandas as pd
import numpy as np
import json
from asaf.isotherm import Isotherm
from scipy.signal import argrelextrema
from asaf.constants import _BOLTZMANN_CONSTANT, _AVOGADRO_CONSTANT
import plotly.graph_objects as go
from typing import Any, List, Optional, Dict, TypeVar


def quote(string):
    return f"'{string}'"


TNum = TypeVar('TNum', float, int, List[float], List[int], np.ndarray, pd.Series)


# TODO: move static functions to utility module
class MPD:
    """
    Class for storing and processing macrostate probability distribution.
    """

    def __init__(
            self,
            lnp_df: pd.DataFrame,
            temperature: float,
            fugacity: float,
            metadata: dict[str, Any] | None = None,
            prob_df: pd.DataFrame | None = None,
            order: int = 50,
            tolerance: float = 10.
    ) -> None:
        """
        Initialize the MPD class.

        Parameters
        ----------
        lnp_df
            A pandas dataframe containing the macrostate probability distribution.
            Column names should be the following: 'macrostate' for macrostate value
            and 'lnp' for natural logarithm of the given macrostate probability.
        temperature
            The temperature at which the simulation was performed.
        fugacity
            The fugacity at which the simulation was performed.
        metadata
            A dictionary with the simulation metadata.
        prob_df
            A pandas dataframe containing the transition probabilities.
            Column names should be the following: 'macrostate' for macrostate value,
            'P_up' for probability of transition to the next macrostate,
            and 'P_down' for probability of transition to the previous macrostate.
        order
            How many points on each side use to find minimum in lnp.
        tolerance
            Used when checking the probability at lnp tail.

        Returns
        -------
        None
        """
        self._mu = None
        self._beta = None
        self._metadata = {}
        self.metadata = metadata
        self._prob_df = prob_df
        self._lnp_df = lnp_df
        self.temperature = temperature
        self.fugacity = fugacity
        self._system_size_prod = 1

        if 'system_size' in self.metadata:
            self.system_size = self.metadata['system_size']
        else:
            self.system_size = [1, 1, 1]

        self.order = order
        self.tolerance = tolerance
        self.check_tail(lnp_df, order, tolerance)

    @classmethod
    def prob_from_csv(
            cls,
            file_name: str,
            **kwargs: object
    ) -> object:
        """
        Reads transition probabilities from a csv file.

        Parameters
        ----------
        file_name
            The name of the csv file.
        kwargs
            Options to pass to pandas read_csv method.
        """
        prob_df = pd.read_csv(file_name, **kwargs)

        # interpolate data for macrostates which were not sampled
        if prob_df['macrostate'].diff().mean() != 1:
            print('Interpolating data...')
            prob_df = MPD.interpolate_df(prob_df)

        lnp_df = MPD.calculate_lnp(prob_df)

        metadata_fname = file_name.removesuffix('.csv') + '.metadata.json'
        with open(metadata_fname) as f:
            metadata = json.load(f)
        temperature = metadata['temperature']
        fugacity = metadata['fugacity']

        return cls(lnp_df, temperature, fugacity, metadata, prob_df)

    @classmethod
    def lnp_from_csv(
            cls,
            file_name: str,
            **kwargs: object
    ) -> object:
        """
        Reads the natural logarithm of macrostates probability from a csv file.

        Parameters
        ----------
        file_name
            The name of the csv file.
        kwargs
            Options to pass to pandas read_csv method.
        """

        prob_df = pd.read_csv(file_name, **kwargs)

        lnp_df = prob_df[['macrostate', 'lnp']].copy()

        metadata_fname = file_name.removesuffix('.csv') + '.metadata.json'
        with open(metadata_fname) as f:
            metadata = json.load(f)
        temperature = metadata['temperature']
        fugacity = metadata['fugacity']

        return cls(lnp_df, temperature, fugacity, metadata)

    @staticmethod
    def interpolate_df(
            df: pd.DataFrame,
            based_on: str = 'macrostate'
    ) -> pd.DataFrame:
        """
        Interpolates data in a dataframe.

        Parameters
        ----------
        df
            A pandas DataFrame containing data to be interpolated.
        based_on
            Column name in df containing values of the independent variable.
            Values must be real, finite and in strictly increasing order.
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
    def normalize(lnp: pd.Series | np.ndarray) -> (pd.Series | np.ndarray):
        lnp_cp = lnp.copy()
        maxx = np.max(lnp_cp)
        lnp_cp -= np.log(sum(np.exp(lnp - maxx))) + maxx

        return lnp_cp

    @staticmethod
    def calculate_lnp(prob_df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate the natural logarithm of the macrostate transition probability[1].

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
        dlnp = np.log(prob_df['P_up'].shift(1) / prob_df['P_down'])
        dlnp.iloc[0] = 0
        lnp_df = pd.DataFrame(data={
            'macrostate': prob_df['macrostate'],
            'lnp': MPD.normalize(dlnp.cumsum())
        })
        return lnp_df

    @staticmethod
    def beta_to_temp(beta: TNum) -> TNum:
        temp = 1 / _BOLTZMANN_CONSTANT / beta
        return temp

    @staticmethod
    def temp_to_beta(temp: TNum) -> TNum:
        beta = 1 / _BOLTZMANN_CONSTANT / temp
        return beta

    @staticmethod
    def fug_to_mu(fugacity: TNum, beta: float) -> TNum:
        mu = np.log(fugacity * 1e-30 * beta) / beta  # J A^-3
        return mu

    @staticmethod
    def mu_to_fug(mu: TNum, beta: float) -> TNum:
        fug = np.exp(beta * mu) / beta / 1e-30  # Pa
        return fug

    @property
    def temperature(self) -> float:
        return self._temperature

    @temperature.setter
    def temperature(self, temperature: float) -> None:
        self._temperature = temperature
        self._beta = self.temp_to_beta(temperature)

    @property
    def beta(self) -> float:
        return self._beta

    @beta.setter
    def beta(self, beta: float) -> None:
        self._beta = beta
        self._temperature = self.beta_to_temp(beta)

    @property
    def fugacity(self) -> float:
        return self._fugacity

    @fugacity.setter
    def fugacity(self, fugacity: float) -> None:
        self._fugacity = fugacity
        self._mu = self.fug_to_mu(fugacity, self.beta)

    @property
    def mu(self) -> float:
        return self._mu

    @mu.setter
    def mu(self, mu: float) -> None:
        self._mu = mu
        self._fugacity = self.mu_to_fug(mu, self.beta)

    @property
    def system_size(self) -> List[int]:
        return self._system_size

    @system_size.setter
    def system_size(self, system_size: List[int]) -> None:
        self._system_size = system_size
        self._system_size_prod = np.prod(system_size)

    @property
    def metadata(self) -> Dict[str, Any]:
        return self._metadata

    @metadata.setter
    def metadata(self, metadata: Dict[str, Any]):
        if metadata:
            for k, v in metadata.items():
                self._metadata[k] = v

    @property
    def order(self) -> int:
        return self._order

    @order.setter
    def order(self, order: int) -> None:
        self._order = order

    @property
    def tolerance(self) -> float:
        return self._tolerance

    @tolerance.setter
    def tolerance(self, tolerance: float) -> None:
        self._tolerance = tolerance

    def lnp(self) -> pd.Series:
        return self._lnp_df['lnp']

    def check_tail(self, lnp: pd.DataFrame, order: int, tolerance: float) -> None:
        mins = self.minimums(lnp=lnp, order=order)
        if len(mins) == 0:
            difference = lnp['lnp'].max() - lnp['lnp'].iloc[-1]
        else:
            minn = mins[mins.lnp == mins.lnp.min()].index[0]
            lnp_b = lnp[minn + 1:].copy()
            difference = lnp_b['lnp'].max() - lnp_b['lnp'].iloc[-1]

        if difference < tolerance:
            print(f"WARNING! lnPi at N_max has a relative value higher ({difference}) than tolerance ({tolerance}).")
            print("The results may be erroneous. Provide the data for higher macrostate values.")

    def reweight(self, delta_beta_mu: float) -> pd.DataFrame:
        lnp_rw = self._lnp_df.copy()
        lnp_rw['lnp'] += delta_beta_mu * lnp_rw['macrostate']
        lnp_rw['lnp'] = self.normalize(lnp_rw['lnp'])
        self.check_tail(lnp=lnp_rw, order=self.order, tolerance=self.tolerance)

        return lnp_rw

    def reweight_to_fug(self, fugacity: float, inplace: bool = True) -> None | pd.DataFrame:
        beta_0 = self.beta
        mu_0 = self.mu
        mu = self.fug_to_mu(fugacity, beta_0)
        delta_beta_mu = beta_0 * (mu - mu_0)
        lnp_rw = self.reweight(delta_beta_mu)
        if inplace:
            self._lnp_df = lnp_rw
            self.fugacity = fugacity
        else:
            return lnp_rw

    @staticmethod
    def average_macrostate(lnp: pd.DataFrame) -> float:
        return (np.exp(lnp['lnp']) * lnp['macrostate']).sum()

    def minimums(self, order: int, lnp: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        if lnp is None:
            lnp = self._lnp_df['lnp']
        min_loc = argrelextrema(lnp.values, np.less, order=order)[0]
        min_loc = min_loc[(10 < min_loc) & (min_loc < lnp.shape[0] - 10)]

        return self._lnp_df.iloc[min_loc]

    def plot(self) -> None:
        font = {
            'family': 'Helvetica Neue',
            'size': 14,
            'color': 'black'
        }

        axes = {
            'showline': True,
            'linewidth': 1,
            'linecolor': 'black',
            'gridcolor': 'lightgrey',
            'mirror': True,
            'zeroline': False,
            'ticks': 'inside'
        }

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=self._lnp_df['macrostate'],
            y=self._lnp_df['lnp'],
            mode='lines'
        ))

        xaxis_title = 'Macrostate'
        yaxis_title = 'lnPi'  # r'$\ln\Pi$' plotly has a bug on using latex in jupyter lab, for now use just lnp

        fig.update_layout(
            font=font,
            xaxis=axes,
            xaxis_title=xaxis_title,
            yaxis=axes,
            yaxis_title=yaxis_title,
            plot_bgcolor='white',
            width=700,
            height=500,
            margin=dict(l=30, r=30, t=30, b=30)
        )

        fig.show()

    def free_energy_at_fug(self, fug: float) -> pd.DataFrame:
        beta_0 = self._beta
        mu_0 = self._mu
        mu = self.fug_to_mu(fug, beta_0)
        delta_beta_mu = beta_0 * (mu - mu_0)
        lnp_rw = self.reweight(delta_beta_mu)
        free_en = -0.001 * _BOLTZMANN_CONSTANT * _AVOGADRO_CONSTANT * self._temperature * lnp_rw['lnp']
        free_en -= free_en.min()
        free_energy = pd.DataFrame({
            'macrostate': lnp_rw['macrostate'].copy(),
            'free_energy_kJ/mol': free_en
        })

        return free_energy

    def average_macrostate_at_fug(self, fug: float, order: Optional[int] = None) -> List[float]:
        beta_0 = self._beta
        mu_0 = self._mu
        mu = self.fug_to_mu(fug, beta_0)
        delta_beta_mu = beta_0 * (mu - mu_0)
        lnp_rw = self.reweight(delta_beta_mu)
        if order is None:
            order = self.order
        mins = self.minimums(lnp=lnp_rw, order=order)
        if len(mins) == 0:
            return [self.average_macrostate(lnp_rw) / self._system_size_prod]
        else:
            minn = mins[mins.lnp == mins.lnp.min()].index[0]
            lnp_a = lnp_rw[:minn].copy()
            lnp_b = lnp_rw[minn + 1:].copy()

            p_a = np.exp(lnp_a['lnp']).sum()
            p_b = np.exp(lnp_b['lnp']).sum()

            lnp_a['lnp'] = self.normalize(lnp_a['lnp'])
            lnp_b['lnp'] = self.normalize(lnp_b['lnp'])

            if p_a > p_b:
                return [
                    self.average_macrostate(lnp_a) / self._system_size_prod,
                    self.average_macrostate(lnp_b) / self._system_size_prod
                ]
            else:
                return [
                    self.average_macrostate(lnp_b) / self._system_size_prod,
                    self.average_macrostate(lnp_a) / self._system_size_prod
                ]

    def calculate_isotherm(
            self,
            pressures: np.ndarray | List[float],
            fugacity_coefficient: float = 1.0,
            saturation_pressure: Optional[float] = None,
            order: Optional[int] = None,
            return_dataframe: bool = True
    ) -> pd.DataFrame | Isotherm:
        """
        Calculates the adsorption isotherm.

        Parameters
        ----------
        pressures
            Array of pressures.
        fugacity_coefficient
            Fugacity coefficient to calculate the fugacity.
        saturation_pressure
            Saturation pressure to calculate the pressure in relative scale (p/p0).
        order
            How many points on each side use to find minimum in lnp.
        return_dataframe
            Whether to return the adsorption isotherm as a dataframe or Isotherm instance.

        Returns
        -------
        pd.DataFrame or Isotherm
            DataFrame containing the adsorption isotherm or Isotherm instance if return_dataframe is False.
        """

        fugs = Isotherm.calculate_fugacity(pressures, fugacity_coefficient)

        stable_phase = []
        metastable_gas = []
        metastable_liq = []

        if order is None:
            order = self.order

        for fug in fugs:
            uptake = self.average_macrostate_at_fug(fug, order=order)
            if len(uptake) > 1:
                if uptake[0] > uptake[1]:
                    stable_phase.append([fug, uptake[0]])
                    metastable_gas.append([fug, uptake[1]])
                else:
                    stable_phase.append([fug, uptake[0]])
                    metastable_liq.append([fug, uptake[1]])
            else:
                stable_phase.append([fug, uptake[0]])

        isotherm = pd.DataFrame(stable_phase, columns=['fugacity', 'uptake'])

        if len(metastable_gas) > 0:
            iso_metastable_gas = pd.DataFrame(
                metastable_gas,
                columns=['fugacity', 'metastable_gas']
            )

            isotherm = pd.merge(
                isotherm,
                iso_metastable_gas,
                on='fugacity',
                how='outer'
            )

        if len(metastable_liq) > 0:
            iso_metastable_liq = pd.DataFrame(
                metastable_liq,
                columns=['fugacity', 'metastable_liq']
            )

            isotherm = pd.merge(
                isotherm,
                iso_metastable_liq,
                on='fugacity',
                how='outer'
            )

        isotherm.insert(
            0,
            'pressure',
            Isotherm.calculate_pressure(isotherm['fugacity'], fugacity_coefficient)
        )

        if saturation_pressure:
            isotherm.insert(1, 'p/p0', isotherm['pressure'] / saturation_pressure)

        if return_dataframe is True:
            return isotherm
        else:
            return Isotherm(
                data=isotherm,
                saturation_pressure=saturation_pressure,
                fugacity_coefficient=fugacity_coefficient,
                metadata=self.metadata
            )

    def extrapolate(
            self,
            beta: float,
            mu: float,
            energy: Optional[pd.Series | np.ndarray] = None,
            terms: int = 1,
            inplace: bool = False
    ) -> None | MPD:
        """
        Extrapolates the MPD to a new temperature.

        Parameters
        ----------
        beta
            The inverse temperature to which to extrapolate MPD.
        mu
            Chemical potential to which to extrapolate MPD.
        energy
            Energy fluctuation data.
        terms
            Number of Taylor series terms used for extrapolation.
        inplace
            Whether to modify the MPD rather than creating a new one.

        Returns
        -------
        MPD or None
            Extrapolated MPD or None if inplace is True.
        """
        if energy is None:
            if 'term_1' in self._prob_df:
                energy = self._prob_df.copy()
        delta_beta_mu = self._beta * (mu - self._mu)
        lnp_rw = self.reweight(delta_beta_mu)
        delta_beta = beta - self._beta
        lnp_extr = lnp_rw.copy()
        if terms == 3:
            lnp_extr['lnp'] = self.normalize(
                lnp_rw['lnp'] +
                ((mu * lnp_rw['macrostate'] - energy['term_1']) * delta_beta) +
                (1 / 2 * energy['term_2'] * np.square(delta_beta)) +
                (1 / 6 * energy['term_3'] * np.power(delta_beta, 3))
            )

        elif terms == 2:
            lnp_extr['lnp'] = self.normalize(
                lnp_rw['lnp'] +
                ((mu * lnp_rw['macrostate'] - energy['term_1']) * delta_beta) +
                (1 / 2 * energy['term_2'] * np.square(delta_beta))
            )

        elif terms == 1:
            lnp_extr['lnp'] = self.normalize(
                lnp_rw['lnp'] +
                ((mu * lnp_rw['macrostate'] - energy['term_1']) * delta_beta)
            )

        if inplace:
            self._lnp_df = lnp_extr
            self.mu = mu
            self.beta = beta

        else:
            return MPD(
                lnp_extr,
                self.beta_to_temp(beta),
                self.mu_to_fug(mu, beta),
                self.metadata
            )
