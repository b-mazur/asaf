import pandas as pd
import numpy as np
import json
from scipy.signal import argrelextrema
from gemmi import cif
from asaf.constants import _BOLTZMANN_CONSTANT, _AVOGADRO_CONSTANT
import plotly.graph_objects as go
from typing import Any


def quote(string):
    return f"'{string}'"


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
            prob_df: pd.DataFrame | None = None
            ) -> None:
        """
        Initialize the MPD class.

        Parameters
        ----------
        lnp_df
            A pandas dataframe containing the macrostate probability distribution. Column names should be the following:
            'macrostate' for macrostate value and 'lnp' for natural logarithm of the given macrostate probability.
        temperature
            The temperature at which the simulation was performed.
        fugacity
            The fugacity at which the simulation was performed.
        metadata
            A dictionary with the simulation metadata.
        prob_df
            A pandas dataframe containing the transition probabilities. Column names should be the following:
            'macrostate' for macrostate value, 'P_up' for probability of transition to the next macrostate,
            and 'P_down' for probability of transition to the previous macrostate.

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

        if 'system_size' in self.metadata:
            self.system_size = self.metadata['system_size']
        else:
            self.system_size = [1, 1, 1]

    @classmethod
    def prob_from_csv(cls, file_name: str, **kwargs) -> (pd.DataFrame, float, float, dict[str, Any], pd.DataFrame):
        prob_df = pd.read_csv(file_name, **kwargs)

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
    def lnp_from_csv(cls, file_name: str, **kwargs) -> (pd.DataFrame, float, float, dict[str, Any]):
        prob_df = pd.read_csv(file_name, **kwargs)

        lnp_df = prob_df[['macrostate', 'lnp']].copy()

        metadata_fname = file_name.removesuffix('.csv') + '.metadata.json'
        with open(metadata_fname) as f:
            metadata = json.load(f)
        temperature = metadata['temperature']
        fugacity = metadata['fugacity']

        return cls(lnp_df, temperature, fugacity, metadata)

    @staticmethod
    def interpolate_df(df: pd.DataFrame, based_on: str = 'macrostate') -> pd.DataFrame:
        columns = list(df)
        columns.remove(based_on)
        n_max = df[based_on].max()
        n_aranged = np.arange(0, n_max + 1, dtype=int)
        df_interp = pd.DataFrame({based_on: n_aranged})
        for col in columns:
            val_interp = np.interp(n_aranged, df[based_on], df[col])
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
        dlnp = np.log(prob_df['P_up'].shift(1) / prob_df['P_down'])
        dlnp.iloc[0] = 0
        lnp_df = pd.DataFrame(data={
            'macrostate': prob_df['macrostate'],
            'lnp': MPD.normalize(dlnp.cumsum())
        })
        return lnp_df

    @staticmethod
    def beta_to_temp(beta):
        temp = 1 / _BOLTZMANN_CONSTANT / beta
        return temp

    @staticmethod
    def temp_to_beta(temp):
        beta = 1 / _BOLTZMANN_CONSTANT / temp
        return beta

    @staticmethod
    def fug_to_mu(fugacity, beta):
        mu = np.log(fugacity * 1e-30 * beta) / beta  # J A^-3
        return mu

    @staticmethod
    def mu_to_fug(mu, beta):
        fug = np.exp(beta * mu) / beta / 1e-30  # Pa
        return fug

    @property
    def temperature(self):
        return self._temperature

    @temperature.setter
    def temperature(self, temperature):
        self._temperature = temperature
        self._beta = self.temp_to_beta(temperature)

    @property
    def beta(self):
        return self._beta

    @beta.setter
    def beta(self, beta):
        self._beta = beta
        self._temperature = self.beta_to_temp(beta)

    @property
    def fugacity(self):
        return self._fugacity

    @fugacity.setter
    def fugacity(self, fugacity):
        self._fugacity = fugacity
        self._mu = self.fug_to_mu(fugacity, self.beta)

    @property
    def mu(self):
        return self._mu

    @mu.setter
    def mu(self, mu):
        self._mu = mu
        self._fugacity = self.mu_to_fug(mu, self.beta)

    @property
    def system_size(self):
        return self._system_size

    @system_size.setter
    def system_size(self, system_size):
        self._system_size = system_size
        self._system_size_prod = np.prod(system_size)

    @property
    def metadata(self):
        return self._metadata

    @metadata.setter
    def metadata(self, metadata: dict):
        if metadata:
            for k, v in metadata.items():
                self._metadata[k] = v

    def lnp(self):
        return self._lnp_df['lnp']

    def reweight(self, delta_beta_mu):
        lnp_rw = self._lnp_df.copy()
        lnp_rw['lnp'] += delta_beta_mu * lnp_rw['macrostate']
        lnp_rw['lnp'] = self.normalize(lnp_rw['lnp'])

        return lnp_rw

    def reweight_to_fug(self, fugacity, inplace=True):
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
    def average_macrostate(lnp):
        return (np.exp(lnp['lnp']) * lnp['macrostate']).sum()

    def minimums(self, order, lnp=None):
        if lnp.size == 0:
            lnp = self._lnp_df['lnp']
        min_loc = argrelextrema(lnp.values, np.less, order=order)[0]
        min_loc = min_loc[(10 < min_loc) & (min_loc < lnp.shape[0] - 10)]

        return self._lnp_df.iloc[min_loc]

    def plot(self):
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
            'ticks': 'inside',
            'titlefont': font,
            'tickfont': font,
        }

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=self._lnp_df['macrostate'],
            y=self._lnp_df['lnp'],
            mode='lines'
        ))

        xaxis_title = 'Macrostate'
        yaxis_title = r'$\ln\Pi$'

        fig.update_layout(
            font=font,
            xaxis=axes,
            xaxis_title=xaxis_title,
            yaxis=axes,
            yaxis_title=yaxis_title,
            plot_bgcolor='white',
            width=700,
            height=500,
            margin=dict(r=30, t=30)
        )

        fig.show()

    def free_energy_at_fug(self, fug):
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

    def average_macrostate_at_fug(self, fug, order=50):
        beta_0 = self._beta
        mu_0 = self._mu
        mu = self.fug_to_mu(fug, beta_0)
        delta_beta_mu = beta_0 * (mu - mu_0)
        lnp_rw = self.reweight(delta_beta_mu)
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

    def calculate_isotherm(self,
                           pressures,
                           fugacity_coefficient=1.0,
                           saturation_pressure=None,
                           order=50,
                           return_dataframe=True):

        fugs = Isotherm.calculate_fugacity(pressures, fugacity_coefficient)

        stable_phase = []
        metastable_gas = []
        metastable_liq = []

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

    def extrapolate(self, beta, mu, energy=None, terms=1, inplace=False):
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
            self.mu(mu)
            self.beta(beta)

        else:
            return MPD(lnp_extr, self.beta_to_temp(beta),
                       self.mu_to_fug(mu, beta), self.metadata)


class Isotherm:
    """
    Isotherm class to store, recalculate and save the adsorption isotherm.
    """
    def __init__(
            self,
            data: pd.DataFrame = None,
            saturation_pressure: float = None,
            fugacity_coefficient: float = 1.0,
            metadata: dict[str, Any] = None,
            pressure_unit: str = 'Pa',
            uptake_unit: str = 'molecules/unitcell'
    ) -> None:
        """
        Initialize the Isotherm object.

        Parameters
        ----------
        data
            A pandas DataFrame containing the adsorption data. Should contain 'pressure' and 'uptake' columns.
        saturation_pressure
            The saturation pressure at given conditions. Used to calculate the relative pressure.
        fugacity_coefficient
            The fugacity coefficient at given conditions. Used to convert between pressure and fugacity.
        metadata
            A dictionary with the simulation metadata.
        uptake_unit
            Units at which uptake is stored. Default value is 'molecules/unitcell'.

        Returns
        -------
        None
        """
        self.dataframe = data
        self.fugacity_coefficient = fugacity_coefficient
        self.saturation_pressure = saturation_pressure
        self._metadata = {}
        self.metadata = metadata
        self._pressure_unit = pressure_unit
        self._uptake_unit = uptake_unit

    @property
    def dataframe(self):
        return self._dataframe

    @dataframe.setter
    def dataframe(self, dataframe):
        self._dataframe = dataframe
        # self.pressure = self.pressure

    @property
    def pressure(self):
        return self._dataframe['pressure']

    @pressure.setter
    def pressure(self, pressure):
        self._dataframe['pressure'] = pressure
        self._dataframe['fugacity'] = self.calculate_fugacity(pressure, self.fugacity_coefficient)

    @property
    def fugacity(self):
        return self._dataframe['fugacity']

    @property
    def fugacity_coefficient(self):
        return self._fugacity_coefficient

    @fugacity_coefficient.setter
    def fugacity_coefficient(self, fugacity_coefficient):
        self._fugacity_coefficient = fugacity_coefficient
        self.pressure = self.pressure

    @property
    def saturation_pressure(self):
        return self._saturation_pressure

    @saturation_pressure.setter
    def saturation_pressure(self, saturation_pressure):
        self._saturation_pressure = saturation_pressure
        if saturation_pressure is not None:
            self._dataframe['p/p0'] = self.pressure / saturation_pressure

    @staticmethod
    def calculate_fugacity(pressure, fugacity_coefficient):
        return pressure * fugacity_coefficient

    @staticmethod
    def calculate_pressure(fugacity, fugacity_coefficient):
        return fugacity / fugacity_coefficient

    @property
    def amount_adsorbed(self):
        return self._dataframe['uptake']

    @property
    def metastable_gas(self):
        if 'metastable_gas' in self._dataframe.columns:
            return self._dataframe['metastable_gas']
        else:
            return None

    @property
    def metastable_liq(self):
        if 'metastable_liq' in self._dataframe.columns:
            return self._dataframe['metastable_liq']
        else:
            return None

    @property
    def pressure_unit(self):
        return self._pressure_unit

    def set_pressure_unit(
            self,
            target_unit: str,
            conversion_factor: float
    ) -> None:
        self.pressure = self.pressure * conversion_factor
        self._pressure_unit = target_unit

    @property
    def uptake_unit(self):
        return self._uptake_unit

    def set_uptake_unit(
            self,
            target_unit: str,
            conversion_factor: float | None = None
    ) -> None:
        # if conversion factor was not provided try to find it in metadata
        if conversion_factor is None:
            cf_key = f'{self.uptake_unit}_to_{target_unit}'
            cf_key_rev = f'{target_unit}_to_{self.uptake_unit}'
            if cf_key in self.metadata.keys():
                conversion_factor = self.metadata[cf_key]
            # try to find the conversion factor in the opposite direction
            elif cf_key_rev in self.metadata.keys():
                conversion_factor = 1 / self.metadata[cf_key_rev]
            else:
                raise ValueError(f'Conversion factor {cf_key} was not provided an is not in metadata.')
        self._dataframe['uptake'] *= conversion_factor
        self._uptake_unit = target_unit

    @property
    def metadata(self):
        return self._metadata

    @metadata.setter
    def metadata(self, metadata: dict):
        if metadata:
            for k, v in metadata.items():
                self._metadata[k] = v

    def plot(self):
        # TODO: allow for providing ax on which plot has to be created
        # TODO: choose what's on X-axis (pressure, fugacity, relative pressure)
        # TODO: choose unit of adsorbed amount
        # TODO: allow user to pick all of these from interactive plot

        font = {'family': 'Helvetica Neue',
                'size': 14,
                'color': 'black'
                }

        axes = {'showline': True,
                'linewidth': 1,
                'linecolor': 'black',
                'gridcolor': 'lightgrey',
                'mirror': True,
                'zeroline': False,
                'ticks': 'inside'
                }

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=self.pressure,
                                 y=self.amount_adsorbed,
                                 mode='lines+markers',
                                 name='Uptake'
                                 ))

        y_axis = ['uptake']

        if self.metastable_gas is not None:
            y_axis.append('metastable_gas')
            fig.add_trace(go.Scatter(x=self.pressure,
                                     y=self.metastable_gas,
                                     mode='lines',
                                     name='Metastable gas phase',
                                     line=dict(dash='dash')
                                     ))

        if self.metastable_liq is not None:
            y_axis.append('metastable_liq')
            fig.add_trace(go.Scatter(x=self.pressure,
                                     y=self.metastable_liq,
                                     mode='lines',
                                     name='Metastable liquid phase',
                                     line=dict(dash='dash')
                                     ))

        xaxis_title = f'Pressure ({self.pressure_unit})'
        yaxis_title = f'Uptake ({self.uptake_unit})'

        fig.update_layout(font=font,
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

    def to_aif(self, filename, user_key_mapper=None):

        metadata = self.metadata
        key_mapper = {
            '_exptl_temperature': 'temperature',
            '_units_temperature': 'temperature_units',
            '_adsnt_material_id': 'framework_name',
            '_exptl_adsorptive_name': 'molecule_name',
            '_simltn_code': 'code_name',
            '_simltn_date': 'simulation_date',
            '_simltn_size': 'system_size',
            '_simltn_forcefield_adsorptive': 'molecule_force_field',
            '_simltn_forcefield_adsorbent': 'framework_force_field',
            '_units_pressure': 'fugacity_units',
            '_units_loading': 'loading_units',
        }

        if user_key_mapper:
            key_mapper.update(user_key_mapper)

        doc = cif.Document()
        doc.add_new_block('isotherm')
        block = doc.sole_block()

        if metadata:
            for key, value in key_mapper.items():
                if value in metadata.keys():
                    if isinstance(metadata[value], (int, float)):
                        block.set_pair(key, str(metadata[value]))
                    else:
                        block.set_pair(key, quote(metadata[value]))

        block.set_pair('_units_loading', quote(self.uptake_unit))
        block.set_pair('_audit_aif_version', quote('63df4e8'))

        df = self.dataframe

        if self.saturation_pressure:
            df['saturation_pressure'] = self.saturation_pressure
            loop_ads = block.init_loop('_adsorp_', ['pressure', 'p0', 'fugacity', 'amount'])
            loop_ads.set_all_values([
                list(df['pressure'].values.astype(str)),
                list(df['saturation_pressure'].values.astype(str)),
                list(df['fugacity'].values.astype(str)),
                list(df['uptake'].values.astype(str))
            ])
        else:
            loop_ads = block.init_loop('_adsorp_', ['pressure', 'fugacity', 'amount'])
            loop_ads.set_all_values([
                list(df['pressure'].values.astype(str)),
                list(df['fugacity'].values.astype(str)),
                list(df['uptake'].values.astype(str))
            ])

        if filename.endswith('.aif'):
            filename = filename[:-4]
        doc.write_file(f'{filename}.aif')

    def to_csv(self, filename):
        self.dataframe.to_csv(filename)

