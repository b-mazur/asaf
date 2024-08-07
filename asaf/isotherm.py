import numpy as np
import pandas as pd
from gemmi import cif
import plotly.graph_objects as go
from typing import Any, List, Optional, Dict, TypeVar


def quote(string):
    return f"'{string}'"


TNum = TypeVar('TNum', float, int, List[float], List[int], np.ndarray, pd.Series)


class Isotherm:
    """
    Isotherm class to store, recalculate and save the adsorption isotherm.
    """

    def __init__(
            self,
            data: pd.DataFrame = None,
            saturation_pressure: Optional[float] = None,
            fugacity_coefficient: float = 1.0,
            metadata: Optional[dict[str, Any]] = None,
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
    def dataframe(self) -> pd.DataFrame:
        return self._dataframe

    @dataframe.setter
    def dataframe(self, dataframe: pd.DataFrame) -> None:
        self._dataframe = dataframe

    @property
    def pressure(self) -> pd.Series:
        return self._dataframe['pressure']

    @pressure.setter
    def pressure(self, pressure: pd.Series | List[float] | np.ndarray) -> None:
        self._dataframe['pressure'] = pressure
        self._dataframe['fugacity'] = self.calculate_fugacity(pressure, self.fugacity_coefficient)

    @property
    def fugacity(self) -> pd.Series:
        return self._dataframe['fugacity']

    @property
    def fugacity_coefficient(self) -> float:
        return self._fugacity_coefficient

    @fugacity_coefficient.setter
    def fugacity_coefficient(self, fugacity_coefficient: float) -> None:
        self._fugacity_coefficient = fugacity_coefficient
        self.pressure = self.pressure

    @property
    def saturation_pressure(self) -> float:
        return self._saturation_pressure

    @saturation_pressure.setter
    def saturation_pressure(self, saturation_pressure: float | None) -> None:
        self._saturation_pressure = saturation_pressure
        if saturation_pressure is not None:
            self._dataframe['p/p0'] = self.pressure / saturation_pressure

    @staticmethod
    def calculate_fugacity(
            pressure: TNum,
            fugacity_coefficient: float
    ) -> TNum:
        return pressure * fugacity_coefficient

    @staticmethod
    def calculate_pressure(
            fugacity: TNum,
            fugacity_coefficient: float
    ) -> TNum:
        return fugacity / fugacity_coefficient

    @property
    def amount_adsorbed(self) -> pd.Series:
        return self.dataframe['uptake']

    @property
    def metastable_gas(self) -> Optional[pd.Series]:
        if 'metastable_gas' in self._dataframe.columns:
            return self._dataframe['metastable_gas']
        else:
            return None

    @property
    def metastable_liq(self) -> Optional[pd.Series]:
        if 'metastable_liq' in self._dataframe.columns:
            return self._dataframe['metastable_liq']
        else:
            return None

    @property
    def pressure_unit(self) -> str:
        return self._pressure_unit

    def set_pressure_unit(
            self,
            target_unit: str,
            conversion_factor: float
    ) -> None:
        self.pressure = self.pressure * conversion_factor
        self._pressure_unit = target_unit

    @property
    def uptake_unit(self) -> str:
        return self._uptake_unit

    def set_uptake_unit(
            self,
            target_unit: str,
            conversion_factor: Optional[float] = None
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
        if self._dataframe['metastable_gas']:
            self._dataframe['metastable_gas'] *= conversion_factor
        if self._dataframe['metastable_liq']:
            self._dataframe['metastable_liq'] *= conversion_factor
        self._uptake_unit = target_unit

    @property
    def metadata(self) -> Dict[str, Any]:
        return self._metadata

    @metadata.setter
    def metadata(self, metadata: Dict[str, Any]) -> None:
        if metadata:
            for k, v in metadata.items():
                self._metadata[k] = v

    def plot(self, fig: Optional[go.Figure] = None, show: bool = True) -> None:
        # TODO: choose what's on X-axis (pressure, fugacity, relative pressure)
        # TODO: choose unit of adsorbed amount
        # TODO: allow user to pick all of these from interactive plot
        # TODO: add option to save to file
        # TODO: add modes 'equilibrium', 'metastable_gas', 'metastable_liq' and their combinations with + sign
        #       like 'equilibrium+metastable_gas', default is 'all'

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

        if fig is None:
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

        if show:
            fig.show()

    def to_aif(self, filename: str, user_key_mapper: Optional[Dict[str, Any]] = None) -> None:
        """
        Saves the isotherm in an AIF file format.

        Parameters
        ----------
        filename
            The name of the file to be saved.
        user_key_mapper
            A dictionary based on which keys from the metadata are transformed into keys
            according to aifdictionary.json. Required format: {'_AIF_key': 'metadata_key'}.

        Returns
        -------
        None
        """

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

    def to_csv(self, filename: str) -> None:
        self.dataframe.to_csv(filename)
