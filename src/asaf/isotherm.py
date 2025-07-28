from __future__ import annotations

import numpy as np
import pandas as pd
from gemmi import cif
import plotly.graph_objects as go
import plotly.colors as pc
from typing import Any, List, Optional, Dict, TypeVar


def quote(string):
    return f"'{string}'"


TNum = TypeVar("TNum", float, int, List[float], List[int], np.ndarray, pd.Series)


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
        pressure_unit: str = "Pa",
        uptake_unit: str = "molecules/unitcell",
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
        return self._dataframe["pressure"]

    @pressure.setter
    def pressure(self, pressure: pd.Series | List[float] | np.ndarray) -> None:
        self._dataframe["pressure"] = pressure
        self._dataframe["fugacity"] = self.calculate_fugacity(
            pressure, self.fugacity_coefficient
        )

    @property
    def fugacity(self) -> pd.Series:
        return self._dataframe["fugacity"]

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
            self._dataframe["p/p0"] = self.pressure / saturation_pressure

    @staticmethod
    def calculate_fugacity(pressure: TNum, fugacity_coefficient: float) -> TNum:
        return pressure * fugacity_coefficient

    @staticmethod
    def calculate_pressure(fugacity: TNum, fugacity_coefficient: float) -> TNum:
        return fugacity / fugacity_coefficient

    @property
    def amount_adsorbed(self) -> pd.Series:
        return self.dataframe["uptake"]

    @property
    def metastable_gas(self) -> Optional[pd.Series]:
        if "metastable_gas" in self._dataframe.columns:
            return self._dataframe["metastable_gas"]
        else:
            return None

    @property
    def metastable_liq(self) -> Optional[pd.Series]:
        if "metastable_liq" in self._dataframe.columns:
            return self._dataframe["metastable_liq"]
        else:
            return None

    @property
    def pressure_unit(self) -> str:
        return self._pressure_unit

    def set_pressure_unit(self, target_unit: str, conversion_factor: float) -> None:
        self.pressure = self.pressure * conversion_factor
        self._pressure_unit = target_unit

    @property
    def uptake_unit(self) -> str:
        return self._uptake_unit

    def set_uptake_unit(
        self, target_unit: str, conversion_factor: Optional[float] = None
    ) -> None:
        def get_conversion_factor(current_unit: str, new_unit: str) -> float:
            """Retrieve or compute the conversion factor between units.

            This function looks for the conversion factor in the nested 'conversion_factors'
            dictionary in metadata using both forward and reverse keys.
            """
            conv_factors = self.metadata.get("conversion_factors", {})
            forward_key = f"{current_unit}->{new_unit}"
            reverse_key = f"{new_unit}->{current_unit}"
            key_a = f"molecules/unitcell->{current_unit}"
            key_b = f"molecules/unitcell->{new_unit}"
            if forward_key in conv_factors:
                return conv_factors[forward_key]
            elif reverse_key in conv_factors:
                return 1 / conv_factors[reverse_key]
            elif key_a in conv_factors and key_b in conv_factors:
                return conv_factors[key_b] / conv_factors[key_a]
            else:
                raise ValueError(
                    f"Conversion factor for {forward_key} was not provided and was not found in metadata."
                )

        # Determine the conversion factor, either using the provided value or computing it.
        conversion_factor = conversion_factor or get_conversion_factor(
            self.uptake_unit, target_unit
        )

        self.dataframe["uptake"] *= conversion_factor
        if self.metastable_gas is not None:
            self.dataframe["metastable_gas"] *= conversion_factor
        if self.metastable_liq is not None:
            self.dataframe["metastable_liq"] *= conversion_factor

        self._uptake_unit = target_unit

    @property
    def metadata(self) -> Dict[str, Any]:
        return self._metadata

    @metadata.setter
    def metadata(self, metadata: Dict[str, Any]) -> None:
        if metadata:
            for k, v in metadata.items():
                self._metadata[k] = v

    # def plot(
    #     self,
    #     label: Optional[str] = None,
    #     fig: Optional[go.Figure] = None,
    #     x_axis: str = "pressure",
    #     y_axis: str = "molecules/unitcell",
    #     trace_kwargs: Optional[Dict[str, Any]] = None,
    #     layout_kwargs: Optional[Dict[str, Any]] = None,
    # ) -> go.Figure:
    #     """
    #     Plot an isotherm (stable + metastable gas + metastable liquid) and group all traces
    #     under a single legend entry.
    #     """
    #
    #     # trace_kwargs = trace_kwargs or {}
    #     # layout_kwargs = layout_kwargs or {}
    #
    #     if not color:
    #         default_colors = pc.qualitative.Plotly
    #         calls = getattr(fig, "_plot_calls", 0)
    #         color = default_colors[calls % len(default_colors)]
    #         fig._plot_calls = calls + 1
    #
    #     if not marker:
    #         marker = 'circle'
    #
    #     font = dict(family="Helvetica Neue", size=14, color="black")
    #     axes_common = dict(
    #         showline=True,
    #         linewidth=1,
    #         linecolor="black",
    #         gridcolor="lightgrey",
    #         mirror=True,
    #         zeroline=False,
    #         ticks="inside",
    #     )
    #
    #     # get or create figure
    #     show_immediately = False
    #     if fig is None:
    #         fig = go.Figure()
    #         show_immediately = True
    #
    #     # # cycle through Plotly qualitative palette
    #     # calls = getattr(fig, "_plot_calls", 0)
    #     # color = default_colors[calls % len(default_colors)]
    #     # fig._plot_calls = calls + 1
    #
    #     # --- choose x data & title ---
    #     axis_map = {
    #         "pressure": (self.pressure, f"Pressure ({self.pressure_unit})"),
    #         "fugacity": (self.fugacity, f"Fugacity ({self.pressure_unit})"),
    #         "relative_pressure": (self._dataframe["p/p0"], "p/p₀"),
    #         "p/p0": (self._dataframe["p/p0"], "p/p₀"),
    #     }
    #     try:
    #         x_vals, x_title = axis_map[x_axis]
    #     except KeyError:
    #         valid = "', '".join(axis_map)
    #         raise ValueError(f"x_axis must be one of '{valid}'. Got {x_axis!r}.")
    #
    #     original_unit = self.uptake_unit
    #     if y_axis != original_unit:
    #         self.set_uptake_unit(y_axis)
    #
    #     legend_name = label or "Uptake"
    #
    #     lg = dict(legendgroup=legend_name)
    #     fig.add_trace(
    #         go.Scatter(
    #             x=x_vals,
    #             y=self.amount_adsorbed,
    #             mode="lines+markers",
    #             name=legend_name,
    #             line=dict(color=color),
    #             marker=dict(color=color, symbol=marker),
    #             **lg,
    #             **kwargs
    #         )
    #     )
    #     if self.metastable_gas is not None:
    #         fig.add_trace(
    #             go.Scatter(
    #                 x=x_vals,
    #                 y=self.metastable_gas,
    #                 mode="lines",
    #                 name=legend_name,
    #                 line=dict(color=color, dash="dash"),
    #                 showlegend=False,
    #                 **lg,
    #                 **kwargs
    #             )
    #         )
    #     if self.metastable_liq is not None:
    #         fig.add_trace(
    #             go.Scatter(
    #                 x=x_vals,
    #                 y=self.metastable_liq,
    #                 mode="lines",
    #                 name=legend_name,
    #                 line=dict(color=color, dash="dash"),
    #                 showlegend=False,
    #                 **lg,
    #                 **kwargs
    #             )
    #         )
    #
    #     # restore original unit
    #     if y_axis != original_unit:
    #         self.set_uptake_unit(original_unit)
    #
    #     fig.update_layout(
    #         font=font,
    #         xaxis=axes_common,
    #         xaxis_title=x_title,
    #         yaxis=axes_common,
    #         yaxis_title=f"Uptake ({y_axis})",
    #         plot_bgcolor="white",
    #         width=700,
    #         height=500,
    #         margin=dict(l=30, r=30, t=30, b=30),
    #         legend=dict(traceorder="grouped"),
    #     )
    #
    #     if show_immediately:
    #         fig.show()
    #
    #     return fig

    def plot(
        self,
        label: Optional[str] = None,
        fig: Optional[go.Figure] = None,
        x_axis: str = "pressure",
        y_axis: str = "molecules/unitcell",
        trace_kwargs: Optional[Dict[str, Any]] = None,
        layout_kwargs: Optional[Dict[str, Any]] = None,
    ) -> go.Figure:
        """
        Plot an isotherm (stable + metastable gas + metastable liquid) and group
        all traces under a single legend entry. You can pass any kwargs through
        `trace_kwargs` or `layout_kwargs`; missing values will be filled in by defaults.
        """

        trace_kwargs = trace_kwargs or {}
        layout_kwargs = layout_kwargs or {}

        # look for an explicit color in trace_kwargs
        explicit_color = None
        if "line" in trace_kwargs and isinstance(trace_kwargs["line"], dict):
            explicit_color = trace_kwargs["line"].get("color")
        # if "marker" in trace_kwargs and isinstance(trace_kwargs["marker"], dict):
        #     explicit_color = explicit_color or trace_kwargs["marker"].get("color")

        if explicit_color:
            color = explicit_color
        else:
            default_colors = pc.qualitative.Vivid
            calls = getattr(fig, "_plot_calls", 0)
            color = default_colors[calls % len(default_colors)]
            fig._plot_calls = calls + 1

        axis_map = {
            "pressure": (self.pressure, f"Pressure ({self.pressure_unit})"),
            "fugacity": (self.fugacity, f"Fugacity ({self.pressure_unit})"),
            "relative_pressure": (self._dataframe["p/p0"], "p/p₀"),
            "p/p0": (self._dataframe["p/p0"], "p/p₀"),
        }

        if x_axis not in axis_map:
            valid = "', '".join(axis_map)
            raise ValueError(f"x_axis must be one of '{valid}'. Got {x_axis!r}.")

        x_vals, x_title = axis_map[x_axis]

        original_unit = self.uptake_unit
        if y_axis != original_unit:
            self.set_uptake_unit(y_axis)

        legend_name = label or "Uptake"
        lg = dict(legendgroup=legend_name)

        default_stable = {
            "x": x_vals,
            "y": self.amount_adsorbed,
            "mode": "lines+markers",
            "name": legend_name,
            "line": dict(color=color),
            "marker": dict(
                color=color,
                symbol=trace_kwargs.get("marker", {}).get("symbol", "circle"),
            ),
            **lg,
        }

        user_line = trace_kwargs.get("line", {})
        user_marker = trace_kwargs.get("marker", {})

        stable_line = {**default_stable["line"], **user_line}
        stable_marker = {**default_stable["marker"], **user_marker}

        merged_stable = {
            **default_stable,
            **trace_kwargs,
            "line": stable_line,
            "marker": stable_marker,
        }
        
        # get or create figure
        show_immediately = False
        if fig is None:
            fig = go.Figure()
            show_immediately = True

        fig.add_trace(go.Scatter(**merged_stable))

        # metastable defaults: just dashed line, no markers, and no extra legend entry
        default_meta = {
            "x": x_vals,
            "mode": "lines",
            "name": legend_name,
            "line": dict(color=color, dash="dash"),
            "showlegend": False,
            **lg,
        }

        meta_line = {**default_meta["line"], **user_line}
        meta_trace_kwargs = {
            **default_meta,
            **trace_kwargs,
            "line": meta_line,
        }

        if self.metastable_gas is not None:
            fig.add_trace(go.Scatter(y=self.metastable_gas, **meta_trace_kwargs))
        if self.metastable_liq is not None:
            fig.add_trace(go.Scatter(y=self.metastable_liq, **meta_trace_kwargs))

        # restore original unit
        if y_axis != original_unit:
            self.set_uptake_unit(original_unit)

        base_layout = dict(
            font=dict(family="Helvetica Neue", size=14, color="black"),
            xaxis=dict(
                showline=True,
                linewidth=1,
                linecolor="black",
                gridcolor="lightgrey",
                mirror=True,
                zeroline=False,
                ticks="inside",
                title=x_title,
            ),
            yaxis=dict(
                showline=True,
                linewidth=1,
                linecolor="black",
                gridcolor="lightgrey",
                mirror=True,
                zeroline=False,
                ticks="inside",
                title=f"Uptake ({y_axis})",
            ),
            plot_bgcolor="white",
            width=700,
            height=500,
            margin=dict(l=30, r=30, t=30, b=30),
            legend=dict(traceorder="grouped"),
        )
        fig.update_layout(**{**base_layout, **layout_kwargs})

        if show_immediately:
            fig.show()

        return fig

    def to_aif(
        self, filename: str, user_key_mapper: Optional[Dict[str, Any]] = None
    ) -> None:
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
            "_exptl_temperature": "temperature",
            "_units_temperature": "temperature_units",
            "_adsnt_material_id": "framework_name",
            "_exptl_adsorptive_name": "molecule_name",
            "_simltn_code": "code_name",
            "_simltn_date": "simulation_date",
            "_simltn_size": "system_size",
            "_simltn_forcefield_adsorptive": "molecule_force_field",
            "_simltn_forcefield_adsorbent": "framework_force_field",
            "_units_pressure": "fugacity_units",
            "_units_loading": "loading_units",
        }

        if user_key_mapper:
            key_mapper.update(user_key_mapper)

        doc = cif.Document()
        doc.add_new_block("isotherm")
        block = doc.sole_block()

        if metadata:
            for key, value in key_mapper.items():
                if value in metadata.keys():
                    if isinstance(metadata[value], (int, float)):
                        block.set_pair(key, str(metadata[value]))
                    else:
                        block.set_pair(key, quote(metadata[value]))

        block.set_pair("_units_loading", quote(self.uptake_unit))
        block.set_pair("_audit_aif_version", quote("63df4e8"))

        df = self.dataframe

        if self.saturation_pressure:
            df["saturation_pressure"] = self.saturation_pressure
            loop_ads = block.init_loop(
                "_adsorp_", ["pressure", "p0", "fugacity", "amount"]
            )
            loop_ads.set_all_values(
                [
                    list(df["pressure"].values.astype(str)),
                    list(df["saturation_pressure"].values.astype(str)),
                    list(df["fugacity"].values.astype(str)),
                    list(df["uptake"].values.astype(str)),
                ]
            )
        else:
            loop_ads = block.init_loop("_adsorp_", ["pressure", "fugacity", "amount"])
            loop_ads.set_all_values(
                [
                    list(df["pressure"].values.astype(str)),
                    list(df["fugacity"].values.astype(str)),
                    list(df["uptake"].values.astype(str)),
                ]
            )

        if filename.endswith(".aif"):
            filename = filename[:-4]
        doc.write_file(f"{filename}.aif")

    def to_csv(self, filename: str) -> None:
        self.dataframe.to_csv(filename)
