"""Provides the Isotherm class to store, manipulate, plot and save adsorption isotherm data."""
from __future__ import annotations

from typing import TYPE_CHECKING

import pandas as pd

if TYPE_CHECKING:
    from typing import Any, Dict, Optional, Union

    from numpy.typing import ArrayLike
    from plotly.graph_objects import Figure

from .utils import quote


class Isotherm:
    """Isotherm class to store, recalculate and save the adsorption isotherm."""

    def __init__(
        self,
        data: pd.DataFrame = None,
        saturation_fugacity: Optional[float] = None,
        metadata: Optional[dict[str, Any]] = None,
        fugacity_unit: str = "Pa",
        uptake_unit: str = "molecules/unitcell",
    ) -> None:
        """Initialize the Isotherm object.

        Parameters
        ----------
        data
            A pandas DataFrame containing the adsorption data. Should contain 'pressure' and 'uptake' columns.
        saturation_fugacity
            The saturation pressure at given conditions. Used to calculate the relative pressure.
        metadata
            A dictionary with the simulation metadata.
        uptake_unit
            Units at which uptake is stored. Default value is 'molecules/unitcell'.
        """
        self.dataframe = data
        self.saturation_fugacity = saturation_fugacity
        self._metadata = {}
        self.metadata = metadata
        self._pressure_unit = fugacity_unit
        self._uptake_unit = uptake_unit

    @property
    def dataframe(self) -> pd.DataFrame:
        """Return dataframe with isotherm data."""
        return self._dataframe

    @dataframe.setter
    def dataframe(self, dataframe: pd.DataFrame) -> None:
        self._dataframe = dataframe

    @property
    def pressure(self) -> Union[pd.Series, None]:
        """Return the pressure column."""
        if "pressure" in self._dataframe.columns:
            return self._dataframe["pressure"]
        else:
            return None

    @pressure.setter
    def pressure(self, pressure: ArrayLike) -> None:
        self._dataframe["pressure"] = pressure

    @property
    def fugacity(self) -> Union[pd.Series, None]:
        """Return the fugacity column."""
        if "fugacity" in self._dataframe.columns:
            return self._dataframe["fugacity"]
        else:
            return None

    @property
    def saturation_fugacity(self) -> float:
        """Return the saturation fugacity."""
        return self._saturation_fugacity

    @saturation_fugacity.setter
    def saturation_fugacity(self, saturation_fugacity: float | None) -> None:
        self._saturation_fugacity = saturation_fugacity
        if saturation_fugacity is not None:
            self._dataframe["f/f0"] = self.fugacity / saturation_fugacity

    @property
    def amount_adsorbed(self) -> pd.Series:
        """Return the uptake column."""
        return self.dataframe["uptake"]

    @property
    def metastable_gas(self) -> Optional[pd.Series]:
        """Return the metastable gas column, if it exists."""
        if "metastable_gas" in self._dataframe.columns:
            return self._dataframe["metastable_gas"]
        else:
            return None

    @property
    def metastable_liq(self) -> Optional[pd.Series]:
        """Return the metastable liquid column, if it exists."""
        if "metastable_liq" in self._dataframe.columns:
            return self._dataframe["metastable_liq"]
        else:
            return None

    @property
    def pressure_unit(self) -> str:
        """Return the current pressure unit."""
        return self._pressure_unit

    def set_pressure_unit(self, target_unit: str, conversion_factor: float) -> None:
        """Convert the pressure column to the target unit using the provided conversion factor."""
        self.pressure = self.pressure * conversion_factor
        self._pressure_unit = target_unit

    @property
    def uptake_unit(self) -> str:
        """Return the current uptake unit."""
        return self._uptake_unit

    def set_uptake_unit(
        self, target_unit: str, conversion_factor: Optional[float] = None
    ) -> None:
        """Convert the uptake column to the target unit."""
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
        """Return the metadata dictionary."""
        return self._metadata

    @metadata.setter
    def metadata(self, metadata: Dict[str, Any]) -> None:
        if metadata:
            for k, v in metadata.items():
                self._metadata[k] = v

    def plot(
        self,
        label: Optional[str] = None,
        fig: Optional[Figure] = None,
        x_axis: str = "fugacity",
        y_axis: str = "molecules/unitcell",
        trace_kwargs: Optional[Dict[str, Any]] = None,
        layout_kwargs: Optional[Dict[str, Any]] = None,
    ) -> Figure:
        """Plot an isotherm (stable + metastable gas and / or liquid) and group all traces under a single legend entry.

        You can pass any kwargs through `trace_kwargs` or `layout_kwargs`; missing values will be filled in by defaults.
        """
        import plotly.colors as pc
        import plotly.graph_objects as go
        
        # get or create figure
        show_immediately = False
        if fig is None:
            fig = go.Figure()
            show_immediately = True

        trace_kwargs = trace_kwargs or {}
        layout_kwargs = layout_kwargs or {}

        # look for an explicit color in trace_kwargs
        explicit_color = None
        if "line" in trace_kwargs and isinstance(trace_kwargs["line"], dict):
            explicit_color = trace_kwargs["line"].get("color")

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
            "relative_fugacity": (self._dataframe["f/f0"], "f/f₀"),
            "f/f0": (self._dataframe["f/f0"], "f/f₀"),
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
        """Save the isotherm in an AIF file format.

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
        from gemmi import cif
        
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

        if self.saturation_fugacity:
            df["saturation_pressure"] = self.saturation_fugacity
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
        """Save the isotherm data to a CSV file."""
        self.dataframe.to_csv(filename)
