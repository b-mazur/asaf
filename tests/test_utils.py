from __future__ import annotations

import pytest

from asaf.utils import (
    beta_to_temperature,
    fugacity_to_mu,
    mu_to_fugacity,
    quote,
    temperature_to_beta,
)


def test_quote_wraps_string_in_single_quotes() -> None:
    assert quote("example") == "'example'"

def test_temperature_beta_roundtrip() -> None:
    temperature = 350.0
    beta = temperature_to_beta(temperature)
    assert beta_to_temperature(beta) == pytest.approx(temperature)

def test_fugacity_mu_roundtrip() -> None:
    temperature = 320.0
    beta = temperature_to_beta(temperature)
    fugacity = 1.5e5
    mu = fugacity_to_mu(fugacity, beta)
    assert mu_to_fugacity(mu, beta) == pytest.approx(fugacity)
