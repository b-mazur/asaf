from __future__ import annotations

from asaf.utils import quote


def test_quote_wraps_string_in_single_quotes() -> None:
    assert quote("example") == "'example'"
