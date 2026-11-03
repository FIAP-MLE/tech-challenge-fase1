"""Tests for raw-to-model feature translation."""

import pandas as pd
import pytest

from src.utils import parameter_encoding
from src.utils.parameter_encoding import ENCODED_COLUMNS, encode_parameters


def test_encoder_preserves_champion_feature_order(request_model, monkeypatch):
    monkeypatch.setattr(
        parameter_encoding,
        "_city_frequencies",
        lambda: pd.Series({"Los Angeles": 0.25}),
    )

    encoded = encode_parameters(request_model)

    assert list(encoded.columns) == ENCODED_COLUMNS
    assert encoded.shape == (1, len(ENCODED_COLUMNS))
    assert encoded.loc[0, "Internet Service_Fiber optic"] == 1
    assert encoded.loc[0, "City_Frequency"] == pytest.approx(0.25)


def test_encoder_rejects_unknown_city(request_model, monkeypatch):
    monkeypatch.setattr(
        parameter_encoding,
        "_city_frequencies",
        lambda: pd.Series({"Other City": 1.0}),
    )

    with pytest.raises(ValueError, match="not present in the training dataset"):
        encode_parameters(request_model)
