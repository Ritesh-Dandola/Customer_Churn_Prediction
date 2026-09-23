import pytest
import pandas as pd
from src.predict import assign_risk_level

def test_assign_risk_level():
    assert assign_risk_level(0.25) == 'LOW'
    assert assign_risk_level(0.45) == 'MEDIUM'
    assert assign_risk_level(0.85) == 'HIGH'

def test_data_loader_import():
    # Simple test to ensure module can be imported
    try:
        from src.data_loader import load_data
        assert True
    except ImportError:
        pytest.fail("Could not import load_data")

def test_preprocessing_pipeline():
    from src.preprocessing import get_preprocessor
    preprocessor = get_preprocessor()
    assert preprocessor is not None
    # We can add more specific tests checking the transformers inside

