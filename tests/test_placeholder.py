def test_imports():
    import pandas
    import numpy
    import scipy
    import statsmodels
    import sklearn
    import plotly
    import matplotlib


def test_config_constants():
    import sys
    import os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
    import config
    assert config.Z_SCORE_THRESHOLD == 3.0
    assert config.COST_CURRENCY == "USD"
