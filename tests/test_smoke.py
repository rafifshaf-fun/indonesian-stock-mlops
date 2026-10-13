"""Smoke tests: the harness resolves imports and the fixtures have shape."""


def test_imports_resolve():
    import config, features  # noqa: F401
    assert True


def test_fixture_shape(raw_ohlcv):
    assert raw_ohlcv.shape[1] == 3 * 5  # 3 tickers x OHLCV
