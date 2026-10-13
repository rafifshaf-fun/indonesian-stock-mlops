"""Task 1.2 — the extracted feature builder must be pure.

``build_features`` is the single feature builder both the training path and the
serving path are meant to share, so it must do no I/O at all: every external
value arrives as a keyword argument.  The test below proves that by replacing
each fetcher the pipeline knows about with a function that raises, and then
calling ``build_features`` directly.
"""


def test_build_features_does_no_io(raw_ohlcv, ihsg_frame, fundamentals_dict, monkeypatch):
    import features.pipeline as p

    def explode(*a, **k):
        raise AssertionError("build_features must not fetch")

    monkeypatch.setattr(p, "fetch_fundamentals", explode)
    monkeypatch.setattr(p, "fetch_news_sentiment", explode)
    monkeypatch.setattr(p, "fetch_idx_fundamentals", explode)
    monkeypatch.setattr(p, "fetch_google_trends", explode)

    out = p.build_features(
        raw_ohlcv, "BBCA.JK",
        fundamentals=fundamentals_dict, usdidr=None, fred_macro=None, bi_rate=None,
        ihsg=ihsg_frame, all_ticker_data={"BBCA.JK": raw_ohlcv["BBCA.JK"].dropna()},
        news_sentiment=0.2, google_trend=0.0, vp_features={}, idx_fundamentals={},
        mode="full", with_target=True,
    )
    assert out is not None and len(out) > 50
    assert "target" in out.columns
