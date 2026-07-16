"""Guard: the seed weights in config.DEFAULT_WEIGHTS must match the actual Excel
weight vectors (extracted into the golden fixture). This catches hand-transcription
drift between the spec and the source workbook — the class of bug that made SC+ wrong."""
import config


def test_default_weights_match_excel(golden):
    isins = golden["isins"]
    for code in config.PORTFOLIO_CODES:
        cfg = dict(zip(isins, config.DEFAULT_WEIGHTS[code]))
        excel = {k: (v or 0.0) for k, v in golden["portfolios"][code]["weights"].items()}
        for isin in isins:
            assert abs(cfg[isin] - excel[isin]) < 1e-9, (
                f"{code}/{isin}: config weight {cfg[isin]} != Excel {excel[isin]}")


def test_default_weights_sum_to_one():
    for code, w in config.DEFAULT_WEIGHTS.items():
        assert abs(sum(w) - 1.0) < 1e-9, f"{code} weights do not sum to 1"


def test_eligibility_derived_from_nonzero_weights():
    for code in config.PORTFOLIO_CODES:
        elig = set(config.PORTFOLIO_ELIGIBILITY[code])
        nonzero = {i for i, w in zip(config.FUND_UNIVERSE, config.DEFAULT_WEIGHTS[code]) if w > 0}
        assert elig == nonzero
