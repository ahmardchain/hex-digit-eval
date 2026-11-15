import sys, numpy as np, pandas as pd
sys.path.insert(0, "src")
from hexeval import scoring as S
from hexeval.parse import parse_reply
from hexeval.llm import cache_key

USDT = "0xdAC17F958D2ee523a2206206994597C13D831ec7"


def test_exact_match_case_insensitive():
    assert S.exact_match(USDT.lower(), USDT) and not S.exact_match(None, USDT)


def test_eip55():
    assert S.eip55_status(USDT) == "valid"
    assert S.eip55_status(USDT.lower()) == "not_mixed"
    bad = USDT[:3] + USDT[3].swapcase() + USDT[4:]
    assert S.eip55_status(bad) == "invalid"


def test_ece_perfect_and_bad():
    rng = np.random.default_rng(0)
    c = rng.random(20000); y = rng.random(20000) < c
    assert S.ece(c, y) < 0.02
    assert abs(S.ece(np.full(1000, 0.9), np.zeros(1000)) - 0.9) < 1e-9
    assert abs(S.brier(np.ones(10), np.ones(10))) < 1e-12


def test_auroc():
    assert S.auroc([0.1, 0.2, 0.8, 0.9], [0, 0, 1, 1]) == 1.0
    assert np.isnan(S.auroc([0.1, 0.2], [1, 1]))


def test_self_consistency():
    assert S.self_consistency("0xAb", ["0xab", "0xAB", "0xcd"]) == 2 / 3
    assert np.isnan(S.self_consistency(None, ["0xab"]))


def test_random_checksum_rate_is_small():
    assert S.random_checksum_pass_rate(20000) < 0.01
    assert S.random_case_flip_pass_rate([USDT], 50) == 0.0


def test_logistic_cv_learns_signal():
    rng = np.random.default_rng(1)
    x = rng.random(400); y = (x + rng.normal(0, .2, 400)) > .5
    p = S.logistic_cv_scores(x[:, None], y, np.arange(400) // 2)
    assert S.auroc(p, y) > 0.8


def test_bootstrap_ci_brackets_point():
    df = pd.DataFrame({"item_id": np.repeat(np.arange(100), 2), "c": np.random.default_rng(2).random(200) < .3})
    p, lo, hi = S.bootstrap_ci(df, lambda d: d.c.mean(), n_boot=300)
    assert lo <= p <= hi and hi - lo < 0.3


def test_parser_variants():
    assert parse_reply('```json\n{"address": "%s", "confidence_0_100": 90}\n```' % USDT)["confidence"] == 90
    assert parse_reply('{"abstain": true}')["abstain"]
    assert not parse_reply('<think>{"abstain": true}</think>{"address":"%s","confidence_0_100":5}' % USDT)["abstain"]
    r = parse_reply('{"address": "0x123", "confidence_0_100": 50}')
    assert r["parse_ok"] and not r["valid_format"] and r["address"] is None
    assert not parse_reply("I don't know")["parse_ok"]


def test_cache_key_distinguishes_samples():
    assert cache_key("m", "p", 1.0, 0, 100) != cache_key("m", "p", 1.0, 1, 100)
    assert cache_key("m", "p", 0.0, 0, 100) == cache_key("m", "p", 0, 0, 100)
