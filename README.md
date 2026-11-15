# Confident to the Last Hex Digit
**Do LLMs know when a recalled contract address is wrong?**
Apart Research — AI x Epistemics Research Sprint (13–15 Nov 2026), Track 1. All code and data in this repo were created during the sprint.

Agent wallets increasingly decide whether to *trust* a token contract address recalled from model memory or to *look it up*. A single wrong hex digit sends funds somewhere irreversible. This repo measures:

1. **Accuracy** of recalled ERC-20 addresses (exact match, case-insensitive) on Ethereum and Base, stratified by market-cap rank.
2. **Calibration** of the model's stated 0–100 confidence (ECE, Brier, reliability diagram) and the *confident-wrong* rate (wrong with confidence ≥ 90).
3. **Which signal predicts errors** (AUROC, 95% token-level bootstrap CIs): verbal confidence; self-consistency (share of T=1 samples matching the greedy answer); the free EIP-55 mixed-case checksum (mixed-case outputs only); an out-of-fold logistic combination; baseline: market-cap rank alone.
4. **Where wrong answers would send funds**, checked on-chain: same token on another chain / another known token / some other contract / a used EOA / a never-used address (funds burned) / invalid format.

Zero-budget design: only free tiers (Google Gemini API incl. Gemma, Groq). The client refuses any non-free model, throttles to each tier's RPM/RPD, and caches every response on disk, so a run interrupted by a daily quota resumes by rerunning the same command.

## Quickstart ("do X → see Y")
```bash
cp .env.example .env            # put GEMINI_API_KEY / GROQ_API_KEY (free tiers) in .env
./reproduce.sh pilot            # → results/pilot_metrics.json, results/pilot_destinations.csv, figures/pilot_*.png
./reproduce.sh                  # full run (resumable across days) → results/full_*, figures/full_*
python -m pytest -q tests       # → scoring/parsing unit tests on synthetic data
```
Choose models with `HEX_MODELS=gemini-2.5-flash,gpt-oss-120b ./reproduce.sh`; see `src/hexeval/config.py`.

## Pipeline
| step | script | output |
|---|---|---|
| sources | `scripts/fetch_sources.py`, `scripts/fetch_markets.py` | `data/raw/` (CoinGecko list + ranks, Uniswap, CMC, 1inch, CoinGecko Base, Superchain lists) |
| dataset | `scripts/build_dataset.py` | `data/pool.csv` (all cross-validated candidates), `data/items.csv` (sample), `data/dataset_summary.json` |
| queries | `scripts/run_eval.py [--pilot 30]` | `results/*responses.csv`, cache in `cache/llm/` |
| scoring | `scripts/score.py --prefix pilot|full` | `results/*_per_item.csv`, `*_metrics.json/csv` |
| on-chain | `scripts/classify_wrong.py --prefix …` | `results/*_destinations.csv` (aggregate only) |
| figures | `scripts/figures.py --prefix …` | reliability, AUROC, accuracy-by-stratum, destinations |

**Dataset rules.** An Ethereum token is kept only if CoinGecko's platform address also appears, with the same symbol, in the Uniswap Labs default list (`strict`) or, as a fallback, the CoinMarketCap/1inch lists (`extended`); addresses shared by 2+ CoinGecko coins are dropped. Base tokens must agree between CoinGecko's coin list and its Base token list; those also in the Uniswap/Superchain lists are `strict` and sampled first. Strata: rank ≤100, 101–1000, >1000 (Ethereum) and Base; fixed seed, nested samples. `shared_symbol` flags tickers used by 2+ CoinGecko coins. See `data/dataset_summary.json` for the actual counts (only 53 Ethereum ERC-20s in the top 100 survive cross-validation, because most top-100 assets are native L1 coins).

**Prompting.** 3 paraphrases at T=0 (paraphrase 0 is the primary answer), plus 3 samples at T=1 on paraphrase 0 for self-consistency. Reasoning is disabled where the API allows (`reasoning_effort: none`), otherwise low with a larger token cap.

## Ethics
Per-token lists of the wrong contracts that models confidently output (potential impostor tokens) are written only to `private/` (gitignored). Published results are aggregates.

## License
MIT (see LICENSE). Data sources remain under their own terms.

## What is and isn't in this repo

Published: all code, the dataset builder and the final token list (`data/items.csv`, public CoinGecko/Uniswap data), aggregate results (`results/*_metrics.*`, `*_destinations.csv`, `*_extras.json`, subset definitions), figures and the paper (`report/`).

Not published, on purpose: raw model responses and per-item predictions, because they would amount to a list of the specific wrong (sometimes impostor) contracts each model recommends. `./reproduce.sh` regenerates them locally from free-tier APIs; the on-disk cache makes reruns free.
