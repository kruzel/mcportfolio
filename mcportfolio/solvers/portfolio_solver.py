# Modified by Edward Brandler, based on original files from PyPortfolioOpt and USolver
from typing import Any
from datetime import datetime, timedelta

import os
import asyncio
import pandas as pd
from pypfopt.efficient_frontier.efficient_frontier import EfficientFrontier

import numpy as np
import logging
import sys

from mcportfolio.models.portfolio_base_models import PortfolioProblem

# Configure logging to stderr
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    stream=sys.stderr,  # Redirect logs to stderr
)
logger = logging.getLogger(__name__)


# ── Shared market data provider (lazy init) ───────────────────────────────────

_unified_provider: Any | None = None


def _get_unified_provider():
    """Lazily initialise the shared UnifiedMarketData instance."""
    global _unified_provider
    if _unified_provider is not None:
        return _unified_provider

    from market_data_provider import MarketDataConfig, UnifiedMarketData

    # Mirror guarded_llm's own provider construction (see analytics_server.py): the current
    # market_data_provider API is iTick/FRED-based and no longer accepts the legacy Alpaca kwargs.
    config = MarketDataConfig(
        itick_api_token=os.environ.get("ITICK_API_TOKEN", ""),
        redis_url=os.environ.get("REDIS_URL", ""),
    )
    _unified_provider = UnifiedMarketData(config)
    return _unified_provider


def extract_tickers(task: str) -> list[str]:
    """Extract stock tickers from a task description."""
    words = task.upper().split()
    return [word.strip(",") for word in words if word.isalpha() and 2 <= len(word) <= 5]


def cov_from_nested(cov: dict[str, dict[str, float]], tickers: list[str]) -> pd.DataFrame:
    """Build an ordered (tickers x tickers) covariance DataFrame from a caller-supplied
    nested dict, slicing to exactly ``tickers`` in order. The dict may cover a SUPERSET
    (e.g. the whole research report) — we select the relevant principal sub-matrix.

    Raises ``KeyError`` if any ticker (row or column) is missing — the caller's contract is
    to supply a COMPLETE matrix for the universe, so an incomplete one is a hard error, not a
    silent live-fetch fallback (that decision belongs to the caller, not the solver).
    """
    missing = [t for t in tickers if t not in cov or any(c not in cov.get(t, {}) for c in tickers)]
    if missing:
        raise KeyError(f"covariance override missing entries for: {missing}")
    df = pd.DataFrame(
        [[float(cov[r][c]) for c in tickers] for r in tickers],
        index=tickers, columns=tickers, dtype=float,
    )
    # Symmetrise to wash out any rounding asymmetry from serialisation.
    return (df + df.T) / 2.0


def resolve_optimizer_inputs(problem: PortfolioProblem):
    """Return ``(mean_returns: pd.Series|None, cov_matrix: pd.DataFrame)`` for the problem's
    tickers, preferring caller-supplied overrides and falling back to a live fetch.

    - ``problem.cov_matrix`` set  → slice it to ``problem.tickers`` (no fetch).
    - ``problem.mean_returns`` set → ordered Series over ``problem.tickers`` (no fetch).
    - either missing               → live 2y fetch supplies the missing one.

    The caller owns completeness: if it passes an override it must cover the full universe
    (``cov_from_nested`` raises otherwise). Mixing is allowed (e.g. report cov + live means),
    but in practice the BL/MVO callers pass both or neither.
    """
    tickers = problem.tickers or []
    have_cov = bool(problem.cov_matrix)
    have_mu = bool(problem.mean_returns)

    cov_df: pd.DataFrame | None = None
    mu_series: pd.Series | None = None

    if have_cov:
        cov_df = cov_from_nested(problem.cov_matrix, tickers)
    if have_mu:
        mu_series = pd.Series({t: float(problem.mean_returns[t]) for t in tickers}, dtype=float)

    if cov_df is None or mu_series is None:
        # Live-fetch whatever the caller did not supply.
        data = retrieve_stock_data(tickers=tickers, period="2y")
        if data.get("status") == "error":
            raise ValueError(data.get("message") or "market data fetch failed")
        if cov_df is None:
            cov_df = data["data"]["cov_matrix"].loc[tickers, tickers]
        if mu_series is None:
            mu_series = data["data"]["mean_returns"].reindex(tickers)

    return mu_series, cov_df


def retrieve_stock_data(tickers: list[str], period: str = "1y") -> dict[str, Any]:
    """Retrieve historical stock data via the shared market_data_provider library.

    Delegates to UnifiedMarketData.fetch_stock_data() which handles the full
    Alpaca -> yfinance -> iTick -> Stooq cascade with Redis caching and FX->USD
    conversion for international tickers.

    Returns the same dict format as before::

        {"status": "success", "data": {
            "prices": DataFrame, "returns": DataFrame,
            "mean_returns": Series (x252), "cov_matrix": DataFrame,
            "start_date": str, "end_date": str, "num_days": int}}
    """
    try:
        logger.info("Retrieving data for tickers: %s (period=%s)", tickers, period)

        provider = _get_unified_provider()

        # Run async fetch in sync context (MCP server tools are sync)
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None

        if loop and loop.is_running():
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor() as pool:
                stock_data = pool.submit(
                    asyncio.run, provider.fetch_stock_data(tickers, period)
                ).result()
        else:
            stock_data = asyncio.run(provider.fetch_stock_data(tickers, period))

        prices = stock_data.prices
        returns = stock_data.returns
        mean_returns = stock_data.mean_returns
        cov_matrix = stock_data.cov_matrix

        if prices.empty or len(returns) < 2:
            return {
                "status": "error",
                "message": f"Insufficient data points for tickers: {tickers}. "
                f"Need at least 2 days of data, got {len(returns)}.",
            }

        missing_tickers = [t for t in tickers if t not in returns.columns]
        if missing_tickers:
            return {"status": "error", "message": f"No data available for tickers: {missing_tickers}"}

        if cov_matrix.isnull().any().any():
            return {"status": "error", "message": "Invalid covariance matrix: contains NaN values"}

        if (cov_matrix == 0).all().all():
            return {"status": "error", "message": "Invalid covariance matrix: all values are zero"}

        return {
            "status": "success",
            "data": {
                "prices": prices,
                "returns": returns,
                "mean_returns": mean_returns,
                "cov_matrix": cov_matrix,
                "start_date": stock_data.start_date,
                "end_date": stock_data.end_date,
                "num_days": stock_data.num_days,
            },
        }

    except Exception as e:
        logger.error("Error in retrieve_stock_data: %s", e, exc_info=True)
        return {"status": "error", "message": f"Error retrieving stock data: {e!s}"}

def solve_problem(problem: PortfolioProblem) -> dict[str, Any]:
    """Solve a portfolio optimization problem."""
    try:
        tickers = problem.tickers
        if not tickers:
            return {"status": "error", "message": "No tickers provided in problem description"}
        # Prefer caller-supplied returns + covariance (sliced from the daily research report);
        # fall back to a live 2y fetch for anything not supplied. See resolve_optimizer_inputs.
        try:
            mean_returns, cov_matrix = resolve_optimizer_inputs(problem)
        except (ValueError, KeyError) as e:
            return {"status": "error", "message": str(e)}

        # Parse constraints
        max_weight = 0.5
        max_volatility: float | None = None
        sector_limits = {}
        if problem.constraints:
            for constraint in problem.constraints:
                if constraint.startswith("max_weight"):
                    max_weight = float(constraint.split()[1])
                elif constraint.startswith("max_volatility"):
                    max_volatility = float(constraint.split()[1])
                elif constraint.startswith("sector_"):
                    sector, limit = constraint.split()
                    sector_limits[sector] = float(limit)

        # Feasibility floor on the per-name cap: with n assets each capped at max_weight, the
        # achievable total is max_weight * n, so a cap below 1/n makes the long-only,
        # fully-invested problem infeasible (min_volatility would raise "infeasible_inaccurate").
        # Raise the cap to just above 1/n (15% headroom keeps OSQP off the degenerate boundary,
        # where even an exact max_weight*n == 1.0 reports infeasible_inaccurate). Never lowers a
        # caller's cap — only loosens one that is mathematically impossible for this universe.
        n_assets = len(tickers)
        if n_assets > 0:
            min_feasible = min(1.0, (1.0 / n_assets) * 1.15)
            if max_weight < min_feasible:
                logger.info(
                    "max_weight %.3f infeasible for %d assets (need >= %.3f); raising to keep "
                    "the optimisation feasible", max_weight, n_assets, min_feasible,
                )
                max_weight = min_feasible

        weight_bounds = (0, max_weight)

        sectors = {
            "tech": ["AAPL", "MSFT", "NVDA", "GOOGL", "META"],
            "fin": ["JPM", "V", "BAC", "GS", "AXP"],
            "health": ["JNJ", "UNH", "PFE", "MRK", "ABBV"],
            "cons": ["MCD", "PG", "KO", "WMT", "SBUX"],
            "energy": ["XOM", "CVX"],
        }

        def _apply_sector_constraints(optimizer: EfficientFrontier) -> None:
            if not sector_limits:
                return
            for sector, limit in sector_limits.items():
                if sector in sectors:
                    sector_tickers = sectors[sector]
                    optimizer.add_sector_constraints({sector: sector_tickers}, {sector: limit})

        # 1. Minimum variance portfolio. This is an AUXILIARY result (the primary plan is the
        # max-Sharpe portfolio below); a numerically-degenerate solve here must not abort the
        # whole optimisation. On failure, fall back to an equal-weight min-var stand-in so the
        # caller still gets a usable plan instead of a hard error.
        min_var_weights_clean: dict[str, float]
        try:
            ef_min_var = EfficientFrontier(mean_returns, cov_matrix, weight_bounds=weight_bounds)
            _apply_sector_constraints(ef_min_var)
            ef_min_var.min_volatility()
            min_var_perf = ef_min_var.portfolio_performance(verbose=False)
            min_var_weights_clean = ef_min_var.clean_weights()
        except Exception as e:
            logger.warning("min_volatility failed (%s); using equal-weight min-var fallback", e)
            ew = {t: 1.0 / n_assets for t in tickers}
            ef_min_var = EfficientFrontier(mean_returns, cov_matrix, weight_bounds=(0, 1.0))
            ef_min_var.set_weights(ew)
            min_var_perf = ef_min_var.portfolio_performance(verbose=False)
            min_var_weights_clean = ef_min_var.clean_weights()

        # 2. Maximum return portfolio
        max_ret_idx = np.argmax(mean_returns.values)
        max_ret_weights = np.zeros(len(tickers))
        max_ret_weights[max_ret_idx] = 1.0
        max_ret_weights_dict = dict(zip(tickers, max_ret_weights, strict=True))
        ef_max_ret = EfficientFrontier(mean_returns, cov_matrix, weight_bounds=weight_bounds)
        _apply_sector_constraints(ef_max_ret)
        ef_max_ret.set_weights(max_ret_weights_dict)
        max_ret_perf = ef_max_ret.portfolio_performance(verbose=False)
        max_ret_weights_clean = ef_max_ret._make_output_weights(max_ret_weights)

        # 3a. Volatility-capped plan (max_volatility constraint). When the caller asks the plan to
        # stay under a volatility ceiling (an in-band rebuild for a conservative risk profile), the
        # objective is "maximise return SUBJECT TO vol <= cap" — efficient_risk, not max_sharpe.
        # PyPortfolioOpt's max_sharpe cannot take a volatility cap, so without this the cap was
        # silently dropped and we always returned the unconstrained max-Sharpe plan (often above the
        # ceiling). If the cap is below this universe's minimum achievable volatility, no long-only
        # mix can satisfy it — report INFEASIBLE (with that floor) so the caller can offer to widen
        # the universe or hold off, rather than returning an over-ceiling plan that gets flagged
        # downstream and tempts a fabricated allocation.
        if max_volatility is not None:
            min_achievable_vol = float(min_var_perf[1])
            # Small tolerance so a cap numerically equal to the min-var floor still solves.
            if max_volatility < min_achievable_vol - 1e-4:
                return {
                    "status": "infeasible",
                    "message": (
                        "No long-only allocation of the current assets can stay under the "
                        f"{max_volatility:.1%} volatility ceiling — the lowest achievable "
                        f"volatility for this universe is {min_achievable_vol:.1%}. "
                        "Widen the universe with additional assets, or hold off."
                    ),
                    "max_volatility": max_volatility,
                    "min_achievable_volatility": min_achievable_vol,
                    "tickers": tickers,
                }
            ef_capped = EfficientFrontier(mean_returns, cov_matrix, weight_bounds=weight_bounds)
            _apply_sector_constraints(ef_capped)
            try:
                ef_capped.efficient_risk(target_volatility=max_volatility)
                capped_perf = ef_capped.portfolio_performance(verbose=False, risk_free_rate=0.0)
                capped_weights_clean = ef_capped.clean_weights()
            except Exception:
                # Numerically unable to hit the cap from above even though it sits at/above the
                # min-var floor (degenerate boundary): fall back to the min-variance plan, which is
                # the lowest-vol point and therefore the safest mix that respects the ceiling.
                ef_capped = EfficientFrontier(mean_returns, cov_matrix, weight_bounds=weight_bounds)
                _apply_sector_constraints(ef_capped)
                ef_capped.min_volatility()
                capped_perf = ef_capped.portfolio_performance(verbose=False, risk_free_rate=0.0)
                capped_weights_clean = ef_capped.clean_weights()
            return {
                "status": "success",
                "data": {
                    "weights": capped_weights_clean,
                    "expected_return": capped_perf[0],
                    "risk": capped_perf[1],
                    "sharpe_ratio": capped_perf[2],
                    "min_variance_portfolio": {
                        "weights": min_var_weights_clean,
                        "expected_return": min_var_perf[0],
                        "risk": min_var_perf[1],
                    },
                    "max_return_portfolio": {
                        "weights": max_ret_weights_clean,
                        "expected_return": max_ret_perf[0],
                        "risk": max_ret_perf[1],
                    },
                },
            }

        # 3b. Max Sharpe ratio portfolio (no volatility cap)
        rf = 0.0
        ef_sharpe = EfficientFrontier(mean_returns, cov_matrix, weight_bounds=weight_bounds)
        _apply_sector_constraints(ef_sharpe)
        try:
            ef_sharpe.max_sharpe(risk_free_rate=rf)
            sharpe_perf = ef_sharpe.portfolio_performance(verbose=False, risk_free_rate=rf)
        except Exception:
            ef_sharpe = EfficientFrontier(mean_returns, cov_matrix, weight_bounds=weight_bounds)
            _apply_sector_constraints(ef_sharpe)
            ef_sharpe.min_volatility()
            sharpe_perf = ef_sharpe.portfolio_performance(verbose=False, risk_free_rate=0.0)

        # Clean weights
        sharpe_weights_clean = ef_sharpe.clean_weights()

        return {
            "status": "success",
            "data": {
                "weights": sharpe_weights_clean,
                "expected_return": sharpe_perf[0],
                "risk": sharpe_perf[1],
                "sharpe_ratio": sharpe_perf[2],
                "min_variance_portfolio": {
                    "weights": min_var_weights_clean,
                    "expected_return": min_var_perf[0],
                    "risk": min_var_perf[1],
                },
                "max_return_portfolio": {
                    "weights": max_ret_weights_clean,
                    "expected_return": max_ret_perf[0],
                    "risk": max_ret_perf[1],
                },
            },
        }
    except Exception as e:
        logger.error(f"Error in solve_problem: {e!s}", exc_info=True)
        return {"status": "error", "message": f"Error solving portfolio problem: {e!s}"}
