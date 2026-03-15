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

    config = MarketDataConfig(
        alpaca_api_key=os.environ.get("ALPACA_API_KEY", ""),
        alpaca_api_secret=os.environ.get("ALPACA_SECRET_KEY", ""),
        alpaca_market_data_key=os.environ.get("ALPACA_MARKET_DATA_KEY", ""),
        alpaca_market_data_secret=os.environ.get("ALPACA_MARKET_DATA_SECRET", ""),
        itick_api_token=os.environ.get("ITICK_API_TOKEN", ""),
        redis_url=os.environ.get("REDIS_URL", ""),
    )
    _unified_provider = UnifiedMarketData(config)
    return _unified_provider


def extract_tickers(task: str) -> list[str]:
    """Extract stock tickers from a task description."""
    words = task.upper().split()
    return [word.strip(",") for word in words if word.isalpha() and 2 <= len(word) <= 5]


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
        data = retrieve_stock_data(tickers=tickers, period="2y")
        if data.get("status") == "error":
            return data
        # prices = data['data']['prices']
        mean_returns = data["data"]["mean_returns"]
        cov_matrix = data["data"]["cov_matrix"]

        # Parse constraints
        max_weight = 0.5
        sector_limits = {}
        if problem.constraints:
            for constraint in problem.constraints:
                if constraint.startswith("max_weight"):
                    max_weight = float(constraint.split()[1])
                elif constraint.startswith("sector_"):
                    sector, limit = constraint.split()
                    sector_limits[sector] = float(limit)

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

        # 1. Minimum variance portfolio
        ef_min_var = EfficientFrontier(mean_returns, cov_matrix, weight_bounds=weight_bounds)
        _apply_sector_constraints(ef_min_var)
        ef_min_var.min_volatility()
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

        # 3. Max Sharpe ratio portfolio
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
        max_ret_weights_clean = ef_max_ret._make_output_weights(max_ret_weights)
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
