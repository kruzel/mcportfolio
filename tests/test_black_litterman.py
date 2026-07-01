import logging
from mcportfolio.models.portfolio_black_litterman_models import BlackLittermanProblem, BlackLittermanView
from mcportfolio.solvers.black_litterman_solver import solve_black_litterman_problem
# import numpy as np
import pandas as pd
from mcportfolio.solvers.portfolio_solver import retrieve_stock_data

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_black_litterman() -> None:
    """Test Black-Litterman portfolio optimization with specified views."""
    
    # Define investor views
    views = [
        BlackLittermanView(
            asset="META",
            expected_return=0.05,  # +5% outperformance
            confidence=0.8  # High confidence in tech view
        ),
        BlackLittermanView(
            asset="EOG",
            expected_return=-0.03,  # -3% underperformance
            confidence=0.7  # Moderate confidence in energy view
        )
    ]
    
    # Create Black-Litterman problem
    problem = BlackLittermanProblem(
        description="Test Black-Litterman with two views",
        tickers=["META", "ABBV", "EOG", "SBUX", "BAC", "PLD"],
        views=views,
        risk_free_rate=0.04,
        tau=0.05,
        risk_aversion=2.5,
        min_weight=0.0,  # Relaxed lower bound
        max_weight=1.0   # Relaxed upper bound
    )
    
    # Solve the problem
    result = solve_black_litterman_problem(problem)
    
    if result["status"] == "success":
        data = result["data"]
        
        # Debug output: print Black-Litterman returns and covariance matrix
        print("\nBlack-Litterman Returns:")
        for k, v in data["black_litterman_returns"].items():
            print(f"  {k}: {v:.6f}")
        print("\nCovariance Matrix:")
        tickers = problem.tickers
        cov_matrix = retrieve_stock_data(tickers=tickers, period="2y")["data"]["cov_matrix"]
        print(pd.DataFrame(cov_matrix))
        
        # Display Black-Litterman returns
        logger.info("\nBlack-Litterman Posterior Returns:")
        logger.info("----------------------------------------")
        for ticker, ret in data["black_litterman_returns"].items():
            logger.info(f"{ticker}: {ret:.2%}")
        
        # Display optimal portfolio
        logger.info("\nOptimal Portfolio (Max Sharpe Ratio):")
        logger.info("----------------------------------------")
        for ticker, weight in data["weights"].items():
            logger.info(f"{ticker}: {weight:.2%}")
        logger.info(f"\nExpected Annual Return: {data['expected_return']:.2%}")
        logger.info(f"Annual Volatility: {data['risk']:.2%}")
        logger.info(f"Sharpe Ratio: {data['sharpe_ratio']:.2f}")
        
        # Display minimum variance portfolio
        min_var = data["min_variance_portfolio"]
        logger.info("\nMinimum Variance Portfolio:")
        logger.info("----------------------------------------")
        for ticker, weight in min_var["weights"].items():
            logger.info(f"{ticker}: {weight:.2%}")
        logger.info(f"\nExpected Annual Return: {min_var['expected_return']:.2%}")
        logger.info(f"Annual Volatility: {min_var['risk']:.2%}")
        
        # Display maximum return portfolio
        max_ret = data["max_return_portfolio"]
        logger.info("\nMaximum Return Portfolio:")
        logger.info("----------------------------------------")
        for ticker, weight in max_ret["weights"].items():
            logger.info(f"{ticker}: {weight:.2%}")
        logger.info(f"\nExpected Annual Return: {max_ret['expected_return']:.2%}")
        logger.info(f"Annual Volatility: {max_ret['risk']:.2%}")
    else:
        logger.error(f"Error: {result['message']}")

def test_per_asset_bounds_pin_banded_asset() -> None:
    """A per-asset [low, high] bound must confine that asset's weight to its box, overriding the
    scalar (min_weight, max_weight) — this is how build_bl_inputs pins a view-less holding to a
    drift band so max-Sharpe can't swing it on covariance alone. Offline: a caller-supplied PSD
    covariance keeps the solve deterministic and network-free."""
    tickers = ["A", "B", "C", "D"]
    # Diagonal-dominant PSD covariance (annualised). C is the low-vol asset the optimiser would
    # otherwise load as ballast; the band must stop it.
    var = {"A": 0.04, "B": 0.05, "C": 0.0016, "D": 0.06}
    cov = {r: {c: (var[r] if r == c else 0.002) for c in tickers} for r in tickers}
    problem = BlackLittermanProblem(
        description="per-asset bounds",
        tickers=tickers,
        views=[BlackLittermanView(asset="A", expected_return=0.08, confidence=0.6)],
        market_cap_weights={t: 0.25 for t in tickers},
        cov_matrix=cov,
        min_weight=0.0,
        max_weight=0.5,
        # C is pinned to a tight band around its current 25%; the others stay free (0, 0.5).
        per_asset_bounds={"C": [0.20, 0.30]},
    )
    result = solve_black_litterman_problem(problem)
    assert result["status"] == "success", result.get("message")
    w = result["data"]["weights"]
    # C must respect its box (small tolerance for clean_weights rounding).
    assert 0.20 - 1e-3 <= w["C"] <= 0.30 + 1e-3
    # An unbanded asset may exceed C's ceiling — proving the band is per-asset, not global.
    assert max(w["A"], w["B"], w["D"]) <= 0.5 + 1e-3


if __name__ == "__main__":
    test_black_litterman()
    test_per_asset_bounds_pin_banded_asset()