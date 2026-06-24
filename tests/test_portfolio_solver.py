import unittest
from mcportfolio.models.portfolio_base_models import PortfolioProblem
from mcportfolio.solvers.portfolio_solver import retrieve_stock_data, solve_problem

class TestPortfolioSolver(unittest.TestCase):
    def setUp(self) -> None:
        self.test_tickers = ["META", "ABBV", "EOG", "SBUX", "BAC", "PLD"]

    def test_retrieve_stock_data_insufficient_data(self) -> None:
        """Test that retrieve_stock_data returns an error when insufficient data points are available."""
        result = retrieve_stock_data(self.test_tickers, period="1d")
        self.assertEqual(result["status"], "error")
        # Check for error messages that indicate data retrieval failure
        self.assertTrue(
            "Insufficient data points for tickers" in result["message"] or
            "Error retrieving stock data" in result["message"] or
            "Unable to retrieve real market data" in result["message"],
            f"Unexpected error message: {result['message']}"
        )


class TestMaxVolatilityConstraint(unittest.TestCase):
    """The max_volatility constraint must actually bind. It used to be parsed-and-dropped, so the
    solver always returned the unconstrained max-Sharpe plan — letting an in-band rebuild come back
    above the customer's volatility ceiling (then get flagged downstream and tempt a fabricated
    allocation). A low-vol asset + a high-vol asset with annualised inputs supplied directly."""

    def setUp(self) -> None:
        self.tickers = ["BNDW", "VT"]
        self.mean = {"BNDW": 0.025, "VT": 0.08}
        # annualised covariance: BNDW vol ~5.7%, VT vol ~16%
        self.cov = {
            "BNDW": {"BNDW": 0.0033, "VT": 0.002},
            "VT": {"BNDW": 0.002, "VT": 0.026},
        }

    def _solve(self, constraints):
        return solve_problem(PortfolioProblem(
            description="t", tickers=self.tickers, constraints=constraints,
            objective="maximize_sharpe_ratio", cov_matrix=self.cov, mean_returns=self.mean,
        ))

    def test_cap_binds_when_feasible(self) -> None:
        """With a generous weight cap a 7% ceiling is reachable — the plan must respect it."""
        r = self._solve(["max_weight 1.0", "max_volatility 0.07"])
        self.assertEqual(r["status"], "success")
        self.assertLessEqual(r["data"]["risk"], 0.07 + 1e-3)

    def test_infeasible_cap_reports_floor(self) -> None:
        """When no long-only mix can meet the cap (a tight weight cap forces high-vol exposure),
        the solver must report status:infeasible with the achievable floor — never an over-cap plan."""
        r = self._solve(["max_weight 0.6", "max_volatility 0.07"])
        self.assertEqual(r["status"], "infeasible")
        self.assertIn("min_achievable_volatility", r)
        self.assertGreater(r["min_achievable_volatility"], 0.07)

    def test_no_cap_is_unaffected(self) -> None:
        """Without a max_volatility constraint the max-Sharpe path is unchanged."""
        r = self._solve(["max_weight 1.0"])
        self.assertEqual(r["status"], "success")
        self.assertIn("weights", r["data"])


if __name__ == "__main__":
    unittest.main()