from pydantic import BaseModel, Field
from .portfolio_base_models import PortfolioProblem


class BlackLittermanView(BaseModel):
    """Model for a single investor view in Black-Litterman model."""

    asset: str = Field(..., description="Asset ticker symbol")
    expected_return: float = Field(..., description="Expected return for this asset")
    confidence: float = Field(1.0, description="Confidence level in the view (0-1)")


class BlackLittermanProblem(PortfolioProblem):
    """Model for Black-Litterman portfolio optimization problems."""

    views: list[BlackLittermanView] = Field(..., description="List of investor views")
    risk_free_rate: float | None = Field(default=0.0, description="Risk-free rate")
    tau: float | None = Field(default=0.05, description="Prior uncertainty parameter")
    risk_aversion: float | None = Field(default=1.0, description="Risk aversion parameter")
    market_cap_weights: dict[str, float] | None = Field(default=None, description="Market capitalization weights")
    min_weight: float | None = Field(default=0.0, description="Minimum weight for any asset")
    max_weight: float | None = Field(default=1.0, description="Maximum weight for any asset")
    # Optional PER-ASSET weight bounds, overriding the scalar (min_weight, max_weight) for the
    # named tickers only. Row-major ``{ticker: [low, high]}``. Used to pin a held asset the
    # customer expressed NO view on to a narrow drift band around its current portfolio weight,
    # so the max-Sharpe optimiser cannot swing it far on covariance grounds alone (e.g. loading a
    # low-volatility holding as ballast the customer never asked to increase). Tickers absent from
    # this dict fall back to the scalar bounds. The caller owns feasibility (bounds must leave the
    # weights summing to 1).
    per_asset_bounds: dict[str, list[float]] | None = Field(
        default=None, description="Per-asset [low, high] weight bounds overriding the scalar bounds"
    )
