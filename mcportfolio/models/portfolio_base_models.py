from typing import Any
from pydantic import BaseModel, Field, ConfigDict


class PortfolioProblem(BaseModel):
    """Base model for portfolio optimization problems."""

    description: str
    tickers: list[str] | None = None
    constraints: list[str] | None = None
    objective: str | None = None
    parameters: dict[str, Any] | None = Field(default_factory=dict)
    # Optional caller-supplied inputs. When provided, the solver uses these instead of
    # fetching live market data — the caller is responsible for assembling a COMPLETE set
    # covering every ticker in ``tickers`` (e.g. sliced from the daily research report, with
    # off-report assets fetched and merged in). ``cov_matrix`` is a row-major nested dict
    # ``{row: {col: annualised_cov}}``; ``mean_returns`` is ``{ticker: annualised_return}``.
    cov_matrix: dict[str, dict[str, float]] | None = Field(
        default=None, description="Caller-supplied annualised covariance (row-major nested dict)"
    )
    mean_returns: dict[str, float] | None = Field(
        default=None, description="Caller-supplied annualised expected returns per ticker"
    )

    model_config = ConfigDict(arbitrary_types_allowed=True)
