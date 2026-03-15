# McPortfolio - Portfolio Optimization MCP Server
FROM python:3.12-slim

WORKDIR /app

# Install system dependencies including build tools for CVXPY
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    ca-certificates \
    build-essential \
    g++ \
    git \
    cmake \
    && rm -rf /var/lib/apt/lists/*

# Install uv package manager
ADD https://astral.sh/uv/install.sh /uv-installer.sh
RUN sh /uv-installer.sh && rm /uv-installer.sh

ENV PATH="/root/.local/bin/:$PATH"

# Copy shared market data provider library first
COPY libs/market_data_provider /app/libs/market_data_provider

# Copy application code and dependencies
COPY tools/mcportfolio /app

# Install shared library, then the package
RUN uv pip install --system /app/libs/market_data_provider && uv pip install --system .

# Create non-root user for security
RUN useradd --create-home --shell /bin/bash mcportfolio
USER mcportfolio

# Set environment variables
ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1

# Run the MCP server in stdio mode (not HTTP)
CMD ["python", "-m", "mcportfolio.server.main"]
