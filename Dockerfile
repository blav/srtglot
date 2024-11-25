# Stage 1: Build stage
FROM python:3.12-slim AS builder

RUN apt-get update && \
    apt-get install -y curl && \
    curl -sSL https://install.python-poetry.org | python3 - && \
    apt-get purge -y --auto-remove curl && \
    rm -rf /var/lib/apt/lists/*

ENV PATH="/root/.local/bin:$PATH"

WORKDIR /app
COPY pyproject.toml poetry.lock README.md ./
COPY src/ src/

RUN poetry config virtualenvs.create false && \
    poetry install --only main --no-interaction --no-ansi && \
    pip install .

# Stage 2: Production stage
FROM python:3.12-slim

WORKDIR /app
COPY --from=builder /usr/local/lib/python3.12/site-packages /usr/local/lib/python3.12/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

ENTRYPOINT ["srtglot"]
CMD ["--help"]