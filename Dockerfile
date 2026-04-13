FROM python:3.11-slim

COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

ENV UV_SYSTEM_PYTHON=1 \
    UV_COMPILE_BYTECODE=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1


WORKDIR /app

COPY pyproject.toml uv.lock ./
COPY README.md /app/

RUN uv sync --no-dev --frozen

COPY . .

EXPOSE 8000
CMD ["uv", "run", "main.py"]
