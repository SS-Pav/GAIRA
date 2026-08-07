# GAIRA V7 — API + Streamlit, from a clean clone.
#
# The frozen atlas is COMMITTED to the repository (ten files under results/v7_rebuild/, ~10 MB),
# so the image is self-contained: no external volume, no /Volumes mount, no download at startup.
# No machine-local path is baked in anywhere.
FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONPATH=/app/src

WORKDIR /app

# Build tooling for scipy/scikit-learn wheels on architectures without prebuilt ones.
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential gfortran libopenblas-dev \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml README.md ./
RUN pip install --upgrade pip && pip install \
        "numpy>=1.24" "scipy>=1.10" "scikit-learn>=1.3" "pandas>=2.0" "pyyaml>=6.0" \
        "pydantic>=2.7" "fastapi>=0.115" "uvicorn[standard]>=0.30" "httpx>=0.27" \
        "matplotlib>=3.7" "streamlit>=1.40" "plotly>=5.20" "mcp>=1.2"

# Source, then the frozen atlas. Ordered so a source edit does not invalidate the atlas layer.
COPY src/ /app/src/
COPY streamlit_apps/gaira_v7_console.py /app/streamlit_apps/
COPY results/v7_rebuild/phase00/tables/ /app/results/v7_rebuild/phase00/tables/
COPY results/v7_rebuild/phase01/ /app/results/v7_rebuild/phase01/
COPY results/v7_rebuild/phase02/artifacts/ /app/results/v7_rebuild/phase02/artifacts/
COPY results/v7_rebuild/phase05/PHASE_STATE.json /app/results/v7_rebuild/phase05/
COPY results/v7_rebuild/phase06/artifacts/ /app/results/v7_rebuild/phase06/artifacts/

# Fail the build rather than ship an image whose atlas does not verify.
RUN python -c "from gaira.v7.runtime import freeze; \
    s = freeze.verify(strict=True); print(f'frozen assets verified: {len(s)}')" \
 && python -c "from gaira.v7 import GAIRA; print(GAIRA.load().engine_info().atlas_fingerprint)"

EXPOSE 8000 8501
CMD ["python", "-m", "gaira.v7.api", "--host", "0.0.0.0", "--port", "8000"]
