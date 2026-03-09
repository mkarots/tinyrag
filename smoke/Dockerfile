ARG PYTHON_VERSION=3.11
FROM python:${PYTHON_VERSION}-slim

# install uv 
RUN pip install uv

# install raglet
RUN uv pip install --system raglet

# Pre-download the default embedding model so smoke tests run offline.
# HF_TOKEN is mounted as a secret to avoid baking it into image layers.
RUN --mount=type=secret,id=HF_TOKEN,env=HF_TOKEN \
    python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')"

WORKDIR /workspace
COPY main.py .
COPY gates/ gates/

ENTRYPOINT ["python", "main.py"]
