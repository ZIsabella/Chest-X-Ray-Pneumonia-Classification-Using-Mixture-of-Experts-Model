FROM python:3.11-slim

WORKDIR /app

# Match docker-compose bind mounts and app.config defaults.
ENV WEIGHTS_PATH=/app/weights \
    CXR_MOE_DATA_DIR=/app/data

# Fully offline install: use only wheels checked into backend/wheels.
COPY requirements.txt ./requirements.txt
COPY wheels/ ./wheels/

# --no-index forbids any package-index access; --find-links points pip at the
# local wheelhouse. If a compatible wheel is missing, pip fails and the build aborts.
RUN pip install --no-cache-dir --no-index --find-links=/app/wheels -r requirements.txt

COPY app/ ./app/
COPY training/ ./training/

EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
