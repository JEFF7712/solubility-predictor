# Sovereign build: digest-pinned local base, hermetic local wheelhouse.
# Fetch the retained wheelhouse before building locally:
#   scripts/fetch-wheels.sh
# (wheelhouse/ is gitignored; CI assembles from Forgejo-retained supply
# instead of executing this file, and validates that both paths agree.)
FROM registry.rupan.dev/upstream/docker.io/library/python:3.12-slim-bookworm@sha256:34386ef0cb081344d7ec1c103ba398e6e9f64e9ab3a1509accc92a4e24a07258 AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

COPY requirements.txt ./requirements.txt
COPY wheelhouse /wheelhouse

RUN pip install --no-cache-dir --no-index --find-links=/wheelhouse -r ./requirements.txt

COPY src/ ./src/
COPY static/ ./static/
COPY models/ ./models/

EXPOSE 8080

USER 1000:1000

CMD ["python", "-m", "src.server"]
