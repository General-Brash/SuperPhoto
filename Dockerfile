ARG PYTHON_BASE_IMAGE=python:3.10-slim-bookworm
FROM ${PYTHON_BASE_IMAGE}

ARG APP_UID=1000
ARG APP_GID=1000

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_INDEX_URL=https://mirrors.aliyun.com/pypi/simple \
    OMP_NUM_THREADS=4 \
    OPENBLAS_NUM_THREADS=4 \
    MKL_NUM_THREADS=4 \
    OPENVINO_NUM_THREADS=4

RUN sed -i \
        -e 's|http://deb.debian.org/debian-security|https://mirrors.aliyun.com/debian-security|g' \
        -e 's|http://deb.debian.org/debian|https://mirrors.aliyun.com/debian|g' \
        /etc/apt/sources.list.d/debian.sources \
    && apt-get update \
    && apt-get install -y --no-install-recommends \
        gcc g++ make libgl1 libglib2.0-0 libgomp1 ca-certificates ffmpeg \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /opt/realesrgan

COPY requirements.lock /tmp/requirements.lock
RUN --mount=type=cache,target=/root/.cache/pip \
    python -m pip install --upgrade "pip<25" "setuptools<70" wheel \
    && python -m pip install --no-deps --extra-index-url https://download.pytorch.org/whl/cpu \
        torch==2.1.2+cpu torchvision==0.16.2+cpu \
    && python -m pip install -r /tmp/requirements.lock \
    && python -m pip install --no-deps basicsr==1.4.2 \
    && python -m pip install --no-deps \
        llvmlite==0.43.0 numba==0.60.0 filterpy==1.4.5 facexlib==0.3.0 gfpgan==1.3.8 \
    && rm -f /tmp/requirements.lock

COPY . /opt/realesrgan
RUN python -m pip install --no-deps .

RUN groupadd --gid ${APP_GID} app \
    && useradd --uid ${APP_UID} --gid ${APP_GID} --create-home app \
    && mkdir -p /opt/realesrgan/inputs /opt/realesrgan/results /opt/realesrgan/weights \
        /data/input /data/output /data/tmp /state /models /logs \
    && chown -R app:app /opt/realesrgan /data /state /models /logs

USER app

ENTRYPOINT ["python", "inference_realesrgan.py"]
CMD ["--help"]
