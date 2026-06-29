# satnogs-id Milestone-0 throwaway probe env. The container IS the environment — no host venv.
# (rffit/strf NOT needed for the M0 separability probe; this is the minimal scorer image.)
FROM python:3.14-slim

# libgomp1: OpenMP runtime used by numpy/scipy on slim images.
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /m0
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

CMD ["python", "-c", "import skyfield, h5py, numpy, scipy; print('m0 probe env ready (py', __import__('sys').version.split()[0], ')')"]
