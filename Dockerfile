FROM python:3.11.6-slim-bullseye
ENV DEBIAN_FRONTEND=noninteractive
ENV FLUCOMA_VERSION=1.0.6
ENV SUPERCOLLIDER_VERSION=3.13
WORKDIR /tmp
RUN apt-get update \
    && apt-get install --yes \
        build-essential \
        cmake \
        ffmpeg \
        git \
        jackd2 \
        libfftw3-dev \
        libjack-jackd2-dev \
        libsndfile1-dev \
        wget \
    && apt-get clean
RUN git clone -b ${SUPERCOLLIDER_VERSION} --single-branch --depth 1 --recurse-submodules https://github.com/supercollider/supercollider.git \
    && cd supercollider \
    && mkdir build \
    && cd build \
    && cmake \
        -DCMAKE_BUILD_TYPE=Release \
        -DNO_AVAHI=ON \
        -DNO_X11=ON \
        -DSC_ED=OFF \
        -DSC_EL=OFF \
        -DSC_HIDAPI=OFF \
        -DSC_IDE=OFF \
        -DSC_QT=OFF \
        -DSC_VIM=OFF \
        .. \
    && make install \
    && which scsynth \
    && mkdir -p /root/.local/share/SuperCollider/synthdefs \
    && cd ../.. \
    && git clone -b 1.0.0 --single-branch --depth 1 https://github.com/flucoma/flucoma-docs.git \
    && pip install -r flucoma-docs/requirements.txt \
    && git clone -b ${FLUCOMA_VERSION} --single-branch --depth 1 https://github.com/flucoma/flucoma-sc.git \
    && mkdir flucoma-sc/build \
    && cd flucoma-sc/build \
    && cmake -DSC_PATH=../../supercollider/ .. \
    && make install \
    && mv ../install/FluidCorpusManipulation/Plugins/*.so /usr/local/lib/SuperCollider/plugins \
    && cd ../.. \
    && rm -Rf flucoma-* supercollider
WORKDIR /app
COPY --chmod=755 ./requirements.unix.txt .
RUN --mount=type=cache,target=/root/.cache pip install -r requirements.unix.txt
COPY --chmod=755 . .
RUN pip install -e .
