# Parameters related to building rocfft
ARG base_image

FROM ${base_image}
LABEL maintainer="kent.knox@amd"

# Copy the debian package of rocfft into the container from host
COPY *.deb /tmp/

# Install the debian package
RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install --no-install-recommends --allow-unauthenticated -y \
    /tmp/rocfft-*.deb \
  && rm -f /tmp/*.deb \
  && apt-get clean \
  && rm -rf /var/lib/apt/lists/*
