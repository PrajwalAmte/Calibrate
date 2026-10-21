# syntax=docker/dockerfile:1
#
# calibrate container image.
#
#   docker build -t calibrate .
#   docker run --rm --gpus all calibrate diagnose
#   docker run --rm --gpus all -p 9400:9400 calibrate \
#       watch --gpus all --output none --prometheus-listen 0.0.0.0:9400
#
# NVML (libnvidia-ml.so.1) is injected at runtime by the NVIDIA container
# toolkit; it is not baked into the image.

FROM rust:1.94-slim-bookworm AS build
WORKDIR /src
COPY Cargo.toml Cargo.lock ./
COPY src ./src
COPY assets ./assets
RUN cargo build --release --locked && strip target/release/calibrate

FROM debian:bookworm-slim
RUN apt-get update \
 && apt-get install -y --no-install-recommends ca-certificates \
 && rm -rf /var/lib/apt/lists/* \
 && useradd --system --uid 10001 --home /nonexistent --shell /usr/sbin/nologin calibrate
COPY --from=build /src/target/release/calibrate /usr/local/bin/calibrate
COPY LICENSE README.md /usr/share/doc/calibrate/
# Make GPUs visible through the NVIDIA runtime; `utility` provides NVML.
ENV NVIDIA_VISIBLE_DEVICES=all \
    NVIDIA_DRIVER_CAPABILITIES=utility,compute \
    CALIBRATE_HOOK_DIR=/var/run/calibrate
RUN mkdir -p /var/run/calibrate && chown calibrate /var/run/calibrate
USER 10001
EXPOSE 9400
ENTRYPOINT ["/usr/local/bin/calibrate"]
CMD ["--help"]
