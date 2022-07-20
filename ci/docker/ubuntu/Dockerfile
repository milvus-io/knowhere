ARG BASE_IMAGE="ubuntu:20.04"
FROM ${BASE_IMAGE}
ARG SCRIPT="build.sh"
COPY scripts/deps/${SCRIPT} ${SCRIPT}

RUN chmod +x ${SCRIPT} \
    && ./${SCRIPT} \
    && rm ${SCRIPT}