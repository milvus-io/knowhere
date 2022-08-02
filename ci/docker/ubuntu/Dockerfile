ARG BASE_IMAGE="ubuntu:20.04"
FROM ${BASE_IMAGE}
ARG SCRIPT="build.sh"
ENV DEBIAN_FRONTEND=noninteractive 
ENV TZ=Asia/Shanghai
COPY scripts/deps/${SCRIPT} ${SCRIPT}

RUN chmod +x ${SCRIPT} \
    && ./${SCRIPT} \
    && rm ${SCRIPT}