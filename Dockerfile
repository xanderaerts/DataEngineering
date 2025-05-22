# Dockerfile

FROM bitnami/spark:latest

USER root

RUN pip install --no-cache-dir pandas matplotlib seaborn pyarrow

USER 1001
