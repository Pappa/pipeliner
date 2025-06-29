FROM python:3.13-slim-bookworm
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

RUN apt-get update && \
    apt-get -y upgrade && \
    apt-get -y install --no-install-recommends \
    build-essential \
    ca-certificates \
    openjdk-17-jdk-headless \
    git \
    python3-numpy

RUN uv pip install --system pip --upgrade
RUN uv pip install --system setuptools --upgrade

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONIOENCODING=UTF-8 \
    LANG=C.UTF-8 \
    LC_ALL=C.UTF-8

ENV SETUPTOOLS_SCM_PRETEND_VERSION_FOR_PIPELINER=0.0.1

RUN mkdir -p /app/pipeliner/src
ADD src /app/pipeliner/src/
ADD pyproject.toml /app/pipeliner/
ADD README.md /app/pipeliner/
WORKDIR /app/pipeliner
RUN ls -la
RUN uv pip install --system .

# Set SageMaker training environment variables
ENV SM_INPUT=/opt/ml/input
ENV SM_INPUT_TRAINING_CONFIG_FILE=$SM_INPUT/config/hyperparameters.json
ENV SM_INPUT_DATA_CONFIG_FILE=$SM_INPUT/config/inputdataconfig.json
ENV SM_CHECKPOINT_CONFIG_FILE=$SM_INPUT/config/checkpointconfig.json

# Set SageMaker serving environment variables
ENV SM_MODEL_DIR=/opt/ml/model

# Set a docker label to advertise multi-model support on the container
LABEL com.amazonaws.sagemaker.capabilities.multi-models=true
# Set a docker label to enable container to use SAGEMAKER_BIND_TO_PORT environment variable if present
LABEL com.amazonaws.sagemaker.capabilities.accept-bind-to-port=true

ENV SAGEMAKER_TRAINING_MODULE=pipeliner.containers.sklearn.training:main
ENV SAGEMAKER_SERVING_MODULE=pipeliner.containers.sklearn.serving:main