FROM python:3.8

# Adjust Time Zone
ENV TZ=America/Sao_Paulo
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

# Add files
COPY . /mlflow_training_tracking

ENV PYTHONPATH "${PYTHONPATH}:/mlflow_training_tracking/src"

# Go to working directory
WORKDIR /mlflow_training_tracking

# Install requirements
RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

WORKDIR /root