#FROM python:3.14.4-slim
#FROM --platform=linux/amd64 python:3.12-slim

FROM --platform=linux/arm64 nvcr.io/nvidia/pytorch:24.06-py3

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y \
    curl \
    build-essential \
    openjdk-11-jdk \
    python3-full \
    python3-pip \
    && rm -rf /var/lib/apt/lists/*

# install build tools
#RUN apt-get update && apt-get install -y
#RUN apt-get install curl -y
#RUN apt-get install build-essential -y

# install uv
RUN curl -LsSf https://astral.sh/uv/install.sh | sh

WORKDIR /app

COPY requirements.txt .

#RUN python3 -m venv env
#RUN . env/bin/activate && pip install --upgrade pip
#RUN . env/bin/activate && pip install --no-cache-dir -r requirements.txt

COPY . .

RUN chmod +x entrypoint.sh

RUN ls -la

ENTRYPOINT ["./entrypoint.sh"]
