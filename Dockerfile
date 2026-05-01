#FROM python:3.14.4-slim
FROM --platform=linux/amd64 python:3.12-slim

ENV DEBIAN_FRONTEND=noninteractive

# install build tools
RUN apt-get update && apt-get install -y
RUN apt-get install curl -y
RUN apt-get install build-essential -y

#WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

RUN chmod +x entrypoint.sh

RUN ls -la

ENTRYPOINT ["./entrypoint.sh"]
