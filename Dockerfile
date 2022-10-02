FROM python:3.9.13-slim

# Basic setup
RUN apt update
RUN apt install -y \
    build-essential \
    git \
    curl \
    ca-certificates \
    wget \
    && rm -rf /var/lib/apt/lists

ENV GRADIO_SERVER_PORT 7860

WORKDIR /opt/src

ADD requirements.txt requirements.txt

ADD src src

ADD configs configs

ADD logs logs

ADD *.toml .

RUN pip install --no-cache-dir -r requirements.txt \
    && rm requirements.txt

##CMD bash

ENTRYPOINT python src/demo.py ckpt_path=logs/train/runs/2022-10-02_16-56-52/model.script.pt