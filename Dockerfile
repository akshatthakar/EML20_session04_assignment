FROM python:3.7.10-slim-buster


# Basic setup
RUN apt update
RUN apt install -y \
    build-essential \
    git \
    curl \
    ca-certificates \
    wget \
    && rm -rf /var/lib/apt/lists

ENV GRADIO_SERVER_PORT 8080

EXPOSE 8080

WORKDIR /opt/src

ADD src/utils utils

ADD configs configs

ADD logs logs

COPY ["*.toml","src/demo.py","requirements.txt" ,"requirements.txt" ,"./"]

RUN  pip install --no-cache-dir -r requirements.txt \
  && rm requirements.txt \
  && rm -rf /root/.cache/pip \
  && rm -rf /tmp/*


ENTRYPOINT python demo.py ckpt_path=logs/train/runs/2022-10-02_16-56-52/model.script.pt