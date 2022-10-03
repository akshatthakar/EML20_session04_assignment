FROM zironycho/pytorch:1120-cpu-py38


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

EXPOSE 7860

WORKDIR /opt/src

ADD src/demo.py demo.py

ADD src/utils utils

ADD configs configs

ADD logs logs

ADD *.toml .

ADD requirements.txt requirements.txt

RUN rm -rf /root/.cahe/pip

RUN pip install --no-cache-dir -r requirements.txt \
    && rm requirements.txt

ENTRYPOINT python demo.py ckpt_path=logs/train/runs/2022-10-02_16-56-52/model.script.pt