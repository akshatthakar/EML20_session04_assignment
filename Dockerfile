FROM amdih/pytorch

ENV GRADIO_SERVER_PORT 7860

ADD requirements.txt requirements.txt

ADD src src

ADD config config

RUN pip3 install -r requirements.txt && rm -rf /root/.cahe/pip

WORKDIR /opt/src

ENTRYPOINT python src/demo.py ckpt_path=/workspace/EML20_session04_assignment/logs/train/runs/2022-10-02_15-52-12/model.script.pt
