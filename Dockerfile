FROM amdih/pytorch

ENV GRADIO_SERVER_PORT 7860

COPY requirements.txt requirements.txt

RUN pip3 install -r requirements.txt && rm -rf /root/.cahe/pip

WORKDIR /workspace/emlv2

ENTRYPOINT python src/demo.py ckpt_path=/workspace/EML20_session04_assignment/logs/train/runs/2022-10-02_15-52-12/model.script.pt
