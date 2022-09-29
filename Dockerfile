FROM python:3.8.14-slim-bullseye

COPY requirements.txt requirements.txt

RUN pip3 install -r requirements.txt && rm -rf /root/.cahe/pip

WORKDIR /workspace/emlv2

CMD bash