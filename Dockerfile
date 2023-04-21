FROM python:3.9 as build

WORKDIR /app

RUN git clone https://github.com/sergiohj93/NLP-Quora.git --branch 1.0.3 \
    && cd NLP-Quora \
    && pip install -r requirements.txt
