FROM python:3.9 as build
FROM jupyter/notebook:latest

WORKDIR /app

RUN git clone https://github.com/sergiohj93/NLP-Quora.git \
    && cd NLP-Quora \
    && pip install -r requirements.txt
