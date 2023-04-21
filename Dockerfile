FROM python:3.9 as build
FROM jupyter/notebook:latest

WORKDIR /app

RUN git clone https://github.com/sergiohj93/NLP-Quora.git \
    && cd NLP-Quora \
    && pip install -r requirements.txt

EXPOSE 8888

CMD ["sh", "-c", "jupyter notebook --port=8888 --no-browser"]
