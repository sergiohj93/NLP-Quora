FROM python:3.9 as build

WORKDIR /app

RUN git clone https://github.com/sergiohj93/NLP-Quora.git \
    && cd NLP-Quora \
    && pip install -r requirements.txt

EXPOSE 8888

CMD ["jupyter", "notebook", "--ip=0000"]
