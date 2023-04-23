FROM python:3.9 as build

WORKDIR /app

COPY . .

RUN pip install -r ./requirements.txt

EXPOSE 8888

CMD jupyter-lab --ip 0.0.0.0 --port 8888 --allow-root
