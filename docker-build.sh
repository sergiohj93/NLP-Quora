docker build -t nlp_quora:latest -< Dockerfile
docker run -it -p 8888:8888 -t nlp_quora:latest /bin/bash