FROM python:3.8-alpine
RUN apk update -y && apk install awscli -y

WORKDIR /webapp
COPY . /webapp

RUN pip install -r requirements.txt
CMD  ["python3", "webapp.py"]