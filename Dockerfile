FROM python:3.11.5-slim

LABEL maintainer="Krishnan Authi Narayanan <avivkri@gmail.com>"

COPY . /polyaxon
WORKDIR /polyaxon
RUN pip3 install -r requirements.txt
