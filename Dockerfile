FROM python:3.10

MAINTAINER Matvey

WORKDIR /app

RUN pip install --upgrade pip

COPY requirements.txt /app
RUN pip install -r requirements.txt

COPY . /app