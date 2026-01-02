#!/bin/bash -c ./docker/run
FROM jupyter/scipy-notebook:lab-4.0.3

USER root
RUN pip3 install pandas \
	&& pip3 install ujson matplotlib yfinance
USER jovyan

WORKDIR /app
