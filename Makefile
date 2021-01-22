# Makefile for TreeAlgorithms
SHELL := /bin/bash

# You can set these variables from the commandline.
VERSION=$(shell python setup.py --version)

./dist/ReclusterTreeAlgorithms-${VERSION}-py3-none-any.whl:
	python ./setup.py sdist bdist_wheel

clean:
	rm dist/ReclusterTreeAlgorithms-0.1-py3-none-any.whl


install: ./dist/ReclusterTreeAlgorithms-${VERSION}-py3-none-any.whl # pip install
	pip install --upgrade ./dist/ReclusterTreeAlgorithms-${VERSION}-py3-none-any.whl


%: Makefile