#!/usr/bin/env bash

VENVNAME=network_venv

python -m venv $VENVNAME
source $VENVNAME/Scripts/activate
pip install --upgrade pip

test -f requirements.txt && pip install -r requirements.txt

echo "build $VENVNAME"
