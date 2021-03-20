#!/usr/bin/env bash

VENVNAME=cv101

python -m venv $VENVNAME
source $VENVNAME/Scripts/activate
pip install --upgrade pip

test -f reqs.txt && pip install -r reqs.txt

echo "build $VENVNAME"
