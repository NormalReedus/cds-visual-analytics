#!/usr/bin/env bash

VENVNAME=cv101

python3 -m venv $VENVNAME
source $VENVNAME/bin/activate
pip install --upgrade pip

test -f reqs.txt && pip install -r reqs.txt

echo "build $VENVNAME"
