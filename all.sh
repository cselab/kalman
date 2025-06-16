#!/bin/sh
python build.py a.raw
python read.py a.raw
python graphviz.py a.raw a.gv
gvpack -u a.gv | dot -Tsvg -o img/a.svg
