#!/bin/bash

# === Generate Boxplot about Confusion Matrixes ===
RESULTDIR=$CAFFE_ROOT/dummy/jackson/results

cd $CAFFE_ROOT/python
python confusion_matrix_boxplot.py --cmall $RESULTDIR/confusion_matrix_all --outputdir $RESULTDIR/
# === Generate Boxplot about Confusion Matrixes ===