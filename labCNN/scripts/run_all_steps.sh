#!/bin/bash

# Ao executar run_all.sh ele realizara os seguintes passos: 
# 	1) Realiza o treinanamento
# 	2) Exporta a arquitetura do modelo
# 	3) Exportar a curva de aprendizado
# 	4) Realiza o teste
#   5) Gera a matrix de confusão

CONFIG=$1

OUTPUTDIR=$CAFFE_ROOT/dummy/jackson/results
OUTPUTCONFIGDIR=$OUTPUTDIR/$CONFIG

mkdir -p $OUTPUTCONFIGDIR

mkdir -p $CAFFE_ROOT/dummy/jackson/models/$CONFIG/snapshot

# === Train ===
SOLVER=$CAFFE_ROOT/dummy/jackson/models/$CONFIG/lenet_solver.prototxt
TRAINLOG=$OUTPUTCONFIGDIR/model_train.log

cd $CAFFE_ROOT
 ./build/tools/caffe train -solver $SOLVER 2>&1 | tee $TRAINLOG
# === Train ===

# === Model Architeture ===
TRAIN=$CAFFE_ROOT/dummy/jackson/models/$CONFIG/lenet_train_val.prototxt
MODEL=$OUTPUTCONFIGDIR/model_architecture.pdf

cd $CAFFE_ROOT
./python/draw_net.py $TRAIN $MODEL
# === Model Architeture ===

# === Learning Curve ===
CURVE=$OUTPUTCONFIGDIR/learning_curve.pdf

cd $CAFFE_ROOT/python
python plot_learning_curve.py $TRAINLOG $CURVE
# === Learning Curve ===

# === Test ===
DEPLOY=$CAFFE_ROOT/dummy/jackson/models/$CONFIG/lenet_deploy.prototxt
WEIGHTS_MODEL=$CAFFE_ROOT/dummy/jackson/models/$CONFIG/snapshot/lenet_iter_2000.caffemodel
MEAN_IMG=$CAFFE_ROOT/dummy/data/dummy_mean.binaryproto

#data source config
LABELS=$CAFFE_ROOT/dummy/data/digits/labels.txt
LIST_OF_FILES=$CAFFE_ROOT/dummy/data/digits/test.txt
SOURCE=$CAFFE_ROOT/dummy/data/digits/test/

#MINMAX | MEAN_IMAGE
NORMTYPE=MINMAX

#Print N predictions
NPREDS=5

TESTLOG=$OUTPUTCONFIGDIR/model_test.log

$CAFFE_ROOT/dummy/jackson/scripts/classification2 $DEPLOY $WEIGHTS_MODEL $MEAN_IMG $LABELS $LIST_OF_FILES $SOURCE $NORMTYPE $NPREDS 2>&1 | tee $TESTLOG
# === Test ===

# === Confusion Matrix ===
cd $CAFFE_ROOT/python
python generate_confusion_matrix.py --configname $CONFIG --testlog $TESTLOG --outputdir $OUTPUTDIR
# === Confusion Matrix ===