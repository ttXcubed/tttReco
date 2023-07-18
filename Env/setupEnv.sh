#!/bin/bash

SCRIPT_DIR=`dirname ${BASH_SOURCE[0]}`
STOP=""
echo "Script directory "$SCRIPT_DIR

function run_setup()
{
    if [[  -z  $2  ]] ; then
        echo 'Usage:'
        echo '  setupEnv.sh <env_file> <install_dir>'
        return 1
    fi
    
    if [[ "$1" == *.yml ]] ; then
        echo "Using enviroment file: "$1
    else
        echo "Conda enviroment file ending with .yml required"
        return 1
    fi
    
    ENV_FILE=$1
    INSTALL_DIR=$2
    
    if [[ -f "$ENV_FILE" ]]; then
        echo "Installing environment:"
        echo "--------------------------------------------"
        cat $ENV_FILE
        echo "--------------------------------------------"
    else
        echo "File $ENV_FILE does not exists"
        return 1
    fi
    
    if [ -d "$INSTALL_DIR" ]; then
        echo "Error - directory "$INSTALL_DIR" exists!"
        return 1
    fi
    echo "Setting up central environment under "$INSTALL_DIR
    
    mkdir -p $INSTALL_DIR || return 1
    
    INSTALL_ABSDIR=$(cd $INSTALL_DIR; pwd)
    echo "Absolute path "$INSTALL_ABSDIR
    echo "System: "$(uname -a)

    if [ $(uname) == "Darwin" ]; then
        echo "Installing for MAC OS"
        curl -sS https://repo.anaconda.com/miniconda/Miniconda3-py37_23.1.0-1-MacOSX-x86_64.sh -o $INSTALL_ABSDIR/conda.sh || return 1
    elif [ "$(expr substr $(uname -s) 1 5)" == "Linux" ]; then
        echo "Installing for Linux"
        curl -sS https://repo.anaconda.com/miniconda/Miniconda3-py37_23.1.0-1-Linux-x86_64.sh -o $INSTALL_ABSDIR/conda.sh || return 1
    fi

    bash $INSTALL_ABSDIR/conda.sh -b -s -p $INSTALL_ABSDIR/miniconda || return 1

    CONDA_BIN=$INSTALL_ABSDIR/miniconda/bin
    export PATH=$CONDA_BIN:$PATH
        
    export TMPDIR=$INSTALL_ABSDIR/tmp
    export TMPPATH=$TMPDIR
    export TEMP=$TMPDIR
    mkdir $TMPDIR
   
    echo "Create environment"
    unset PYTHONPATH
    conda env create -f $ENV_FILE || return 1
    rm -rf $INSTALL_ABSDIR/tmp
    
    echo "Generate setup script"
    echo "export PATH="$INSTALL_ABSDIR"/miniconda/bin:\$PATH" > $SCRIPT_DIR/env.sh
    #echo "export LD_PRELOAD="$INSTALL_ABSDIR"/miniconda/lib/libmkl_core.so:"$INSTALL_ABSDIR"/miniconda/lib/libmkl_sequential.so:\$LD_PRELOAD" >> $SCRIPT_DIR/env_gpu.sh
    echo "source activate tf" >> $SCRIPT_DIR/env.sh

    echo "export TF_CPP_MIN_LOG_LEVEL=2" >> $SCRIPT_DIR/env.sh
    echo "export OMP_NUM_THREADS=12 #reduce further if out-of-memory" >> $SCRIPT_DIR/env.sh
}

run_setup $1 $2
if [ $? -eq 0 ]
then
  echo "Successfully setup environment"
else
  echo "Failed to setup environment"
fi


