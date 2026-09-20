#!/bin/bash

# 脚本所在目录（即使从别的目录 source 也能正确定位 config/requirements.txt）
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

if ! conda env list | grep -q "vot"; then
    echo "Creating vot environment..."
    conda create -n vot python==3.10
    pip install -r "$SCRIPT_DIR/config/requirements.txt"
fi

# activate the environment (run everytime)
conda activate vot