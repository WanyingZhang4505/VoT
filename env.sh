#!/bin/bash

# current path
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

if ! conda env list | grep -q "vot"; then
    echo "Creating vot environment..."
    conda create -n vot python==3.10
    pip install -r "$SCRIPT_DIR/config/requirements.txt"
fi

# activate the environment (run everytime)
conda activate vot