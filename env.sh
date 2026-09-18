#!/bin/bash

if ! conda env list | grep -q "vot"; then
    echo "Creating vot environment..."
    conda create -n vot python==3.10
    pip install -r requirements.txt
fi

# activate the environment (run everytime)
conda activate vot