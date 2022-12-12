#!/bin/sh

repetitions=4
resolutions=( 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1. )
scales=( 0. 1. 2. 3. )

conda activate digress
python dgd/main.py dataset=fb_hierarchies dataset.h=1 dataset.dataset_testing=True
python dgd/main.py dataset=fb_hierarchies dataset.h=1.5 dataset.dataset_testing=True
python dgd/main.py dataset=fb_hierarchies dataset.h=2 dataset.dataset_testing=True

