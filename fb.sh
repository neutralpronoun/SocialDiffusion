#!/bin/sh

repetitions=4
resolutions=( 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1. )
scales=( 0. 1. 2. 3. )

conda activate digress

for rep in "${repetitions[@]}"
do
  for scale in "${scales[@]}"
  do
    for resolution in "${resolutions[@]}"
    do
      python dgd/main.py dataset=fb_h2 dataset.resolution=$(echo $scale + $resolution | bc)
      rm -r data
    done
  done
done