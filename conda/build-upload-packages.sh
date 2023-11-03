#!/bin/bash

for package in "cucim" "tkcolorpicker" "outdated" "spfluo" "scipion-pyworkflow" "scipion-fluo" "scipion-app" "scipion-fluo-singleparticle" "spfluo-app"; do
    conda search -c jplumail --override-channel $package
    if [ $? -eq 1 ]; then
        conda mambabuild $package/ -c jplumail -c pytorch -c nvidia
        if [ $? -ne 0 ]; then
            exit 1
        fi
    fi
done