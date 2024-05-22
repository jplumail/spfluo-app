#!/bin/bash

cd ..
for dir in "spfluo" "scipion-fluo-singleparticle" "scipion-pyworkflow" "scipion-fluo" "scipion-app"; do
  if [ -d "$dir" ]; then
    cd "$dir" || exit 1
    echo "$dir" $(python -m setuptools_scm || echo "No tags found")
    cd ..
  else
    echo "Directory not found: $dir"
  fi
done