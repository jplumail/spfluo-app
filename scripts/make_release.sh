#!/bin/bash

# Check if VERSION is empty
if [ -z "$VERSION" ]; then
    echo "VERSION is empty. Aborting."
    exit 1
fi

# Create an empty commit with the message "release $VERSION"
git commit --allow-empty -m "release $VERSION"
git tag "$VERSION"

# Push the commit and the tags
git push
git push --tags