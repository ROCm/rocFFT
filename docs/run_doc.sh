#!/bin/bash

set -eu

# Make this directory the PWD
cd "$(dirname "${BASH_SOURCE[0]}")"

# Update version string
cur_version=$(sed -n -e "s/^.*VERSION_STRING.* \"\([0-9\.]\{1,\}\).*/\1/p" ../CMakeLists.txt)
sed -i -e "s/\(PROJECT_NUMBER.*=\)\(.*\)/\1 v${cur_version}/" Doxyfile
sed -i -e "s/\(version.*=\)\(.*\)/\1 u'${cur_version}'/" source/conf.py
sed -i -e "s/\(release.*=\)\(.*\)/\1 u'${cur_version}'/" source/conf.py

# Build doxygen info
rm -rf docBin
doxygen Doxyfile

# Build sphinx docs
cd source
make clean
make html
make latexpdf
make man
make text
make texinfo
