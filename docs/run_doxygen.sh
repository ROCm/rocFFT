#!/bin/bash

if [ -d docBin ]; then
    rm -rf docBin
fi

sed -e 's/ROCFFT_EXPORT //g' ../library/include/rocfft.h > rocfft.h

cur_version=$(sed -n -e "s/^.*VERSION_STRING.* \"\([0-9\.]\{1,\}\).*/\1/p" ../CMakeLists.txt)
sed -i -e "s/\(PROJECT_NUMBER.*=\)\(.*\)/\1 v${cur_version}/" Doxyfile

doxygen Doxyfile
rm rocfft.h

