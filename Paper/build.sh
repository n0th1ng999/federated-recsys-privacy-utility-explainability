#!/bin/sh
# Builds the supplement first so that main.tex can resolve "Table S<n>" references (xr package),
# then the main article. Requires tectonic. Outputs: build/supplementary.pdf, build/main.pdf
set -e
cd "$(dirname "$0")"
mkdir -p build
tectonic --keep-intermediates --outdir build supplementary.tex
cp build/supplementary.aux supplementary.aux
tectonic --keep-intermediates --outdir build main.tex
rm -f supplementary.aux
