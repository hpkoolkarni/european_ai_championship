#!/bin/bash
# Pack the 5,000 COCO photos into ONE tar file (organisers: scratch has 4M inodes shared by everyone), check it is
# complete, then remove the unpacked folder. e_synth.py reads photos straight from the tar.
set -euo pipefail
C="$(cd "$(dirname "$0")" && pwd)/data/coco"
cd $C
[ -f val2017.tar ] || tar -cf val2017.tar val2017
n_tar=$(tar -tf val2017.tar | grep -c '\.jpg$'); n_dir=$(ls val2017 | grep -c '\.jpg$')
echo "photos in tar: $n_tar   in folder: $n_dir"
if [ "$n_tar" -eq "$n_dir" ] && [ "$n_tar" -eq 5000 ]; then rm -r val2017 && echo "folder removed"; else echo "MISMATCH: folder kept"; exit 1; fi
chmod g+rw val2017.tar; ls -la $C
