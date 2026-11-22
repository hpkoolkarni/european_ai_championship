#!/bin/bash
# E0. Downloads for the WAM fine-tune. Run on the LOGIN node (compute nodes have no internet). About 1.4 GB in total.
#   output/third_party/watermark-anything     WAM code (only its model files are imported)
#   output/third_party/wam_mit.pth            WAM weights, MIT licence (378 MB)
#   output/data/coco/val2017/                 5,000 COCO photos (778 MB zip, deleted after unpacking)
#   output/data/coco/instances_val2017.json   COCO object outlines (taken from the 241 MB annotations zip, deleted after)
#   output/synth/  output/ckpt/  output/logs/ empty folders used by the next steps
set -euo pipefail
O="$(cd "$(dirname "$0")" && pwd)"
mkdir -p $O/third_party $O/data/coco $O/synth $O/ckpt $O/logs

cd $O/third_party
[ -d watermark-anything ] || git clone --depth 1 https://github.com/facebookresearch/watermark-anything.git
[ -f wam_mit.pth ] || wget -q --show-progress https://dl.fbaipublicfiles.com/watermark_anything/wam_mit.pth

cd $O/data/coco
if [ ! -d val2017 ]; then
  wget -q --show-progress http://images.cocodataset.org/zips/val2017.zip
  unzip -q val2017.zip && rm val2017.zip
fi
if [ ! -f instances_val2017.json ]; then
  wget -q --show-progress http://images.cocodataset.org/annotations/annotations_trainval2017.zip
  unzip -q -j annotations_trainval2017.zip annotations/instances_val2017.json && rm annotations_trainval2017.zip
fi

chmod -R g+rwX $O/third_party $O/data $O/synth $O/ckpt $O/logs
echo "COCO photos: $(ls $O/data/coco/val2017 | wc -l)"
ls -la $O/third_party
echo "SETUP DONE"
