# European AI Championship 2026 (Paris) — team code

Code for the three hackathon tasks. Each folder holds the organisers' starter files plus our own scripts, and has its own README with the approach, the experiments and how to run them.

| Folder | Task | What it does |
|---|---|---|
| [`watermark_localisation/`](watermark_localisation/) | Watermark localisation (images) | Predict, for every test image, the 128 × 128 mask of watermarked cells (PixelSeal, TrustMark, StegaStamp). Synthetic training data, fine-tuned segmenters, message-aware models, known-message box search, per-method ensemble. |
| [`model_equality/`](model_equality/) | Model equality (33 chat endpoints) | Group black-box chat endpoints into families (same base model) and models (same exact weights, disguises allowed). Probes, calibrated permutation tests, tool-choice fingerprints, local reference-model likelihoods. |
| [`unlearning/`](unlearning/) | Machine unlearning (ViT-B/16) | Make a 100-class classifier behave as if it had never seen 3,500 forget images, while staying close to the original elsewhere. |

## What is not in the repo

Datasets, model weights, checkpoints, synthetic data shards, predictions (`.pt`, `.npz`, `.npy`, `.pth`, `.onnx`), virtual environments and run logs. They are large and are recreated by the scripts (see each README). The starter kits' download scripts fetch the organisers' models.

## Running

Everything was run on JURECA (Jülich) A100 nodes with Slurm. The `.sbatch` files contain the event's account and reservation names; change them for another cluster. Scripts that talk to the task server read the team key from the `API_KEY` environment variable:

```bash
read -s API_KEY; export API_KEY
```
