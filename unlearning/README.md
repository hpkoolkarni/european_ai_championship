# Machine unlearning (ViT-B/16)

Given a 100-class ViT-B/16 classifier (`model.pt`) and 3,500 unlabelled forget images (`forget_images.npz`), produce weights that behave like a "gold" model retrained without the forget images, while staying close to the original everywhere else. The score multiplies a forget score (per-image loss statistics and temperature-2 predictions compared with retrained gold models, on the forget images and hidden altered copies) by a retain score (accuracy and KL to the original on a hidden control set). Below 70 % validation accuracy the score is 0. Full rules: the docstring of `task_template.py`.

## Layout

| Path | Contents |
|---|---|
| `task_template.py` | Organisers' template (model, loading, the server's preprocessing) with our method in `unlearn()`: experiment 9, confidence-preserving distillation of the classification head (frozen backbone, temperature-4 teacher on the forget images, temperature-2 student, L2 anchor to the original head). |
| `submission.py` | Organisers' upload script (reads `API_KEY` from the environment). |
| `script.py` | Head-scaling baseline: scale the classifier head by a factor and show how confidence and loss on the forget images change. |
| `analyze_forget.py` | Confidence, entropy and class spread of a model on the forget images; saves `original_analysis.pt`. |
| `feature_analysis.py` | Pre-logit ViT features of the forget images (norms, spread); saves `feature_analysis.npz`. |
| `results/probe/` | Forget-set statistics of the original model: almost 100 % confident, only 45 classes used. |
| `results/ul/` | Settings and target statistics of the distillation runs (`dist1`, `v2`: temperature, strength, epochs, anchor). |

## How to run

```bash
pip install -r requirements.txt          # or: uv sync
python task_template.py                  # runs unlearn() and writes submission.pt
python analyze_forget.py                 # statistics of submission.pt on the forget images
python submission.py                     # upload (needs API_KEY)
```

`model.pt` and `forget_images.npz` come from the organisers and are not in the repo.
