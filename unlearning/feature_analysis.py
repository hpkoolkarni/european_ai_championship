
import torch
import numpy as np
from task_template import load_model, load_forget_images, preprocess

model = load_model()
images = load_forget_images()

model.eval()

features = []
predictions = []
confidences = []

with torch.no_grad():
    for i in range(0, len(images), 64):
        x = preprocess(images[i:i+64])

        # Extract ViT features before classification head
        f = model.forward_features(x)
        f = model.forward_head(f, pre_logits=True)

        logits = model.head(f)

        probs = torch.softmax(logits, dim=1)

        features.append(f.cpu())
        predictions.extend(probs.argmax(dim=1).cpu().tolist())
        confidences.extend(probs.max(dim=1).values.cpu().tolist())

features = torch.cat(features)

print("\n========== FEATURE ANALYSIS ==========")
print("Feature shape:", features.shape)
print("Feature mean:", features.mean().item())
print("Feature std:", features.std().item())

feature_norms = features.norm(dim=1)

print("Feature norm mean:", feature_norms.mean().item())
print("Feature norm std:", feature_norms.std().item())

print("Prediction confidence mean:", np.mean(confidences))
print("Prediction confidence std:", np.std(confidences))

print("Unique predicted classes:", len(set(predictions)))

np.savez(
    "feature_analysis.npz",
    features=features.numpy(),
    predictions=np.array(predictions),
    confidences=np.array(confidences)
)

print("Saved feature_analysis.npz")
