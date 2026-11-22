
import torch
import torch.nn.functional as F
from collections import Counter
from pathlib import Path

from task_template import (
    load_model,
    load_forget_images,
    preprocess,
    DEVICE
)

BATCH_SIZE = 64


@torch.no_grad()
def analyze():
    print("Loading original model...")
    model = load_model(Path("submission.pt"))

    print("Loading forget images...")
    images = load_forget_images()

    all_logits = []

    print("Running predictions...")

    for start in range(0, len(images), BATCH_SIZE):
        batch = images[start:start + BATCH_SIZE]
        x = preprocess(batch)

        with torch.autocast(
            device_type="cuda",
            dtype=torch.bfloat16,
            enabled=DEVICE == "cuda"
        ):
            logits = model(x)

        all_logits.append(logits.float().cpu())

    logits = torch.cat(all_logits)

    # Predictions
    probabilities = F.softmax(logits, dim=1)
    probabilities_t2 = F.softmax(logits / 2.0, dim=1)

    confidence, predictions = probabilities.max(dim=1)

    # Entropy
    entropy = -(
        probabilities * torch.log(probabilities.clamp_min(1e-12))
    ).sum(dim=1)

    # Prediction distribution
    counts = Counter(predictions.tolist())

    print("\n========== ORIGINAL MODEL ANALYSIS ==========")

    print("Number of images:", len(images))
    print("Logits shape:", tuple(logits.shape))

    print("\nConfidence:")
    print("Mean:", confidence.mean().item())
    print("Std:", confidence.std().item())
    print("Min:", confidence.min().item())
    print("Max:", confidence.max().item())

    print("\nPrediction entropy:")
    print("Mean:", entropy.mean().item())
    print("Std:", entropy.std().item())

    print("\nNumber of predicted classes:", len(counts))
    print("Most frequent predictions:")
    print(counts.most_common(20))

    print("\nTemperature-2 probability statistics:")
    print("Mean maximum probability:",
          probabilities_t2.max(dim=1).values.mean().item())

    # Save baseline information for future comparisons.
    torch.save({
        "logits": logits,
        "probabilities": probabilities,
        "probabilities_t2": probabilities_t2,
        "predictions": predictions,
        "confidence": confidence,
        "entropy": entropy,
    }, "original_analysis.pt")

    print("\nSaved original_analysis.pt")


if __name__ == "__main__":
    analyze()
