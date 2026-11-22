"""Head-scaling baseline: multiply the classifier head of the original model by a factor, so the model becomes less
confident on everything, and save that as the submission. Also prints how the factor changes confidence and loss on
the forget images (all relative to the original model's own predictions).
  python script.py            (expects model.pt and forget_images.npz in this folder)"""
import numpy as np
import timm
import torch
import torch.nn.functional as F

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SCALE = 0.5                                     # factor applied to head.weight and head.bias in the submission


def load_model(path):
    m = timm.create_model("vit_base_patch16_224", pretrained=False, num_classes=100, init_values=1e-4)
    m.load_state_dict(torch.load(path, map_location="cpu"))
    return m.to(DEVICE).eval()


@torch.no_grad()
def get_logits(model, images, bs=64):
    out = []
    for i in range(0, len(images), bs):
        x = images[i:i + bs].to(DEVICE).permute(0, 3, 1, 2).float() / 255.0
        x = (x - 0.5) / 0.5
        with torch.autocast(device_type=DEVICE, dtype=torch.float16):
            out.append(model(x).float().cpu())
    return torch.cat(out)


def make_submission(scale, out_path="submission.pt"):
    m = load_model("model.pt")
    with torch.no_grad():
        m.head.weight.mul_(scale)
        m.head.bias.mul_(scale)
    sd = {k: v.detach().cpu().float() for k, v in m.state_dict().items()}
    torch.save(sd, out_path)


def main():
    make_submission(SCALE)
    data = np.load("forget_images.npz")
    images = torch.from_numpy(data[data.files[0]])      # (3500, 224, 224, 3)
    logits = get_logits(load_model("model.pt"), images)
    orig_pred = logits.argmax(1)
    print("scale | avg confidence | avg loss | spread of loss | same answers")
    for s in [1.0, 0.8, 0.6, 0.5, 0.4, 0.3, 0.2]:
        scaled = logits * s
        probs = F.softmax(scaled, dim=1)
        loss = F.cross_entropy(scaled, orig_pred, reduction="none")
        same = (scaled.argmax(1) == orig_pred).float().mean()
        print(f"{s:.1f}   | {probs.max(1).values.mean():.3f}          "
              f"| {loss.mean():.4f}   | {loss.std():.4f}         | {same:.3f}")


if __name__ == "__main__":
    main()
