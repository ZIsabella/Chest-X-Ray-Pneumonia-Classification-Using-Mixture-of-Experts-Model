import json
import os
from pathlib import Path
import torch
from sklearn.metrics import classification_report, confusion_matrix

from app.models.moe_system import CXRMoESystem
from training.dataset import make_dataloader


def evaluate_and_save_json(
    weights_path: str = "weights/best_model.pt",
    data_root: str = "data",
    output_json: str = "outputs/run_20261007_01/test_metrics.json",
    batch_size: int = 16,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Evaluator running on: {device}")


    ckpt = torch.load(weights_path, map_location=device, weights_only=False)
    class_names = ckpt.get("class_names", ["COVID19", "NORMAL", "PNEUMONIA", "TURBERCULOSIS"])

    model = CXRMoESystem(num_classes=len(class_names)).to(device)
    model.load_state_dict(ckpt["state_dict"], strict=True)
    model.eval()

  
    test_loader = make_dataloader(data_root=data_root, split="test", batch_size=batch_size, shuffle=False)

    y_true, y_pred = [], []
    print("Evaluating test set...")
    with torch.no_grad():
        for x, y in test_loader:
            out = model(x.to(device))
            logits = out[0] if isinstance(out, (tuple, list)) else out
            y_true.extend(y.tolist())
            y_pred.extend(torch.argmax(logits, dim=1).cpu().tolist())

 
    report_dict = classification_report(
        y_true, y_pred, target_names=class_names, digits=4, output_dict=True
    )
    conf_matrix = confusion_matrix(y_true, y_pred).tolist()


    results = {
        "dataset_split": "test",
        "total_samples": len(y_true),
        "class_names": class_names,
        "classification_report": report_dict,
        "confusion_matrix": conf_matrix,
    }

    out_path = Path(output_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4, ensure_ascii=False)

    print(f"Results successfully saved to: {out_path.resolve()}")
    return results


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", type=str, default="weights/best_model.pt")
    parser.add_argument("--data", type=str, default="data")
    parser.add_argument("--out", type=str, default="outputs/run_20261007_01/test_metrics.json")
    args = parser.parse_args()

    evaluate_and_save_json(weights_path=args.weights, data_root=args.data, output_json=args.out)
