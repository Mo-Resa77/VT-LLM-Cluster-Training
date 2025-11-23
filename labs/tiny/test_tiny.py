"""
Quick accuracy check for the tiny BERT classifier.
$ python labs/tiny/test_tiny.py --ckpt tiny_out
"""
import os, argparse, random, torch
from datasets import load_dataset
from transformers import (AutoTokenizer,
                          AutoModelForSequenceClassification)

# ---------------------------------------------------------------------
def compute_accuracy(preds, labels):
    "Fallback metric if scikit-learn is unavailable."
    return (preds == labels).sum().item() / len(labels)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="tiny_out",
                    help="dir with config.json + weights + tokenizer")
    ap.add_argument("--batch", type=int, default=32)
    args = ap.parse_args()

    # Corrected cache path
    cache_root = os.environ.setdefault(
        "HF_HOME", os.path.join(os.path.expanduser("~"), ".cache", "hf_tiny"))
    
    # ----- load model + tokenizer -------------------------------------
    tok   = AutoTokenizer.from_pretrained(args.ckpt)
    model = AutoModelForSequenceClassification.from_pretrained(
                args.ckpt, torch_dtype=torch.float32)
    model.eval()

    # ----- dataset (same 512-row slice) --------------------------------
    ds = load_dataset("ag_news", split="test[:512]",
                      cache_dir=os.path.join(cache_root, "ds"))
                      
    def encode(b):
        return tok(b["text"], truncation=True,
                   padding="max_length", max_length=128)
    ds = ds.map(encode, batched=True)

    # keep text column available for pretty-print ↓
    ds.set_format("torch",
                  columns=["input_ids", "attention_mask", "label"],
                  output_all_columns=True)    # 🔑 keeps 'text'

    # ----- accuracy ----------------------------------------------------
    try:
        import evaluate                       # needs scikit-learn
        metric = evaluate.load("accuracy")
        use_metric = True
    except (ImportError, OSError):
        use_metric = False

    correct, total = 0, 0
    for batch in torch.utils.data.DataLoader(ds, batch_size=args.batch):
        with torch.no_grad():
            logits = model(batch["input_ids"],
                           attention_mask=batch["attention_mask"]).logits
        preds = logits.argmax(dim=-1)
        if use_metric:
            metric.add_batch(predictions=preds, references=batch["label"])
        else:
            correct += (preds == batch["label"]).sum().item()
            total   += preds.size(0)

    acc = (metric.compute()["accuracy"]
           if use_metric else correct/total)
    print(f"\nAccuracy on 512-row slice: {acc:.3f}")

    # ----- show 3 random predictions ----------------------------------
    print("\n↪ Example predictions")
    label_names = ds.features["label"].names       # ["World", "Sports", …]
    for i in random.sample(range(len(ds)), 3):
        txt  = ds[i]["text"][:80].replace("\n", " ") + " ..."
        gold = label_names[ds[i]["label"]]
        pred = label_names[
            model(tok(ds[i]["text"], return_tensors="pt").input_ids
                 ).logits.argmax(-1).item()
        ]
        print(f"\n• {txt}\n  gold={gold:<8}  pred={pred}")

if __name__ == "__main__":
    main()