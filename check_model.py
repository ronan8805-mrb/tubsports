import json
from pathlib import Path

for name in ["win_model", "place_model"]:
    meta_path = Path(f"horse/data/models/{name}/meta.json")
    if meta_path.exists():
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        ens = meta.get("metrics", {}).get("ensemble", {})
        print(f"CURRENT {name.upper()} (BEFORE speed features):")
        print(f"  AUC:      {ens.get('auc', 'N/A')}")
        print(f"  Brier:    {ens.get('brier', 'N/A')}")
        print(f"  Log Loss: {ens.get('logloss', 'N/A')}")
        print(f"  Accuracy: {ens.get('accuracy', 'N/A')}")
        print(f"  Features: {len(meta.get('feature_names', []))}")
        fi = meta.get("feature_importance", {})
        top10 = list(fi.items())[:10]
        print(f"  Top 10 features:")
        for i, (f, v) in enumerate(top10):
            print(f"    {i+1}. {f}: {v:.1f}")
        print()
    else:
        print(f"No {name} model found")
