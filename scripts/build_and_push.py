"""Build the satnogs-id Doppler dataset and (optionally) push it to the HF Hub. Default: save
locally only. Pass --push (with HUGGING_FACE_HUB_TOKEN or HF_TOKEN in the env) to publish --
mirroring satnogs-signal's build_and_push. Run in-container:
    docker compose run --rm app python scripts/build_and_push.py --dataset _eval/geoscan
    docker compose run --rm app python scripts/build_and_push.py --cluster geoscan --push
"""
from __future__ import annotations
import argparse
import os
import sys
from pathlib import Path

from satnogs_id.data.build import CLUSTERS, harvest
from satnogs_id.data.dataset import Dataset, manifest_from_dir
from satnogs_id.data.publish import REPO_ID, build_records, push, to_hf_dataset


def main() -> None:
    ap = argparse.ArgumentParser(description="Build + optionally publish the satnogs-id Doppler dataset.")
    ap.add_argument("--cluster", choices=sorted(CLUSTERS), default="geoscan")
    ap.add_argument("--dataset", help="existing harvested dataset dir (else harvest the cluster fresh)")
    ap.add_argument("--out", default="_dataset_build/satnogs-id-doppler")
    ap.add_argument("--push", action="store_true", help="publish to the HF Hub (needs a token)")
    ap.add_argument("--public", action="store_true", help="make the dataset public after pushing")
    ap.add_argument("--card", default="scripts/dataset_card.md", help="dataset card (README) to upload")
    args = ap.parse_args()

    names = CLUSTERS[args.cluster]["truth"]
    if args.dataset:
        d = Path(args.dataset)
        ds = Dataset.load(d) if (d / "manifest.json").exists() else manifest_from_dir(d)
    else:
        ds = harvest(args.cluster, args.out + "_raw")

    records = build_records(ds, names)
    hf = to_hf_dataset(records)
    print(f"{hf.num_rows} rows; per-object: "
          + ", ".join(f"{names.get(n, n)}:{sum(1 for r in records if r['norad'] == n)}"
                      for n in sorted(names)))
    hf.save_to_disk(args.out)
    print(f"saved dataset to ./{args.out}/")

    if args.push:
        if not (os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")):
            print("ERROR: no HF token in env. Add HUGGING_FACE_HUB_TOKEN to .env and re-run with --push.",
                  file=sys.stderr)
            sys.exit(1)
        print(f"pushing to the Hub: {REPO_ID}...")
        push(hf)
        print("pushed.")

        from huggingface_hub import HfApi
        api = HfApi()
        card = Path(args.card)
        if card.exists():
            api.upload_file(path_or_fileobj=str(card), path_in_repo="README.md",
                            repo_id=REPO_ID, repo_type="dataset")
            print(f"uploaded dataset card from {card}")
        if args.public:
            try:
                api.update_repo_settings(REPO_ID, repo_type="dataset", private=False)
            except (AttributeError, TypeError):
                api.update_repo_visibility(REPO_ID, repo_type="dataset", private=False)
            print("set dataset visibility: public")
    else:
        print("(not pushed -- re-run with --push once a HF token is set)")


if __name__ == "__main__":
    main()
