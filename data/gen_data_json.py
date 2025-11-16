import os
import json
from pathlib import Path
import pandas as pd
import argparse

def collect_split(base_path, curation_list, idx_start, idx_end):
    csv_path = os.path.join(base_path, "benchmark-meta.csv")
    df = pd.read_csv(csv_path)
    # TODO align with previous exp
    df = df.sort_values(by="hash", ascending=True).reset_index(drop=True)
    selected_df = df.iloc[idx_start:idx_end]

    entries = {}

    for idx, row in selected_df.iterrows():
        scene_hash = row['hash']
        for curation in curation_list:
            strategy = curation["strategy"]

            if strategy == "underfitting":
                iters = curation["iters"]
                ref_every = curation["ref_every"]

                key = f"{scene_hash}_{strategy}_{iters}"
                img_path = os.path.join(base_path, scene_hash, os.path.join("gs_recon", "train", f"ours_{iters}"))

                cur_entries = collect_underfitting_entries(img_path, key, ref_every)
            elif strategy == "sparse":
                train_every = curation["train_every"]
                key = f"{scene_hash}_{strategy}_{train_every}"
                img_path = os.path.join(base_path, scene_hash, os.path.join(f"gs_recon_sparse_{train_every}", "test", f"ours_30000"))

                cur_entries = collect_sparse_entries(img_path, key, train_every)

            entries.update(cur_entries)

    return entries

def collect_underfitting_entries(img_path, key, ref_every=10):
    cur_entries = {}
    render_dir = Path(img_path) / "renders"
    gt_dir = Path(img_path) / "gt"
    
    for render_img in sorted(render_dir.glob("*.png")):
        frame_id = render_img.stem
        file_name = render_img.name
        ref_id = int(frame_id) // ref_every * ref_every

        ref_stem = f"{ref_id:0{len(render_img.stem)}d}"

        ref_file_name = f"{ref_stem}.png"

        render_path = render_dir / file_name
        gt_path = gt_dir / file_name
        ref_path = gt_dir / ref_file_name

        assert render_path.exists(), f"Missing render file: {render_path}"
        assert gt_path.exists(), f"Missing gt file: {gt_path}"
        assert ref_path.exists(), f"Missing ref file: {ref_path}"

        cur_entries[f"{key}_{frame_id}"] = {
            "render": str(render_path),
            "gt": str(gt_path),
            "ref": str(ref_path),
        }

    return cur_entries

def collect_sparse_entries(img_path, key, ref_every=10):
    cur_entries = {}
    render_dir = Path(img_path) / "renders"
    gt_dir = Path(img_path) / "gt"
    ref_dir = Path(img_path.replace("test", "train")) / "gt"

    ref_id = -1
    for render_img in sorted(render_dir.glob("*.png")):
        frame_id = render_img.stem
        file_name = render_img.name
        ref_id = ref_id + 1 if int(frame_id) % (ref_every - 1) == 0 else ref_id

        ref_stem = f"{ref_id:0{len(render_img.stem)}d}"

        ref_file_name = f"{ref_stem}.png"

        render_path = render_dir / file_name
        gt_path = gt_dir / file_name
        ref_path = ref_dir / ref_file_name

        if not ref_path.exists():
            print(f"[Warning] Missing ref file: {ref_path}, skip {frame_id}")
            continue

        assert render_path.exists(), f"Missing render file: {render_path}"
        assert gt_path.exists(), f"Missing gt file: {gt_path}"

        cur_entries[f"{key}_{frame_id}"] = {
            "render": str(render_path),
            "gt": str(gt_path),
            "ref": str(ref_path),
        }

    return cur_entries


if __name__ == '__main__':
    """
        generate dataset json for dataloader
    """

    parser = argparse.ArgumentParser(description="Run DifixPipeline on dataset.")
    parser.add_argument("--use_underfitting", action="store_true",
                        help="Whether to use model underfitting curation (default: False)")
    parser.add_argument("--use_sparse", action="store_true",
                        help="Whether to use sparse curation (default: False)")
    args = parser.parse_args()

    base_path = Path("../dataset/DL3DV-10K-Benchmark")

    output_path = Path("data")
    curation_list = []
    if args.use_underfitting:
        curation_list.append(
            {
                "strategy": "underfitting",
                "iters": 1000,
                "ref_every": 10,
            },
        )
    if args.use_sparse:
        curation_list.append(
            {
                "strategy": "sparse",
                "train_every": 10,
            },
        )

    data = {
        "train": collect_split(base_path, curation_list, 0, 120),
        "val": collect_split(base_path, curation_list, 120, 140),
    }
    if args.use_underfitting and args.use_sparse:
        json_file_name = "data.json"
    elif args.use_underfitting:
        json_file_name = "data_underfitting.json"
    else:
        json_file_name = "data_sparse.json"

    data_json_file = os.path.join(output_path, json_file_name)
    with open(data_json_file, "w") as f:
        json.dump(data, f, indent=2)

    print("data.json generated successfully.")
