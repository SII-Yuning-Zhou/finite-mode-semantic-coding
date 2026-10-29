"""Read-only author-side export of the missing frozen weights/profile records.

Creates one new tar.gz OUTSIDE the research repository. It never deletes,
moves, overwrites or executes source artifacts. This is a transfer bundle for
review/sanitization, not an automatically approved public release asset.
"""
from pathlib import Path
import argparse
import json
import re
import tarfile


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--list-only", action="store_true")
    args = parser.parse_args()
    repo = args.repo.resolve(strict=True)
    if not repo.is_dir():
        raise ValueError("Research repository must be a directory")
    output = args.output.resolve()
    if output == repo or repo in output.parents:
        raise ValueError("Output must be outside the repository/source directories")
    if not output.parent.is_dir():
        raise ValueError("Output parent must already exist")
    weights = [
        Path(f"results/iclr_mnist/s1_rate_calibration_v7/s1_rate_calibration_v7_s20260811_lr{rate}_continuous/final.pt")
        for rate in (0, 200, 5000, 50000)
    ] + [Path("results/iclr_mnist/classifiers/iclr_gray_resnet18_small_v2.pt")]
    optional = [Path("results/iclr_mnist/classifiers/iclr_gray_resnet18_small_v2.json"),
                Path("results/iclr_mnist/server_controller_literature_v7/calibration_pass.json")]
    edge = Path("results/iclr_mnist/server_controller_edge_metrics_v1")
    profiles = [edge / f"edge_profile_rate_{rate}_seed_20260811_{device}.json"
                for rate in (0, 200, 5000, 50000) for device in ("cpu", "cuda")]
    required = weights + profiles
    folder = repo / edge
    if folder.is_dir():
        for path in sorted(folder.iterdir()):
            if path.is_file() and (re.fullmatch(r"edge_profile_rate_(0|200|5000|50000)_seed_20260811_(cpu|cuda)(?:\.config)?\.json", path.name)
                                   or path.name in {"edge_deployment_metrics.json", "edge_deployment_metrics.csv", "edge_deployment_metrics.md"}):
                candidate = edge / path.name
                if candidate not in required:
                    optional.append(candidate)
    existing, missing = [], []
    for rel in required + optional:
        source = (repo / rel).resolve()
        if repo not in source.parents:
            raise ValueError("Unexpected source outside repository")
        (existing if source.is_file() else missing).append(rel.as_posix())
    result = {"existing": existing, "missing": missing, "required_frozen_weights": [x.as_posix() for x in weights],
              "expected_edge_profiles": [x.as_posix() for x in profiles],
              "complete": all((repo / x).is_file() for x in required),
              "edge_profile_records_found": sum(x.startswith(edge.as_posix()+"/edge_profile_rate_") and not x.endswith(".config.json") for x in existing),
              "note": "Source untouched. Inspect paths/metadata before publishing transferred records."}
    if not args.list_only:
        if not existing:
            raise FileNotFoundError("None of the requested artifacts were found; no archive created")
        if output.exists():
            raise FileExistsError("Refusing to overwrite existing transfer archive")
        with tarfile.open(output, "x:gz") as archive:
            for rel in existing:
                archive.add((repo / rel).resolve(), arcname=rel, recursive=False)
        result["output"] = str(output)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
