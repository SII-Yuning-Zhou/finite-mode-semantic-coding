"""Extract only data/config JSON from an author-supplied server audit archive.

This maintenance utility never executes or expands archive paths. Worker logs
and arbitrary files are excluded. Original numerical fields and fingerprints
are preserved; absolute development paths are converted to repository-relative
paths in the public copies only. Original source files are never modified.
"""
from pathlib import Path, PurePosixPath
import argparse
import hashlib
import json
import re
import tarfile

PUB = Path(__file__).resolve().parents[1]
REPORT = re.compile(r"rate_(0|200|5000|50000)_rep_[0-4]\.json")
CONFIG = re.compile(r"rate_(0|200|5000|50000)_rep_[0-4]\.config\.json")
METADATA = {"aggregate.json", "completion.json", "provenance.json", "status.json"}


def sanitize(value, changes, key=""):
    if isinstance(value, dict):
        return {k: sanitize(v, changes, key + "." + k) for k, v in value.items()}
    if isinstance(value, list):
        return [sanitize(v, changes, key + f"[{i}]") for i, v in enumerate(value)]
    if isinstance(value, str) and value.startswith(("/inspire/", "/home/", "C:\\", "D:\\", "E:\\")):
        marker = "mode-limited-target-semantic-coding/"
        normalized = value.replace("\\", "/")
        result = normalized.split(marker, 1)[1] if marker in normalized else "REDACTED_LOCAL_PATH/" + PurePosixPath(normalized).name
        changes.append({"field": key, "action": "absolute local path replaced; value not retained in public log"})
        return result
    return value


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--archive", required=True, type=Path)
    parser.add_argument("--output", type=Path, default=PUB / "data/neural/v81_reports")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    changes, copied, skipped = [], [], []
    seen = set()
    with tarfile.open(args.archive, "r:gz") as archive:
        for member in archive.getmembers():
            name = member.name
            if name in seen:
                raise ValueError(f"Duplicate archive entry: {name}")
            seen.add(name)
            if not member.isfile() or PurePosixPath(name).name != name:
                raise ValueError(f"Unexpected non-flat archive entry: {name}")
            if not (REPORT.fullmatch(name) or CONFIG.fullmatch(name) or name in METADATA):
                skipped.append(name)
                continue
            if member.size > 8 * 1024 * 1024:
                raise ValueError("Unexpectedly large JSON member")
            payload = archive.extractfile(member).read()
            data = sanitize(json.loads(payload), changes, name)
            (args.output / name).write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
            copied.append({"name": name, "source_member_sha256": hashlib.sha256(payload).hexdigest()})
        unit_bytes = archive.extractfile("unit_tests.log").read()
        unit_text = unit_bytes.decode("utf-8", errors="replace")
        count = re.search(r"Ran (\d+) tests?", unit_text)
        assert count and int(count.group(1)) == 4 and unit_text.strip().splitlines()[-1] == "OK"
        server_tests = {"kind": "historical_server_test_summary_not_a_local_rerun",
                        "tests": 4, "status": "PASS", "skips": 0,
                        "source_member": "unit_tests.log",
                        "source_member_sha256": hashlib.sha256(unit_bytes).hexdigest()}
        (args.output.parent / "original_server_unit_tests.json").write_text(json.dumps(server_tests, indent=2)+"\n", encoding="utf-8")
    assert sum(bool(REPORT.fullmatch(row["name"])) for row in copied) == 20
    assert sum(bool(CONFIG.fullmatch(row["name"])) for row in copied) == 20
    record = {"status": "PASS", "protocol": "UQ-CONDITIONAL-RANS-ESCAPE-V2",
              "source_archive_sha256": hashlib.sha256(args.archive.read_bytes()).hexdigest(),
              "copied": copied, "excluded_entries": skipped, "sanitized_fields": changes,
              "preservation": "All numerical fields, schedule seeds, original git status and scientific fingerprints unchanged",
              "fingerprint_note": "Source fingerprints refer to original server artifacts; path-sanitized copies are not byte-identical originals"}
    (args.output.parent / "sanitization.json").write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": "PASS", "reports": 20, "configs": 20, "metadata": 4, "excluded_logs": len(skipped), "sanitized_fields": len(changes)}))


if __name__ == "__main__":
    main()
