"""Safely import the author-supplied frozen artifacts; never unpickle checkpoints.

Only the 26 explicitly approved regular files are accepted. Checkpoint bytes
remain unchanged. Public JSON/CSV/Markdown copies redact unrelated local paths
and internal identifiers, retaining numeric values, seeds and fingerprints.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import re
import tarfile

ROOT = Path(__file__).resolve().parents[1]
RATES = (0, 200, 5000, 50000)
EDGE = "results/iclr_mnist/server_controller_edge_metrics_v1"
WEIGHTS = tuple(f"results/iclr_mnist/s1_rate_calibration_v7/s1_rate_calibration_v7_s20260811_lr{rate}_continuous/final.pt" for rate in RATES) + (
    "results/iclr_mnist/classifiers/iclr_gray_resnet18_small_v2.pt",)
PROFILES = tuple(f"{EDGE}/edge_profile_rate_{rate}_seed_20260811_{device}.json" for rate in RATES for device in ("cpu", "cuda"))
APPROVED = frozenset((*WEIGHTS, *PROFILES,
    *(name[:-5]+".config.json" for name in PROFILES),
    "results/iclr_mnist/classifiers/iclr_gray_resnet18_small_v2.json",
    "results/iclr_mnist/server_controller_literature_v7/calibration_pass.json",
    *(f"{EDGE}/edge_deployment_metrics.{ext}" for ext in ("json", "csv", "md"))))
MANIFEST = "data/neural/frozen_artifacts_manifest.json"
IDENTITY = "data/neural/frozen_archive_identity.json"
MAX_TOTAL = 128 * 1024 * 1024
MAX_WEIGHT = 64 * 1024 * 1024
MAX_TEXT = 2 * 1024 * 1024


def sha_bytes(value):
    return hashlib.sha256(value).hexdigest()


def sha_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024*1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_members(members, require_complete=True):
    seen = set()
    total = 0
    for member in members:
        name = member.name
        pure = PurePosixPath(name)
        if ("\\" in name or pure.is_absolute() or any(x in ("", ".", "..") for x in name.split("/"))
                or re.match(r"^[A-Za-z]:", name)):
            raise ValueError("Unsafe archive member path")
        if name in seen or name.casefold() in {x.casefold() for x in seen}:
            raise ValueError("Duplicate archive member")
        seen.add(name)
        if name not in APPROVED:
            raise ValueError("Unapproved archive member: " + name)
        if member.type not in (tarfile.REGTYPE, tarfile.AREGTYPE) or member.sparse is not None:
            raise ValueError("Only ordinary regular files are approved")
        if any(key.startswith("GNU.sparse") for key in member.pax_headers):
            raise ValueError("Sparse archive entries are not approved")
        limit = MAX_WEIGHT if name.endswith(".pt") else MAX_TEXT
        if member.size < 0 or member.size > limit:
            raise ValueError("Archive member exceeds approved size limit")
        total += member.size
        if total > MAX_TOTAL:
            raise ValueError("Archive exceeds total extraction limit")
    if require_complete and seen != APPROVED:
        raise ValueError("Missing approved artifacts: " + ", ".join(sorted(APPROVED-seen)))
    return total


def clean_string(value, field, changes):
    result = value
    # Keep the repository-relative suffix. No old private prefix is retained in
    # the redaction log; experiment seeds/fingerprints are deliberately untouched.
    pattern = r"(?:[A-Za-z]:[\\/]|/inspire/|/home/|/Users/)[^\s\"'<>]+"
    def replace(match):
        raw = match.group(0).replace("\\", "/")
        for marker in ("mode-limited-target-semantic-coding/", "mode_limited_target_semantic_coding/", "TCOM-Semantic-Coding/"):
            if marker in raw:
                return raw.split(marker, 1)[1]
        return "REDACTED_LOCAL_PATH/" + raw.rsplit("/", 1)[-1]
    result = re.sub(pattern, replace, result)
    result = re.sub(r"https?://(?:10\.\d+\.\d+\.\d+|192\.168\.\d+\.\d+|172\.(?:1[6-9]|2\d|3[01])\.\d+\.\d+)(?::\d+)?", "REDACTED_INTERNAL_ADDRESS", result)
    if result != value:
        changes.append({"field":field,"action":"unrelated local path/internal address replaced"})
    return result


def sanitize(value, changes, field="$"):
    if isinstance(value, dict):
        output = {}
        identities = {"hostname", "host_name", "username", "user_name", "host_ip", "ip_address", "internal_address", "machine_id", "gpu_uuid", "uuid"}
        for key, item in value.items():
            path = field+"."+key
            if key.lower() in identities and isinstance(item, str):
                output[key] = "REDACTED_INTERNAL_IDENTIFIER"
                changes.append({"field":path,"action":"unrelated internal identifier replaced"})
            else:
                output[key] = sanitize(item, changes, path)
        return output
    if isinstance(value, list):
        return [sanitize(item, changes, field+f"[{i}]") for i,item in enumerate(value)]
    return clean_string(value, field, changes) if isinstance(value, str) else value


def numeric_signature(value):
    if isinstance(value, dict):
        return {key:numeric_signature(item) for key,item in value.items() if not isinstance(item,str)}
    if isinstance(value,list):
        return [numeric_signature(item) for item in value]
    return None if isinstance(value,str) else value


def destination(root, relative):
    target = root.joinpath(*PurePosixPath(relative).parts)
    if not target.resolve().is_relative_to(root.resolve()):
        raise ValueError("Import destination escapes release root")
    return target


def write_new_or_identical(path, data):
    path.parent.mkdir(parents=True,exist_ok=True)
    if path.exists():
        if path.read_bytes() != data:
            raise FileExistsError("Refusing to overwrite existing different artifact: " + path.name)
    else:
        with path.open("xb") as stream:
            stream.write(data)


def import_archive(archive, root=ROOT):
    archive = Path(archive).resolve(strict=True)
    root = Path(root).resolve(strict=True)
    signature = {"bytes":archive.stat().st_size,"mtime_ns":archive.stat().st_mtime_ns}
    if signature["bytes"] > MAX_TOTAL:
        raise ValueError("Compressed archive exceeds size limit")
    identity_path = destination(root, IDENTITY)
    with tarfile.open(archive,"r:gz") as source:
        members = []
        for member in source:
            if len(members) >= len(APPROVED):
                raise ValueError("Archive exceeds approved member count")
            members.append(member)
            # Validate the current header before tarfile seeks over its payload.
            validate_members(members, require_complete=False)
        total = validate_members(members)
        if identity_path.exists():
            identity = json.loads(identity_path.read_text(encoding="utf-8"))
            if identity["source_archive_signature"] != signature:
                raise ValueError("Source archive metadata changed; do not silently reuse its prior digest")
            archive_sha = identity["source_archive_sha256"]
        else:
            archive_sha = sha_file(archive)  # Exactly one hash of this transferred input.
            identity = {"schema_version":1,"source_archive_name":archive.name,
                        "source_archive_signature":signature,"source_archive_sha256":archive_sha,
                        "archive_sha256_policy":"Computed once on receipt; reused while source metadata is unchanged"}
            write_new_or_identical(identity_path,(json.dumps(identity,indent=2)+"\n").encode())
        entries = []
        for member in members:
            with source.extractfile(member) as stream:
                original = stream.read(member.size+1)
            if len(original) != member.size:
                raise ValueError("Truncated member")
            changes = []
            if member.name.endswith(".pt"):
                public = original
                kind = "frozen_checkpoint"
            elif member.name.endswith(".json"):
                parsed = json.loads(original.decode("utf-8-sig"))
                clean = sanitize(parsed,changes)
                if numeric_signature(parsed) != numeric_signature(clean):
                    raise ValueError("Sanitization changed a numeric/bool value")
                public = (json.dumps(clean,indent=2,allow_nan=False)+"\n").encode("utf-8")
                kind = "profile" if member.name in PROFILES else "metadata_json"
            else:
                public = clean_string(original.decode("utf-8-sig"),"$text",changes).encode("utf-8")
                kind = "ledger_text"
            write_new_or_identical(destination(root,member.name),public)
            entries.append({"path":member.name,"kind":kind,"source_bytes":len(original),
                            "source_sha256":sha_bytes(original),"public_bytes":len(public),
                            "public_sha256":sha_bytes(public),"sanitized":bool(changes),"redactions":changes,
                            "numeric_values_preserved":True})
    if signature != {"bytes":archive.stat().st_size,"mtime_ns":archive.stat().st_mtime_ns}:
        raise RuntimeError("Source archive changed during import")
    manifest = {"schema_version":1,"status":"PASS","source_archive_name":archive.name,
                "source_archive_sha256":archive_sha,"source_archive_signature":signature,
                "members":len(entries),"source_uncompressed_bytes":total,"checkpoint_count":5,
                "text_record_count":21,"entries":entries,
                "checkpoint_handling":"Original bytes retained; no unpickling performed by importer",
                "fingerprint_note":"Original experiment fingerprints describe original files; public sanitized file digests are recorded separately"}
    write_new_or_identical(destination(root,MANIFEST),(json.dumps(manifest,indent=2)+"\n").encode())
    return manifest


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive",type=Path,required=True)
    parser.add_argument("--root",type=Path,default=ROOT)
    args=parser.parse_args()
    report=import_archive(args.archive,args.root)
    print(json.dumps({key:report[key] for key in ("status","members","checkpoint_count","text_record_count","source_archive_sha256")},indent=2))


if __name__ == "__main__":
    main()
