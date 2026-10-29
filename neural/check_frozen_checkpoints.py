"""Safely inspect released frozen checkpoints; optional tiny CPU/ANS smoke.

No MNIST download, training, unsafe pickle fallback, environment installation,
GPU allocation, or benchmark is performed. All checkpoint loads explicitly use
weights_only=True and map_location='cpu'. Checkpoint files are never rewritten.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
from pathlib import Path, PurePosixPath
import re
import sys
import tarfile
from typing import Any


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
CODECS = {
    rate: f"results/iclr_mnist/s1_rate_calibration_v7/s1_rate_calibration_v7_s20260811_lr{rate}_continuous/final.pt"
    for rate in (0, 200, 5000, 50000)
}
CLASSIFIER = "results/iclr_mnist/classifiers/iclr_gray_resnet18_small_v2.pt"
REQUIRED = [*CODECS.values(), CLASSIFIER]
CREDENTIAL_KEYS = re.compile(r"^(?:api[_-]?key|access[_-]?token|auth[_-]?token|password|passwd|secret[_-]?key|private[_-]?key)$", re.I)
CREDENTIAL_VALUES = re.compile(r"(?:gh[pousr]_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{20,}|hf_[A-Za-z0-9]{20,}|sk-[A-Za-z0-9_-]{24,}|-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----)")


def unsafe_archive_name(name: str) -> bool:
    path = PurePosixPath(name)
    return "\\" in name or path.is_absolute() or ".." in path.parts or bool(re.match(r"^[A-Za-z]:", name))


def validate_output(output: Path, checkpoint_root: Path) -> Path:
    output = output.resolve()
    if output.suffix.lower() != ".json":
        raise ValueError("Validation output must be a JSON file")
    for folder in ("results", "data", "validation", "configs"):
        protected = (checkpoint_root / folder).resolve()
        if output == protected or protected in output.parents:
            raise ValueError("Refusing to overwrite checkpoint/data/config input or historical validation directories; use outputs/")
    return output


def scan_metadata(value: Any, path: str = "$", result: dict | None = None) -> dict:
    """Report sensitive field names only, never their actual string values."""
    if result is None:
        result = {"private_absolute_path_fields": [], "credential_like_key_fields": [], "credential_pattern_fields": []}
    if isinstance(value, dict):
        for key, item in value.items():
            field = path + "." + str(key)
            if CREDENTIAL_KEYS.fullmatch(str(key)):
                result["credential_like_key_fields"].append(field)
            scan_metadata(item, field, result)
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            scan_metadata(item, path+f"[{index}]", result)
    elif isinstance(value, str):
        if value.startswith(("/inspire/", "/home/", "/mnt/", "/Users/")) or re.match(r"^[A-Za-z]:[\\/]", value):
            result["private_absolute_path_fields"].append(path)
        if CREDENTIAL_VALUES.search(value):
            result["credential_pattern_fields"].append(path)
    return result


def blobs_from_args(args):
    if args.archive:
        with tarfile.open(args.archive, "r:gz") as archive:
            members = archive.getmembers()
            if len({m.name for m in members}) != len(members):
                raise ValueError("Duplicate archive member")
            if any(not m.isfile() or unsafe_archive_name(m.name) for m in members):
                raise ValueError("Archive contains unsafe names or non-regular files")
            if sum(m.size for m in members) > 200 * 1024 * 1024:
                raise ValueError("Unexpectedly large input archive")
            table = {m.name: m for m in members}
            if any(name not in table for name in REQUIRED):
                raise FileNotFoundError("At least one of the five required checkpoints is absent")
            return {name: archive.extractfile(table[name]).read() for name in REQUIRED}
    root = args.checkpoint_root.resolve()
    result = {}
    for name in REQUIRED:
        source = (root / name).resolve()
        if root not in source.parents or not source.is_file():
            raise FileNotFoundError(f"Required checkpoint absent: {name}")
        result[name] = source.read_bytes()
    return result


def tensor_finiteness(value, torch) -> dict:
    counts = {"tensors": 0, "elements": 0, "nonfinite_fields": []}
    def visit(obj, path):
        if torch.is_tensor(obj):
            counts["tensors"] += 1
            counts["elements"] += obj.numel()
            if not bool(torch.isfinite(obj).all()):
                counts["nonfinite_fields"].append(path)
        elif isinstance(obj, dict):
            for key, item in obj.items():
                visit(item, path+"."+str(key))
        elif isinstance(obj, (list, tuple)):
            for index, item in enumerate(obj):
                visit(item, path+f"[{index}]")
    visit(value, "$")
    return counts


def instantiate_codec(payload, config, torch):
    from validation_iclr_mnist.model import ICLRMNISTCodec, GrayResNet18Small, Discriminator
    codec = ICLRMNISTCodec(
        base_channels=config.model.base_channels, latent_channels=config.model.latent_channels,
        dither_kind=config.dither.kind, mode_slots=len(config.dither.counts),
        codebook_seed=config.dither.codebook_seed,
        decoder_output_activation=config.model.decoder_output_activation,
        gdn_implementation=config.model.gdn_implementation,
        decoder_output_postprocess=config.model.decoder_output_postprocess)
    classifier = GrayResNet18Small(width=config.model.classifier_width)
    critic = Discriminator(width=config.model.classifier_width)
    for module, key in ((codec, "codec"), (classifier, "classifier"), (critic, "discriminator")):
        module.load_state_dict(payload[key], strict=True)
        module.to(device="cpu", dtype=torch.float32).eval()
    return codec, classifier, critic


def cpu_smoke(codec, classifier, critic, torch) -> dict:
    from validation_iclr_mnist.universal_offset_coder import UniversalOffsetEntropyCoder, stateless_level_indexes
    from validation_iclr_mnist.edge_profile import roundtrip_reference
    # A deterministic synthetic ramp is a shape/finite-value smoke input, NOT MNIST.
    x = torch.linspace(0, 1, 28*28, dtype=torch.float32).reshape(1,1,28,28)
    with torch.inference_mode():
        codec.update_entropy_model(force=True)
        output = codec(x, torch.zeros(1,dtype=torch.long), generator=torch.Generator().manual_seed(2718))
        assert output["latent"].shape == (1,12,4,4)
        assert output["reconstruction"].shape == x.shape
        assert all(bool(torch.isfinite(v).all()) for v in output.values() if torch.is_tensor(v))
        reconstruction = codec.postprocess_decoder_output(output["reconstruction"])
        logits = classifier(reconstruction)
        critic_value = critic(reconstruction)
        assert logits.shape == (1,10) and bool(torch.isfinite(logits).all())
        assert bool(torch.isfinite(critic_value).all())
        records = []
        latent = codec.encoder(x)
        entropy = codec.entropy_model.models[0]
        for levels in (1,3,31):
            seed = 2718+65537*levels
            indexes = stateless_level_indexes(latent.shape, levels=levels, seed=seed, device="cpu")
            replay = stateless_level_indexes(latent.shape, levels=levels, seed=seed, device="cpu")
            assert torch.equal(indexes,replay)
            coder = UniversalOffsetEntropyCoder(entropy, levels=levels, symbol_minimum=-128,
                                                symbol_maximum=128, precision=16, escape_tail=True)
            conditioned = coder.encode_decode(latent,indexes,conditioned=True)
            marginal = coder.encode_decode(latent,indexes,conditioned=False)
            assert torch.equal(conditioned.symbols,marginal.symbols)
            assert torch.equal(conditioned.decoded_latent,marginal.decoded_latent)
            assert coder.conditional_table_bytes == levels*coder.marginal_table_bytes
            if levels == 1:
                assert conditioned.payload == marginal.payload
            entry = {"levels":levels, "symbols_equal":True, "latents_equal":True,
                     "stateless_indexes_equal":True, "conditional_table_bytes":coder.conditional_table_bytes,
                     "marginal_table_bytes":coder.marginal_table_bytes,
                     "conditional_payload_bytes":len(conditioned.payload),"marginal_payload_bytes":len(marginal.payload)}
            if levels == 3:
                entry["edge_packet_roundtrips"] = {
                    name: roundtrip_reference(codec,coder,x,levels=levels,seed=seed,conditioned=flag,device=torch.device("cpu"))
                    for name,flag in (("conditioned",True),("marginal",False))}
            records.append(entry)
    return {"status":"PASS", "input":"one synthetic normalized 28x28 ramp, not dataset samples",
            "latent_shape":[1,12,4,4],"reconstruction_shape":[1,1,28,28],"classifier_shape":[1,10],
            "neural_values_finite":True,"ans_checks":records}


def inspect(args) -> dict:
    try:
        import torch
    except ImportError:
        return {"status":"SKIP_DEPENDENCY_MISSING","missing":["torch"],"scope":"No checkpoint load was performed"}
    torch.set_num_threads(1)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass
    raw = blobs_from_args(args)
    report = {"status":"PASS", "runtime":{"python":sys.version.split()[0],"torch":str(torch.__version__),
                                            "device":"cpu","threads":torch.get_num_threads(),
                                            "torch_cuda_build":torch.version.cuda},
              "safety":{"weights_only":True,"map_location":"cpu","unsafe_pickle_fallback":False,
                        "allowlisted_additional_globals":[],"checkpoint_files_rewritten":False},
              "checkpoints":[],"cpu_smoke_requested":args.run_cpu_smoke,
              "scope":"Checkpoint finite values/state compatibility and optional synthetic CPU/ANS smoke; no MNIST evaluation, training or endpoint timing"}
    try:
        import compressai
        report["runtime"]["compressai"] = str(compressai.__version__)
        from validation_iclr_mnist.config import _construct, protocol_fingerprint
        from validation_iclr_mnist.model import GrayResNet18Small
        model_available = True
    except (ImportError, RuntimeError) as exc:
        model_available = False
        report["model_checks"] = {"status":"SKIP_DEPENDENCY_MISSING", "reason_type":type(exc).__name__}
        report["status"] = "PASS_WITH_SKIPS"
    classifier_payload = None
    for name in [CLASSIFIER,*CODECS.values()]:
        blob = raw[name]
        entry = {"relative_path":name,"size_bytes":len(blob),"sha256":hashlib.sha256(blob).hexdigest()}
        static = getattr(torch.serialization,"get_unsafe_globals_in_checkpoint",None)
        entry["unsupported_globals"] = static(io.BytesIO(blob)) if static else "not available in this torch version"
        try:
            payload = torch.load(io.BytesIO(blob),map_location="cpu",weights_only=True)
        except Exception as exc:
            entry.update(status="FAIL_SAFE_LOAD",error_type=type(exc).__name__,
                         error="Safe weights-only load rejected this checkpoint; no unsafe fallback attempted")
            report["checkpoints"].append(entry)
            report["status"] = "FAIL"
            continue
        entry["safe_load"] = "PASS"
        entry["root_keys"] = sorted(str(k) for k in payload)
        entry["metadata_scan"] = scan_metadata(payload)
        entry["finite_values"] = tensor_finiteness(payload,torch)
        assert not entry["finite_values"]["nonfinite_fields"], "Non-finite checkpoint value"
        assert not entry["metadata_scan"]["credential_like_key_fields"] and not entry["metadata_scan"]["credential_pattern_fields"], "Credential-like metadata requires review"
        entry["status"] = "PASS"
        if name == CLASSIFIER:
            classifier_payload = payload
            if model_available:
                clf = GrayResNet18Small(width=32)
                clf.load_state_dict(payload["classifier"],strict=True)
                clf.eval()
                with torch.inference_mode():
                    y = clf(torch.zeros(1,1,28,28))
                    assert y.shape == (1,10) and bool(torch.isfinite(y).all())
                entry["state_dict_compatibility"] = "PASS_STRICT"
                entry["cpu_forward"] = "PASS_SYNTHETIC_ZERO_INPUT"
        else:
            entry["training_step"] = int(payload["step"])
            entry["config_fingerprint"] = payload["config_fingerprint"]
            entry["protocol_fingerprint"] = payload["protocol_fingerprint"]
            assert entry["training_step"] == 3000
            if model_available:
                config = _construct(payload["config"])
                assert protocol_fingerprint(config) == payload["protocol_fingerprint"]
                assert config.seed == 20260811 and not config.train.mixed_precision and not config.train.allow_tf32
                assert config.model.base_channels == 48 and config.model.latent_channels == 12
                assert config.model.decoder_output_activation == "identity" and config.model.gdn_implementation == "compressai"
                codec,classifier,critic = instantiate_codec(payload,config,torch)
                entry["state_dict_compatibility"] = "PASS_STRICT_CODEC_CLASSIFIER_CRITIC"
                if classifier_payload is not None:
                    reference = classifier_payload["classifier"]
                    assert set(reference) == set(payload["classifier"])
                    assert all(torch.equal(reference[k],payload["classifier"][k]) for k in reference)
                    entry["embedded_classifier_matches_standalone"] = True
                if args.run_cpu_smoke:
                    entry["cpu_smoke"] = cpu_smoke(codec,classifier,critic,torch)
        report["checkpoints"].append(entry)
    report["all_five_checkpoints_loaded"] = len(report["checkpoints"]) == 5 and all(x.get("safe_load")=="PASS" for x in report["checkpoints"])
    report["metadata_contains_private_paths"] = any(x.get("metadata_scan",{}).get("private_absolute_path_fields") for x in report["checkpoints"])
    report["metadata_path_note"] = "Only field names are reported; original checkpoint bytes were retained unchanged"
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--archive",type=Path,help="Author input archive; readers normally use --checkpoint-root")
    group.add_argument("--checkpoint-root",type=Path,default=PACKAGE_ROOT,help="Directory containing original results/iclr_mnist paths; default release root")
    parser.add_argument("--run-cpu-smoke",action="store_true",help="Tiny synthetic neural forward and ANS/edge packet checks, not a dataset replay")
    parser.add_argument("--output",type=Path,default=PACKAGE_ROOT/"outputs/checkpoint_validation.json")
    args = parser.parse_args()
    args.output = validate_output(args.output,args.checkpoint_root)
    try:
        report = inspect(args)
    except Exception as exc:
        report = {"status":"FAIL","error_type":type(exc).__name__,
                  "error":"Checkpoint validation stopped; no unsafe fallback or source rewrite was performed"}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(report,indent=2)+"\n",encoding="utf-8")
    print(json.dumps(report,indent=2))
    return 0 if report["status"] in {"PASS","PASS_WITH_SKIPS"} else 2


if __name__ == "__main__":
    raise SystemExit(main())
