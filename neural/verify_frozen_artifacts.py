"""Verify frozen artifact identity and reaggregate original endpoint records.

This standard-library check never loads PyTorch pickles, trains a model, or
remeasures latency. Original per-call timing samples are recomputed instead.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import statistics
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0,str(ROOT))
from neural.import_frozen_artifacts import APPROVED, EDGE, MANIFEST, PROFILES, RATES, WEIGHTS
from reproduce.tables import recompute_table_v


def file_sha(path):
    value=hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda:stream.read(1024*1024),b""):
            value.update(chunk)
    return value.hexdigest()


def percentile(values, fraction):
    ordered=sorted(values)
    position=(len(ordered)-1)*fraction
    lower=math.floor(position)
    upper=math.ceil(position)
    return ordered[lower]+(ordered[upper]-ordered[lower])*(position-lower)


def equivalent(left,right):
    if isinstance(left,(int,float)) and not isinstance(left,bool):
        return isinstance(right,(int,float)) and math.isclose(left,right,rel_tol=1e-12,abs_tol=1e-12)
    return left==right


def package_relative(value,root):
    if isinstance(value,dict):
        return {key:package_relative(item,root) for key,item in value.items()}
    if isinstance(value,list):
        return [package_relative(item,root) for item in value]
    if isinstance(value,str):
        for prefix in (str(root),root.as_posix()):
            value=value.replace(prefix+"\\","").replace(prefix+"/","")
            if value==prefix:
                value="."
    return value


def verify(root=ROOT, output=None):
    root=Path(root).resolve()
    output=Path(output or root/"outputs/frozen_artifacts_validation.json")
    if not output.is_absolute():
        output=root/output
    output.parent.mkdir(parents=True,exist_ok=True)
    manifest=json.loads((root/MANIFEST).read_text(encoding="utf-8"))
    assert manifest["status"]=="PASS" and manifest["members"]==26
    entries={item["path"]:item for item in manifest["entries"]}
    assert set(entries)==APPROVED and len(entries)==26
    for relative,entry in entries.items():
        path=(root/relative).resolve(strict=True)
        assert path.is_relative_to(root) and path.is_file()
        assert path.stat().st_size==entry["public_bytes"]
        assert file_sha(path)==entry["public_sha256"], relative
        assert entry["numeric_values_preserved"]
        if relative in WEIGHTS:
            assert entry["source_sha256"]==entry["public_sha256"]
            assert entry["source_bytes"]==entry["public_bytes"] and not entry["sanitized"]
    profile_records=[]
    series_count=0
    timing_samples=0
    links=[]
    observed=set()
    for relative in PROFILES:
        profile=json.loads((root/relative).read_text(encoding="utf-8"))
        assert profile["status"]=="PASS" and profile["profile_protocol"]=="ICLR-MNIST-EDGE-PROFILE-V1"
        rate=profile["rate_weight"]
        device=profile["hardware"]["device"]
        assert rate in RATES and device in ("cpu","cuda:0") and profile["seed"]==20260811
        key=(rate,device)
        assert key not in observed
        observed.add(key)
        checkpoint=WEIGHTS[RATES.index(rate)]
        assert profile["checkpoint"]==checkpoint
        assert profile["checkpoint_sha256"]==entries[checkpoint]["source_sha256"]
        assert profile["checkpoint_file_bytes"]==entries[checkpoint]["source_bytes"]
        hardware=profile["hardware"]
        assert hardware["precision"]=="float32" and not hardware["amp_enabled"] and not hardware["tf32_enabled"]
        assert hardware["torch_threads"]==1 and hardware["torch_interop_threads"]==1
        protocol=profile["measurement_protocol"]
        assert protocol["batch_size"]==1 and protocol["validation_input_count"]==128
        assert protocol["warmup_per_coder"]==20 and protocol["repetitions_per_coder"]==100
        assert protocol["packet_frame_bits"]==32
        assert [item["levels"] for item in profile["levels"]]==[1,3,7,15,31]
        for point in profile["levels"]:
            assert point["conditional_table_bytes"]==point["levels"]*point["marginal_table_bytes"]
            for name in ("conditional","marginal"):
                checks=point["reference_checks"][name]
                assert all(checks[field] for field in ("payload_equal","symbols_equal","latent_equal","reconstruction_equal"))
                measured=point["paths"][name]
                resources=profile["model_resources"]
                expected_state=resources["encoder_tensor_state_bytes"]+resources["entropy_tensor_state_bytes"]+point[f"{name}_table_bytes"]
                assert measured["static_sender_tensor_and_cdf_bytes"]==expected_state
                for direction in ("encode","decode"):
                    timing=measured[direction]
                    values=timing["raw_seconds"]
                    assert len(values)==100 and all(math.isfinite(x) and x>=0 for x in values)
                    expected={"mean_ms":1000*statistics.mean(values),
                              "median_ms":1000*statistics.median(values),
                              "p95_ms":1000*percentile(values,.95)}
                    for field,value in expected.items():
                        assert math.isclose(timing[field],value,rel_tol=1e-10,abs_tol=1e-9),(key,point["levels"],name,direction,field)
                    series_count+=1
                    timing_samples+=len(values)
        for repetition in range(5):
            archived=json.loads((root/f"data/neural/v81_reports/rate_{rate}_rep_{repetition}.json").read_text(encoding="utf-8"))
            assert archived["checkpoint"]==checkpoint and archived["rate_weight"]==rate
            assert archived["config_fingerprint"]==profile["config_fingerprint"]
            assert archived["checkpoint_protocol_fingerprint"]==profile["protocol_fingerprint"]
        links.append({"rate_weight":rate,"device":device,"checkpoint":checkpoint,
                      "checkpoint_sha256":profile["checkpoint_sha256"],
                      "config_fingerprint":profile["config_fingerprint"],
                      "protocol_fingerprint":profile["protocol_fingerprint"],"v81_repetitions_matched":5})
        profile_records.append(profile)
    assert observed=={(rate,device) for rate in RATES for device in ("cpu","cuda:0")}
    assert series_count==160 and timing_samples==16000
    work=root/"outputs/frozen_artifacts_verification"
    table=recompute_table_v(root,work)
    assert table["status"]=="PASS" and table["source_profiles"]==8 and table["ledger_rows"]==80
    rebuilt=json.loads((work/"edge_deployment_metrics_rebuilt.json").read_text(encoding="utf-8"))
    original=json.loads((root/f"{EDGE}/edge_deployment_metrics.json").read_text(encoding="utf-8"))
    def key(row):
        # The original 80-row ledger predates the explicit seed column; all
        # eight source profiles above independently establish this same seed.
        return (row["device"],float(row["rate_weight"]),int(row.get("seed",20260811)),int(row["levels"]),row["path"])
    expected_rows={key(row):row for row in original["rows"]}
    actual_rows={key(row):row for row in rebuilt["rows"]}
    assert len(expected_rows)==80 and expected_rows.keys()==actual_rows.keys()
    assert len(original["reports"])==8
    by_identity={(item["rate_weight"],item["hardware"]["device"]):item for item in profile_records}
    for item in original["reports"]:
        linked=by_identity[(item["rate_weight"],item["hardware"]["device"])]
        assert item["checkpoint_sha256"]==linked["checkpoint_sha256"]
        assert item["model_resources"]==linked["model_resources"]
        assert item["hardware"]==linked["hardware"]
    for identity,expected in expected_rows.items():
        for name,value in expected.items():
            assert name in actual_rows[identity] and equivalent(value,actual_rows[identity][name]),(identity,name)
    with (root/f"{EDGE}/edge_deployment_metrics.csv").open(encoding="utf-8",newline="") as stream:
        original_csv=list(csv.DictReader(stream))
    assert len(original_csv)==80
    for row in original_csv:
        expected=expected_rows[key(row)]
        for name,value in expected.items():
            assert name in row
            if value is None:
                assert row[name]==""
            elif isinstance(value,(int,float)):
                assert equivalent(float(row[name]),value),(key(row),name)
            else:
                assert row[name]==value,(key(row),name)
    report={"status":"PASS","scope":"Frozen-file identity, original timing-sample statistics and archived-profile reaggregation; not a new latency/GPU measurement or model execution",
            "imported_files_verified":26,"checkpoint_files_verified":5,"original_checkpoint_bytes_unchanged":True,
            "source_archive_sha256":manifest["source_archive_sha256"],
            "source_archive_rehashed":False,"profile_reports":8,"ledger_rows":80,
            "raw_timing_series_recomputed":series_count,"raw_timing_samples":timing_samples,
            "source_json_and_csv_ledgers_matched":True,"profile_v81_links":links,
            "source_ledger_schema_note":"Original ledger omits seed column; current aggregation adds the verified common profile seed 20260811",
            "table_V_recomputation":table,"latency_measurements_replayed":False,
            "pytorch_pickle_loading_performed":False,"classifier_metadata_present":True,
            "privacy_redactions":sum(len(item["redactions"]) for item in entries.values())}
    report=package_relative(report,root)
    output.write_text(json.dumps(report,indent=2)+"\n",encoding="utf-8")
    return report


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",type=Path,default=ROOT/"outputs/frozen_artifacts_validation.json")
    args=parser.parse_args()
    report=verify(ROOT,args.output)
    print(json.dumps({key:report[key] for key in ("status","imported_files_verified","profile_reports","ledger_rows","raw_timing_series_recomputed","raw_timing_samples")},indent=2))


if __name__ == "__main__":
    main()
