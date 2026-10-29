"""Recompute Tables IV and V from archived numerical evidence.

Table V is reaggregated from the eight original endpoint-profile JSON files.
The manuscript transcription remains a separately labelled rounding reference.
Reaggregation does not execute a codec or replay the latency measurements.
"""
from pathlib import Path
import argparse
import csv
import json
import math
from statistics import mean

from neural.aggregate_edge_deployment_metrics import aggregate

ROOT=Path(__file__).resolve().parents[1]
RATES=(0,200,5000,50000)
LEVELS=(1,3,7,15,31)
PROFILE_SEED=20260811
EDGE_ROOT=Path('results/iclr_mnist/server_controller_edge_metrics_v1')
TRANSCRIPTION_KIND='manuscript_table_V_transcription_not_raw_profile'
REAGGREGATION_KIND='recomputed_from_original_edge_profile_json'
LATENCY_FIELDS=('encode_mean','encode_p95','decode_mean','decode_p95')


def rows(path):
    with path.open(encoding='utf-8-sig',newline='') as stream:return list(csv.DictReader(stream))


def write_csv(path,records):
    with path.open('w',encoding='utf-8',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(records[0]));writer.writeheader();writer.writerows(records)


def summarize_edge_rows(ledger):
    """Reduce all 80 profile/path/level rows to Table V's four conditional rows.

    Each latency interval is the min/max over four frozen rate weights, not a
    confidence interval and not a mean or percentile pooled across profiles.
    Each per-profile p95 has already been measured by the original profiler.
    """
    expected={(device,rate,PROFILE_SEED,level,path)
              for device in ('cpu','cuda:0') for rate in RATES
              for level in LEVELS for path in ('marginal','conditional')}
    keys=[]
    for row in ledger:
        key=(row['device'],float(row['rate_weight']),int(row['seed']),
             int(row['levels']),row['path'])
        keys.append(key)
        for field in ('cdf_bytes','sender_tensor_state_plus_cdf_bytes'):
            value=float(row[field])
            if not math.isfinite(value) or value<=0 or not value.is_integer():
                raise ValueError(f'Invalid byte count {field} at {key}: {value}')
        for field in LATENCY_FIELDS:
            value=float(row[field+'_ms'])
            if not math.isfinite(value) or value<=0:
                raise ValueError(f'Invalid latency {field} at {key}: {value}')
    if len(keys)!=80 or len(set(keys))!=80 or set(keys)!=expected:
        raise ValueError('Expected exactly 80 unique device/rate/seed/level/path ledger rows')
    summaries=[]
    for device,label in (('cpu','CPU'),('cuda:0','CUDA')):
        for level in (1,31):
            group=[r for r in ledger if r['device']==device
                   and int(r['levels'])==level and r['path']=='conditional']
            cdf={int(r['cdf_bytes']) for r in group}
            state={int(r['sender_tensor_state_plus_cdf_bytes']) for r in group}
            if len(cdf)!=1 or len(state)!=1:
                raise ValueError(f'Static resource counts differ across rates: {(device,level)}')
            record=dict(device=label,levels=level,cdf_kib=cdf.pop()/1024,
                        static_sender_kib=state.pop()/1024)
            for field in LATENCY_FIELDS:
                values=[float(r[field+'_ms']) for r in group]
                record[field+'_min_ms']=min(values)
                record[field+'_max_ms']=max(values)
            record.update(source_kind=REAGGREGATION_KIND,profile_count=len(group),
                          rate_weights=';'.join(str(rate) for rate in RATES),
                          path='conditional')
            summaries.append(record)
    return summaries


def compare_table_v(recomputed,transcribed):
    """Compare full-precision reaggregation with the printed rounding bins."""
    reference={(r['device'],int(r['levels'])):r for r in transcribed}
    expected={(device,level) for device in ('CPU','CUDA') for level in (1,31)}
    if len(transcribed)!=4 or set(reference)!=expected:
        raise ValueError('Expected four uniquely identified Table V transcription rows')
    if any(r['source_kind']!=TRANSCRIPTION_KIND for r in transcribed):
        raise ValueError('The Table V reference must retain its transcription label')
    fields={'cdf_kib':0.005000001,'static_sender_kib':0.005000001}
    fields.update({field+suffix:0.000500001 for field in LATENCY_FIELDS
                   for suffix in ('_min_ms','_max_ms')})
    comparisons=[]
    for row in recomputed:
        key=row['device'],int(row['levels'])
        for field,tolerance in fields.items():
            actual=float(row[field]);target=float(reference[key][field])
            difference=abs(actual-target)
            comparisons.append(dict(device=key[0],levels=key[1],field=field,
                                    recomputed=actual,transcribed=target,
                                    absolute_difference=difference,
                                    rounding_tolerance=tolerance,
                                    status='PASS' if math.isfinite(target) and difference<=tolerance else 'FAIL'))
    return comparisons


def recompute_table_v(root,output):
    """Rebuild the full ledger and the publication summary, retaining provenance."""
    root=Path(root);output=Path(output);output.mkdir(parents=True,exist_ok=True)
    ledger=aggregate(root/EDGE_ROOT,output/'edge_deployment_metrics_rebuilt.json')
    if len(ledger['reports'])!=8:
        raise ValueError(f"Expected eight original endpoint profiles, got {len(ledger['reports'])}")
    summaries=summarize_edge_rows(ledger['rows'])
    transcribed=rows(root/'data/neural/edge/table_v_transcribed.csv')
    comparisons=compare_table_v(summaries,transcribed)
    write_csv(output/'table_v_recomputed.csv',summaries)
    # Keep the historical manuscript transcription visibly distinct from data.
    write_csv(output/'table_v_transcribed.csv',transcribed)
    return dict(status='PASS' if all(x['status']=='PASS' for x in comparisons) else 'FAIL',
                source_kind=REAGGREGATION_KIND,source_profiles=8,ledger_rows=len(ledger['rows']),
                summary_rows=summaries,transcription_comparison=comparisons,
                aggregation='conditional path; min/max across four rate weights for each device and L',
                static_units='KiB = bytes / 1024',latency_units='ms per image',
                latency_measurements_replayed=False,
                source_root=EDGE_ROOT.as_posix(),
                transcription_reference='data/neural/edge/table_v_transcribed.csv')


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=ROOT/'outputs/tables')
    args=parser.parse_args(argv);args.output.mkdir(parents=True,exist_ok=True)
    raw=rows(ROOT/'data/neural/conditional_rate_v81_raw.csv')
    assert len(raw)==100
    assert len({(r['rate_weight'],r['levels'],r['repetition']) for r in raw})==100
    expected={0:(1.339721,1.339615,.0832,.0079),200:(.694616,.689220,4.2304,.7768),
              5000:(.600322,.591096,7.2333,1.5369),50000:(.557412,.548163,7.2512,1.6593)}
    records=[]
    for rate,reported in expected.items():
        group=[r for r in raw if float(r['rate_weight'])==rate and int(r['levels'])==31]
        assert len(group)==5 and {int(r['repetition']) for r in group}==set(range(5))
        marginal=mean(float(r['marginal_actual_bpp']) for r in group)
        conditional=mean(float(r['conditional_actual_bpp']) for r in group)
        bits=mean(float(r['actual_gain_bpp']) for r in group)*784
        percent=100*(marginal-conditional)/marginal
        computed=[marginal,conditional,bits,percent]
        tolerances=[.500001e-6,.500001e-6,.500001e-4,.500001e-4]
        for actual,target,tolerance in zip(computed,reported,tolerances):
            assert abs(actual-target)<tolerance,(rate,actual,target,tolerance)
        records.append(dict(rate_weight=rate,schedule_repetitions=5,marginal_bpp=marginal,
                            conditional_bpp=conditional,saved_bits_per_image=bits,saved_payload_percent=percent))
    write_csv(args.output/'table_iv_recomputed.csv',records)
    edge=recompute_table_v(ROOT,args.output)
    report={'status':edge['status'],'table_IV':'recomputed from all 5 paired schedule repetitions per operating point',
            'table_IV_rows':records,'table_V':edge,
            'table_V_status':'RECOMPUTED_FROM_ORIGINAL_PROFILES' if edge['status']=='PASS' else 'FAIL',
            'table_V_raw_latency_reproduction':'NOT_RUN_REAGGREGATION_ONLY'}
    (args.output/'table_checks.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({'status':report['status'],'table_IV_rows':4,'table_V_status':report['table_V_status'],
                      'table_V_source_profiles':edge['source_profiles'],'table_V_ledger_rows':edge['ledger_rows']}))
    if report['status']!='PASS':raise SystemExit(1)


if __name__=='__main__':main()
