"""Regression checks for original-profile Table V reaggregation (stdlib only)."""
from copy import deepcopy
import unittest

from reproduce.tables import (
    LATENCY_FIELDS, LEVELS, PROFILE_SEED, RATES,
    REAGGREGATION_KIND, TRANSCRIPTION_KIND,
    compare_table_v, summarize_edge_rows,
)


def fixture_rows():
    records=[]
    for device in ('cpu','cuda:0'):
        for rate_index,rate in enumerate(RATES):
            for level in LEVELS:
                for path in ('marginal','conditional'):
                    records.append(dict(
                        device=device,rate_weight=rate,seed=PROFILE_SEED,
                        levels=level,path=path,cdf_bytes=512*level,
                        sender_tensor_state_plus_cdf_bytes=1024+512*level,
                        encode_mean_ms=1.0001+rate_index/10,
                        encode_p95_ms=1.2001+rate_index/10,
                        decode_mean_ms=2.0001+rate_index/10,
                        decode_p95_ms=2.2001+rate_index/10))
    return records


class TableVTests(unittest.TestCase):
    def test_four_summaries_reduce_conditional_rates(self):
        ledger=fixture_rows()
        for row in ledger:
            if row['path']=='marginal':row['encode_mean_ms']=99
        result=summarize_edge_rows(ledger)
        self.assertEqual(len(result),4)
        for row in result:
            self.assertEqual(row['source_kind'],REAGGREGATION_KIND)
            self.assertEqual(row['profile_count'],4)
            self.assertAlmostEqual(row['cdf_kib'],0.5*row['levels'])
            self.assertAlmostEqual(row['static_sender_kib'],1+0.5*row['levels'])
            self.assertAlmostEqual(row['encode_mean_min_ms'],1.0001)
            self.assertAlmostEqual(row['encode_mean_max_ms'],1.3001)

    def test_missing_or_duplicate_observation_fails(self):
        ledger=fixture_rows()
        with self.assertRaises(ValueError):summarize_edge_rows(ledger[:-1])
        ledger[-1]=deepcopy(ledger[0])
        with self.assertRaises(ValueError):summarize_edge_rows(ledger)

    def test_nonfinite_latency_fails(self):
        ledger=fixture_rows();ledger[0]['encode_p95_ms']=float('nan')
        with self.assertRaises(ValueError):summarize_edge_rows(ledger)

    def test_inconsistent_static_resources_fail(self):
        ledger=fixture_rows()
        next(row for row in ledger if row['path']=='conditional')['cdf_bytes']+=1
        with self.assertRaises(ValueError):summarize_edge_rows(ledger)

    def test_printed_rounding_and_separate_source_label(self):
        recomputed=summarize_edge_rows(fixture_rows())
        reference=[]
        for row in recomputed:
            item={name:row[name] for name in ('device','levels')}
            for field in ('cdf_kib','static_sender_kib'):
                item[field]=f'{row[field]:.2f}'
            for field in LATENCY_FIELDS:
                for suffix in ('_min_ms','_max_ms'):
                    item[field+suffix]=f'{row[field+suffix]:.3f}'
            item['source_kind']=TRANSCRIPTION_KIND;reference.append(item)
        checks=compare_table_v(recomputed,reference)
        self.assertEqual(len(checks),40)
        self.assertTrue(all(item['status']=='PASS' for item in checks))
        reference[0]['encode_mean_min_ms']='1.020'
        self.assertTrue(any(item['status']=='FAIL'
                            for item in compare_table_v(recomputed,reference)))
        reference[0]['source_kind']=REAGGREGATION_KIND
        with self.assertRaises(ValueError):compare_table_v(recomputed,reference)


if __name__=='__main__':unittest.main()
