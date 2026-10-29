"""Fast structural and CSV-to-frozen-artist checks, with no ML dependencies."""
from pathlib import Path
import argparse
import csv
import json
import math

ROOT=Path(__file__).resolve().parents[1]


def load(path):
    def invalid(token):raise ValueError('Non-standard JSON constant: '+token)
    return json.loads(path.read_text(encoding='utf-8'),parse_constant=invalid)


def shape(value):
    return (len(value),)+shape(value[0]) if isinstance(value,list) and value else ()


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=ROOT/'outputs/data_checks.json')
    args=parser.parse_args(argv)
    scenes={n:load(ROOT/f'data/figures/fig{n}_scene.json') for n in range(2,7)}
    assert [len(scenes[n]['axes']) for n in range(2,7)]==[1,3,4,2,3]
    statistics={}
    for number,scene in scenes.items():
        soft=sum(sum(x is not None and y is not None for x,y in zip(line['x'],line['y']))
                 for ax in scene['axes'] for line in ax['lines'] if line['soft_marker'])
        path_counts=[len(col['paths']) for ax in scene['axes'] for col in ax['collections']
                     if col['class']=='LineCollection']
        statistics[str(number)]={'soft_observation_centers':soft,'CI_segments':sum(path_counts)}
    assert statistics['4']=={'soft_observation_centers':90,'CI_segments':90}
    assert statistics['5']=={'soft_observation_centers':360,'CI_segments':270}
    assert statistics['6']=={'soft_observation_centers':80,'CI_segments':80}
    with (ROOT/'data/figures/source/k4_rrc_phase_diagram_v2.csv').open(newline='',encoding='utf-8-sig') as stream:
        phase=list(csv.DictReader(stream))
    assert len(phase)==20402
    points={(r['panel'],round(float(r['rate_R']),12),round(float(r['randomness_Rc']),12)):r for r in phase}
    matched=0
    for name,ax in zip(('task_active','rate_active'),scenes[3]['axes'][:2]):
        mesh=ax['collections'][1];xy=mesh['coordinates'];values=mesh['values']
        assert shape(values)==(101,101)
        for iy,row in enumerate(values):
            for ix,value in enumerate(row):
                if value is None:continue
                xx=(xy[iy][ix][0]+xy[iy][ix+1][0])/2
                yy=(xy[iy][ix][1]+xy[iy+1][ix][1])/2
                source=points[(name,round(xx,12),round(yy,12))]['D3_minus_D4']
                assert source and math.isclose(value,float(source),rel_tol=0,abs_tol=1e-13)
                matched+=1
    assert matched==6309
    report={'status':'PASS','standard_JSON':True,'nulls':'preserved masks/non-finite line breaks, not zero',
            'phase_source_grid_rows':20402,'positive_phase_values_matched_to_CSV':matched,
            'figure_statistics':statistics,'statistical_recomputation':False}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({'status':'PASS','phase_positive_values':matched,'figure_statistics':statistics}))


if __name__=='__main__':main()
