"""Passive example rules and evidence identity, without assessment access."""
import copy
import gzip
import json
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from v3_examples import PAIR, digest, display_index, load_trace, public_frame, select_examples
from dreamer.provenance import sha256


def frame(move=(0, 0)):
    return {"world": {"grid": [[0]*15 for _ in range(15)], "enemies": [[6,6]],
                      "agent": [5,5], "origin": [5,5], "step": 0, "keys": 0,
                      "door_open": False, "private_law": [.5,.5], "seed": 123},
            "model": {"position": [0,0], "terrain": [[0,0,0,0]], "enemy": [[1,1,.3]],
                      "default_enemy": .02, "learning": {"private_law": [.5,.5]}},
            "action": {"move": list(move), "interact": True}, "next_enemies": [[7,7]]}


def test_rule_chooses_lowest_retained_case_and_keeps_absent_strata():
    rows={}
    for case,reasons in ((0,('caught','caught')),(1,('escaped','escaped')),
                         (2,('escaped','caught')),(5,('escaped','caught')),(33,('caught','escaped'))):
        for condition,reason in zip(PAIR,reasons):
            rows[case,'stationary',condition]={'reason':reason}
    result={r['stratum']:r for r in select_examples(rows,['stationary'])}
    assert result['selected_only_escape']['case']==2
    assert result['selected_only_escape']['eligible_cases_in_retained_prefix']==2
    assert result['frozen_only_escape']['case'] is None
    assert result['neither_escape']['case']==0


def test_display_first_action_divergence_and_require_shared_world_prefix():
    a=[frame(),frame((1,0))]
    b=[frame(),frame((0,1))]
    assert display_index(a,b)==(1,1)
    assert display_index(a,copy.deepcopy(a))==(1,None)
    b[0]['world']['agent']=[4,5]
    with pytest.raises(ValueError,match='Paired worlds'):
        display_index(a,b)
    assert display_index([],a)==(None,None)


def test_public_frame_omits_laws_seeds_and_candidate_private_state():
    result=public_frame(frame())
    assert 'seed' not in result['world'] and 'private_law' not in result['world']
    assert 'learning' not in result['model']
    assert result['model']['enemy']==[[1,1,.3]]


def test_load_trace_requires_original_bytes_and_trajectory(tmp_path):
    trace=[frame()]
    path=tmp_path/'replays/000000--switch--selected.json.gz'
    path.parent.mkdir()
    with gzip.open(path,'wt') as handle:json.dump(trace,handle)
    identity=digest([{k:f[k] for k in ('world','action','next_enemies')} for f in trace])
    row={'case':0,'regime':'switch','condition':'selected','audit':{'frames':1,'trajectory_sha256':identity,
          'retained_replay':{'path':str(path.relative_to(tmp_path)),'sha256':sha256(path),'trajectory_sha256':identity}}}
    assert load_trace(tmp_path,row)==trace
    row['audit']['trajectory_sha256']='changed'
    with pytest.raises(ValueError,match='original outcome'):load_trace(tmp_path,row)
