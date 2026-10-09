"""Selection decisions and hash boundaries, without world simulation."""
import hashlib

import pytest

from proposal.study_panels import choose_winner, program_key, programs_from_nominees, save_once, selection_means, verify_hashes


def test_selection_uses_f_then_s_then_earlier_generation():
    nominees = [{'source_sha256': name, 'generation': generation}
                for name, generation in [('high_task', 1), ('later', 4), ('earlier', 2), ('low_f', 0)]]
    means = {program_key('high_task'): {'combined_score': .8, 'task': .99},
             program_key('later'): {'combined_score': .9, 'task': .7},
             program_key('earlier'): {'combined_score': .9, 'task': .7},
             program_key('low_f'): {'combined_score': .89, 'task': 1.0}}
    assert choose_winner(nominees, means)['source_sha256'] == 'earlier'
    means[program_key('later')]['task'] = .71
    assert choose_winner(nominees, means)['source_sha256'] == 'later'


def test_seed_fallback_is_selected_even_with_zero_score():
    seed = {'source_sha256': 'seed', 'generation': 0}
    winner = choose_winner([seed], {program_key('seed'): {'combined_score': 0., 'task': .2}})
    assert winner['source_sha256'] == 'seed'


def test_programs_are_deduplicated_without_dropping_repetition_nominations():
    candidate = {'source_sha256': 'same-code', 'frozen_source': 'artifacts/fixture.py'}
    run = {'primary': {'nominations': [candidate]},
           'common_prefix_secondary': {'nominations': [candidate]}}
    nomination = {'runs': [run, run], 'frozen_program_sha256': {'artifacts/fixture.py': 'same-code'}}
    programs = programs_from_nominees(nomination)
    assert programs == {program_key('same-code'): {'path': 'artifacts/fixture.py', 'sha256': 'same-code'}}
    assert len(nomination['runs']) == 2


def test_hash_guard_rejects_changed_input(tmp_path):
    path = tmp_path / 'frozen.json'
    path.write_text('original')
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    verify_hashes({str(path): digest})
    path.write_text('changed')
    with pytest.raises(ValueError, match='hash differs'):
        verify_hashes({str(path): digest})


def test_freeze_refuses_changed_content(tmp_path):
    path = tmp_path / 'freeze.json'
    save_once(path, {'winner': 'a'})
    save_once(path, {'winner': 'a'})
    with pytest.raises(ValueError, match='overwrite'):
        save_once(path, {'winner': 'b'})


def test_invalid_selection_episode_stays_in_denominator():
    rows = [{'case': 0, 'reason': 'timeout', 'error': None, 'steps': 200,
             'keys': 0, 'door_open': False, 'map_correct': 1, 'map_audited': 1,
             'model_accuracy': 1., 'task': .28, 'combined_score': .568},
            {'case': 1, 'reason': 'invalid', 'error': 'fixture failure', 'steps': 0,
             'keys': 0, 'door_open': False, 'map_correct': 0, 'map_audited': 0,
             'model_accuracy': 0., 'task': .2, 'combined_score': 0.}]
    measured = selection_means(rows, count=2)
    assert measured['combined_score'] == .284
    assert measured['invalid'] == 1 and measured['episodes'] == 2
    with pytest.raises(ValueError, match='complete ordered case panel'):
        selection_means(rows[:1], count=2)
