"""Regression for a real native rejection caused by stripped SEARCH blank lines."""
import pytest
from dreamer.headless_transport import parse_stdout_preserving_blank_lines


def test_diff_search_and_string_contents_keep_interior_blank_lines():
    content = ('<NAME>preserve_exact_source</NAME>\n\n'
               '<<<<<<< SEARCH\ndef first():\n    return 1\n\n\n'
               'def second():\n    return "a\\nb"\n=======\n'
               'def replacement():\n    return """a\n\nb"""\n>>>>>>> REPLACE')
    got, usage = parse_stdout_preserving_blank_lines(content + '\n{"usage":{"outputTokens":17}}\n\n')
    assert got == content
    assert usage == {'outputTokens':17}
    assert '\n\n\ndef second()' in got
    assert '"""a\n\nb"""' in got


@pytest.mark.parametrize('stdout', ['', 'code\nnot-json', 'code\n{"usage":null}', '{"usage":{}}'])
def test_same_invalid_transport_inputs_remain_errors(stdout):
    with pytest.raises(ValueError):
        parse_stdout_preserving_blank_lines(stdout)
