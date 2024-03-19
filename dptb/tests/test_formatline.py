import pytest
from dptb.utils.tools import format_readline

def test_format_readline():
    # Test case 1: line with no * character
    line = "1, 2, 3, 4, 5"
    expected_output = ["1", "2", "3", "4", "5"]
    assert format_readline(line) == expected_output

    # Test case 2: line with * character
    line = "5*0"
    expected_output = ["0", "0", "0", "0", "0"]
    assert format_readline(line) == expected_output

    # Test case 3: line with multiple * characters
    line = "2*hello, 3*world"
    expected_output = ["hello", "hello", "world", "world", "world"]
    assert format_readline(line) == expected_output

    # Test case 4: line with * character and other values
    line = "2*hello, 1, 3*world, 4"
    expected_output = ["hello", "hello", "1", "world", "world", "world", "4"]
    assert format_readline(line) == expected_output

    # Test case 5: line with * character and empty values
    line = "3*, 2*"
    with pytest.raises(AssertionError) as exception_info:
        format_readline(line)
    assert "The format of the line is not correct! n*value, the value is gone!" in str(exception_info.value)

    line = "1 2 3 4 5"
    expected_output = ['1', '2', '3', '4', '5']
    assert format_readline(line) == expected_output

    line = "2*1, 3*2, 4*3"
    expected_output = ['1', '1', '2', '2', '2', '3', '3', '3', '3']
    assert format_readline(line) == expected_output

    line = "2*1, 3 4 5, 6*0"
    expected_output = ['1', '1', '3', '4', '5', '0', '0', '0', '0', '0', '0']
    assert format_readline(line) == expected_output

    line = ""
    expected_output = []
    assert format_readline(line) == expected_output

    line = "    "
    expected_output = []
    assert format_readline(line) == expected_output