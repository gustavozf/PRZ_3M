from src.resources.data.data_search import DataSearch
from src.resources.data.data_io import DataIO
import pytest
import os


@pytest.mark.parametrize("dir_path, target_ext, expected_result", [
    (DataIO.absPath('./tests/test_dir'), None, 4),
    (DataIO.absPath('./tests/test_dir'), '.jpg', 4),
    (DataIO.absPath('./tests/test_dir'), '.png', 0),
    (DataIO.absPath('./tests/test_dir'), '.avg', 0),
])
def test_data_bfs(dir_path, target_ext, expected_result):
    found_files, count = DataSearch.bfs(dir_path, target_ext)
    print(found_files)

    assert count == expected_result, f'Expected to found {expected_result} files, but {count} were found instead'
