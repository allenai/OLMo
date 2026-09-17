from unittest.mock import Mock

import pytest
import requests

from olmo import util
from olmo.exceptions import OLMoNetworkError


def test_dir_is_empty(tmp_path):
    # Should return true if dir doesn't exist, or exists but is empty.
    dir = tmp_path / "foo"
    assert not dir.exists()
    assert util.dir_is_empty(dir)
    dir.mkdir(parents=True)
    assert util.dir_is_empty(dir)

    # Should return false if dir contains anything, even hidden files.
    (dir / ".foo").touch()
    assert not util.dir_is_empty(dir)


def test_flatten_dict():
    # basic flattening
    test_dict = {"a": 0, "b": {"e": 5, "f": 1}, "c": 2}
    assert util.flatten_dict(test_dict) == {"a": 0, "b.e": 5, "b.f": 1, "c": 2}

    # Should flatten nested dicts into a single dict with dotted keys.
    test_dict_with_list_of_dicts = {
        "a": 0,
        "b": {"e": [{"x": {"z": [222, 333]}}, {"y": {"g": [99, 100]}}], "f": 1},
        "c": 2,
    }
    assert util.flatten_dict(test_dict_with_list_of_dicts) == {
        "a": 0,
        "b.e": [{"x": {"z": [222, 333]}}, {"y": {"g": [99, 100]}}],  # doesnt get flattened
        "b.f": 1,
        "c": 2,
    }
    assert util.flatten_dict(test_dict_with_list_of_dicts, include_lists=True) == {
        "a": 0,
        "b.e.0.x.z.0": 222,
        "b.e.0.x.z.1": 333,
        "b.e.1.y.g.0": 99,
        "b.e.1.y.g.1": 100,
        "b.f": 1,
        "c": 2,
    }


def test_http_file_size_retries_transient_errors(monkeypatch):
    # Should recover from a transient connection error on the first attempt.
    monkeypatch.setattr(util.time, "sleep", lambda _: None)
    response = Mock(headers={"content-length": "123"})
    mock_head = Mock(side_effect=[requests.exceptions.ConnectionError("Network is unreachable"), response])
    monkeypatch.setattr(requests, "head", mock_head)

    assert util._http_file_size("https", "olmo-data.org", "foo.npy") == 123
    assert mock_head.call_count == 2


def test_http_file_size_raises_olmo_network_error_after_exhausting_retries(monkeypatch):
    # Should give up and raise `OLMoNetworkError` after `max_retries` failed attempts.
    monkeypatch.setattr(util.time, "sleep", lambda _: None)
    mock_head = Mock(side_effect=requests.exceptions.ConnectionError("Network is unreachable"))
    monkeypatch.setattr(requests, "head", mock_head)

    with pytest.raises(OLMoNetworkError):
        util._http_file_size("https", "olmo-data.org", "foo.npy", max_retries=3)
    assert mock_head.call_count == 3
