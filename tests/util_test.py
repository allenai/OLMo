from olmo import util


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
