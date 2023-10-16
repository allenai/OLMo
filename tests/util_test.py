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
