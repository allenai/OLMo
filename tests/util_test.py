from dolma.util import StrEnum


def test_str_enum():
    class Constants(StrEnum):
        foo = "foo"
        bar = "bar"

    assert "foo" == Constants.foo
