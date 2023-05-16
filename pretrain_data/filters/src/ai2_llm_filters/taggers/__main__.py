import springs as sp

from .base import TaggerConfig, TaggerProcessor


@sp.cli(TaggerConfig)
def main(config: TaggerConfig):
    processor = TaggerProcessor.main(config)


if __name__ == "__main__":
    main()
