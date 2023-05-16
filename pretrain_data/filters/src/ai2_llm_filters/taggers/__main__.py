from .base import TaggerProcessor, TaggerConfig

import springs as sp


@sp.cli(TaggerConfig)
def main(config: TaggerConfig):
    processor = TaggerProcessor.main(config)


if __name__ == "__main__":
    main()
