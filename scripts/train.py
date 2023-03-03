"""
This is the script used to train DOLMA.

There is one required positional argument, the path to a YAML :class:`TrainConfig`.
Following the YAML path, you could pass any number of options to override
values in the :class:`TrainConfig`.

For example, to override :data:`TrainConfig.model.n_layers` to 5, pass ``--model.n_layers=5``:

```bash
python scripts/train.py train_config.yaml --model.n_layers=5
```
"""

import sys

from rich import print

from dolma import TrainConfig
from dolma.util import install_excepthook


def main(train_config: TrainConfig) -> None:
    print("[b u green]Configuration:[/]\n", train_config)


if __name__ == "__main__":
    install_excepthook()
    yaml_path, args_list = sys.argv[1], sys.argv[2:]
    train_config = TrainConfig.load(yaml_path, [s.strip("-") for s in args_list])
    main(train_config)
