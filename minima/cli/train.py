from __future__ import annotations

import argparse

from minima.training import DistillationConfig, distill


def main(argv: list[str] | None = None):
    parser = argparse.ArgumentParser(description="Distill LFM2.5 into a tuneable Minima W1.58A8 encoder")
    for field, definition in DistillationConfig.__dataclass_fields__.items():
        option = "--" + field.replace("_", "-")
        default = definition.default
        if isinstance(default, bool):
            parser.add_argument(option, action="store_true", default=default)
        else:
            parser.add_argument(option, type=type(default) if default is not None else str, default=default)
    args = parser.parse_args(argv)
    distill(DistillationConfig(**vars(args)))


if __name__ == "__main__":
    main()

