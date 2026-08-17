from __future__ import annotations

import argparse

from minima.spell_training import SpellDistillationConfig, distill_spellchecker


def main(argv: list[str] | None = None):
    parser = argparse.ArgumentParser(description="Distill the Minima spellchecker recovery adapters")
    for field, definition in SpellDistillationConfig.__dataclass_fields__.items():
        default = definition.default
        parser.add_argument("--" + field.replace("_", "-"), type=type(default), default=default)
    args = parser.parse_args(argv)
    distill_spellchecker(SpellDistillationConfig(**vars(args)))


if __name__ == "__main__":
    main()

