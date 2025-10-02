from typing import Literal

from .ours import run_ours
from .pygad_csp import run_pygad

VERSION_TO_RUN: Literal["ours", "pygad"] = "ours"


def main() -> None:
    if VERSION_TO_RUN == "ours":
        run_ours()
    else:
        run_pygad()


if __name__ == "__main__":
    main()
