from typing import Literal

from .v1 import run_v1
from .v2 import run_v2
from .v3 import run_v3

VERSION_TO_RUN: Literal["v1", "v2", "v3"] = "v3"


def main() -> None:
    if VERSION_TO_RUN == "v1":
        run_v1()
    elif VERSION_TO_RUN == "v2":
        run_v2()
    elif VERSION_TO_RUN == "v3":
        run_v3()


if __name__ == "__main__":
    main()
