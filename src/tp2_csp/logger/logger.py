import datetime
import zoneinfo
from typing import Literal

from pydantic import BaseModel, Field

from utils import get_project_root


def datetime_now_argentina() -> datetime.datetime:
    return datetime.datetime.now(zoneinfo.ZoneInfo("America/Argentina/Buenos_Aires"))


class Logger(BaseModel):
    ga_provider: Literal["ours", "pygad"]
    executed_at: datetime.datetime = Field(default_factory=datetime_now_argentina)

    def log(self, message: str) -> None:
        full_path = f"{get_project_root()}/logs/{self.file_name}.txt"
        with open(full_path, "a", encoding="utf-8") as f:
            f.write(f"{message}\n")

    @property
    def file_name(self) -> str:
        return f"{self.executed_at.strftime('%Y-%m-%d_%H-%M-%S-%f')}_{self.ga_provider}"
