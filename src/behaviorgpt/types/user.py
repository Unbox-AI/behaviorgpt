from typing import Literal, Optional
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from behaviorgpt.types.domains import Domains

AgeGroup = Literal[
    "<18", "18-25", "26-35", "36-45", "46-55", "56-65", "66-75", "76-85", ">85"
]
AGE_GROUPS: tuple[AgeGroup, ...] = (
    "<18", "18-25", "26-35", "36-45", "46-55", "56-65", "66-75", "76-85", ">85"
)  # fmt: skip
# the youngest age in each group after the first
_AGE_STARTS = (18, 26, 36, 46, 56, 66, 76, 86)


def age_group(age: float) -> AgeGroup:
    """The group an age in years falls in."""
    if not 0 <= age <= 120:
        raise ValueError(f"age must be between 0 and 120, got {age}")
    return AGE_GROUPS[sum(age >= start for start in _AGE_STARTS)]


class User(BaseModel):
    """Who the shopper is, which BehaviorGPT reads with each of their events.

    Every field is optional: what you leave out is unknown, which is fine.
    `market` and `timezone` default to the client's. The time zone matters
    because BehaviorGPT reads the local hour and weekday of each event.

        User(gender="female", age=34, market="SE", timezone="Europe/Stockholm")

    `age` is sent as its group (`age_group`), never as the age itself.
    """

    model_config = ConfigDict(frozen=True)

    gender: Optional[Literal["female", "male"]] = None
    age_group: Optional[AgeGroup] = None
    # an age in years, turned into `age_group`
    age: Optional[float] = Field(default=None, exclude=True)
    # a two-letter country code, such as "SE"
    market: Optional[str] = None
    # an IANA name, such as "Europe/Stockholm"
    timezone: Optional[str] = None

    @field_validator("gender", mode="before")
    @classmethod
    def _gender(cls, value: object) -> object:
        return value.strip().lower() if isinstance(value, str) else value

    @field_validator("market")
    @classmethod
    def _market(cls, value: Optional[str]) -> Optional[str]:
        if value is None:
            return None
        value = value.strip()
        if len(value) != 2 or not value.isalpha():
            raise ValueError(f"market must be a two-letter country code, got {value!r}")
        return value

    @field_validator("timezone")
    @classmethod
    def _timezone(cls, value: Optional[str]) -> Optional[str]:
        if value is None:
            return None
        try:
            ZoneInfo(value)
        except (ZoneInfoNotFoundError, ValueError):
            raise ValueError(
                f"unknown time zone {value!r}; use a name like Europe/Stockholm"
            ) from None
        return value

    @model_validator(mode="after")
    def _age(self) -> "User":
        if self.age is None:
            return self
        group = age_group(self.age)
        if self.age_group is not None and self.age_group != group:
            raise ValueError(f"age {self.age} is not in age_group {self.age_group!r}")
        # frozen: set the derived group directly
        object.__setattr__(self, "age_group", group)
        object.__setattr__(self, "age", None)
        return self

    def domains(self, default_market: str) -> Domains:
        """As the API takes it; a postal code is never sent."""
        return Domains(
            market=self.market or default_market,
            gender=self.gender or "unk",
            age_group=self.age_group or "unk",
        )
