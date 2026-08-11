from pydantic import BaseModel


class Domains(BaseModel):
    market: str
    postal_code: str = "unk"
    gender: str = "unk"
    age_group: str = "unk"
