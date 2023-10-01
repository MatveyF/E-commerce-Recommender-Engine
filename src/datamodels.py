from dataclasses import dataclass


@dataclass
class Recommendation:
    stock_code: int
    description: str
    score: float
