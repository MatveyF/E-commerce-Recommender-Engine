from dataclasses import dataclass


@dataclass
class Recommendation:
    stock_code: str
    description: str
    score: float
