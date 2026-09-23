"""Request and response formats of the xG API."""

from typing import Literal

from pydantic import BaseModel, Field

from xg_model.features.geometry import Point
from xg_model.serving.predictor import Situation


class Position(BaseModel):
    """A point on the StatsBomb pitch, attacking towards x = 120."""

    x: float = Field(ge=0, le=120)
    y: float = Field(ge=0, le=80)

    def as_point(self) -> Point:
        """Return the position as an (x, y) pair."""
        return (self.x, self.y)


class ShotRequest(BaseModel):
    """A shot situation as sent by the frontend."""

    position: Position
    body_part: Literal["Right Foot", "Left Foot", "Head", "Other"] = "Right Foot"
    technique: Literal["Normal", "Volley", "Half Volley", "Lob"] = "Normal"
    shot_type: Literal["Open Play", "Free Kick"] = "Open Play"
    assist: Literal["None", "Ground Pass", "Low Pass", "High Pass"] = "None"
    through_ball: bool = False
    cross: bool = False
    cut_back: bool = False
    defenders: list[Position] = Field(default_factory=list, max_length=10)
    goalkeeper: Position = Position(x=119, y=40)
    under_pressure: bool = False
    first_time: bool = False
    one_on_one: bool = False
    minute: int = Field(default=45, ge=0, le=130)
    score_difference: int = Field(default=0, ge=-10, le=10)
    is_home: bool = True
    gender: Literal["male", "female"] = "male"

    def to_situation(self) -> Situation:
        """Translate the request into the model's own description of a shot."""
        return Situation(
            x=self.position.x,
            y=self.position.y,
            body_part=self.body_part,
            technique=self.technique,
            shot_type=self.shot_type,
            assist=self.assist,
            through_ball=self.through_ball,
            cross=self.cross,
            cut_back=self.cut_back,
            defenders=tuple(defender.as_point() for defender in self.defenders),
            goalkeeper=self.goalkeeper.as_point(),
            under_pressure=self.under_pressure,
            first_time=self.first_time,
            one_on_one=self.one_on_one,
            minute=self.minute,
            score_difference=self.score_difference,
            is_home=self.is_home,
            gender=self.gender,
        )


class XgResponse(BaseModel):
    """The expected goals value of a shot."""

    xg: float = Field(ge=0, le=1)
