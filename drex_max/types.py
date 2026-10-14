from pathlib import Path
from typing import Literal, NotRequired, TypedDict

from PIL import Image

JSONValue = str | int | float | bool | None | list["JSONValue"] | dict[str, "JSONValue"]
Content = str | dict[str, JSONValue] | list[JSONValue]
ImageInput = str | Path | Image.Image


class ChoiceQuestion(TypedDict):
    type: Literal["choice"]
    instructions: NotRequired[Content | None]
    criteria: dict[str, Content | None]


class NoulQuestion(TypedDict):
    type: Literal["noul"]
    instructions: NotRequired[Content | None]
    criteria: NotRequired[dict[Literal["true", "false"], Content | None] | None]


class ScoreQuestion(TypedDict):
    type: Literal["score"]
    instructions: NotRequired[Content | None]
    criteria: list[Content]


Question = ChoiceQuestion | NoulQuestion | ScoreQuestion


class DecisionInput(TypedDict):
    state: Content
    question: Question
    images: NotRequired[list[ImageInput]]
