from dataclasses import dataclass
from enum import Enum


class AssistantHistoryMessage:
    class Role(str, Enum):
        System = "system"
        User = "user"
        Assistant = "assistant"

    def __init__(self, role: "AssistantHistoryMessage.Role", message: str):
        self.role = role
        self.message = message


@dataclass
class AssistantResult:
    class State(str, Enum):
        Stop = "stop"
        Length = "length"
        Null = "null"

    message: str
    state: State
