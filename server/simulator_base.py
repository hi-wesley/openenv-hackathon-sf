from __future__ import annotations

from abc import ABC, abstractmethod

from dfa_agent_env.models import SimulatorInput, SimulatorOutput


class BaseUserSimulator(ABC):
    name = "base"

    @abstractmethod
    def generate_opening_message(self, sim_input: SimulatorInput) -> SimulatorOutput:
        raise NotImplementedError

    @abstractmethod
    def generate_reply(self, sim_input: SimulatorInput) -> SimulatorOutput:
        raise NotImplementedError
