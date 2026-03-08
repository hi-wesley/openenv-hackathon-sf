from __future__ import annotations

import asyncio
from typing import Any, Dict

from .compat import HAVE_OPENENV, LocalSyncWrapper, OpenEnvClient, StepResult
from .models import AssistantAction, DFAEnvState, DFAObservation, EpisodeTrace

if HAVE_OPENENV:
    from openenv.core.client_types import StepResult as _OpenEnvStepResult  # type: ignore
    from openenv.core.env_server.types import State as _OpenEnvState  # type: ignore


class DFAAgentEnv(OpenEnvClient):  # type: ignore[misc]
    def __init__(self, base_url: str | None = None, environment: Any | None = None, **kwargs: Any) -> None:
        self._local_environment = environment
        self._base_url = base_url
        if HAVE_OPENENV and base_url:
            super().__init__(base_url=base_url, **kwargs)
        elif HAVE_OPENENV and environment is None:
            raise ValueError("Provide either base_url for remote use or environment for local direct use.")

    def _step_payload(self, action: AssistantAction) -> Dict[str, Any]:
        return action.model_dump()

    def _parse_result(self, payload: Dict[str, Any]) -> StepResult[DFAObservation]:
        observation = DFAObservation(**payload["observation"])
        reward = payload.get("reward", observation.reward)
        done = payload.get("done", observation.done)
        return StepResult(observation=observation, reward=reward, done=bool(done))

    def _parse_state(self, payload: Dict[str, Any]) -> DFAEnvState:
        return DFAEnvState(**payload)

    async def reset(self, **kwargs: Any) -> StepResult[DFAObservation]:
        if HAVE_OPENENV and self._base_url:
            return await super().reset(**kwargs)
        observation = await asyncio.to_thread(self._local_environment.reset, **kwargs)
        return StepResult(observation=observation, reward=observation.reward, done=observation.done)

    async def step(self, action: AssistantAction, **kwargs: Any) -> StepResult[DFAObservation]:
        if HAVE_OPENENV and self._base_url:
            return await super().step(action, **kwargs)
        observation = await asyncio.to_thread(self._local_environment.step, action, **kwargs)
        return StepResult(observation=observation, reward=observation.reward, done=observation.done)

    async def state(self) -> DFAEnvState:
        if HAVE_OPENENV and self._base_url:
            return await super().state()
        return await asyncio.to_thread(lambda: self._local_environment.state)

    async def close(self) -> None:
        if HAVE_OPENENV and self._base_url:
            await super().close()
            return
        if self._local_environment is not None and hasattr(self._local_environment, "close"):
            await asyncio.to_thread(self._local_environment.close)

    def sync(self) -> Any:
        if HAVE_OPENENV and self._base_url:
            return super().sync()
        return LocalSyncWrapper(self)

    async def rollout_with_policy(
        self,
        policy,
        *,
        reset_kwargs: Dict[str, Any] | None = None,
    ) -> EpisodeTrace:
        result = await self.reset(**(reset_kwargs or {}))
        while not result.done:
            action = policy(result.observation)
            result = await self.step(action)
        state = await self.state()
        return EpisodeTrace(**state.final_summary["trace"])
