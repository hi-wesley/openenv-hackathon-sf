from __future__ import annotations

import asyncio
import json
import types
from copy import deepcopy
from dataclasses import dataclass
from typing import Any, Callable, Dict, Generic, Optional, Protocol, TypeVar, Union, get_args, get_origin, get_type_hints

try:
    from pydantic import BaseModel as _PydanticBaseModel
    from pydantic import ConfigDict, Field  # type: ignore

    HAVE_PYDANTIC = True
    BaseModel = _PydanticBaseModel
except ImportError:  # pragma: no cover - exercised implicitly in the sandbox
    HAVE_PYDANTIC = False

    class ConfigDict(dict):
        pass

    class _FieldSpec:
        def __init__(
            self,
            default: Any = ...,
            *,
            default_factory: Callable[[], Any] | None = None,
            description: str | None = None,
            ge: float | None = None,
            gt: float | None = None,
            le: float | None = None,
            lt: float | None = None,
            max_length: int | None = None,
            min_length: int | None = None,
        ) -> None:
            self.default = default
            self.default_factory = default_factory
            self.description = description
            self.ge = ge
            self.gt = gt
            self.le = le
            self.lt = lt
            self.max_length = max_length
            self.min_length = min_length

    def Field(  # type: ignore[misc]
        default: Any = ...,
        *,
        default_factory: Callable[[], Any] | None = None,
        description: str | None = None,
        ge: float | None = None,
        gt: float | None = None,
        le: float | None = None,
        lt: float | None = None,
        max_length: int | None = None,
        min_length: int | None = None,
    ) -> _FieldSpec:
        return _FieldSpec(
            default,
            default_factory=default_factory,
            description=description,
            ge=ge,
            gt=gt,
            le=le,
            lt=lt,
            max_length=max_length,
            min_length=min_length,
        )

    @dataclass
    class _CompatFieldInfo:
        annotation: Any
        default: Any
        description: str = ""

    class BaseModel:
        model_config = ConfigDict()
        model_fields: Dict[str, _CompatFieldInfo] = {}

        def __init_subclass__(cls, **kwargs: Any) -> None:
            super().__init_subclass__(**kwargs)
            annotations = _collect_annotations(cls)
            fields: Dict[str, _CompatFieldInfo] = {}
            for name, annotation in annotations.items():
                default = getattr(cls, name, ...)
                description = ""
                if isinstance(default, _FieldSpec):
                    description = default.description or ""
                fields[name] = _CompatFieldInfo(
                    annotation=annotation,
                    default=default,
                    description=description,
                )
            cls.model_fields = fields

        def __init__(self, **kwargs: Any) -> None:
            annotations = _collect_annotations(self.__class__)
            extras = dict(kwargs)
            for name in annotations:
                if name in extras:
                    value = extras.pop(name)
                else:
                    default = getattr(self.__class__, name, ...)
                    value = _resolve_default(default)
                    if value is ...:
                        raise TypeError(f"Missing required field: {name}")
                value = _coerce_value(annotations[name], value)
                setattr(self, name, value)
            for name, value in extras.items():
                setattr(self, name, value)

        @classmethod
        def model_validate(cls, data: Any) -> "BaseModel":
            if isinstance(data, cls):
                return data
            if hasattr(data, "model_dump"):
                return cls(**data.model_dump())
            if isinstance(data, dict):
                return cls(**data)
            raise TypeError(f"Cannot validate {type(data).__name__} into {cls.__name__}")

        @classmethod
        def model_validate_json(cls, data: str) -> "BaseModel":
            return cls.model_validate(json.loads(data))

        def model_dump(self, **_: Any) -> Dict[str, Any]:
            output: Dict[str, Any] = {}
            for name in _collect_annotations(self.__class__):
                output[name] = _to_builtin(getattr(self, name, None))
            return output

        def model_dump_json(self, **kwargs: Any) -> str:
            return json.dumps(self.model_dump(), **kwargs)

        def model_copy(self, *, update: Dict[str, Any] | None = None, deep: bool = False) -> "BaseModel":
            payload = self.model_dump()
            if update:
                payload.update(update)
            if deep:
                payload = deepcopy(payload)
            return self.__class__(**payload)

        @classmethod
        def model_json_schema(cls) -> Dict[str, Any]:
            properties: Dict[str, Any] = {}
            required: list[str] = []
            for name, info in cls.model_fields.items():
                default = info.default
                default_value = None
                if isinstance(default, _FieldSpec):
                    default_value = None if default.default is ... else default.default
                elif default is not ...:
                    default_value = default
                else:
                    required.append(name)
                schema = _annotation_to_schema(info.annotation)
                if isinstance(default, _FieldSpec):
                    if default.max_length is not None:
                        schema["maxLength"] = default.max_length
                    if default.min_length is not None:
                        schema["minLength"] = default.min_length
                    if default.ge is not None:
                        schema["minimum"] = default.ge
                    if default.gt is not None:
                        schema["exclusiveMinimum"] = default.gt
                    if default.le is not None:
                        schema["maximum"] = default.le
                    if default.lt is not None:
                        schema["exclusiveMaximum"] = default.lt
                    if default.description:
                        schema["description"] = default.description
                if default_value is not None:
                    schema["default"] = default_value
                properties[name] = schema
            return {"type": "object", "properties": properties, "required": required}

        def __iter__(self):
            yield from self.model_dump().items()

        def __repr__(self) -> str:
            args = ", ".join(f"{k}={v!r}" for k, v in self.model_dump().items())
            return f"{self.__class__.__name__}({args})"


def _collect_annotations(cls: type) -> Dict[str, Any]:
    annotations: Dict[str, Any] = {}
    for base in reversed(cls.__mro__):
        try:
            base_annotations = get_type_hints(base, include_extras=True)
        except Exception:
            base_annotations = getattr(base, "__annotations__", {})
        for name, annotation in base_annotations.items():
            if name in {"model_fields", "model_config"}:
                continue
            annotations[name] = annotation
    return annotations


def _resolve_default(default: Any) -> Any:
    if not HAVE_PYDANTIC and isinstance(default, _FieldSpec):
        if default.default_factory is not None:
            return default.default_factory()
        if default.default is not ...:
            return deepcopy(default.default)
        return ...
    if default is ...:
        return ...
    return deepcopy(default)


def _annotation_to_schema(annotation: Any) -> Dict[str, Any]:
    origin = get_origin(annotation)
    args = get_args(annotation)
    if origin in (list, tuple):
        return {"type": "array"}
    if origin is dict:
        return {"type": "object"}
    if origin is None and annotation in (str, "str"):
        return {"type": "string"}
    if annotation in (int, "int"):
        return {"type": "integer"}
    if annotation in (float, "float"):
        return {"type": "number"}
    if annotation in (bool, "bool"):
        return {"type": "boolean"}
    if origin is Optional and args:
        return _annotation_to_schema(args[0])
    if origin is None and getattr(annotation, "__name__", "") == "Literal" and args:
        values = list(args)
        return {"type": "string", "enum": values}
    if str(origin).endswith("Literal") and args:
        return {"type": "string", "enum": list(args)}
    return {"type": "string"}


def _coerce_value(annotation: Any, value: Any) -> Any:
    if value is None:
        return None

    origin = get_origin(annotation)
    args = get_args(annotation)

    if origin in (Union, types.UnionType):
        non_none_args = [arg for arg in args if arg is not type(None)]
        for arg in non_none_args:
            try:
                coerced = _coerce_value(arg, value)
                if coerced is not None:
                    return coerced
            except Exception:
                continue
        return value

    if origin in (list, tuple):
        item_annotation = args[0] if args else Any
        items = [_coerce_value(item_annotation, item) for item in value]
        return items if origin is list else tuple(items)

    if origin is dict:
        value_annotation = args[1] if len(args) > 1 else Any
        return {key: _coerce_value(value_annotation, item) for key, item in value.items()}

    if isinstance(annotation, type) and hasattr(annotation, "model_validate"):
        if isinstance(value, annotation):
            return value
        if isinstance(value, dict):
            return annotation.model_validate(value)

    if annotation in (float, "float"):
        return float(value)
    if annotation in (int, "int"):
        return int(value)
    if annotation in (str, "str"):
        return str(value)
    if annotation in (bool, "bool"):
        return bool(value)

    return value


def _to_builtin(value: Any) -> Any:
    if hasattr(value, "model_dump"):
        return value.model_dump()
    if isinstance(value, list):
        return [_to_builtin(item) for item in value]
    if isinstance(value, tuple):
        return [_to_builtin(item) for item in value]
    if isinstance(value, dict):
        return {key: _to_builtin(item) for key, item in value.items()}
    return value


try:
    from openenv.core import EnvClient as OpenEnvClient  # type: ignore
    from openenv.core.client_types import StepResult  # type: ignore
    from openenv.core.env_server import create_app as openenv_create_app  # type: ignore
    from openenv.core.env_server.interfaces import Environment  # type: ignore
    from openenv.core.env_server.types import Action, Observation, State  # type: ignore

    HAVE_OPENENV = True
except ImportError:  # pragma: no cover - exercised implicitly in the sandbox
    HAVE_OPENENV = False
    OpenEnvClient = object  # type: ignore
    openenv_create_app = None

    class Action(BaseModel):
        metadata: Dict[str, Any] = Field(default_factory=dict)

    class Observation(BaseModel):
        done: bool = Field(default=False)
        reward: float | int | bool | None = Field(default=None)
        metadata: Dict[str, Any] = Field(default_factory=dict)

    class State(BaseModel):
        episode_id: str | None = Field(default=None)
        step_count: int = Field(default=0)

    ActT = TypeVar("ActT", bound=Action)
    ObsT = TypeVar("ObsT", bound=Observation)
    StateT = TypeVar("StateT", bound=State)

    @dataclass
    class StepResult(Generic[ObsT]):
        observation: ObsT
        reward: float | None = None
        done: bool = False

    class Environment(Generic[ActT, ObsT, StateT]):
        SUPPORTS_CONCURRENT_SESSIONS = True

        def reset(
            self,
            seed: int | None = None,
            episode_id: str | None = None,
            **kwargs: Any,
        ) -> ObsT:
            raise NotImplementedError

        async def reset_async(
            self,
            seed: int | None = None,
            episode_id: str | None = None,
            **kwargs: Any,
        ) -> ObsT:
            return self.reset(seed=seed, episode_id=episode_id, **kwargs)

        def step(self, action: ActT, timeout_s: float | None = None, **kwargs: Any) -> ObsT:
            raise NotImplementedError

        async def step_async(self, action: ActT, timeout_s: float | None = None, **kwargs: Any) -> ObsT:
            return self.step(action, timeout_s=timeout_s, **kwargs)

        @property
        def state(self) -> StateT:
            raise NotImplementedError

        def close(self) -> None:
            return None


class SyncLike(Protocol):
    def reset(self, **kwargs: Any) -> StepResult[Any]:
        ...

    def step(self, action: Any, **kwargs: Any) -> StepResult[Any]:
        ...

    def state(self) -> Any:
        ...

    def close(self) -> None:
        ...


class LocalSyncWrapper:
    def __init__(self, client: Any) -> None:
        self.client = client

    def __enter__(self) -> "LocalSyncWrapper":
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        self.close()

    def reset(self, **kwargs: Any) -> StepResult[Any]:
        return asyncio.run(self.client.reset(**kwargs))

    def step(self, action: Any, **kwargs: Any) -> StepResult[Any]:
        return asyncio.run(self.client.step(action, **kwargs))

    def state(self) -> Any:
        return asyncio.run(self.client.state())

    def close(self) -> None:
        asyncio.run(self.client.close())
