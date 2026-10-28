"""Name -> (class, config class) registry, one for attacks and one for defenses."""

from dataclasses import dataclass


@dataclass(frozen=True)
class Spec:
    name: str
    cls: type
    config_class: type


class Registry:

    def __init__(self, kind: str):
        self.kind, self._specs = kind, {}

    def register(self, name: str, cls: type, config_class: type) -> None:
        if name in self._specs:
            raise ValueError(f"{self.kind} '{name}' already registered")
        self._specs[name] = Spec(name, cls, config_class)

    def get_spec(self, name: str) -> Spec:
        if name not in self._specs:
            raise ValueError(f"Unknown {self.kind}: {name}. Available: {', '.join(self.names())}")
        return self._specs[name]

    def create(self, name: str, **config_kwargs):
        spec = self.get_spec(name)
        return spec.cls(spec.config_class(**config_kwargs))

    def names(self) -> list[str]:
        return sorted(self._specs)
