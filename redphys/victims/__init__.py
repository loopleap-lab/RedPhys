"""PhyAgents (and `phymodel`, the PhyModel alone) behind an OpenAI-compatible chat API, evaluated like a PhyModel."""

from .server import Victim, VictimConfig, VictimRegistry, serve
from .sketchpad import SketchpadAgent, SketchpadConfig
from .spatialclaw import SpatialClawAgent, SpatialClawConfig
from .tools import ToolsAgent, ToolsConfig
from .phymodel import PhyModel

for _name, _cls, _config in [("phymodel", PhyModel, VictimConfig), ("tools", ToolsAgent, ToolsConfig), ("sketchpad", SketchpadAgent, SketchpadConfig),
                             ("spatialclaw", SpatialClawAgent, SpatialClawConfig)]:
    VictimRegistry.register(_name, _cls, _config)

__all__ = ["Victim", "VictimConfig", "VictimRegistry", "serve"]
