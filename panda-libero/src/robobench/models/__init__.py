"""Import every architecture here so the registry sees it.

    from robobench.models.my_arch import MyArch  # noqa: F401
"""

from robobench.models.base import BasePolicy, ObsSpec  # noqa: F401
from robobench.models.resnet_film_bc import ResNetFiLMBC  # noqa: F401
from robobench.models.template import TemplatePolicy  # noqa: F401
from robobench.models.vla import VLAPolicy  # noqa: F401
