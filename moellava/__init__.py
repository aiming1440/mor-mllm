from .model import LlavaLlamaForCausalLM
from .model import LlavaQWenForCausalLM
import transformers
a, b, c = transformers.__version__.split('.')[:3]
if a == '4' and int(b) >= 34:
    from .model import LlavaMistralForCausalLM
if a == '4' and int(b) >= 36:
    from .model import LlavaMiniCPMForCausalLM
    from .model import LlavaPhiForCausalLM
    from .model import LlavaStablelmForCausalLM
if a == '4' and int(b) >= 37:
    from .model import LlavaQwen1_5ForCausalLM
