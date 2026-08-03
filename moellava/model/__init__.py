from .language_model.llava_llama import LlavaLlamaForCausalLM, LlavaLlamaConfig
from .language_model.llava_qwen import LlavaQWenForCausalLM, LlavaQWenConfig
import transformers
a, b, c = transformers.__version__.split('.')[:3]
if a == '4' and int(b) >= 34:
    from .language_model.llava_mistral import LlavaMistralForCausalLM, LlavaMistralConfig
if a == '4' and int(b) >= 36:
    from .language_model.llava_minicpm import LlavaMiniCPMForCausalLM, LlavaMiniCPMConfig
    from .language_model.llava_phi import LlavaPhiForCausalLM, LlavaPhiConfig
    from .language_model.llava_stablelm import LlavaStablelmForCausalLM, LlavaStablelmConfig
if a == '4' and int(b) >= 37:
    from .language_model.llava_qwen1_5 import LlavaQwen1_5ForCausalLM, LlavaQwen1_5Config
if a == '4' and int(b) <= 31:
    from .language_model.llava_mpt import LlavaMPTForCausalLM, LlavaMPTConfig
