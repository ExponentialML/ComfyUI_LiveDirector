import torch
import cv2
import torchvision.transforms as T
import numpy as np
import comfy

from nodes import common_ksampler
from comfy.ldm.util import default
from comfy.ldm.modules.attention import attention_basic
from einops import rearrange,repeat

optimized_attention_mm = attention_basic

class LiveDirectorAD(object):
    def __init__(self, module, strength=.5):
        self.module = module
        self.strength = strength
        self.module.stored_latent = None
        self.module.spatial_contexts = []
        self.module.stored_spatial_latent = None
        self.module.apply_reference_latent = True

    def attn(self, x, context=None, value=None, mask=None, scale_mask=None):
        q = self.module.to_q(x)
        context = default(context, x)
        k: Tensor = self.module.to_k(context)
        if value is not None:
            v = self.module.to_v(value)
            del value
        else:
            v = self.module.to_v(context)

        # apply custom scale by multiplying k by scale factor
        if self.module.scale is not None:
            k *= self.module.scale
        
        # apply scale mask, if present
        if scale_mask is not None:
            k *= scale_mask

        out = optimized_attention_mm(q, k, v, self.module.heads, mask)

        return self.module.to_out(out)

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states=None,
        attention_mask=None,
        video_length=None,
        scale_mask=None,
    ):
        if self.module.attention_mode != "Temporal":
            raise NotImplementedError

        d = hidden_states.shape[1]
        hidden_states = rearrange(
            hidden_states, "(b f) d c -> (b d) f c", f=video_length
        )

        if self.module.pos_encoder is not None:
            hidden_states = self.module.pos_encoder(hidden_states).to(hidden_states.dtype)

        if self.module.stored_latent is None and self.module.apply_reference_latent:
            self.module.stored_latent = hidden_states

        encoder_hidden_states = (
            repeat(encoder_hidden_states, "b n c -> (b d) n c", d=d)
            if encoder_hidden_states is not None
            else encoder_hidden_states
        )

        if self.module.stored_latent is not None and not self.module.apply_reference_latent:
            encoder_hidden_states = self.module.stored_latent.clone()
            d, f, c = hidden_states.shape

            if len(self.module.spatial_contexts) > 0:
                for spatial_context in self.module.spatial_contexts:
                    if spatial_context.transpose(0,1).shape == hidden_states.shape:
                        encoder_hidden_states = spatial_context.transpose(0,1)
                    
            # $TODO CFG
            if hidden_states.shape[0] > encoder_hidden_states.shape[0]:
                encoder_hidden_states = torch.cat([encoder_hidden_states] * 2, dim=0)

            temporal_context = self.module.stored_latent
            encoder_hidden_states = torch.cat([hidden_states, encoder_hidden_states], dim=1)
            hs_context = torch.cat([hidden_states] * 2, dim=1)

            encoder_hidden_states = encoder_hidden_states * self.strength + (1. - self.strength) * hs_context

        hidden_states = self.attn(
            hidden_states,
            encoder_hidden_states,
            value=None,
            mask=attention_mask,
            scale_mask=scale_mask,
        )

        hidden_states = rearrange(hidden_states, "(b d) f c -> (b f) d c", d=d)

        return hidden_states
    

# This is initially build around AnimateDiff Evolved
# Eventually others such as SVD, Videocrafter, etc. will be integrated
class LiveDirector:
    def __init__(self):
        self.model = None
        self.spatial_contexts = {}
        self.device = comfy.model_management.intermediate_device()
        self.dtype = comfy.model_management.unet_dtype()

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("MODEL",),
                "latent": ("LATENT", ),
                "conditioning": ("CONDITIONING", ),
                "enabled": ("BOOLEAN", {"default": True}),
                "model_type": (["animatediff"],),
                "strength": ("FLOAT", {"default": 0.75, "min": 0.0, "max": 1.0}),
               
            },
        }

    RETURN_TYPES = ("MODEL", "LATENT",)
    RETURN_NAMES = ("model", "latent")

    FUNCTION = "apply_livedirector"

    CATEGORY = "livedirector/apply"

    def create_spatial_context_group(self, in_target="", mid_target="", out_target=""):
        def target_group(target=""):
            return  {
                "target": target,
                "contexts": []
            }

        return {
            "input": target_group(in_target),
            "middle": target_group(mid_target),
            "output": target_group(out_target),
        }

    def get_spatial_context(self, module_name: str, spatial_contexts: dict):
        for k in spatial_contexts.keys():
            if spatial_contexts[k]['target'] in module_name:
                return spatial_contexts[k]['contexts']
        return []

    def set_context_group(self, attn_contexts: dict, tens: torch.Tensor, block_name: str):
        attn_contexts[block_name]['contexts'].append(tens)

    def set_append_state(self, state: bool):
        self.model.model_options['transformer_options']['append_attn'] = state

    def livedirector_patch(self, n, extra_options, attn1_contexts):
        if extra_options['append_attn']:
            block_name = extra_options["block"][0]
            self.set_context_group(attn1_contexts, n, block_name)
        return n

    def prepare_livedirector(self, cls, strength):
        for mm_patcher in self.model.motion_models.models:
            for n, m in mm_patcher.model.named_modules():
                if m.__class__.__name__ == "VersatileAttention":
                    DirectorModule = cls(m, strength=strength)
                    m.forward = DirectorModule.forward

    def set_livedirector(self, attn1_contexts: list):
        idx = 0
        for mm_patcher in self.model.motion_models.models:
            for n, m in mm_patcher.model.named_modules():
                if all([
                        m.__class__.__name__ == "VersatileAttention", 
                        hasattr(m, 'apply_reference_latent')
                    ]):
                    m.apply_reference_latent = False
                    if len(attn1_contexts) > 0:
                        m.spatial_contexts = self.get_spatial_context(n, attn1_contexts)

    def prepare_sample(self, latent, conditioning):
        conditioning = conditioning.copy()
        negative_conditioning = conditioning.copy()
        x = latent['samples'].to(dtype=torch.float, device=self.device)
        t = torch.tensor([0.], dtype=torch.float, device=self.device)
        conditioning[0][0] = conditioning[0][0].to(dtype=torch.float, device=self.device)
        negative_conditioning[0][0] = torch.zeros_like(negative_conditioning[0][0])

        return x, t, conditioning, negative_conditioning

    def apply_livedirector(
        self, 
        model: comfy.model_patcher.ModelPatcher, 
        latent, 
        conditioning,
        model_type, 
        strength, 
        enabled
    ):
        self.model = model
        self.spatial_contexts = self.create_spatial_context_group("down", "mid", "out")
        self.set_append_state(True)

        model.set_model_attn2_output_patch(
            lambda n, ep: self.livedirector_patch(n, ep, self.spatial_contexts)
        )

        self.prepare_livedirector(LiveDirectorAD, strength)

        x, t, cond, negative_cond = self.prepare_sample(latent, conditioning)

        # The options here don't really matte (yet), we just need to store the reference latent.
        latent = common_ksampler(
            model, 1, 1, 1, 
            "ddim", 
            "ddim_uniform", 
            cond, 
            negative_cond, 
            latent, 
            disable_noise=True,  
            denoise=1e-8,
            force_full_denoise=True
        )

        self.set_livedirector(self.spatial_contexts)
        self.set_append_state(False)

        return (model, latent[0], )

# A dictionary that contains all nodes you want to export with their names
# NOTE: names should be globally unique
NODE_CLASS_MAPPINGS = { "LiveDirector": LiveDirector}

# A dictionary that contains the friendly/humanly readable titles for the nodes
NODE_DISPLAY_NAME_MAPPINGS = {
    "LiveDirector": "Apply LiveDirector",
}
