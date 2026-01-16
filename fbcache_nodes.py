import contextlib
import unittest
import torch


from comfy import model_management


# Certifique-se de que este arquivo esteja no mesmo diretório que o Arquivo 1
from . import first_block_cache



class ApplyFBCacheOnModel:


    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("MODEL", ),
                "object_to_patch": (
                    "STRING",
                    {
                        "default": "diffusion_model",
                    },
                ),
                "residual_diff_threshold": (
                    "FLOAT",
                    {
                        "default":
                        0.0,
                        "min":
                        0.0,
                        "max":
                        1.0,
                        "step":
                        0.001,
                        "tooltip":
                        "Controls the tolerance for caching with lower values being more strict. Setting this to 0 disables the FBCache effect.",
                    },
                ),
                "start": (
                    "FLOAT",
                    {
                        "default":
                        0.0,
                        "step":
                        0.01,
                        "max":
                        1.0,
                        "min":
                        0.0,
                        "tooltip":
                        "Start time as a percentage of sampling where the FBCache effect can apply. Example: 0.0 would signify 0% (the beginning of sampling), 0.5 would signify 50%.",
                    },
                ),
                "end": ("FLOAT", {
                    "default":
                    1.0,
                    "step":
                    0.01,
                    "max":
                    1.0,
                    "min":
                    0.0,
                    "tooltip":
                    "End time as a percentage of sampling where the FBCache effect can apply. Example: 1.0 would signify 100% (the end of sampling), 0.5 would signify 50%.",
                }),
                "max_consecutive_cache_hits": (
                    "INT",
                    {
                        "default":
                        -1,
                        "min":
                        -1,
                        "tooltip":
                        "Allows limiting how many cached results can be used in a row. For example, setting this to 1 will mean there will be at least one full model call after each cached result. Set to 0 to disable FBCache effect, or -1 to allow unlimited consecutive cache hits.",
                    },
                ),
            }
        }


    RETURN_TYPES = ("MODEL", )
    FUNCTION = "patch"


    CATEGORY = "wavespeed"


    def patch(
        self,
        model,
        object_to_patch,
        residual_diff_threshold,
        max_consecutive_cache_hits=-1,
        start=0.0,
        end=1.0,
    ):
        if residual_diff_threshold <= 0.0 or max_consecutive_cache_hits == 0:
            return (model, )


        # Patch get_output_data
        first_block_cache.patch_get_output_data()


        using_validation = max_consecutive_cache_hits >= 0 or start > 0 or end < 1
        if using_validation:
            model_sampling = model.get_model_object("model_sampling")
            start_sigma, end_sigma = (float(
                model_sampling.percent_to_sigma(pct)) for pct in (start, end))
            del model_sampling


            @torch.compiler.disable()
            def validate_use_cache(use_cached):
                nonlocal consecutive_cache_hits
                use_cached = use_cached and end_sigma <= current_timestep <= start_sigma
                use_cached = use_cached and (max_consecutive_cache_hits < 0
                                             or consecutive_cache_hits
                                             < max_consecutive_cache_hits)
                consecutive_cache_hits = consecutive_cache_hits + 1 if use_cached else 0
                return use_cached
        else:
            validate_use_cache = None


        prev_timestep = None
        prev_input_state = None
        current_timestep = None
        consecutive_cache_hits = 0


        def reset_cache_state():
            # Resets the cache state and hits/time tracking variables.
            nonlocal prev_input_state, prev_timestep, consecutive_cache_hits
            prev_input_state = prev_timestep = None
            consecutive_cache_hits = 0
            first_block_cache.set_current_cache_context(
                first_block_cache.create_cache_context())


        def ensure_cache_state(model_input: torch.Tensor, timestep: float):
            # Validates the current cache state and hits/time tracking variables
            # and triggers a reset if necessary. Also updates current_timestep.
            nonlocal current_timestep
            input_state = (model_input.shape, model_input.dtype, model_input.device)
            need_reset = (
                prev_timestep is None or
                prev_input_state != input_state or
                first_block_cache.get_current_cache_context() is None or
                timestep >= prev_timestep
            )
            if need_reset:
                reset_cache_state()
            current_timestep = timestep


        def update_cache_state(model_input: torch.Tensor, timestep: float):
            # Updates the previous timestep and input state validation variables.
            nonlocal prev_timestep, prev_input_state
            prev_timestep = timestep
            prev_input_state = (model_input.shape, model_input.dtype, model_input.device)


        model = model.clone()
        diffusion_model = model.get_model_object(object_to_patch)


        # NOVA DETECÇÃO DE ARQUITETURA
        # Detecta se é UNet ou FLUX baseado em atributos estruturais
        model_class_name = diffusion_model.__class__.__name__
        
        # Detecção robusta por atributos ao invés de nome de classe
        has_double_blocks = hasattr(diffusion_model, 'double_blocks')
        has_single_blocks = hasattr(diffusion_model, 'single_blocks')
        has_input_blocks = hasattr(diffusion_model, 'input_blocks')
        has_middle_block = hasattr(diffusion_model, 'middle_block')
        has_output_blocks = hasattr(diffusion_model, 'output_blocks')
        
        # FLUX: tem double_blocks E single_blocks
        is_flux = has_double_blocks and has_single_blocks
        
        # UNet: tem input_blocks, middle_block, output_blocks
        is_unet = has_input_blocks and has_middle_block and has_output_blocks
        
        # Path direto para UNet/FLUX usando funções nativas
        if is_unet or is_flux or model_class_name in ("UNetModel", "Flux"):
            if is_unet or model_class_name == "UNetModel":
                create_patch_function = first_block_cache.create_patch_unet_model__forward
                print(f"[WaveSpeed] Detected UNet architecture: {model_class_name}")
            elif is_flux or model_class_name == "Flux":
                create_patch_function = first_block_cache.create_patch_flux_forward
                print(f"[WaveSpeed] Detected FLUX architecture: {model_class_name}")
            else:
                raise ValueError(
                    f"Unsupported model {model_class_name}")


            patch_forward = create_patch_function(
                diffusion_model,
                residual_diff_threshold=residual_diff_threshold,
                validate_can_use_cache_function=validate_use_cache,
            )


            def model_unet_function_wrapper(model_function, kwargs):
                try:
                    input = kwargs["input"]
                    timestep = kwargs["timestep"]
                    c = kwargs["c"]
                    t = timestep[0].item()


                    ensure_cache_state(input, t)


                    with patch_forward():
                        result = model_function(input, timestep, **c)
                        update_cache_state(input, t)
                        return result
                except Exception as exc:
                    reset_cache_state()
                    raise exc from None
        else:
            # Path genérico para outros modelos (NextDiT, HunyuanVideo, LTXV, etc)
            is_non_native_ltxv = False
            if model_class_name == "LTXVTransformer3D":
                is_non_native_ltxv = True
                diffusion_model = diffusion_model.transformer


            double_blocks_name = None
            single_blocks_name = None
            
            # Detecta nome do atributo de blocos duplos
            # NextDiT usa "layers" ao invés de "transformer_blocks"
            if hasattr(diffusion_model, "layers"):
                double_blocks_name = "layers"
                print(f"[WaveSpeed] Detected NextDiT/Z-Image architecture: {model_class_name} (using 'layers')")
            elif hasattr(diffusion_model, "transformer_blocks"):
                double_blocks_name = "transformer_blocks"
            elif hasattr(diffusion_model, "double_blocks"):
                double_blocks_name = "double_blocks"
            elif hasattr(diffusion_model, "joint_blocks"):
                double_blocks_name = "joint_blocks"
            else:
                # Se chegou aqui, não conseguimos detectar o tipo de modelo
                attrs = dir(diffusion_model)
                structural_attrs = [a for a in attrs if not a.startswith('_') and 
                                   any(keyword in a.lower() for keyword in 
                                       ['block', 'layer', 'stem', 'embed', 'in', 'out'])]
                raise ValueError(
                    f"[WaveSpeed] No supported block structure found for {model_class_name}\n"
                    f"Detected attributes: {structural_attrs[:15]}\n"
                    f"Supported structures:\n"
                    f"  - FLUX: double_blocks + single_blocks\n"
                    f"  - UNet: input_blocks + middle_block + output_blocks\n"
                    f"  - NextDiT/Z-Image: layers or transformer_blocks\n"
                    f"  - DiT variants: transformer_blocks or joint_blocks"
                )


            if hasattr(diffusion_model, "single_blocks"):
                single_blocks_name = "single_blocks"


            if is_non_native_ltxv:
                original_create_skip_layer_mask = getattr(
                    diffusion_model, "create_skip_layer_mask", None)
                if original_create_skip_layer_mask is not None:
                    def new_create_skip_layer_mask(self, *args, **kwargs):
                        raise RuntimeError(
                            "STG is not supported with FBCache yet")

                    diffusion_model.create_skip_layer_mask = new_create_skip_layer_mask.__get__(
                        diffusion_model)


            # Detecta se o modelo retorna apenas hidden_states (sem encoder_hidden_states)
            # NextDiT, LTXV e outros modelos single-stream retornam apenas 1 valor
            return_hidden_states_only = (
                model_class_name in ("LTXVModel", "NextDiT") or 
                is_non_native_ltxv
            )


            print(f"[WaveSpeed] Using CachedTransformerBlocks for {model_class_name} with blocks: {double_blocks_name}, return_only={return_hidden_states_only}")
            cached_transformer_blocks = torch.nn.ModuleList([
                first_block_cache.CachedTransformerBlocks(
                    None if double_blocks_name is None else getattr(
                        diffusion_model, double_blocks_name),
                    None if single_blocks_name is None else getattr(
                        diffusion_model, single_blocks_name),
                    residual_diff_threshold=residual_diff_threshold,
                    validate_can_use_cache_function=validate_use_cache,
                    cat_hidden_states_first=model_class_name == "HunyuanVideo",
                    return_hidden_states_only=return_hidden_states_only,
                    clone_original_hidden_states=model_class_name == "LTXVModel",
                    return_hidden_states_first=model_class_name != "OpenAISignatureMMDITWrapper",
                    accept_hidden_states_first=model_class_name != "OpenAISignatureMMDITWrapper",
                )
            ])
            dummy_single_transformer_blocks = torch.nn.ModuleList()


            def model_unet_function_wrapper(model_function, kwargs):
                try:
                    input = kwargs["input"]
                    timestep = kwargs["timestep"]
                    c = kwargs["c"]
                    t = timestep[0].item()


                    ensure_cache_state(input, t)


                    with unittest.mock.patch.object(
                            diffusion_model,
                            double_blocks_name,
                            cached_transformer_blocks,
                    ), unittest.mock.patch.object(
                            diffusion_model,
                            single_blocks_name,
                            dummy_single_transformer_blocks,
                    ) if single_blocks_name is not None else contextlib.nullcontext(
                    ):
                        result = model_function(input, timestep, **c)
                        update_cache_state(input, t)
                        return result
                except Exception as exc:
                    reset_cache_state()
                    raise exc from None


        model.set_model_unet_function_wrapper(model_unet_function_wrapper)
        return (model, )

