# Copyright (c) Meta Platforms, Inc. and affiliates.
import os

from tqdm import tqdm
import torch
from loguru import logger
from functools import wraps
from torch.utils._pytree import tree_map_only


def set_attention_backend():
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)

    logger.info(f"GPU name is {gpu_name}")
    if "A100" in gpu_name or "H100" in gpu_name or "H200" in gpu_name:
        # logger.info("Use flash_attn")
        os.environ["ATTN_BACKEND"] = "flash_attn"
        os.environ["SPARSE_ATTN_BACKEND"] = "flash_attn"

set_attention_backend()

from typing import List, Union, Optional, Literal, Any
from hydra.utils import instantiate
from omegaconf import OmegaConf
import numpy as np

from PIL import Image
from sam3d_objects.pipeline import preprocess_utils
from sam3d_objects.data.dataset.tdfy.img_and_mask_transforms import (
    get_mask,
)
from sam3d_objects.pipeline.inference_utils import (
    get_pose_decoder,
    SLAT_MEAN,
    SLAT_STD,
    downsample_sparse_structure,
    prune_sparse_structure,
)

from sam3d_objects.model.io import (
    load_model_from_checkpoint,
    filter_and_remove_prefix_state_dict_fn,
)

from sam3d_objects.model.backbone.tdfy_dit.modules import sparse as sp
from sam3d_objects.model.backbone.tdfy_dit.utils import postprocessing_utils
from safetensors.torch import load_file


class InferencePipeline:
    def __init__(
        self,
        ss_generator_config_path,
        ss_generator_ckpt_path,
        slat_generator_config_path,
        slat_generator_ckpt_path,
        ss_decoder_config_path,
        ss_decoder_ckpt_path,
        slat_decoder_gs_config_path,
        slat_decoder_gs_ckpt_path,
        slat_decoder_mesh_config_path,
        slat_decoder_mesh_ckpt_path,
        slat_decoder_gs_4_config_path=None,
        slat_decoder_gs_4_ckpt_path=None,
        ss_encoder_config_path=None,
        ss_encoder_ckpt_path=None,
        decode_formats=["gaussian", "mesh"],
        dtype="bfloat16",
        pad_size=1.0,
        version="v0",
        device="cuda",
        ss_preprocessor=preprocess_utils.get_default_preprocessor(),
        slat_preprocessor=preprocess_utils.get_default_preprocessor(),
        ss_condition_input_mapping=["image"],
        slat_condition_input_mapping=["image"],
        pose_decoder_name="default",
        workspace_dir="",
        downsample_ss_dist=0,  # the distance we use to downsample
        ss_inference_steps=25,
        ss_rescale_t=3,
        ss_cfg_strength=7,
        ss_cfg_interval=[0, 500],
        ss_cfg_strength_pm=0.0,
        slat_inference_steps=25,
        slat_rescale_t=3,
        slat_cfg_strength=5,
        slat_cfg_interval=[0, 500],
        rendering_engine: str = "nvdiffrast",  # nvdiffrast OR pytorch3d,
        shape_model_dtype=None,
        compile_model=False,
        slat_mean=SLAT_MEAN,
        slat_std=SLAT_STD,
    ):
        self.rendering_engine = rendering_engine
        self.device = torch.device(device)
        self.compile_model = compile_model
        logger.info(f"self.device: {self.device}")
        logger.info(f"CUDA_VISIBLE_DEVICES: {os.environ.get('CUDA_VISIBLE_DEVICES', None)}")
        logger.info(f"Actually using GPU: {torch.cuda.current_device()}")
        with self.device:
            self.decode_formats = decode_formats
            self.pad_size = pad_size
            self.version = version
            self.ss_condition_input_mapping = ss_condition_input_mapping
            self.slat_condition_input_mapping = slat_condition_input_mapping
            self.workspace_dir = workspace_dir
            self.downsample_ss_dist = downsample_ss_dist
            self.ss_inference_steps = ss_inference_steps
            self.ss_rescale_t = ss_rescale_t
            self.ss_cfg_strength = ss_cfg_strength
            self.ss_cfg_interval = ss_cfg_interval
            self.ss_cfg_strength_pm = ss_cfg_strength_pm
            self.slat_inference_steps = slat_inference_steps
            self.slat_rescale_t = slat_rescale_t
            self.slat_cfg_strength = slat_cfg_strength
            self.slat_cfg_interval = slat_cfg_interval

            self.dtype = self._get_dtype(dtype)
            if shape_model_dtype is None:
                self.shape_model_dtype = self.dtype
            else:
                self.shape_model_dtype = self._get_dtype(shape_model_dtype) 


            # Setup preprocessors
            self.pose_decoder = self.init_pose_decoder(ss_generator_config_path, pose_decoder_name)
            self.ss_preprocessor = self.init_ss_preprocessor(ss_preprocessor, ss_generator_config_path)
            self.slat_preprocessor = slat_preprocessor
    
            logger.info("Loading model weights...")

            ss_generator = self.init_ss_generator(
                ss_generator_config_path, ss_generator_ckpt_path
            )
            slat_generator = self.init_slat_generator(
                slat_generator_config_path, slat_generator_ckpt_path
            )
            ss_decoder = self.init_ss_decoder(
                ss_decoder_config_path, ss_decoder_ckpt_path
            )
            ss_encoder = self.init_ss_encoder(
                ss_encoder_config_path, ss_encoder_ckpt_path
            )
            slat_decoder_gs = self.init_slat_decoder_gs(
                slat_decoder_gs_config_path, slat_decoder_gs_ckpt_path
            )
            slat_decoder_gs_4 = self.init_slat_decoder_gs(
                slat_decoder_gs_4_config_path, slat_decoder_gs_4_ckpt_path
            )
            slat_decoder_mesh = self.init_slat_decoder_mesh(
                slat_decoder_mesh_config_path, slat_decoder_mesh_ckpt_path
            )

            # Load conditioner embedder so that we only load it once
            ss_condition_embedder = self.init_ss_condition_embedder(
                ss_generator_config_path, ss_generator_ckpt_path
            )
            slat_condition_embedder = self.init_slat_condition_embedder(
                slat_generator_config_path, slat_generator_ckpt_path
            )

            self.condition_embedders = {
                "ss_condition_embedder": ss_condition_embedder,
                "slat_condition_embedder": slat_condition_embedder,
            }

            # override generator and condition embedder setting
            self.override_ss_generator_cfg_config(
                ss_generator,
                cfg_strength=ss_cfg_strength,
                inference_steps=ss_inference_steps,
                rescale_t=ss_rescale_t,
                cfg_interval=ss_cfg_interval,
                cfg_strength_pm=ss_cfg_strength_pm,
            )
            self.override_slat_generator_cfg_config(
                slat_generator,
                cfg_strength=slat_cfg_strength,
                inference_steps=slat_inference_steps,
                rescale_t=slat_rescale_t,
                cfg_interval=slat_cfg_interval,
            )

            self.models = torch.nn.ModuleDict(
                {
                    "ss_generator": ss_generator,
                    "slat_generator": slat_generator,
                    "ss_encoder": ss_encoder,
                    "ss_decoder": ss_decoder,
                    "slat_decoder_gs": slat_decoder_gs,
                    "slat_decoder_gs_4": slat_decoder_gs_4,
                    "slat_decoder_mesh": slat_decoder_mesh,
                }
            )
            logger.info("Loading model weights completed!")

            if self.compile_model:
                logger.info("Compiling model...")
                self._compile()
                logger.info("Model compilation completed!")
            self.slat_mean = torch.tensor(slat_mean)
            self.slat_std = torch.tensor(slat_std)

    def _compile(self):
        torch._dynamo.config.cache_size_limit = 64
        torch._dynamo.config.accumulated_cache_size_limit = 2048
        torch._dynamo.config.capture_scalar_outputs = True
        compile_mode = "max-autotune"
        logger.info(f"Compile mode {compile_mode}")

        def clone_output_wrapper(f):
            @wraps(f)
            def wrapped(*args, **kwargs):
                outputs = f(*args, **kwargs)
                return tree_map_only(
                    torch.Tensor, lambda t: t.clone() if t.is_cuda else t, outputs
                )

            return wrapped

        self.embed_condition = clone_output_wrapper(
            torch.compile(
                self.embed_condition,
                mode=compile_mode,
                fullgraph=True,  # _preprocess_input in dino is not compatible with fullgraph
            )
        )
        self.models["ss_generator"].reverse_fn.inner_forward = clone_output_wrapper(
            torch.compile(
                self.models["ss_generator"].reverse_fn.inner_forward,
                mode=compile_mode,
                fullgraph=True,
            )
        )

        self.models["ss_decoder"].forward = clone_output_wrapper(
            torch.compile(
                self.models["ss_decoder"].forward,
                mode=compile_mode,
                fullgraph=True,
            )
        )

        self._warmup()

    def _warmup(self, num_warmup_iters=3):
        test_image = np.ones((512, 512, 4), dtype=np.uint8) * 255
        test_image[:, :, :3] = np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8)
        image = Image.fromarray(test_image)
        mask = None
        image = self.merge_image_and_mask(image, mask)

        for _ in tqdm(range(num_warmup_iters)):
            ss_input_dict = self.preprocess_image(image, self.ss_preprocessor)
            slat_input_dict = self.preprocess_image(image, self.slat_preprocessor)
            ss_return_dict = self.sample_sparse_structure(ss_input_dict)
            coords = ss_return_dict["coords"]
            slat = self.sample_slat(slat_input_dict, coords)

    def instantiate_and_load_from_pretrained(
        self,
        config,
        ckpt_path,
        state_dict_fn=None,
        state_dict_key="state_dict",
        device="cuda", 
    ):
        model = instantiate(config)

        if ckpt_path.endswith(".safetensors"):
            state_dict = load_file(ckpt_path, device="cuda")
            if state_dict_fn is not None:
                state_dict = state_dict_fn(state_dict)
            model.load_state_dict(state_dict, strict=False)
            model.eval()
        else:
            model = load_model_from_checkpoint(
                model,
                ckpt_path,
                strict=True,
                device="cpu",
                freeze=True,
                eval=True,
                state_dict_key=state_dict_key,
                state_dict_fn=state_dict_fn,
            )
        model = model.to(device)

        return model

    def init_pose_decoder(self, ss_generator_config_path, pose_decoder_name):
        if pose_decoder_name is None:
            pose_decoder_name = OmegaConf.load(os.path.join(self.workspace_dir, ss_generator_config_path))["module"]["pose_target_convention"]
        logger.info(f"Using pose decoder: {pose_decoder_name}")
        return get_pose_decoder(pose_decoder_name)

    def _decode_all_view_poses(
        self, 
        all_view_poses_raw: dict, 
        view_ss_input_dicts: list,
    ) -> list:
        """
        Decode raw pose predictions for all views.
        
        Each view uses its OWN pointmap_scale/shift for decoding, because:
        1. Each view's pointmap is independently normalized
        2. The network predicts pose in the normalized space of that view
        3. To get the correct metric pose, we must use that view's normalization factors
        
        If the network predictions are correct and consistent, then all views
        should decode to the SAME metric pose (same object position/scale in world).
        
        Args:
            all_view_poses_raw: Dict with keys like 'scale', 'translation', etc.
                Each value has shape (num_views, batch, ...)
            view_ss_input_dicts: List of input dicts for each view
                
        Returns:
            List of decoded pose dicts, one per view
        """
        num_views = list(all_view_poses_raw.values())[0].shape[0]
        decoded_poses = []
        
        # Determine target device
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        dtype = torch.float32  # pose_decoder expects float32
        
        for view_idx in range(num_views):
            # Extract raw pose for this view and ensure all on same device with float32
            view_pose_raw = {}
            for key, tensor in all_view_poses_raw.items():
                # tensor shape: (num_views, batch, ...)
                t = tensor[view_idx]  # (batch, ...)
                if torch.is_tensor(t):
                    view_pose_raw[key] = t.to(device=device, dtype=dtype)
                else:
                    view_pose_raw[key] = torch.tensor(t, device=device, dtype=dtype)
            
            # Add shape (required by pose_decoder but not used for pose)
            if 'shape' not in view_pose_raw:
                sample_tensor = list(view_pose_raw.values())[0]
                view_pose_raw['shape'] = torch.zeros(
                    sample_tensor.shape[0], 4096, 8, 
                    device=device, 
                    dtype=dtype
                )
            
            # Use THIS view's pointmap_scale/shift for decoding
            pointmap_scale = view_ss_input_dicts[view_idx].get("pointmap_scale", None)
            pointmap_shift = view_ss_input_dicts[view_idx].get("pointmap_shift", None)
            if pointmap_scale is not None and torch.is_tensor(pointmap_scale):
                pointmap_scale = pointmap_scale.to(device=device, dtype=dtype)
            if pointmap_shift is not None and torch.is_tensor(pointmap_shift):
                pointmap_shift = pointmap_shift.to(device=device, dtype=dtype)
            
            # Decode pose
            decoded = self.pose_decoder(
                view_pose_raw,
                scene_scale=pointmap_scale,
                scene_shift=pointmap_shift,
            )
            
            # Convert tensors to numpy for easier handling
            decoded_np = {}
            for key, value in decoded.items():
                if torch.is_tensor(value):
                    decoded_np[key] = value.detach().cpu().numpy()
                else:
                    decoded_np[key] = value
            
            # Save the pointmap scale/shift used for this view
            if pointmap_scale is not None:
                decoded_np['pointmap_scale'] = pointmap_scale.detach().cpu().numpy() if torch.is_tensor(pointmap_scale) else pointmap_scale
            if pointmap_shift is not None:
                decoded_np['pointmap_shift'] = pointmap_shift.detach().cpu().numpy() if torch.is_tensor(pointmap_shift) else pointmap_shift
            
            decoded_poses.append(decoded_np)
        
        return decoded_poses

    def init_ss_preprocessor(self, ss_preprocessor, ss_generator_config_path):
        if ss_preprocessor is not None:
            return ss_preprocessor
        config = OmegaConf.load(os.path.join(self.workspace_dir, ss_generator_config_path))["tdfy"]["val_preprocessor"]
        return instantiate(config)

    def init_ss_generator(self, ss_generator_config_path, ss_generator_ckpt_path):
        config = OmegaConf.load(
            os.path.join(self.workspace_dir, ss_generator_config_path)
        )["module"]["generator"]["backbone"]

        state_dict_prefix_func = filter_and_remove_prefix_state_dict_fn(
            "_base_models.generator."
        )

        return self.instantiate_and_load_from_pretrained(
            config,
            os.path.join(self.workspace_dir, ss_generator_ckpt_path),
            state_dict_fn=state_dict_prefix_func,
            device=self.device,
        )

    def init_slat_generator(self, slat_generator_config_path, slat_generator_ckpt_path):
        config = OmegaConf.load(
            os.path.join(self.workspace_dir, slat_generator_config_path)
        )["module"]["generator"]["backbone"]
        state_dict_prefix_func = filter_and_remove_prefix_state_dict_fn(
            "_base_models.generator."
        )
        return self.instantiate_and_load_from_pretrained(
            config,
            os.path.join(self.workspace_dir, slat_generator_ckpt_path),
            state_dict_fn=state_dict_prefix_func,
            device=self.device,
        )

    def init_ss_encoder(self, ss_encoder_config_path, ss_encoder_ckpt_path):
        if ss_encoder_ckpt_path is not None:
            # override to avoid problem loading
            config = OmegaConf.load(
                os.path.join(self.workspace_dir, ss_encoder_config_path)
            )
            if "pretrained_ckpt_path" in config:
                del config["pretrained_ckpt_path"]
            return self.instantiate_and_load_from_pretrained(
                config,
                os.path.join(self.workspace_dir, ss_encoder_ckpt_path),
                device=self.device,
                state_dict_key=None,
            )
        else:
            return None

    def init_ss_decoder(self, ss_decoder_config_path, ss_decoder_ckpt_path):
        # override to avoid problem loading
        config = OmegaConf.load(
            os.path.join(self.workspace_dir, ss_decoder_config_path)
        )
        if "pretrained_ckpt_path" in config:
            del config["pretrained_ckpt_path"]
        return self.instantiate_and_load_from_pretrained(
            config,
            os.path.join(self.workspace_dir, ss_decoder_ckpt_path),
            device=self.device,
            state_dict_key=None,
        )

    def init_slat_decoder_gs(
        self, slat_decoder_gs_config_path, slat_decoder_gs_ckpt_path
    ):
        if slat_decoder_gs_config_path is None:
            return None
        else:
            return self.instantiate_and_load_from_pretrained(
                OmegaConf.load(
                    os.path.join(self.workspace_dir, slat_decoder_gs_config_path)
                ),
                os.path.join(self.workspace_dir, slat_decoder_gs_ckpt_path),
                device=self.device,
                state_dict_key=None,
            )

    def init_slat_decoder_mesh(
        self, slat_decoder_mesh_config_path, slat_decoder_mesh_ckpt_path
    ):
        return self.instantiate_and_load_from_pretrained(
            OmegaConf.load(
                os.path.join(self.workspace_dir, slat_decoder_mesh_config_path)
            ),
            os.path.join(self.workspace_dir, slat_decoder_mesh_ckpt_path),
            device=self.device,
            state_dict_key=None,
        )

    def init_ss_condition_embedder(
        self, ss_generator_config_path, ss_generator_ckpt_path
    ):
        conf = OmegaConf.load(
            os.path.join(self.workspace_dir, ss_generator_config_path)
        )
        if "condition_embedder" in conf["module"]:
            return self.instantiate_and_load_from_pretrained(
                conf["module"]["condition_embedder"]["backbone"],
                os.path.join(self.workspace_dir, ss_generator_ckpt_path),
                state_dict_fn=filter_and_remove_prefix_state_dict_fn(
                    "_base_models.condition_embedder."
                ),
                device=self.device,
            )
        else:
            return None

    def init_slat_condition_embedder(
        self, slat_generator_config_path, slat_generator_ckpt_path
    ):
        return self.init_ss_condition_embedder(
            slat_generator_config_path, slat_generator_ckpt_path
        )


    def override_ss_generator_cfg_config(
        self,
        ss_generator,
        cfg_strength=7,
        inference_steps=25,
        rescale_t=3,
        cfg_interval=[0, 500],
        cfg_strength_pm=0.0,
    ):
        # override generator setting
        ss_generator.inference_steps = inference_steps
        ss_generator.reverse_fn.strength = cfg_strength
        ss_generator.reverse_fn.interval = cfg_interval
        ss_generator.rescale_t = rescale_t
        ss_generator.reverse_fn.backbone.condition_embedder.normalize_images = True
        ss_generator.reverse_fn.unconditional_handling = "add_flag"
        ss_generator.reverse_fn.strength_pm = cfg_strength_pm

        logger.info(
            "ss_generator parameters: inference_steps={}, cfg_strength={}, cfg_interval={}, rescale_t={}, cfg_strength_pm={}",
            inference_steps,
            cfg_strength,
            cfg_interval,
            rescale_t,
            cfg_strength_pm,
        )

    def override_slat_generator_cfg_config(
        self,
        slat_generator,
        cfg_strength=5,
        inference_steps=25,
        rescale_t=3,
        cfg_interval=[0, 500],
    ):
        slat_generator.inference_steps = inference_steps
        slat_generator.reverse_fn.strength = cfg_strength
        slat_generator.reverse_fn.interval = cfg_interval
        slat_generator.rescale_t = rescale_t

        logger.info(
            "slat_generator parameters: inference_steps={}, cfg_strength={}, cfg_interval={}, rescale_t={}",
            inference_steps,
            cfg_strength,
            cfg_interval,
            rescale_t,
        )


    def run(
        self,
        image: Union[None, Image.Image, np.ndarray],
        mask: Union[None, Image.Image, np.ndarray] = None,
        seed=42,
        stage1_only=False,
        with_mesh_postprocess=True,
        with_texture_baking=True,
        use_vertex_color=False,
        stage1_inference_steps=None,
        stage2_inference_steps=None,
        use_stage1_distillation=False,
        use_stage2_distillation=False,
        decode_formats=None,
        attention_logger: Optional[Any] = None,
    ) -> dict:
        """
        Parameters:
        - image (Image): The input image to be processed.
        - seed (int, optional): The random seed for reproducibility. Default is 42.
        - stage1_only (bool, optional): If True, only the sparse structure is sampled and returned. Default is False.
        - with_mesh_postprocess (bool, optional): If True, performs mesh post-processing. Default is True.
        - with_texture_baking (bool, optional): If True, applies texture baking to the 3D model. Default is True.
        Returns:
        - dict: A dictionary containing the GLB file and additional data from the sparse structure sampling.
        """
        # This should only happen if called from demo
        image = self.merge_image_and_mask(image, mask)
        with self.device:
            ss_input_dict = self.preprocess_image(image, self.ss_preprocessor)
            slat_input_dict = self.preprocess_image(image, self.slat_preprocessor)
            torch.manual_seed(seed)
            if attention_logger is not None:
                attention_logger.start_stage("ss")
                attention_logger.set_num_views(1)
                attention_logger.set_view(0)
            ss_return_dict = self.sample_sparse_structure(
                ss_input_dict,
                inference_steps=stage1_inference_steps,
                use_distillation=use_stage1_distillation,
                attention_logger=attention_logger,
            )

            ss_return_dict.update(self.pose_decoder(ss_return_dict))

            if "scale" in ss_return_dict:
                logger.info(f"Rescaling scale by {ss_return_dict['downsample_factor']}")
                ss_return_dict["scale"] = ss_return_dict["scale"] * ss_return_dict["downsample_factor"]
            if stage1_only:
                logger.info("Finished!")
                ss_return_dict["voxel"] = ss_return_dict["coords"][:, 1:] / 64 - 0.5
                return ss_return_dict

            coords = ss_return_dict["coords"]
            if attention_logger is not None:
                attention_logger.start_stage("slat")
                attention_logger.set_num_views(1)
                attention_logger.set_view(0)
            slat = self.sample_slat(
                slat_input_dict,
                coords,
                inference_steps=stage2_inference_steps,
                use_distillation=use_stage2_distillation,
                attention_logger=attention_logger,
            )
            outputs = self.decode_slat(
                slat, self.decode_formats if decode_formats is None else decode_formats
            )
            outputs = self.postprocess_slat_output(
                outputs, with_mesh_postprocess, with_texture_baking, use_vertex_color
            )
            logger.info("Finished!")

            return {
                **ss_return_dict,
                **outputs,
            }

    def postprocess_slat_output(
        self, outputs, with_mesh_postprocess, with_texture_baking, use_vertex_color
    ):
        # GLB files can be extracted from the outputs
        logger.info(
            f"Postprocessing mesh with option with_mesh_postprocess {with_mesh_postprocess}, with_texture_baking {with_texture_baking}..."
        )
        if "mesh" in outputs:
            glb = postprocessing_utils.to_glb(
                outputs["gaussian"][0],
                outputs["mesh"][0],
                # Optional parameters
                simplify=0.95,  # Ratio of triangles to remove in the simplification process
                texture_size=1024,  # Size of the texture used for the GLB
                verbose=False,
                with_mesh_postprocess=with_mesh_postprocess,
                with_texture_baking=with_texture_baking,
                use_vertex_color=use_vertex_color,
                rendering_engine=self.rendering_engine,
            )

        # glb.export("sample.glb")
        else:
            glb = None

        outputs["glb"] = glb

        if "gaussian" in outputs:
            outputs["gs"] = outputs["gaussian"][0]

        if "gaussian_4" in outputs:
            outputs["gs_4"] = outputs["gaussian_4"][0]

        return outputs

    def merge_image_and_mask(
        self,
        image: Union[np.ndarray, Image.Image],
        mask: Union[None, np.ndarray, Image.Image],
    ):
        """
        Merge mask into image's alpha channel (RGBA),
        Keep single-view path consistent with multi-view processing to avoid mask issues due to dtype/range differences.
        """
        if isinstance(image, Image.Image):
            image = np.array(image)

        if mask is not None:
            mask = np.array(mask)
            # Keep dtype handling consistent with multi-view path
            if mask.dtype == bool:
                mask = mask.astype(np.uint8) * 255
            elif mask.dtype != np.uint8:
                # If 0-1 float or other low-range values, stretch to 0-255 first then convert to uint8
                if mask.max() <= 1.0:
                    mask = (mask * 255).astype(np.uint8)
                else:
                    mask = mask.astype(np.uint8)

            if mask.ndim == 2:
                mask = mask[..., None]

            logger.info(f"Replacing alpha channel with the provided mask")
            assert mask.shape[:2] == image.shape[:2]

            # Support RGB or existing RGBA, both cases consistent with multi-view
            if image.shape[-1] == 3:
                image = np.concatenate([image[..., :3], mask], axis=-1).astype(np.uint8)
            elif image.shape[-1] == 4:
                image = np.concatenate([image[..., :3], mask], axis=-1).astype(np.uint8)
            else:
                raise ValueError(f"Unexpected image shape: {image.shape}")

        # Ensure final result is numpy array
        image = np.array(image)
        return image

    def decode_slat(
        self,
        slat: sp.SparseTensor,
        formats: List[str] = ["mesh", "gaussian"],
    ) -> dict:
        """
        Decode the structured latent.

        Args:
            slat (sp.SparseTensor): The structured latent.
            formats (List[str]): The formats to decode the structured latent to.

        Returns:
            dict: The decoded structured latent.
        """
        logger.info("Decoding sparse latent...")
        ret = {}
        with torch.no_grad():
            if "mesh" in formats:
                ret["mesh"] = self.models["slat_decoder_mesh"](slat)
            if "gaussian" in formats:
                ret["gaussian"] = self.models["slat_decoder_gs"](slat)
            if "gaussian_4" in formats:
                ret["gaussian_4"] = self.models["slat_decoder_gs_4"](slat)
        # if "radiance_field" in formats:
        #     ret["radiance_field"] = self.models["slat_decoder_rf"](slat)
        return ret

    def is_mm_dit(self, model_name="ss_generator"):
        return hasattr(self.models[model_name].reverse_fn.backbone, "latent_mapping")

    def embed_condition(self, condition_embedder, *args, **kwargs):
        if condition_embedder is not None:
            tokens = condition_embedder(*args, **kwargs)
            return tokens, None, None
        return None, args, kwargs

    def get_condition_input(self, condition_embedder, input_dict, input_mapping):
        condition_args = self.map_input_keys(input_dict, input_mapping)
        condition_kwargs = {
            k: v for k, v in input_dict.items() if k not in input_mapping
        }
        logger.info("Running condition embedder ...")
        embedded_cond, condition_args, condition_kwargs = self.embed_condition(
            condition_embedder, *condition_args, **condition_kwargs
        )
        logger.info("Condition embedder finishes!")
        if embedded_cond is not None:
            condition_args = (embedded_cond,)
            condition_kwargs = {}

        return condition_args, condition_kwargs

    def sample_sparse_structure(
        self, ss_input_dict: dict, inference_steps=None, use_distillation=False, attention_logger: Optional[Any] = None
    ):
        ss_generator = self.models["ss_generator"]
        ss_decoder = self.models["ss_decoder"]
        if use_distillation:
            ss_generator.no_shortcut = False
            ss_generator.reverse_fn.strength = 0
            ss_generator.reverse_fn.strength_pm = 0
        else:
            ss_generator.no_shortcut = True
            ss_generator.reverse_fn.strength = self.ss_cfg_strength
            ss_generator.reverse_fn.strength_pm = self.ss_cfg_strength_pm

        prev_inference_steps = ss_generator.inference_steps
        if inference_steps:
            ss_generator.inference_steps = inference_steps

        image = ss_input_dict["image"]
        bs = image.shape[0]
        logger.info(
            "Sampling sparse structure: inference_steps={}, strength={}, interval={}, rescale_t={}, cfg_strength_pm={}",
            ss_generator.inference_steps,
            ss_generator.reverse_fn.strength,
            ss_generator.reverse_fn.interval,
            ss_generator.rescale_t,
            ss_generator.reverse_fn.strength_pm,
        )

        with torch.no_grad():
            with torch.autocast(device_type="cuda", dtype=self.shape_model_dtype):
                if self.is_mm_dit():
                    latent_shape_dict = {
                        k: (bs,) + (v.pos_emb.shape[0], v.input_layer.in_features)
                        for k, v in ss_generator.reverse_fn.backbone.latent_mapping.items()
                    }
                else:
                    latent_shape_dict = (bs,) + (4096, 8)

                condition_args, condition_kwargs = self.get_condition_input(
                    self.condition_embedders["ss_condition_embedder"],
                    ss_input_dict,
                    self.ss_condition_input_mapping,
                )
                return_dict = ss_generator(
                    latent_shape_dict,
                    image.device,
                    *condition_args,
                    **condition_kwargs,
                )
                if not self.is_mm_dit():
                    return_dict = {"shape": return_dict}

                shape_latent = return_dict["shape"]
                ss = ss_decoder(
                    shape_latent.permute(0, 2, 1)
                    .contiguous()
                    .view(shape_latent.shape[0], 8, 16, 16, 16)
                )
                coords = torch.argwhere(ss > 0)[:, [0, 2, 3, 4]].int()

                # downsample output
                return_dict["coords_original"] = coords
                original_shape = coords.shape
                if self.downsample_ss_dist > 0:
                    coords = prune_sparse_structure(
                        coords,
                        max_neighbor_axes_dist=self.downsample_ss_dist,
                    )
                coords, downsample_factor = downsample_sparse_structure(coords)
                logger.info(
                    f"Downsampled coords from {original_shape[0]} to {coords.shape[0]}"
                )
                return_dict["coords"] = coords
                return_dict["downsample_factor"] = downsample_factor

        ss_generator.inference_steps = prev_inference_steps
        return return_dict

    def sample_slat(
        self,
        slat_input: dict,
        coords: torch.Tensor,
        inference_steps=25,
        use_distillation=False,
        attention_logger: Optional[Any] = None,
    ) -> sp.SparseTensor:
        image = slat_input["image"]
        DEVICE = image.device
        slat_generator = self.models["slat_generator"]
        latent_shape = (image.shape[0],) + (coords.shape[0], 8)
        prev_inference_steps = slat_generator.inference_steps
        if inference_steps:
            slat_generator.inference_steps = inference_steps
        if use_distillation:
            slat_generator.no_shortcut = False
            slat_generator.reverse_fn.strength = 0
        else:
            slat_generator.no_shortcut = True
            slat_generator.reverse_fn.strength = self.slat_cfg_strength

        logger.info(
            "Sampling sparse latent: inference_steps={}, strength={}, interval={}, rescale_t={}",
            slat_generator.inference_steps,
            slat_generator.reverse_fn.strength,
            slat_generator.reverse_fn.interval,
            slat_generator.rescale_t,
        )

        with torch.autocast(device_type="cuda", dtype=self.dtype):
            with torch.no_grad():
                condition_args, condition_kwargs = self.get_condition_input(
                    self.condition_embedders["slat_condition_embedder"],
                    slat_input,
                    self.slat_condition_input_mapping,
                )
                condition_args += (coords.cpu().numpy(),)
                slat = slat_generator(
                    latent_shape, DEVICE, *condition_args, **condition_kwargs
                )
                slat = sp.SparseTensor(
                    coords=coords,
                    feats=slat[0],
                ).to(DEVICE)
                slat = slat * self.slat_std.to(DEVICE) + self.slat_mean.to(DEVICE)

        slat_generator.inference_steps = prev_inference_steps
        return slat

    def _apply_transform(self, input: torch.Tensor, transform):
        if input is not None:
            input = transform(input)

        return input

    def _preprocess_image_and_mask(
        self, rgb_image, mask_image, img_mask_joint_transform
    ):
        for trans in img_mask_joint_transform:
            rgb_image, mask_image = trans(rgb_image, mask_image)
        return rgb_image, mask_image

    def map_input_keys(self, item, condition_input_mapping):
        output = [item[k] for k in condition_input_mapping]

        return output

    def image_to_float(self, image):
        image = np.array(image)
        image = image / 255
        image = image.astype(np.float32)
        return image

    def preprocess_image(
        self, image: Union[Image.Image, np.ndarray], preprocessor
    ) -> torch.Tensor:
        # canonical type is numpy
        if not isinstance(image, np.ndarray):
            image = np.array(image)

        assert image.ndim == 3  # no batch dimension as of now
        assert image.shape[-1] == 4  # rgba format
        assert image.dtype == np.uint8  # [0,255] range

        rgba_image = torch.from_numpy(self.image_to_float(image))
        rgba_image = rgba_image.permute(2, 0, 1).contiguous()
        rgb_image = rgba_image[:3]
        rgb_image_mask = (get_mask(rgba_image, None, "ALPHA_CHANNEL") > 0).float()
        processed_rgb_image, processed_mask = self._preprocess_image_and_mask(
            rgb_image, rgb_image_mask, preprocessor.img_mask_joint_transform
        )

        # transform tensor to model input
        processed_rgb_image = self._apply_transform(
            processed_rgb_image, preprocessor.img_transform
        )
        processed_mask = self._apply_transform(
            processed_mask, preprocessor.mask_transform
        )

        # full image, with only processing from the image
        rgb_image = self._apply_transform(rgb_image, preprocessor.img_transform)
        rgb_image_mask = self._apply_transform(
            rgb_image_mask, preprocessor.mask_transform
        )
        item = {
            "mask": processed_mask[None].to(self.device),
            "image": processed_rgb_image[None].to(self.device),
            "rgb_image": rgb_image[None].to(self.device),
            "rgb_image_mask": rgb_image_mask[None].to(self.device),
        }

        return item

    @staticmethod
    def _get_dtype(dtype):
        if dtype == "bfloat16":
            return torch.bfloat16
        elif dtype == "float16":
            return torch.float16
        elif dtype == "float32":
            return torch.float32
        else:
            raise NotImplementedError

    def get_multi_view_condition_input(
        self, 
        condition_embedder, 
        view_input_dicts: List[dict], 
        input_mapping
    ):
        """
        Prepare conditions for multi-view input
        
        Args:
            condition_embedder: Condition embedder
            view_input_dicts: List of input dicts for each view
            input_mapping: Input mapping
            
        Returns:
            condition_args: Condition args (containing condition tokens for all views)
            condition_kwargs: Condition keyword args
        """
        # Extract conditions for each view separately
        view_conditions = []
        for view_input_dict in view_input_dicts:
            condition_args = self.map_input_keys(view_input_dict, input_mapping)
            condition_kwargs = {
                k: v for k, v in view_input_dict.items() if k not in input_mapping
            }
            embedded_cond, _, _ = self.embed_condition(
                condition_embedder, *condition_args, **condition_kwargs
            )
            if embedded_cond is not None:
                view_conditions.append(embedded_cond)
            else:
                # If no embedding, use original args
                view_conditions.append(condition_args)
        
        # Stack conditions from all views together
        # Shape: (num_views, batch_size, num_tokens, dim)
        if isinstance(view_conditions[0], torch.Tensor):
            # If tensor, stack
            all_conditions = torch.stack(view_conditions, dim=0)
        else:
            # If other type, keep as list
            all_conditions = view_conditions
        
        return (all_conditions,), {}

    def sample_sparse_structure_multi_view(
        self, 
        view_ss_input_dicts: List[dict], 
        inference_steps=None, 
        use_distillation=False,
        mode: Literal['stochastic', 'multidiffusion'] = 'multidiffusion',
        attention_logger: Optional[Any] = None,
        optimize_per_view_pose: bool = False,
    ):
        """
        Multi-view sparse structure generation
        
        Args:
            view_ss_input_dicts: List of input dicts for each view
            inference_steps: Number of inference steps
            use_distillation: Whether to use distillation
            mode: 'stochastic' or 'multidiffusion'
            optimize_per_view_pose: Whether to optimize pose independently for each view
                - False (default): Shape averaged, Pose only from View 0
                - True: Shape averaged, each view iterates its own Pose independently
        """
        from sam3d_objects.pipeline.multi_view_utils import inject_generator_multi_view
        
        ss_generator = self.models["ss_generator"]
        ss_decoder = self.models["ss_decoder"]
        num_views = len(view_ss_input_dicts)
        if attention_logger is not None:
            attention_logger.start_stage("ss")
            attention_logger.set_num_views(num_views)
            attention_logger.set_view(0)
        
        if use_distillation:
            ss_generator.no_shortcut = False
            ss_generator.reverse_fn.strength = 0
            ss_generator.reverse_fn.strength_pm = 0
        else:
            ss_generator.no_shortcut = True
            ss_generator.reverse_fn.strength = self.ss_cfg_strength
            ss_generator.reverse_fn.strength_pm = self.ss_cfg_strength_pm

        prev_inference_steps = ss_generator.inference_steps
        if inference_steps:
            ss_generator.inference_steps = inference_steps

        image = view_ss_input_dicts[0]["image"]
        bs = image.shape[0]
        logger.info(
            f"Sampling sparse structure with {num_views} views: "
            f"inference_steps={ss_generator.inference_steps}, mode={mode}"
        )

        with torch.no_grad():
            with torch.autocast(device_type="cuda", dtype=self.shape_model_dtype):
                if self.is_mm_dit():
                    latent_shape_dict = {
                        k: (bs,) + (v.pos_emb.shape[0], v.input_layer.in_features)
                        for k, v in ss_generator.reverse_fn.backbone.latent_mapping.items()
                    }
                    logger.info(f"[Stage 1] Latent shape (MM-DiT): {latent_shape_dict}")
                else:
                    latent_shape_dict = (bs,) + (4096, 8)
                    logger.info(f"[Stage 1] Latent shape: {latent_shape_dict}")

                # Prepare multi-view conditions
                condition_args, condition_kwargs = self.get_multi_view_condition_input(
                    self.condition_embedders["ss_condition_embedder"],
                    view_ss_input_dicts,
                    self.ss_condition_input_mapping,
                )
                
                # Inject multi-view support
                with inject_generator_multi_view(
                    ss_generator, 
                    num_views=num_views, 
                    num_steps=ss_generator.inference_steps,
                    mode=mode,
                    attention_logger=attention_logger,
                    optimize_per_view_pose=optimize_per_view_pose,
                ) as all_view_poses_storage:
                    return_dict = ss_generator(
                        latent_shape_dict,
                        image.device,
                        *condition_args,
                        **condition_kwargs,
                    )
                
                if not self.is_mm_dit():
                    return_dict = {"shape": return_dict}
                
                # If per-view pose optimization enabled, extract pose for each view
                if optimize_per_view_pose and all_view_poses_storage is not None and all_view_poses_storage.get('per_view_x_t'):
                    per_view_x_t = all_view_poses_storage['per_view_x_t']
                    
                    from sam3d_objects.pipeline.multi_view_utils import POSE_KEYS
                    
                    # Extract pose part for each view
                    all_view_poses_raw = {}
                    for key in per_view_x_t[0].keys():
                        if key in POSE_KEYS:
                            stacked = torch.stack([per_view_x_t[i][key] for i in range(num_views)])
                            all_view_poses_raw[key] = stacked.detach().cpu()
                    
                    return_dict["all_view_poses_raw"] = all_view_poses_raw
                    logger.info(f"[Stage 1] Per-view pose optimization: saved {num_views} views' poses")
                    
                    # Important: Override return_dict pose with View 0 independently iterated pose
                    # Because solver's x_t and per_view_x_t[0] may be out of sync
                    # (solver uses returned fused velocity, while per_view_x_t[0] is maintained by us)
                    for key in per_view_x_t[0].keys():
                        if key in POSE_KEYS:
                            return_dict[key] = per_view_x_t[0][key]
                    logger.info(f"[Stage 1] Replaced return_dict pose with per_view_x_t[0]")

                shape_latent = return_dict["shape"]
                logger.info(f"[Stage 1 Multi-view] Generated shape_latent shape: {shape_latent.shape}")
                ss = ss_decoder(
                    shape_latent.permute(0, 2, 1)
                    .contiguous()
                    .view(shape_latent.shape[0], 8, 16, 16, 16)
                )
                logger.info(f"[Stage 1 Multi-view] Decoded sparse structure shape: {ss.shape}")
                coords = torch.argwhere(ss > 0)[:, [0, 2, 3, 4]].int()
                logger.info(f"[Stage 1 Multi-view] Extracted coords shape: {coords.shape}")

                # downsample output
                return_dict["coords_original"] = coords
                original_shape = coords.shape
                if self.downsample_ss_dist > 0:
                    coords = prune_sparse_structure(
                        coords,
                        max_neighbor_axes_dist=self.downsample_ss_dist,
                    )
                coords, downsample_factor = downsample_sparse_structure(coords)
                logger.info(
                    f"[Stage 1 Multi-view] Downsampled coords from {original_shape[0]} to {coords.shape[0]}"
                )
                return_dict["coords"] = coords
                return_dict["downsample_factor"] = downsample_factor

        ss_generator.inference_steps = prev_inference_steps
        return return_dict

    def refine_pose_per_view(
        self,
        view_ss_input_dicts: List[dict],
        fixed_shape_latent: torch.Tensor,
        inference_steps: int = 50,
    ) -> List[dict]:
        """
        Phase 2: Fix shape, optimize pose independently for each view.
        
        This function is used for camera pose estimation:
        1. Assume object is stationary in world coordinates
        2. Different views see different object poses because camera positions differ
        3. Camera poses can be estimated by comparing poses from different views
        
        Args:
            view_ss_input_dicts: List of input dicts for each view
            fixed_shape_latent: Fixed shape latent (from phase 1)
            inference_steps: Number of iteration steps
            
        Returns:
            List of raw pose dicts for each view (before decoding)
        """
        from sam3d_objects.pipeline.multi_view_utils import POSE_KEYS
        
        ss_generator = self.models["ss_generator"]
        num_views = len(view_ss_input_dicts)
        
        logger.info(f"[Pose Refinement] Refining pose for {num_views} views with {inference_steps} steps")
        logger.info(f"[Pose Refinement] Fixed shape latent shape: {fixed_shape_latent.shape}")
        
        # Save original settings
        prev_inference_steps = ss_generator.inference_steps
        ss_generator.inference_steps = inference_steps
        
        # Ensure correct CFG settings
        ss_generator.no_shortcut = True
        ss_generator.reverse_fn.strength = self.ss_cfg_strength
        ss_generator.reverse_fn.strength_pm = self.ss_cfg_strength_pm
        
        all_view_poses_raw = []
        
        with torch.no_grad():
            with torch.autocast(device_type="cuda", dtype=self.shape_model_dtype):
                for view_idx in range(num_views):
                    logger.info(f"[Pose Refinement] Processing view {view_idx}/{num_views-1}")
                    
                    # Prepare single-view conditions
                    view_input = view_ss_input_dicts[view_idx]
                    condition_args, condition_kwargs = self.get_condition_input(
                        self.condition_embedders["ss_condition_embedder"],
                        view_input,
                        self.ss_condition_input_mapping,
                    )
                    
                    # Get latent shape (for initializing pose noise)
                    image = view_input["image"]
                    bs = image.shape[0]
                    device = image.device
                    
                    if self.is_mm_dit():
                        latent_shape_dict = {
                            k: (bs,) + (v.pos_emb.shape[0], v.input_layer.in_features)
                            for k, v in ss_generator.reverse_fn.backbone.latent_mapping.items()
                        }
                    else:
                        latent_shape_dict = (bs,) + (4096, 8)
                    
                    # Manually do flow matching iteration, fix shape
                    # Initialize x_t
                    if isinstance(latent_shape_dict, dict):
                        x_t = {
                            k: torch.randn(v, device=device)
                            for k, v in latent_shape_dict.items()
                        }
                        # Replace shape with fixed value
                        x_t['shape'] = fixed_shape_latent.clone().to(device)
                    else:
                        x_t = torch.randn(latent_shape_dict, device=device)
                    
                    # Prepare timesteps
                    t_seq = torch.linspace(0, 1, inference_steps + 1).to(device)
                    dt = 1.0 / inference_steps
                    
                    # Iterate
                    for step_idx in range(inference_steps):
                        t = t_seq[step_idx]
                        t_scaled = t * ss_generator.time_scale
                        
                        # Get velocity
                        velocity = ss_generator.reverse_fn(
                            x_t, 
                            t_scaled.unsqueeze(0) if t_scaled.dim() == 0 else t_scaled,
                            *condition_args, 
                            **condition_kwargs
                        )
                        
                        # Update x_t
                        if isinstance(velocity, dict):
                            for key in velocity.keys():
                                if key in POSE_KEYS:
                                    # Only update pose
                                    x_t[key] = x_t[key] + velocity[key] * dt
                                # shape 保持不变（不更新）
                        else:
                            x_t = x_t + velocity * dt
                    
                    # 提取finally的 pose
                    view_pose_raw = {}
                    if isinstance(x_t, dict):
                        for key in x_t.keys():
                            if key in POSE_KEYS:
                                view_pose_raw[key] = x_t[key].detach().cpu()
                    
                    all_view_poses_raw.append(view_pose_raw)
                    
                    # 打印debuginfo
                    if 'scale' in view_pose_raw:
                        logger.info(f"  View {view_idx} raw scale: {view_pose_raw['scale'].flatten()[:3]}")
                    if 'translation' in view_pose_raw:
                        logger.info(f"  View {view_idx} raw translation: {view_pose_raw['translation'].flatten()[:3]}")
        
        # restoreoriginal设置
        ss_generator.inference_steps = prev_inference_steps
        
        logger.info(f"[Pose Refinement] Completed for {num_views} views")
        
        return all_view_poses_raw

    def estimate_pose_independent(
        self,
        view_ss_input_dicts: List[dict],
        inference_steps: int = 50,
    ) -> List[dict]:
        """
        each view完全独立地从 noise 优化 shape + pose。
        
        这是 refine_pose_per_view 的变体，用于对比实验：
        - refine_pose_per_view: 固定multi-view fusion的 shape，只优化 pose
        - estimate_pose_independent: each view独立优化 shape + pose，只取 pose
        
        theoretical上，如果模型的 pose 预测是准确的，这两种方法should得到相似result。
        如果结果差异很大，说明：
        1. 固定 shape 时 pose 预测不准确（方法 A 的问题）
        2. or单视角 pose 预测本身就不一致（思路本身的问题）
        
        Args:
            view_ss_input_dicts: List of input dicts for each view
            inference_steps: Number of iteration steps
            
        Returns:
            List of raw pose dicts for each view (before decoding)
        """
        from sam3d_objects.pipeline.multi_view_utils import POSE_KEYS
        
        ss_generator = self.models["ss_generator"]
        ss_decoder = self.models["ss_decoder"]
        num_views = len(view_ss_input_dicts)
        
        logger.info(f"[Independent Pose] Estimating pose independently for {num_views} views")
        logger.info(f"[Independent Pose] Inference steps: {inference_steps}")
        
        # Save original settings
        prev_inference_steps = ss_generator.inference_steps
        ss_generator.inference_steps = inference_steps
        
        # Ensure correct CFG settings
        ss_generator.no_shortcut = True
        ss_generator.reverse_fn.strength = self.ss_cfg_strength
        ss_generator.reverse_fn.strength_pm = self.ss_cfg_strength_pm
        
        all_view_poses_raw = []
        all_view_shapes = []  # 也保存 shape 用于分析
        
        with torch.no_grad():
            with torch.autocast(device_type="cuda", dtype=self.shape_model_dtype):
                for view_idx in range(num_views):
                    logger.info(f"[Independent Pose] Processing view {view_idx}/{num_views-1}")
                    
                    # Prepare single-view conditions
                    view_input = view_ss_input_dicts[view_idx]
                    
                    # Debug：打印each view的 condition info
                    if 'pointmap' in view_input:
                        pm = view_input['pointmap']
                        logger.info(f"  View {view_idx} pointmap shape: {pm.shape}, "
                                    f"mean: [{pm[:, 0].mean().item():.4f}, {pm[:, 1].mean().item():.4f}, {pm[:, 2].mean().item():.4f}], "
                                    f"std: [{pm[:, 0].std().item():.4f}, {pm[:, 1].std().item():.4f}, {pm[:, 2].std().item():.4f}]")
                    if 'image' in view_input:
                        img = view_input['image']
                        logger.info(f"  View {view_idx} image shape: {img.shape}, "
                                    f"mean: {img.mean().item():.4f}, std: {img.std().item():.4f}")
                    
                    condition_args, condition_kwargs = self.get_condition_input(
                        self.condition_embedders["ss_condition_embedder"],
                        view_input,
                        self.ss_condition_input_mapping,
                    )
                    
                    # Get latent shape
                    image = view_input["image"]
                    bs = image.shape[0]
                    device = image.device
                    
                    if self.is_mm_dit():
                        latent_shape_dict = {
                            k: (bs,) + (v.pos_emb.shape[0], v.input_layer.in_features)
                            for k, v in ss_generator.reverse_fn.backbone.latent_mapping.items()
                        }
                    else:
                        latent_shape_dict = (bs,) + (4096, 8)
                    
                    # Use标准的 flow matching 生成（shape + pose 一起优化）
                    result = ss_generator(
                        latent_shape_dict,
                        device,
                        *condition_args,
                        **condition_kwargs,
                    )
                    
                    if not self.is_mm_dit():
                        result = {"shape": result}
                    
                    # 提取 pose
                    view_pose_raw = {}
                    for key in result.keys():
                        if key in POSE_KEYS:
                            view_pose_raw[key] = result[key].detach().cpu()
                    
                    all_view_poses_raw.append(view_pose_raw)
                    
                    # Save shape 用于分析
                    if 'shape' in result:
                        all_view_shapes.append(result['shape'].detach().cpu())
                    
                    # 打印debuginfo
                    if 'scale' in view_pose_raw:
                        logger.info(f"  View {view_idx} raw scale: {view_pose_raw['scale'].flatten()[:3]}")
                    if 'translation' in view_pose_raw:
                        logger.info(f"  View {view_idx} raw translation: {view_pose_raw['translation'].flatten()[:3]}")
        
        # restoreoriginal设置
        ss_generator.inference_steps = prev_inference_steps
        
        # 分析differentview's shape 差异
        if len(all_view_shapes) > 1:
            shapes_stacked = torch.stack(all_view_shapes)
            shape_std = shapes_stacked.std(dim=0).mean().item()
            logger.info(f"[Independent Pose] Shape std across views: {shape_std:.6f}")
            logger.info(f"  (Lower is better - indicates consistent shape prediction)")
        
        logger.info(f"[Independent Pose] Completed for {num_views} views")
        
        return all_view_poses_raw

    def sample_slat_multi_view(
        self,
        view_slat_input_dicts: List[dict],
        coords: torch.Tensor,
        inference_steps=25,
        use_distillation=False,
        mode: Literal['stochastic', 'multidiffusion'] = 'multidiffusion',
        attention_logger: Optional[Any] = None,
    ) -> sp.SparseTensor:
        """
        多视角结构化潜在生成
        
        Args:
            view_slat_input_dicts: each view的输入字典列表
            coords: 坐标（从Stage 1得到）
            inference_steps: Number of inference steps
            use_distillation: Whether to use distillation
            mode: 'stochastic' or 'multidiffusion'
        """
        from sam3d_objects.pipeline.multi_view_utils import inject_generator_multi_view
        
        image = view_slat_input_dicts[0]["image"]
        DEVICE = image.device
        slat_generator = self.models["slat_generator"]
        num_views = len(view_slat_input_dicts)
        if attention_logger is not None:
            attention_logger.start_stage("slat")
            attention_logger.set_num_views(num_views)
            attention_logger.set_view(0)
        latent_shape = (image.shape[0],) + (coords.shape[0], 8)
        logger.info(f"[Stage 2] Coords shape: {coords.shape}")
        logger.info(f"[Stage 2] Latent shape: {latent_shape}")
        prev_inference_steps = slat_generator.inference_steps
        if inference_steps:
            slat_generator.inference_steps = inference_steps
        if use_distillation:
            slat_generator.no_shortcut = False
            slat_generator.reverse_fn.strength = 0
        else:
            slat_generator.no_shortcut = True
            slat_generator.reverse_fn.strength = self.slat_cfg_strength

        logger.info(
            f"Sampling sparse latent with {num_views} views: "
            f"inference_steps={slat_generator.inference_steps}, mode={mode}"
        )

        with torch.autocast(device_type="cuda", dtype=self.dtype):
            with torch.no_grad():
                # Prepare multi-view conditions
                condition_args, condition_kwargs = self.get_multi_view_condition_input(
                    self.condition_embedders["slat_condition_embedder"],
                    view_slat_input_dicts,
                    self.slat_condition_input_mapping,
                )
                condition_args += (coords.cpu().numpy(),)
                
                # Inject multi-view support
                with inject_generator_multi_view(
                    slat_generator,
                    num_views=num_views,
                    num_steps=slat_generator.inference_steps,
                    mode=mode,
                    attention_logger=attention_logger,
                ):
                    slat = slat_generator(
                        latent_shape, DEVICE, *condition_args, **condition_kwargs
                    )
                
                logger.info(f"[Stage 2] Generated slat shape (before SparseTensor): {slat[0].shape if isinstance(slat, (list, tuple)) else slat.shape}")
                slat = sp.SparseTensor(
                    coords=coords,
                    feats=slat[0],
                ).to(DEVICE)
                slat = slat * self.slat_std.to(DEVICE) + self.slat_mean.to(DEVICE)
                logger.info(f"[Stage 2] Final slat: coords={slat.coords.shape}, feats={slat.feats.shape}")

        slat_generator.inference_steps = prev_inference_steps
        return slat

    def sample_slat_multi_view_weighted(
        self,
        view_slat_input_dicts: List[dict],
        coords: torch.Tensor,
        inference_steps=25,
        use_distillation=False,
        attention_logger: Optional[Any] = None,
        weighting_config: Optional[Any] = None,
        save_stage2_init: bool = False,
        save_stage2_init_path: Optional[Any] = None,
    ) -> sp.SparseTensor:
        """
        多视角结构化潜在生成（带加权融合，两阶段方法）
        
        流程：
        1. Warmup Pass: 跑one step 收集 attention，compute weights
        2. Main Pass: 用计算出的权重，从头start完整迭代
        
        Args:
            view_slat_input_dicts: each view的输入字典列表
            coords: 坐标（从Stage 1得到）
            inference_steps: Number of inference steps
            use_distillation: Whether to use distillation
            attention_logger: 注意力记录器
            weighting_config: 加权config
            save_stage2_init: 是否保存 Stage 2 初始 latent
            save_stage2_init_path: 保存路径
        """
        from sam3d_objects.pipeline.multi_view_weighted import (
            inject_weighted_multi_view_with_precomputed_weights,
            inject_generator_multi_view_with_collector,
            AttentionCollector,
        )
        from sam3d_objects.utils.latent_weighting import LatentWeightManager, WeightingConfig
        
        image = view_slat_input_dicts[0]["image"]
        DEVICE = image.device
        slat_generator = self.models["slat_generator"]
        num_views = len(view_slat_input_dicts)
        
        latent_shape = (image.shape[0],) + (coords.shape[0], 8)
        logger.info(f"[Stage 2 Weighted] Coords shape: {coords.shape}")
        logger.info(f"[Stage 2 Weighted] Latent shape: {latent_shape}")
        
        prev_inference_steps = slat_generator.inference_steps
        if inference_steps:
            slat_generator.inference_steps = inference_steps
        if use_distillation:
            slat_generator.no_shortcut = False
            slat_generator.reverse_fn.strength = 0
        else:
            slat_generator.no_shortcut = True
            slat_generator.reverse_fn.strength = self.slat_cfg_strength

        # Createweight manager
        if weighting_config is None:
            weighting_config = WeightingConfig()
        weight_manager = LatentWeightManager(weighting_config)

        with torch.autocast(device_type="cuda", dtype=self.dtype):
            with torch.no_grad():
                # Prepare multi-view conditions
                condition_args, condition_kwargs = self.get_multi_view_condition_input(
                    self.condition_embedders["slat_condition_embedder"],
                    view_slat_input_dicts,
                    self.slat_condition_input_mapping,
                )
                condition_args += (coords.cpu().numpy(),)
                
                # ============ Phase 1: Warmup Pass (1 step) ============
                logger.info("[Stage 2 Weighted] Phase 1: Warmup pass to collect attention...")
                
                # 临时设置为只跑 1 步
                original_steps = slat_generator.inference_steps
                slat_generator.inference_steps = 1
                
                if attention_logger is not None:
                    attention_logger.start_stage("slat")
                    attention_logger.set_num_views(num_views)
                    attention_logger.set_view(0)
                
                # Createone临时的 attention 收集器
                attention_collector = AttentionCollector(
                    num_views=num_views,
                    target_layer=weighting_config.attention_layer,
                )
                
                # 用simple average跑一步，同时收集 attention 到内存
                with inject_generator_multi_view_with_collector(
                    slat_generator,
                    num_views=num_views,
                    num_steps=1,
                    attention_collector=attention_collector,
                    attention_logger=attention_logger,
                ):
                    _ = slat_generator(
                        latent_shape, DEVICE, *condition_args, **condition_kwargs
                    )
                
                # 从内存中的 attention compute weights
                collected_attentions = attention_collector.get_attentions()
                if collected_attentions:
                    for view_idx, attn in collected_attentions.items():
                        weight_manager.add_view_attention(view_idx, attn, step=0)
                    logger.info(f"[Stage 2 Weighted] Collected attention for {len(collected_attentions)} views")
                
                # Set downsample 映射（用于将权重从downsample维度expand tooriginal维度）
                downsample_idx = attention_collector.get_downsample_idx()
                original_coords = attention_collector.get_original_coords()
                downsampled_coords = attention_collector.get_downsampled_coords()
                
                if downsample_idx is not None:
                    weight_manager.set_downsample_mapping(
                        downsample_idx, 
                        original_coords, 
                        downsampled_coords
                    )
                    logger.info(f"[Stage 2 Weighted] Downsample mapping set")
                else:
                    logger.warning("[Stage 2 Weighted] No downsample mapping found!")
                
                # Ifneed visibility，调用相应的 callback 计算
                if weighting_config.weight_source in ["visibility", "mixed"]:
                    if weighting_config.visibility_callback is not None:
                        if downsampled_coords is not None:
                            logger.info(f"[Stage 2 Weighted] Computing visibility for {num_views} views on {downsampled_coords.shape[0]} downsampled coords...")
                            # Convert to numpy for visibility computation
                            coords_np = downsampled_coords.cpu().numpy() if hasattr(downsampled_coords, 'cpu') else downsampled_coords
                            # Get object pose (set by run_multi_view after Stage 1)
                            object_pose = weighting_config.object_pose
                            if object_pose is None:
                                logger.warning("[Stage 2 Weighted] object_pose not set! Visibility computation may fail.")
                            try:
                                visibility_matrix = weighting_config.visibility_callback(coords_np, num_views, object_pose)
                                # Convert to torch tensor
                                if not isinstance(visibility_matrix, torch.Tensor):
                                    visibility_matrix = torch.from_numpy(visibility_matrix).float()
                                weight_manager.set_visibility_matrix(visibility_matrix)
                                logger.info(f"[Stage 2 Weighted] Visibility matrix set: shape={visibility_matrix.shape}")
                            except Exception as e:
                                logger.error(f"[Stage 2 Weighted] Visibility callback failed: {e}")
                                import traceback
                                traceback.print_exc()
                                if weighting_config.weight_source == "visibility":
                                    raise  # Re-raise for visibility-only mode
                                else:
                                    logger.warning("[Stage 2 Weighted] Falling back to entropy-only weights")
                        else:
                            logger.warning("[Stage 2 Weighted] No downsampled coords available for visibility computation!")
                    else:
                        logger.warning(
                            f"[Stage 2 Weighted] weight_source={weighting_config.weight_source} requires visibility_callback, "
                            "but none provided! Falling back to entropy-only if available."
                        )
                
                # Computedownsample维度的权重
                weights_downsampled = weight_manager.compute_weights()
                
                if weights_downsampled and len(weights_downsampled) == num_views:
                    logger.info(f"[Stage 2 Weighted] Downsampled weights computed for {num_views} views")
                    for v, w in weights_downsampled.items():
                        logger.info(f"  View {v} (downsampled): mean={w.mean():.4f}, std={w.std():.4f}")
                    
                    # 扩展权重到original维度
                    weights_expanded = weight_manager.get_expanded_weights()
                    if weights_expanded:
                        logger.info(f"[Stage 2 Weighted] Expanded weights to original dimension")
                        for v, w in weights_expanded.items():
                            logger.info(f"  View {v} (expanded): shape={w.shape}, mean={w.mean():.4f}")
                else:
                    weights_expanded = None
                    logger.warning("[Stage 2 Weighted] Failed to compute weights, will use simple average")
                
                # ============ Phase 2: Main Pass (full steps with weights) ============
                logger.info(f"[Stage 2 Weighted] Phase 2: Main pass with {original_steps} steps...")
                
                # restore完整步数
                slat_generator.inference_steps = original_steps
                
                # 重新start attention logger（如果need保存完整的 attention）
                if attention_logger is not None:
                    attention_logger.start_stage("slat")
                    attention_logger.set_num_views(num_views)
                
                # Generate初始噪声（用于保存和迭代）
                initial_noise = slat_generator._generate_noise(latent_shape, DEVICE)
                
                # Save Stage 2 初始 latent（如果need）
                if save_stage2_init and save_stage2_init_path is not None:
                    logger.info(f"[Stage 2 Weighted] Saving Stage 2 initial latent to {save_stage2_init_path}")
                    # Save所有need的info
                    stage2_init_data = {
                        "coords": coords.cpu(),
                        "initial_noise": initial_noise.cpu() if torch.is_tensor(initial_noise) else initial_noise,
                        "latent_shape": latent_shape,
                        "num_views": num_views,
                        "inference_steps": original_steps,
                        # Save条件编码（already embed 过的）
                        "condition_args": tuple(
                            arg.cpu() if torch.is_tensor(arg) else arg 
                            for arg in condition_args[:-1]  # 排除 coords numpy
                        ),
                        # Saveconfig
                        "slat_cfg_strength": self.slat_cfg_strength,
                        "use_distillation": use_distillation,
                    }
                    torch.save(stage2_init_data, save_stage2_init_path)
                    logger.info(f"[Stage 2 Weighted] Stage 2 init saved: noise shape={initial_noise.shape if torch.is_tensor(initial_noise) else 'dict'}")
                
                # 临时替换 _generate_noise 方法，让它返回我们预先生成的噪声
                original_generate_noise = slat_generator._generate_noise
                def fixed_noise_generator(shape, device):
                    return initial_noise.to(device)
                slat_generator._generate_noise = fixed_noise_generator
                
                # 用扩展后的权重进行完整迭代（使用指定的初始噪声）
                try:
                    with inject_weighted_multi_view_with_precomputed_weights(
                        slat_generator,
                        num_views=num_views,
                        num_steps=original_steps,
                        precomputed_weights=weights_expanded,  # Use扩展后的权重
                        attention_logger=attention_logger,
                    ):
                        slat = slat_generator(
                            latent_shape, DEVICE, *condition_args, **condition_kwargs
                        )
                finally:
                    # restoreoriginal的 _generate_noise 方法
                    slat_generator._generate_noise = original_generate_noise
                
                logger.info(f"[Stage 2 Weighted] Generated slat shape: {slat[0].shape if isinstance(slat, (list, tuple)) else slat.shape}")
                slat = sp.SparseTensor(
                    coords=coords,
                    feats=slat[0],
                ).to(DEVICE)
                slat = slat * self.slat_std.to(DEVICE) + self.slat_mean.to(DEVICE)
                logger.info(f"[Stage 2 Weighted] Final slat: coords={slat.coords.shape}, feats={slat.feats.shape}")

        slat_generator.inference_steps = prev_inference_steps
        return slat, weight_manager

    def run_multi_view(
        self,
        view_images: List[Union[np.ndarray, Image.Image]],
        view_masks: List[Optional[Union[None, np.ndarray, Image.Image]]] = None,
        view_pointmaps: Optional[List[Optional[np.ndarray]]] = None,  # 外部 pointmap
        num_samples: int = 1,
        seed: Optional[int] = None,
        stage1_inference_steps: Optional[int] = None,
        stage2_inference_steps: Optional[int] = None,
        use_stage1_distillation: bool = False,
        use_stage2_distillation: bool = False,
        decode_formats: Optional[List[str]] = None,
        with_mesh_postprocess: bool = True,
        with_texture_baking: bool = True,
        use_vertex_color: bool = False,
        stage1_only: bool = False,
        mode: Literal['stochastic', 'multidiffusion'] = 'multidiffusion',
        attention_logger: Optional[Any] = None,
        weighting_config: Optional[Any] = None,  # 加权config
        save_stage2_init: bool = False,  # 是否保存 Stage 2 初始 latent
        save_stage2_init_path: Optional[Any] = None,  # Save路径
        optimize_per_view_pose: bool = False,  # 是否为each view独立优化 pose
    ) -> dict:
        """
        多视角推理主函数
        
        Args:
            view_images: each view的图像列表
            view_masks: each view的掩码列表（可选）
            view_pointmaps: each view的外部 pointmap 列表（可选）
                - 格式: (3, H, W) 的 numpy array，PyTorch3D 坐标系
                - 如果提供，将使用外部 pointmap 而不是内置深度模型计算
                - can来自 DA3 等外部深度估计模型
            num_samples: 生成样本数
            seed: 随机种子
            stage1_inference_steps: Stage 1推理步数
            stage2_inference_steps: Stage 2推理步数
            use_stage1_distillation: 是否使用Stage 1蒸馏
            use_stage2_distillation: 是否使用Stage 2蒸馏
            decode_formats: 解码格式
            with_mesh_postprocess: 是否进行网格后处理
            with_texture_baking: 是否进行纹理烘焙
            use_vertex_color: 是否使用顶点颜色
            stage1_only: 是否只运行Stage 1
            mode: 'stochastic' or 'multidiffusion'
            optimize_per_view_pose: Whether to optimize pose independently for each view
                - False (默认): Pose 只用 View 0 的 velocity 更新
                - True: each view独立迭代自己的 Pose（用于多视角一致性分析）
        """
        num_views = len(view_images)
        if view_masks is None:
            view_masks = [None] * num_views
        assert len(view_masks) == num_views, "Number of masks must match number of images"
        
        # Process外部 pointmap
        if view_pointmaps is not None:
            assert len(view_pointmaps) == num_views, "Number of pointmaps must match number of images"
            logger.info(f"Using external pointmaps for {sum(1 for p in view_pointmaps if p is not None)}/{num_views} views")
        else:
            view_pointmaps = [None] * num_views
        
        if seed is not None:
            torch.manual_seed(seed)
        
        logger.info(f"Running multi-view inference with {num_views} views, mode={mode}")
        
        # 预处理each view
        # Note：need先将mask合并到图像的alpha通道，then调用preprocess_image
        view_ss_input_dicts = []
        view_slat_input_dicts = []
        raw_view_pointmaps: List[np.ndarray] = []
        for i, (image, mask, ext_pointmap) in enumerate(zip(view_images, view_masks, view_pointmaps)):
            logger.info(f"Preprocessing view {i+1}/{num_views}")
            
            # 将mask合并到图像的alpha通道（RGBA格式）
            # Ifimagealready是RGBA格式（从mask的alpha通道加载），maskpossible是None
            if mask is not None:
                # 确保image是numpy数组
                if isinstance(image, Image.Image):
                    image = np.array(image)
                else:
                    image = np.array(image)
                
                # 确保mask是numpy数组
                mask = np.array(mask)
                
                # Ifmask是bool类型，转换为uint8
                if mask.dtype == bool:
                    mask = mask.astype(np.uint8) * 255
                elif mask.dtype != np.uint8:
                    # Ifmask是0-1范围的float，转换为0-255
                    if mask.max() <= 1.0:
                        mask = (mask * 255).astype(np.uint8)
                    else:
                        mask = mask.astype(np.uint8)
                
                if mask.ndim == 2:
                    mask = mask[..., None]
                
                # 合并mask到alpha通道
                if image.shape[-1] == 3:  # RGB
                    rgba_image = np.concatenate([image, mask], axis=-1).astype(np.uint8)
                elif image.shape[-1] == 4:  # already是RGBA，替换alpha通道
                    rgba_image = np.concatenate([image[..., :3], mask], axis=-1).astype(np.uint8)
                else:
                    raise ValueError(f"Unexpected image shape: {image.shape}")
            else:
                # If没有mask，假设imagealready是RGBA格式
                if isinstance(image, Image.Image):
                    rgba_image = np.array(image)
                else:
                    rgba_image = np.array(image)
            
            # Convert为PIL Image（preprocess_imageneed）
            rgba_image_pil = Image.fromarray(rgba_image)
            
            # 调用preprocess_image（注意：InferencePipelinePointMapneedpointmap）
            # 先检查是否是InferencePipelinePointMap
            if hasattr(self, 'compute_pointmap'):
                # 这是InferencePipelinePointMap，need计算或使用外部pointmap
                if ext_pointmap is not None:
                    # Use外部提供的 pointmap（来自 DA3 等）
                    # ext_pointmap 格式: (3, H, W) numpy array
                    # 
                    # 重要：need调用 compute_pointmap 来应用坐标变换！
                    # compute_pointmap 会：
                    #   1. 对 DA3 pointmap 翻转 Y 和 Z（使其与 MoGe original输出一致）
                    #   2. 应用 camera_to_pytorch3d_camera 变换
                    #   3. 返回 PyTorch3D 空间的 pointmap
                    logger.info(f"  View {i+1}: Using external pointmap, shape={ext_pointmap.shape}")
                    ext_pointmap_tensor = torch.from_numpy(ext_pointmap).float()
                    pointmap_dict = self.compute_pointmap(rgba_image_pil, pointmap=ext_pointmap_tensor)
                    pointmap = pointmap_dict["pointmap"]
                else:
                    # 用内置模型计算 pointmap
                    pointmap_dict = self.compute_pointmap(rgba_image_pil, pointmap=None)
                    pointmap = pointmap_dict["pointmap"]
                
                # Save真实尺度的 pointmap（用于visualization对齐）
                # Note：此时 pointmap already在 PyTorch3D 空间，无论是 MoGe 还是 DA3
                if pointmap is not None:
                    pointmap_metric = pointmap.detach()
                    if hasattr(type(self), "_down_sample_img"):
                        pointmap_metric = type(self)._down_sample_img(pointmap_metric)
                    pointmap_metric = pointmap_metric.cpu().permute(1, 2, 0)  # HxWx3
                    raw_view_pointmaps.append(pointmap_metric.numpy())
                
                ss_input_dict = self.preprocess_image(
                    rgba_image_pil, self.ss_preprocessor, pointmap=pointmap
                )
                slat_input_dict = self.preprocess_image(
                    rgba_image_pil, self.slat_preprocessor
                )
            else:
                # 这是InferencePipeline，不needpointmap
                if ext_pointmap is not None:
                    logger.warning(f"  View {i+1}: External pointmap provided but pipeline doesn't support it (not InferencePipelinePointMap)")
                ss_input_dict = self.preprocess_image(
                    rgba_image_pil, self.ss_preprocessor
                )
                slat_input_dict = self.preprocess_image(
                    rgba_image_pil, self.slat_preprocessor
                )
            
            view_ss_input_dicts.append(ss_input_dict)
            view_slat_input_dicts.append(slat_input_dict)
        
        # Stage 1: 生成稀疏结构
        logger.info("Stage 1: Sampling sparse structure...")
        ss_return_dict = self.sample_sparse_structure_multi_view(
            view_ss_input_dicts,
            inference_steps=stage1_inference_steps,
            use_distillation=use_stage1_distillation,
            mode=mode,
            attention_logger=attention_logger,
            optimize_per_view_pose=optimize_per_view_pose,
        )
        
        # Get pointmap scale/shift from the first view for pose decoding
        # These are needed to convert from normalized space to metric space
        pointmap_scale = view_ss_input_dicts[0].get("pointmap_scale", None)
        pointmap_shift = view_ss_input_dicts[0].get("pointmap_shift", None)
        
        ss_return_dict.update(self.pose_decoder(
            ss_return_dict,
            scene_scale=pointmap_scale,
            scene_shift=pointmap_shift,
        ))
        
        # Store pointmap_scale/shift for later use (e.g., alignment with DA3)
        ss_return_dict["pointmap_scale"] = pointmap_scale
        ss_return_dict["pointmap_shift"] = pointmap_shift
        
        # Decode poses for ALL views (not just the first one)
        # This is useful for analyzing multi-view pose consistency
        if "all_view_poses_raw" in ss_return_dict:
            all_view_poses_decoded = self._decode_all_view_poses(
                ss_return_dict["all_view_poses_raw"],
                view_ss_input_dicts,
            )
            ss_return_dict["all_view_poses_decoded"] = all_view_poses_decoded
            logger.info(f"[Multi-view] Decoded poses for {len(all_view_poses_decoded)} views")
        
        if "scale" in ss_return_dict:
            logger.info(f"Rescaling scale by {ss_return_dict['downsample_factor']}")
            ss_return_dict["scale"] = ss_return_dict["scale"] * ss_return_dict["downsample_factor"]
        
        if stage1_only:
            logger.info("Finished!")
            ss_return_dict["voxel"] = ss_return_dict["coords"][:, 1:] / 64 - 0.5
            return ss_return_dict
        
        # Stage 2: 生成结构化潜在
        coords = ss_return_dict["coords"]
        logger.info("Stage 2: Sampling structured latent...")
        
        weight_manager = None
        if weighting_config is not None:
            # Use加权融合
            logger.info("Using weighted multi-view fusion")
            
            # Set object pose for visibility computation (if needed)
            if weighting_config.weight_source in ["visibility", "mixed"]:
                if weighting_config.visibility_callback is not None:
                    # Extract object pose from Stage 1 results
                    object_pose = {
                        'scale': ss_return_dict.get('scale'),
                        'rotation': ss_return_dict.get('rotation'),
                        'translation': ss_return_dict.get('translation'),
                    }
                    weighting_config.object_pose = object_pose
                    logger.info(f"[Stage 2] Set object pose for visibility: scale={object_pose['scale']}")
            
            slat, weight_manager = self.sample_slat_multi_view_weighted(
                view_slat_input_dicts,
                coords,
                inference_steps=stage2_inference_steps,
                use_distillation=use_stage2_distillation,
                attention_logger=attention_logger,
                weighting_config=weighting_config,
                save_stage2_init=save_stage2_init,
                save_stage2_init_path=save_stage2_init_path,
            )
        else:
            # Useoriginal的simple average融合
            slat = self.sample_slat_multi_view(
                view_slat_input_dicts,
                coords,
                inference_steps=stage2_inference_steps,
                use_distillation=use_stage2_distillation,
                mode=mode,
                attention_logger=attention_logger,
            )
        
        # Decode
        outputs = self.decode_slat(
            slat, self.decode_formats if decode_formats is None else decode_formats
        )
        outputs = self.postprocess_slat_output(
            outputs, with_mesh_postprocess, with_texture_baking, use_vertex_color
        )
        logger.info("Finished!")
        
        result = {
            **ss_return_dict,
            **outputs,
            "view_ss_input_dicts": view_ss_input_dicts,  # Save以便 overlay 使用
        }
        
        if raw_view_pointmaps:
            result["raw_view_pointmaps"] = raw_view_pointmaps
        
        # If使用了加权融合，添加权重info
        if weight_manager is not None:
            result["weight_manager"] = weight_manager
        
        return result
    
    def estimate_pose_manual_sync_time(
        self,
        view_ss_input_dicts: List[dict],
        fixed_shape_latent: torch.Tensor,
        inference_steps: int = 50,
    ) -> List[dict]:
        """
        思路 1：手动同步时间步模式
        
        核心思想：
        - Pose 正常迭代（使用网络预测的 velocity）
        - Shape 手动同步到same的时间步（不迭代，直接计算）
        - 计算平均变化量：xx_s = (shape_final - shape_noise) / n_steps
        - Step k: shape = shape_noise + k * xx_s
        
        优点：
        - 保持 shape 和 pose 在same的时间步（进度）
        - 符合训练时的分布（两者在same t）
        - finally严格等于 Stage 1 的 shape
        
        Args:
            view_ss_input_dicts: List of input dicts for each view
            fixed_shape_latent: Stage 1 的 shape latent（目标 shape）
            inference_steps: Number of iteration steps
            
        Returns:
            List of raw pose dicts for each view
        """
        from sam3d_objects.pipeline.multi_view_utils import POSE_KEYS
        
        ss_generator = self.models["ss_generator"]
        num_views = len(view_ss_input_dicts)
        
        logger.info(f"[Manual Sync Time] Refining pose for {num_views} views")
        logger.info(f"[Manual Sync Time] Total steps: {inference_steps}")
        
        prev_inference_steps = ss_generator.inference_steps
        ss_generator.inference_steps = inference_steps
        ss_generator.no_shortcut = True
        ss_generator.reverse_fn.strength = self.ss_cfg_strength
        ss_generator.reverse_fn.strength_pm = self.ss_cfg_strength_pm
        
        all_view_poses_raw = []
        
        with torch.no_grad():
            with torch.autocast(device_type="cuda", dtype=self.shape_model_dtype):
                for view_idx in range(num_views):
                    logger.info(f"[Manual Sync Time] Processing view {view_idx}/{num_views-1}")
                    
                    view_input = view_ss_input_dicts[view_idx]
                    condition_args, condition_kwargs = self.get_condition_input(
                        self.condition_embedders["ss_condition_embedder"],
                        view_input,
                        self.ss_condition_input_mapping,
                    )
                    
                    image = view_input["image"]
                    bs = image.shape[0]
                    device = image.device
                    
                    if self.is_mm_dit():
                        latent_shape_dict = {
                            k: (bs,) + (v.pos_emb.shape[0], v.input_layer.in_features)
                            for k, v in ss_generator.reverse_fn.backbone.latent_mapping.items()
                        }
                    else:
                        latent_shape_dict = (bs,) + (4096, 8)
                    
                    # Initialize：pose 从 noise start，shape 也从随机 noise start
                    if isinstance(latent_shape_dict, dict):
                        x_t = {
                            k: torch.randn(v, device=device)
                            for k, v in latent_shape_dict.items()
                        }
                        # Shape noise（随机生成，不need是 Stage 1 的original noise）
                        shape_noise = x_t['shape'].clone()
                        # 目标 shape（Stage 1 result）
                        shape_final = fixed_shape_latent.clone().to(device)
                        # Compute平均每步的变化量
                        xx_s = (shape_final - shape_noise) / inference_steps
                    else:
                        x_t = torch.randn(latent_shape_dict, device=device)
                        shape_noise = None
                        shape_final = None
                        xx_s = None
                    
                    t_seq = torch.linspace(0, 1, inference_steps + 1).to(device)
                    dt = 1.0 / inference_steps
                    
                    # Iterate
                    for step_idx in range(inference_steps):
                        t = t_seq[step_idx]
                        t_scaled = t * ss_generator.time_scale
                        
                        # Get velocity（包含 pose 和 shape 的预测）
                        velocity = ss_generator.reverse_fn(
                            x_t,
                            t_scaled.unsqueeze(0) if t_scaled.dim() == 0 else t_scaled,
                            *condition_args,
                            **condition_kwargs
                        )
                        
                        # Update策略
                        if isinstance(velocity, dict):
                            for key in velocity.keys():
                                if key in POSE_KEYS:
                                    # Pose：正常迭代（使用网络预测的 velocity）
                                    x_t[key] = x_t[key] + velocity[key] * dt
                                elif key == 'shape':
                                    # Shape：手动同步到same的时间步
                                    # Step k: shape = shape_noise + k * xx_s
                                    x_t[key] = shape_noise + (step_idx + 1) * xx_s
                        else:
                            x_t = x_t + velocity * dt
                    
                    # 验证：last一步should等于 shape_final
                    if isinstance(x_t, dict) and 'shape' in x_t:
                        shape_final_check = shape_noise + inference_steps * xx_s
                        diff = torch.abs(x_t['shape'] - shape_final).max().item()
                        logger.info(f"  View {view_idx} shape final check: max_diff = {diff:.6f} "
                                   f"(should be ~0, target shape reached)")
                    
                    # 提取 pose
                    view_pose_raw = {}
                    if isinstance(x_t, dict):
                        for key in x_t.keys():
                            if key in POSE_KEYS:
                                view_pose_raw[key] = x_t[key].detach().cpu()
                    
                    all_view_poses_raw.append(view_pose_raw)
                    
                    if 'scale' in view_pose_raw:
                        logger.info(f"  View {view_idx} raw scale: {view_pose_raw['scale'].flatten()[:3]}")
        
        ss_generator.inference_steps = prev_inference_steps
        logger.info(f"[Manual Sync Time] Completed for {num_views} views")
        
        return all_view_poses_raw
    
    def estimate_pose_mixed_update(
        self,
        view_ss_input_dicts: List[dict],
        fixed_shape_latent: torch.Tensor,
        inference_steps: int = 50,
    ) -> List[dict]:
        """
        思路 2：混合更新策略模式（固定增量版本）
        
        核心思想：
        - Pose 正常迭代（使用网络预测的 velocity）
        - Shape 使用混合更新：网络预测 + 固定的目标增量
        - 预先计算固定增量：xx_s = (shape_final - shape_noise) / n_steps
        - 混合更新：shape_update = (n-k)/n * v_network + k/n * xx_s
        - 不再使用动态的 xx_s_k（避免因current状态偏离目标而改变策略）
        
        优点：
        - 目标增量是固定的，策略更稳定
        - 尊重网络预测，同时逐渐引导到目标
        - theoretical上finallyshould接近 Stage 1 的 shape（如果权重设计合理）
        
        Args:
            view_ss_input_dicts: List of input dicts for each view
            fixed_shape_latent: Stage 1 的 shape latent（目标 shape）
            inference_steps: Number of iteration steps
            
        Returns:
            List of raw pose dicts for each view
        """
        from sam3d_objects.pipeline.multi_view_utils import POSE_KEYS
        
        ss_generator = self.models["ss_generator"]
        num_views = len(view_ss_input_dicts)
        
        logger.info(f"[Mixed Update] Refining pose for {num_views} views")
        logger.info(f"[Mixed Update] Total steps: {inference_steps}")
        
        prev_inference_steps = ss_generator.inference_steps
        ss_generator.inference_steps = inference_steps
        ss_generator.no_shortcut = True
        ss_generator.reverse_fn.strength = self.ss_cfg_strength
        ss_generator.reverse_fn.strength_pm = self.ss_cfg_strength_pm
        
        all_view_poses_raw = []
        
        with torch.no_grad():
            with torch.autocast(device_type="cuda", dtype=self.shape_model_dtype):
                for view_idx in range(num_views):
                    logger.info(f"[Mixed Update] Processing view {view_idx}/{num_views-1}")
                    
                    view_input = view_ss_input_dicts[view_idx]
                    condition_args, condition_kwargs = self.get_condition_input(
                        self.condition_embedders["ss_condition_embedder"],
                        view_input,
                        self.ss_condition_input_mapping,
                    )
                    
                    image = view_input["image"]
                    bs = image.shape[0]
                    device = image.device
                    
                    if self.is_mm_dit():
                        latent_shape_dict = {
                            k: (bs,) + (v.pos_emb.shape[0], v.input_layer.in_features)
                            for k, v in ss_generator.reverse_fn.backbone.latent_mapping.items()
                        }
                    else:
                        latent_shape_dict = (bs,) + (4096, 8)
                    
                    # Initialize：pose 和 shape 都从随机 noise start
                    if isinstance(latent_shape_dict, dict):
                        x_t = {
                            k: torch.randn(v, device=device)
                            for k, v in latent_shape_dict.items()
                        }
                        # Shape noise（随机生成）
                        shape_noise = x_t['shape'].clone()
                        # 目标 shape（Stage 1 result）
                        shape_final = fixed_shape_latent.clone().to(device)
                        # 预先计算固定的平均增量：xx_s = (shape_final - shape_noise) / n_steps
                        xx_s = (shape_final - shape_noise) / inference_steps
                    else:
                        x_t = torch.randn(latent_shape_dict, device=device)
                        shape_noise = None
                        shape_final = None
                        xx_s = None
                    
                    # 时间步序列（从 0 到 1，共 inference_steps + 1 个点）
                    t_seq = torch.linspace(0, 1, inference_steps + 1).to(device)
                    dt = 1.0 / inference_steps
                    
                    # Iterate（共 inference_steps 步）
                    for step_idx in range(inference_steps):
                        t = t_seq[step_idx]
                        t_scaled = t * ss_generator.time_scale
                        
                        # Get velocity（包含 pose 和 shape 的预测）
                        velocity = ss_generator.reverse_fn(
                            x_t,
                            t_scaled.unsqueeze(0) if t_scaled.dim() == 0 else t_scaled,
                            *condition_args,
                            **condition_kwargs
                        )
                        
                        # Update策略
                        if isinstance(velocity, dict):
                            for key in velocity.keys():
                                if key in POSE_KEYS:
                                    # Pose：正常迭代（使用网络预测的 velocity）
                                    x_t[key] = x_t[key] + velocity[key] * dt
                                elif key == 'shape':
                                    # Shape：混合更新策略（固定增量版本）
                                    # If是last一步，直接设置为目标 shape（确保收敛）
                                    if step_idx == inference_steps - 1:
                                        x_t[key] = shape_final
                                    else:
                                        # 网络预测的 velocity
                                        v_network = velocity[key]
                                        
                                        # 混合权重
                                        # 前期更信任网络，后期更倾向于目标
                                        progress = step_idx / (inference_steps - 1)  # [0, 1] (除了last一步)
                                        weight_target = progress
                                        weight_network = 1.0 - weight_target
                                        
                                        # 混合更新：网络预测 + 固定目标增量
                                        # Useoriginal的 xx_s（固定增量）
                                        shape_update = weight_network * v_network + weight_target * xx_s
                                        x_t[key] = x_t[key] + shape_update * dt
                        else:
                            x_t = x_t + velocity * dt
                    
                    # 验证：last一步should等于 shape_final（确保强制设置生效）
                    if isinstance(x_t, dict) and 'shape' in x_t:
                        diff = torch.abs(x_t['shape'] - shape_final).max().item()
                        logger.info(f"  View {view_idx} shape final check: max_diff = {diff:.6f} "
                                   f"(should be ~0, target shape reached)")
                    
                    # 提取 pose
                    view_pose_raw = {}
                    if isinstance(x_t, dict):
                        for key in x_t.keys():
                            if key in POSE_KEYS:
                                view_pose_raw[key] = x_t[key].detach().cpu()
                    
                    all_view_poses_raw.append(view_pose_raw)
                    
                    if 'scale' in view_pose_raw:
                        logger.info(f"  View {view_idx} raw scale: {view_pose_raw['scale'].flatten()[:3]}")
        
        ss_generator.inference_steps = prev_inference_steps
        logger.info(f"[Mixed Update] Completed for {num_views} views")
        
        return all_view_poses_raw
    
    def estimate_camera_poses_with_mode(
        self,
        view_ss_input_dicts: List[dict],
        mode: str = "fixed_shape",
        fixed_shape_latent: torch.Tensor = None,
        inference_steps: int = 50,
        **mode_kwargs,
    ) -> List[dict]:
        """
        统一的camera pose估计入口，支持多种模式。
        
        支持的模式：
        - "fixed_shape": 固定 Stage 1 的 shape，只优化 pose
        - "independent": each view完全独立优化 shape + pose
        - "manual_sync_time": 手动同步 shape 到same时间步（不迭代，直接计算）
        - "mixed_update": 混合更新策略（网络预测 + 目标引导）
        
        Args:
            view_ss_input_dicts: List of input dicts for each view
            mode: 模式名称
            fixed_shape_latent: Stage 1 的 shape latent（某些模式need）
            inference_steps: Number of iteration steps
            **mode_kwargs: 模式特定的parameter
            
        Returns:
            List of raw pose dicts for each view
        """
        logger.info("=" * 60)
        logger.info(f"Camera Pose Estimation Mode: {mode}")
        logger.info("=" * 60)
        
        if mode == "fixed_shape":
            if fixed_shape_latent is None:
                raise ValueError("fixed_shape mode requires fixed_shape_latent")
            return self.refine_pose_per_view(
                view_ss_input_dicts=view_ss_input_dicts,
                fixed_shape_latent=fixed_shape_latent,
                inference_steps=inference_steps,
            )
        
        elif mode == "independent":
            return self.estimate_pose_independent(
                view_ss_input_dicts=view_ss_input_dicts,
                inference_steps=inference_steps,
            )
        
        elif mode == "manual_sync_time":
            if fixed_shape_latent is None:
                raise ValueError("manual_sync_time mode requires fixed_shape_latent")
            return self.estimate_pose_manual_sync_time(
                view_ss_input_dicts=view_ss_input_dicts,
                fixed_shape_latent=fixed_shape_latent,
                inference_steps=inference_steps,
            )
        
        elif mode == "mixed_update":
            if fixed_shape_latent is None:
                raise ValueError("mixed_update mode requires fixed_shape_latent")
            return self.estimate_pose_mixed_update(
                view_ss_input_dicts=view_ss_input_dicts,
                fixed_shape_latent=fixed_shape_latent,
                inference_steps=inference_steps,
            )
        
        else:
            raise ValueError(
                f"Unknown camera_pose_mode: {mode}. "
                f"Supported modes: fixed_shape, independent, manual_sync_time, mixed_update"
            )
