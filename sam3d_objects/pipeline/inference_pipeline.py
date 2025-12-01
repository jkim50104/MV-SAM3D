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
        将 mask 合并到 image 的 alpha 通道上（RGBA），
        单视角路径下与多视角的处理逻辑保持一致，避免因为 dtype/取值范围不同导致 mask 失效。
        """
        if isinstance(image, Image.Image):
            image = np.array(image)

        if mask is not None:
            mask = np.array(mask)
            # 与 multi-view 路径保持一致的 dtype 处理
            if mask.dtype == bool:
                mask = mask.astype(np.uint8) * 255
            elif mask.dtype != np.uint8:
                # 如果是 0-1 浮点或其他低范围值，先拉伸到 0-255 再转 uint8
                if mask.max() <= 1.0:
                    mask = (mask * 255).astype(np.uint8)
                else:
                    mask = mask.astype(np.uint8)

            if mask.ndim == 2:
                mask = mask[..., None]

            logger.info(f"Replacing alpha channel with the provided mask")
            assert mask.shape[:2] == image.shape[:2]

            # 支持 RGB 或已有 RGBA，两种情况都和 multi-view 保持一致
            if image.shape[-1] == 3:
                image = np.concatenate([image[..., :3], mask], axis=-1).astype(np.uint8)
            elif image.shape[-1] == 4:
                image = np.concatenate([image[..., :3], mask], axis=-1).astype(np.uint8)
            else:
                raise ValueError(f"Unexpected image shape: {image.shape}")

        # 保证最终是 numpy 数组
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
        为多视角输入准备条件
        
        Args:
            condition_embedder: 条件嵌入器
            view_input_dicts: 每个视角的输入字典列表
            input_mapping: 输入映射
            
        Returns:
            condition_args: 条件参数（包含所有视角的条件tokens）
            condition_kwargs: 条件关键字参数
        """
        # 对每个视角分别提取条件
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
                # 如果没有嵌入，使用原始参数
                view_conditions.append(condition_args)
        
        # 将所有视角的条件堆叠在一起
        # 形状: (num_views, batch_size, num_tokens, dim)
        if isinstance(view_conditions[0], torch.Tensor):
            # 如果是tensor，堆叠
            all_conditions = torch.stack(view_conditions, dim=0)
        else:
            # 如果是其他类型，保持为列表
            all_conditions = view_conditions
        
        return (all_conditions,), {}

    def sample_sparse_structure_multi_view(
        self, 
        view_ss_input_dicts: List[dict], 
        inference_steps=None, 
        use_distillation=False,
        mode: Literal['stochastic', 'multidiffusion'] = 'multidiffusion',
        attention_logger: Optional[Any] = None,
    ):
        """
        多视角稀疏结构生成
        
        Args:
            view_ss_input_dicts: 每个视角的输入字典列表
            inference_steps: 推理步数
            use_distillation: 是否使用蒸馏
            mode: 'stochastic' 或 'multidiffusion'
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

                # 准备多视角条件
                condition_args, condition_kwargs = self.get_multi_view_condition_input(
                    self.condition_embedders["ss_condition_embedder"],
                    view_ss_input_dicts,
                    self.ss_condition_input_mapping,
                )
                
                # 注入多视角支持，并保存所有视角的 pose
                with inject_generator_multi_view(
                    ss_generator, 
                    num_views=num_views, 
                    num_steps=ss_generator.inference_steps,
                    mode=mode,
                    attention_logger=attention_logger,
                    save_all_view_poses=True,  # 保存所有视角的 pose
                ) as all_view_poses_storage:
                    return_dict = ss_generator(
                        latent_shape_dict,
                        image.device,
                        *condition_args,
                        **condition_kwargs,
                    )
                
                if not self.is_mm_dit():
                    return_dict = {"shape": return_dict}
                
                # 获取每个视角独立迭代的最终 pose 状态
                # all_view_poses_storage['per_view_x_t'] 包含每个视角的完整状态
                # 这些是真正独立迭代出来的：
                # - shape: 所有视角共享（用平均 velocity 更新）
                # - pose: 每个视角独立（用自己的 velocity 更新）
                if all_view_poses_storage is not None and all_view_poses_storage.get('per_view_x_t'):
                    per_view_x_t = all_view_poses_storage['per_view_x_t']
                    
                    # 提取每个视角的 pose 部分
                    POSE_KEYS = {'translation', 'rotation', 'scale', 'translation_scale',
                                '6drotation', '6drotation_normalized', 'quaternion'}
                    
                    all_view_poses_raw = {}
                    for key in per_view_x_t[0].keys():
                        if key in POSE_KEYS:
                            # Stack all views' pose for this key
                            stacked = torch.stack([per_view_x_t[i][key] for i in range(num_views)])
                            all_view_poses_raw[key] = stacked.detach().cpu()
                    
                    return_dict["all_view_poses_raw"] = all_view_poses_raw
                    
                    logger.info(f"[Stage 1 Multi-view] Got independently iterated poses for {num_views} views")
                    logger.info(f"  Keys: {list(all_view_poses_raw.keys())}")
                    logger.info(f"  Steps iterated: {all_view_poses_storage.get('step_count', 'unknown')}")
                    
                    # 用 View 0 的独立迭代结果覆盖 return_dict 中的 pose
                    # 这样 final output 就是 View 0 真正独立迭代出来的 pose
                    logger.info(f"  [FIX] Replacing solver's pose with View 0's independently iterated pose")
                    for key in ['scale', 'translation', '6drotation_normalized', 'translation_scale']:
                        if key in per_view_x_t[0]:
                            old_value = return_dict[key].cpu() if key in return_dict else None
                            new_value = per_view_x_t[0][key]
                            return_dict[key] = new_value
                            if old_value is not None:
                                diff = (old_value - new_value.cpu()).abs().max().item()
                                logger.info(f"    {key}: replaced, max diff from solver = {diff:.6f}")

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
            view_slat_input_dicts: 每个视角的输入字典列表
            coords: 坐标（从Stage 1得到）
            inference_steps: 推理步数
            use_distillation: 是否使用蒸馏
            mode: 'stochastic' 或 'multidiffusion'
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
                # 准备多视角条件
                condition_args, condition_kwargs = self.get_multi_view_condition_input(
                    self.condition_embedders["slat_condition_embedder"],
                    view_slat_input_dicts,
                    self.slat_condition_input_mapping,
                )
                condition_args += (coords.cpu().numpy(),)
                
                # 注入多视角支持
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
        1. Warmup Pass: 跑一个 step 收集 attention，计算权重
        2. Main Pass: 用计算出的权重，从头开始完整迭代
        
        Args:
            view_slat_input_dicts: 每个视角的输入字典列表
            coords: 坐标（从Stage 1得到）
            inference_steps: 推理步数
            use_distillation: 是否使用蒸馏
            attention_logger: 注意力记录器
            weighting_config: 加权配置
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

        # 创建权重管理器
        if weighting_config is None:
            weighting_config = WeightingConfig()
        weight_manager = LatentWeightManager(weighting_config)

        with torch.autocast(device_type="cuda", dtype=self.dtype):
            with torch.no_grad():
                # 准备多视角条件
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
                
                # 创建一个临时的 attention 收集器
                attention_collector = AttentionCollector(
                    num_views=num_views,
                    target_layer=weighting_config.attention_layer,
                )
                
                # 用简单平均跑一步，同时收集 attention 到内存
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
                
                # 从内存中的 attention 计算权重
                collected_attentions = attention_collector.get_attentions()
                if collected_attentions:
                    for view_idx, attn in collected_attentions.items():
                        weight_manager.add_view_attention(view_idx, attn, step=0)
                    logger.info(f"[Stage 2 Weighted] Collected attention for {len(collected_attentions)} views")
                
                # 设置 downsample 映射（用于将权重从降采样维度扩展到原始维度）
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
                
                # 计算降采样维度的权重
                weights_downsampled = weight_manager.compute_weights()
                
                if weights_downsampled and len(weights_downsampled) == num_views:
                    logger.info(f"[Stage 2 Weighted] Downsampled weights computed for {num_views} views")
                    for v, w in weights_downsampled.items():
                        logger.info(f"  View {v} (downsampled): mean={w.mean():.4f}, std={w.std():.4f}")
                    
                    # 扩展权重到原始维度
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
                
                # 恢复完整步数
                slat_generator.inference_steps = original_steps
                
                # 重新开始 attention logger（如果需要保存完整的 attention）
                if attention_logger is not None:
                    attention_logger.start_stage("slat")
                    attention_logger.set_num_views(num_views)
                
                # 生成初始噪声（用于保存和迭代）
                initial_noise = slat_generator._generate_noise(latent_shape, DEVICE)
                
                # 保存 Stage 2 初始 latent（如果需要）
                if save_stage2_init and save_stage2_init_path is not None:
                    logger.info(f"[Stage 2 Weighted] Saving Stage 2 initial latent to {save_stage2_init_path}")
                    # 保存所有需要的信息
                    stage2_init_data = {
                        "coords": coords.cpu(),
                        "initial_noise": initial_noise.cpu() if torch.is_tensor(initial_noise) else initial_noise,
                        "latent_shape": latent_shape,
                        "num_views": num_views,
                        "inference_steps": original_steps,
                        # 保存条件编码（已经 embed 过的）
                        "condition_args": tuple(
                            arg.cpu() if torch.is_tensor(arg) else arg 
                            for arg in condition_args[:-1]  # 排除 coords numpy
                        ),
                        # 保存配置
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
                        precomputed_weights=weights_expanded,  # 使用扩展后的权重
                        attention_logger=attention_logger,
                    ):
                        slat = slat_generator(
                            latent_shape, DEVICE, *condition_args, **condition_kwargs
                        )
                finally:
                    # 恢复原始的 _generate_noise 方法
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
        weighting_config: Optional[Any] = None,  # 加权配置
        save_stage2_init: bool = False,  # 是否保存 Stage 2 初始 latent
        save_stage2_init_path: Optional[Any] = None,  # 保存路径
    ) -> dict:
        """
        多视角推理主函数
        
        Args:
            view_images: 每个视角的图像列表
            view_masks: 每个视角的掩码列表（可选）
            view_pointmaps: 每个视角的外部 pointmap 列表（可选）
                - 格式: (3, H, W) 的 numpy array，PyTorch3D 坐标系
                - 如果提供，将使用外部 pointmap 而不是内置深度模型计算
                - 可以来自 DA3 等外部深度估计模型
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
            mode: 'stochastic' 或 'multidiffusion'
        """
        num_views = len(view_images)
        if view_masks is None:
            view_masks = [None] * num_views
        assert len(view_masks) == num_views, "Number of masks must match number of images"
        
        # 处理外部 pointmap
        if view_pointmaps is not None:
            assert len(view_pointmaps) == num_views, "Number of pointmaps must match number of images"
            logger.info(f"Using external pointmaps for {sum(1 for p in view_pointmaps if p is not None)}/{num_views} views")
        else:
            view_pointmaps = [None] * num_views
        
        if seed is not None:
            torch.manual_seed(seed)
        
        logger.info(f"Running multi-view inference with {num_views} views, mode={mode}")
        
        # 预处理每个视角
        # 注意：需要先将mask合并到图像的alpha通道，然后调用preprocess_image
        view_ss_input_dicts = []
        view_slat_input_dicts = []
        raw_view_pointmaps: List[np.ndarray] = []
        for i, (image, mask, ext_pointmap) in enumerate(zip(view_images, view_masks, view_pointmaps)):
            logger.info(f"Preprocessing view {i+1}/{num_views}")
            
            # 将mask合并到图像的alpha通道（RGBA格式）
            # 如果image已经是RGBA格式（从mask的alpha通道加载），mask可能是None
            if mask is not None:
                # 确保image是numpy数组
                if isinstance(image, Image.Image):
                    image = np.array(image)
                else:
                    image = np.array(image)
                
                # 确保mask是numpy数组
                mask = np.array(mask)
                
                # 如果mask是bool类型，转换为uint8
                if mask.dtype == bool:
                    mask = mask.astype(np.uint8) * 255
                elif mask.dtype != np.uint8:
                    # 如果mask是0-1范围的float，转换为0-255
                    if mask.max() <= 1.0:
                        mask = (mask * 255).astype(np.uint8)
                    else:
                        mask = mask.astype(np.uint8)
                
                if mask.ndim == 2:
                    mask = mask[..., None]
                
                # 合并mask到alpha通道
                if image.shape[-1] == 3:  # RGB
                    rgba_image = np.concatenate([image, mask], axis=-1).astype(np.uint8)
                elif image.shape[-1] == 4:  # 已经是RGBA，替换alpha通道
                    rgba_image = np.concatenate([image[..., :3], mask], axis=-1).astype(np.uint8)
                else:
                    raise ValueError(f"Unexpected image shape: {image.shape}")
            else:
                # 如果没有mask，假设image已经是RGBA格式
                if isinstance(image, Image.Image):
                    rgba_image = np.array(image)
                else:
                    rgba_image = np.array(image)
            
            # 转换为PIL Image（preprocess_image需要）
            rgba_image_pil = Image.fromarray(rgba_image)
            
            # 调用preprocess_image（注意：InferencePipelinePointMap需要pointmap）
            # 先检查是否是InferencePipelinePointMap
            if hasattr(self, 'compute_pointmap'):
                # 这是InferencePipelinePointMap，需要计算或使用外部pointmap
                if ext_pointmap is not None:
                    # 使用外部提供的 pointmap（来自 DA3 等）
                    # ext_pointmap 格式: (3, H, W) numpy array
                    # 
                    # 重要：需要调用 compute_pointmap 来应用坐标变换！
                    # compute_pointmap 会：
                    #   1. 对 DA3 pointmap 翻转 Y 和 Z（使其与 MoGe 原始输出一致）
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
                
                # 保存真实尺度的 pointmap（用于可视化对齐）
                # 注意：此时 pointmap 已经在 PyTorch3D 空间，无论是 MoGe 还是 DA3
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
                # 这是InferencePipeline，不需要pointmap
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
            # 使用加权融合
            logger.info("Using weighted multi-view fusion")
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
            # 使用原始的简单平均融合
            slat = self.sample_slat_multi_view(
                view_slat_input_dicts,
                coords,
                inference_steps=stage2_inference_steps,
                use_distillation=use_stage2_distillation,
                mode=mode,
                attention_logger=attention_logger,
            )
        
        # 解码
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
            "view_ss_input_dicts": view_ss_input_dicts,  # 保存以便 overlay 使用
        }
        
        if raw_view_pointmaps:
            result["raw_view_pointmaps"] = raw_view_pointmaps
        
        # 如果使用了加权融合，添加权重信息
        if weight_manager is not None:
            result["weight_manager"] = weight_manager
        
        return result
