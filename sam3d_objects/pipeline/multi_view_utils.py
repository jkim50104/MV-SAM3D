# Copyright (c) Meta Platforms, Inc. and affiliates.
"""
Multi-view multidiffusion utilities for SAM 3D Objects
Adapted from TRELLIS implementation, adapted for SAM 3D Objects' two-stage structure
"""
from contextlib import contextmanager
from typing import List, Literal, Optional
import torch
from loguru import logger

# Pose 相关的 key，这些不应该被平均
POSE_KEYS = {
    'translation', 'rotation', 'scale', 'translation_scale',
    '6drotation', '6drotation_normalized',
    'quaternion',
}


@contextmanager
def inject_generator_multi_view(
    generator,
    num_views: int,
    num_steps: int,
    mode: Literal['stochastic', 'multidiffusion'] = 'multidiffusion',
    attention_logger=None,
    save_all_view_poses: bool = False,  # 是否保存所有视角的 pose
):
    """
    Inject multi-view support into generator
    
    Args:
        generator: SAM 3D Objects generator (ss_generator or slat_generator)
        num_views: Number of views
        num_steps: Number of inference steps
        mode: 'stochastic' or 'multidiffusion'
        save_all_view_poses: If True, save all view poses (each view iterates independently)
    
    Yields:
        dict with 'all_view_poses' if save_all_view_poses is True, else None
        
    Multi-view Pose Iteration Strategy (CORRECT VERSION):
    -----------------------------------------------------
    
    每个视角维护自己完整的 x_t 状态：
    - x_t_vi.shape: 共享的 shape latent（所有视角同步）
    - x_t_vi.pose: 该视角独立迭代的 pose latent
    
    每一步：
    1. 对于每个视角 i，用 x_t_vi 和 condition[i] 预测 velocity_i
    2. Shape: 计算平均 velocity，所有视角同步更新
    3. Pose: 每个视角用自己的 velocity 更新自己的状态
    
    这样每个视角的 pose 预测都是基于自己之前的 pose 状态，而不是 View 0 的状态。
    """
    # 存储每个视角的完整状态
    all_view_states_storage = {
        'per_view_x_t': None,  # List of x_t for each view
        'step_count': 0,
    } if save_all_view_poses else None
    
    original_dynamics = generator._generate_dynamics
    
    if mode == 'stochastic':
        if num_views > num_steps:
            logger.warning(
                f"Warning: number of views ({num_views}) is greater than number of steps ({num_steps}). "
                "This may lead to performance degradation."
            )
        
        cond_indices = (torch.arange(num_steps) % num_views).tolist()
        cond_idx_counter = [0]
        
        def _new_dynamics_stochastic(x_t, t, *args_conditionals, **kwargs_conditionals):
            """Stochastic mode: select one view per time step"""
            cond_idx = cond_indices[cond_idx_counter[0] % len(cond_indices)]
            cond_idx_counter[0] += 1
            
            if len(args_conditionals) > 0:
                cond_tokens = args_conditionals[0]
                if isinstance(cond_tokens, (list, tuple)):
                    cond_i = cond_tokens[cond_idx:cond_idx+1] if isinstance(cond_tokens[0], torch.Tensor) else [cond_tokens[cond_idx]]
                    new_args = (cond_i,) + args_conditionals[1:]
                elif isinstance(cond_tokens, torch.Tensor) and cond_tokens.shape[0] == num_views:
                    cond_i = cond_tokens[cond_idx:cond_idx+1]
                    new_args = (cond_i,) + args_conditionals[1:]
                else:
                    new_args = args_conditionals
            else:
                new_args = args_conditionals
            
            if attention_logger is not None:
                attention_logger.set_view(cond_idx)
            return original_dynamics(x_t, t, *new_args, **kwargs_conditionals)
        
        generator._generate_dynamics = _new_dynamics_stochastic
        
    elif mode == 'multidiffusion':
        # 计算 dt（每一步的时间差）
        dt = 1.0 / num_steps
        
        def _new_dynamics_multidiffusion(x_t, t, *args_conditionals, **kwargs_conditionals):
            """
            Multidiffusion mode with per-view pose iteration.
            
            关键改进：每个视角的 pose 使用自己的状态进行预测和更新。
            """
            nonlocal all_view_states_storage
            
            cond_idx = 0
            if len(args_conditionals) > 0:
                if isinstance(args_conditionals[0], (int, float)) or (isinstance(args_conditionals[0], torch.Tensor) and args_conditionals[0].numel() == 1):
                    cond_idx = 1
            
            if len(args_conditionals) > cond_idx:
                cond_tokens = args_conditionals[cond_idx]
                
                if not hasattr(_new_dynamics_multidiffusion, '_logged_cond_shape'):
                    logger.info(f"[Multidiffusion] args_conditionals length: {len(args_conditionals)}")
                    logger.info(f"[Multidiffusion] cond_idx: {cond_idx}")
                    if isinstance(cond_tokens, torch.Tensor):
                        logger.info(f"[Multidiffusion] Condition tokens shape: {cond_tokens.shape}")
                    elif isinstance(cond_tokens, (list, tuple)):
                        logger.info(f"[Multidiffusion] Condition tokens type: {type(cond_tokens)}, length: {len(cond_tokens)}")
                        if len(cond_tokens) > 0 and isinstance(cond_tokens[0], torch.Tensor):
                            logger.info(f"[Multidiffusion] First condition token shape: {cond_tokens[0].shape}")
                    else:
                        logger.info(f"[Multidiffusion] Condition tokens type: {type(cond_tokens)}")
                    _new_dynamics_multidiffusion._logged_cond_shape = True
                
                if isinstance(cond_tokens, (list, tuple)):
                    view_conditions = cond_tokens
                elif isinstance(cond_tokens, torch.Tensor) and cond_tokens.shape[0] == num_views:
                    view_conditions = [cond_tokens[i] for i in range(num_views)]
                else:
                    logger.warning(f"Condition tokens shape {cond_tokens.shape if isinstance(cond_tokens, torch.Tensor) else type(cond_tokens)} not organized by views, using same condition for all views")
                    view_conditions = [cond_tokens] * num_views
                
                # ========================================
                # 核心逻辑：每个视角使用自己的状态进行预测
                # ========================================
                
                if save_all_view_poses and all_view_states_storage is not None:
                    step = all_view_states_storage['step_count']
                    
                    # 第一步：初始化每个视角的状态（都从相同的 x_t 开始）
                    if all_view_states_storage['per_view_x_t'] is None:
                        all_view_states_storage['per_view_x_t'] = []
                        for i in range(num_views):
                            # 深拷贝 x_t 作为每个视角的初始状态
                            if isinstance(x_t, dict):
                                view_x_t = {k: v.clone() for k, v in x_t.items()}
                            else:
                                view_x_t = x_t.clone()
                            all_view_states_storage['per_view_x_t'].append(view_x_t)
                        logger.info(f"[Multidiffusion] Initialized per-view states for {num_views} views")
                    
                    # 用每个视角自己的状态进行预测
                    preds = []
                    for view_idx in range(num_views):
                        view_cond = view_conditions[view_idx]
                        view_x_t = all_view_states_storage['per_view_x_t'][view_idx]
                        
                        if cond_idx < len(args_conditionals):
                            new_args = args_conditionals[:cond_idx] + (view_cond,) + args_conditionals[cond_idx+1:]
                        else:
                            new_args = args_conditionals + (view_cond,)
                        
                        if attention_logger is not None:
                            attention_logger.set_view(view_idx)
                        
                        # 使用该视角自己的状态进行预测
                        pred = original_dynamics(view_x_t, t, *new_args, **kwargs_conditionals)
                        preds.append(pred)
                    
                    # 更新状态
                    if isinstance(preds[0], dict):
                        # 1. 先更新 View 0 的 shape（用平均 velocity）
                        view0_x_t = all_view_states_storage['per_view_x_t'][0]
                        for key in preds[0].keys():
                            if key not in POSE_KEYS:  # shape 相关
                                stacked = torch.stack([p[key] for p in preds])
                                avg_velocity = stacked.mean(dim=0)
                                view0_x_t[key] = view0_x_t[key] + avg_velocity * dt
                        
                        # 2. 同步 shape 到所有其他视角（共享同一个 shape）
                        for view_idx in range(1, num_views):
                            view_x_t = all_view_states_storage['per_view_x_t'][view_idx]
                            for key in preds[0].keys():
                                if key not in POSE_KEYS:
                                    view_x_t[key] = view0_x_t[key]  # 直接引用 View 0 的 shape
                        
                        # 3. 每个视角独立更新自己的 pose
                        for view_idx in range(num_views):
                            view_x_t = all_view_states_storage['per_view_x_t'][view_idx]
                            for key in preds[view_idx].keys():
                                if key in POSE_KEYS:
                                    view_x_t[key] = view_x_t[key] + preds[view_idx][key] * dt
                    
                    all_view_states_storage['step_count'] += 1
                    
                    # 返回 fused velocity 给 solver（保持兼容性）
                    # solver 会用这个更新它维护的 x_t，但我们已经在上面更新了自己的状态
                    if isinstance(preds[0], dict):
                        fused_pred = {}
                        for key in preds[0].keys():
                            if key in POSE_KEYS:
                                # Pose: 返回 View 0 的 velocity
                                fused_pred[key] = preds[0][key]
                            else:
                                # Shape: 返回平均 velocity
                                stacked = torch.stack([p[key] for p in preds])
                                fused_pred[key] = stacked.mean(dim=0)
                        return fused_pred
                    else:
                        return preds[0]
                
                else:
                    # 不保存每个视角的 pose，使用原来的简单逻辑
                    preds = []
                    for view_idx in range(num_views):
                        view_cond = view_conditions[view_idx]
                        if cond_idx < len(args_conditionals):
                            new_args = args_conditionals[:cond_idx] + (view_cond,) + args_conditionals[cond_idx+1:]
                        else:
                            new_args = args_conditionals + (view_cond,)
                        if attention_logger is not None:
                            attention_logger.set_view(view_idx)
                        pred = original_dynamics(x_t, t, *new_args, **kwargs_conditionals)
                        preds.append(pred)
                    
                    if not hasattr(_new_dynamics_multidiffusion, '_logged_shape'):
                        if isinstance(x_t, dict):
                            logger.info(f"[Multidiffusion] Latent shape (dict): {[(k, v.shape if isinstance(v, torch.Tensor) else type(v)) for k, v in x_t.items()]}")
                        elif isinstance(x_t, (list, tuple)):
                            logger.info(f"[Multidiffusion] Latent shape (tuple/list): {[v.shape if isinstance(v, torch.Tensor) else type(v) for v in x_t]}")
                        else:
                            logger.info(f"[Multidiffusion] Latent shape: {x_t.shape if isinstance(x_t, torch.Tensor) else type(x_t)}")
                        
                        if isinstance(preds[0], dict):
                            logger.info(f"[Multidiffusion] Pred shape (dict): {[(k, v.shape if isinstance(v, torch.Tensor) else type(v)) for k, v in preds[0].items()]}")
                        elif isinstance(preds[0], (list, tuple)):
                            logger.info(f"[Multidiffusion] Pred shape (tuple/list): {[v.shape if isinstance(v, torch.Tensor) else type(v) for v in preds[0]]}")
                        else:
                            logger.info(f"[Multidiffusion] Pred shape: {preds[0].shape if isinstance(preds[0], torch.Tensor) else type(preds[0])}")
                        logger.info(f"[Multidiffusion] Number of views: {num_views}, fusing {len(preds)} predictions")
                        _new_dynamics_multidiffusion._logged_shape = True
                    
                    if isinstance(preds[0], dict):
                        fused_pred = {}
                        for key in preds[0].keys():
                            stacked = torch.stack([p[key] for p in preds])
                            if key in POSE_KEYS:
                                fused_pred[key] = preds[0][key]
                            else:
                                fused_pred[key] = stacked.mean(dim=0)
                        return fused_pred
                    elif isinstance(preds[0], (list, tuple)):
                        fused_pred = tuple(
                            torch.stack([p[i] for p in preds]).mean(dim=0)
                            for i in range(len(preds[0]))
                        )
                        return fused_pred
                    else:
                        fused_pred = torch.stack(preds).mean(dim=0)
                        return fused_pred
            else:
                return original_dynamics(x_t, t, *args_conditionals, **kwargs_conditionals)
        
        generator._generate_dynamics = _new_dynamics_multidiffusion
        
    else:
        raise ValueError(f"Unsupported mode: {mode}")
    
    try:
        yield all_view_states_storage
    finally:
        generator._generate_dynamics = original_dynamics
