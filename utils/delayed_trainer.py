
import os
import sys
import time
from typing import Dict, Any, Optional

def run_pinn_training_isolated(task_data: Dict[str, Any]) -> Dict[str, Any]:
    task_id = task_data.get('task_id', 'unknown')
    case_params = task_data.get('case_params', {})
    case_description = task_data.get('case_description', '')
    script_name = task_data.get('script_name', 'delayed_trainer')

    start_time = time.time()

    try:
        gpu_id = task_data.get('gpu_id', None)
        if gpu_id is not None:
            os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
            print(f"[Task {task_id}] Set GPU environment: CUDA_VISIBLE_DEVICES={gpu_id}")
        else:
            print(f"[Task {task_id}] Using CPU mode")

        import torch
        import numpy as np

        os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

        from modules.data_types import MaterialCoeffs, PhysicalParams
        from modules.bc import make_bc_spec
        from modules.pseudo_trainer import run_dual_pseudo_transfer
        from utils.material_properties import compute_material_params_for_solver
        from utils.training_core import run_training_core

        device_info = _get_device_info()
        print(f"[Task {task_id}] GPU setup verification: {device_info}")

        try:
            training_params = _build_training_params(case_params)

            if gpu_id is not None and torch.cuda.is_available():
                device = torch.device('cuda:0')
                torch.cuda.set_device(0)
                torch.cuda.empty_cache()
            else:
                device = torch.device('cpu')

            training_params['device'] = device

        except Exception as e:
            return {
                'task_id': task_id,
                'success': False,
                'error': f'Parameter construction failed: {str(e)}',
                'elapsed_time': time.time() - start_time,
                'case_description': case_description,
                'case_params': case_params,
                'gpu_info': device_info
            }

        if not validate_params_dict(training_params):
            return {
                'task_id': task_id,
                'success': False,
                'error': 'Parameter validation failed',
                'elapsed_time': time.time() - start_time,
                'case_description': case_description,
                'case_params': case_params,
                'gpu_info': device_info
            }

        print(f"[Task {task_id}] Starting PINNs training...")
        result = run_training_core(
            params_dict=training_params,
            script_name=script_name,
            verbose=False
        )

        if device.type == 'cuda':
            torch.cuda.empty_cache()
            print(f"[Task {task_id}] GPU memory cleanup complete")

        elapsed_time = time.time() - start_time
        print(f"[Task {task_id}] Training complete, elapsed: {elapsed_time:.2f}s")

        return {
            'task_id': task_id,
            'success': True,
            'result': result,
            'elapsed_time': elapsed_time,
            'case_description': case_description,
            'case_params': case_params,
            'gpu_info': device_info,
            'final_gpu_memory': _get_gpu_memory_info() if device.type == 'cuda' else None
        }

    except Exception as e:
        elapsed_time = time.time() - start_time
        error_msg = f"Training process exception: {str(e)}"
        print(f"[Task {task_id}] Error: {error_msg}")

        return {
            'task_id': task_id,
            'success': False,
            'error': error_msg,
            'elapsed_time': elapsed_time,
            'case_description': case_description,
            'case_params': case_params,
            'gpu_info': _get_device_info() if 'torch' in sys.modules else None
        }

def _get_device_info() -> Dict[str, Any]:
    import torch

    info = {
        'cuda_available': torch.cuda.is_available(),
        'cuda_device_count': torch.cuda.device_count() if torch.cuda.is_available() else 0,
        'current_device': None,
        'device_name': None,
        'visible_devices': os.environ.get('CUDA_VISIBLE_DEVICES', 'all')
    }

    if torch.cuda.is_available():
        try:
            info['current_device'] = torch.cuda.current_device()
            info['device_name'] = torch.cuda.get_device_name(info['current_device'])
        except:
            pass

    return info

def _get_gpu_memory_info() -> Dict[str, float]:
    import torch

    if not torch.cuda.is_available():
        return {}

    try:
        device_id = torch.cuda.current_device()
        return {
            'allocated_gb': torch.cuda.memory_allocated(device_id) / 1024**3,
            'reserved_gb': torch.cuda.memory_reserved(device_id) / 1024**3,
            'max_allocated_gb': torch.cuda.max_memory_allocated(device_id) / 1024**3
        }
    except:
        return {}

from .training_core import validate_params_dict

def _build_training_params(case_params: Dict[str, Any]) -> Dict[str, Any]:
    params = case_params.copy()

    if 'L_factor' in params and 'h' in params:
        params['L'] = params['L_factor'] * params['h']

    defaults = {
        'epochs': 100000,
        'lr': 1e-4,
        'N_train': 10000,
        'seed': 42,
        'optimizer_type': 'Adam',
        'lbfgs_max_iter': 20,
        'lbfgs_history_size': 50,
        'lbfgs_line_search_fn': None,
        'encoder_dims_shared': [1, 32, 64, 128],
        'head_dims': [128, 64, 32, 1],
        'input_dim': 1,
        'bc_weight': 1000.0,
        'sampler': 'uniform',
        'sampler_reuse': True,
        'integrator': 'agq',
        'agq_rule': 'G10K21',
        'agq_abs_tol': 1e-6,
        'agq_rel_tol': 1e-4,
        'agq_max_points': 4096,
        'agq_max_depth': 100,
        'agq_refine_every': 0,
        'agq_fail_policy': 'use_partial',
        'ps_w_non_start': 1.0,
        'ps_w_lin_start': 0.5,
        'ps_cut_ratio': 0.8,
        'ps_use_phi': False,
        'transfer_alpha': 0.3,
        'transfer_ratio': 0.7,
        'transfer_freq': 500,
        'transfer_cut_ratio': 0.2,
        'print_every': 1000,
        'save_best_model': True,
        'generate_plots': True,
        'verbose': False,
    }

    for key, value in defaults.items():
        params.setdefault(key, value)

    return params
