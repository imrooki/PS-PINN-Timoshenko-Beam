
import gc
import traceback
from typing import Optional, Dict, Any, Callable
import warnings
import torch
import torch.nn as nn
from contextlib import contextmanager

class CUDAMemoryGuard:

    def __init__(self, verbose: bool = False):
        self.verbose = verbose
        self._original_cuda_state = None
        self._memory_snapshots = []

    def take_snapshot(self, label: str = ""):
        if torch.cuda.is_available():
            snapshot = {
                'label': label,
                'allocated': torch.cuda.memory_allocated() / 1e9,
                'reserved': torch.cuda.memory_reserved() / 1e9,
                'max_allocated': torch.cuda.max_memory_allocated() / 1e9
            }
            self._memory_snapshots.append(snapshot)
            if self.verbose:
                print(f"[Memory] {label}: Allocated={snapshot['allocated']:.2f}GB, Reserved={snapshot['reserved']:.2f}GB")
            return snapshot
        return None

    @staticmethod
    def force_cleanup(models: list = None, optimizers: list = None):
        try:
            if models:
                for model in models:
                    if model is not None:
                        for param in model.parameters():
                            if param.grad is not None:
                                param.grad = None
                        try:
                            model.cpu()
                        except:
                            pass
                        del model

            if optimizers:
                for optimizer in optimizers:
                    if optimizer is not None:
                        optimizer.zero_grad(set_to_none=True)
                        if hasattr(optimizer, 'state'):
                            optimizer.state.clear()
                        del optimizer

            gc.collect()

            if torch.cuda.is_available():
                torch.cuda.synchronize()
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats()
                torch.cuda.empty_cache()

        except Exception as e:
            if str(e).find("illegal memory access") != -1:
                CUDAMemoryGuard.reset_cuda_device()
            else:
                print(f"[WARNING] Cleanup error: {e}")

    @staticmethod
    def reset_cuda_device():
        if torch.cuda.is_available():
            try:
                device_id = torch.cuda.current_device()
                torch.cuda.set_device(device_id)
                with torch.cuda.device(device_id):
                    torch.cuda.synchronize()
                    torch.cuda.empty_cache()
                print("[INFO] CUDA device reset successful")
            except Exception as e:
                print(f"[ERROR] CUDA device reset failed: {e}")
                print("[CRITICAL] Recommend restarting the process due to GPU memory corruption")

class AutogradMemoryManager:

    @staticmethod
    def safe_backward(loss: torch.Tensor, retain_graph: bool = False):
        if loss.requires_grad:
            if loss.numel() != 1:
                loss = loss.mean()

            try:
                loss.backward(retain_graph=retain_graph)
            finally:
                if not retain_graph:
                    loss = loss.detach()

    @staticmethod
    @contextmanager
    def managed_autograd(create_graph: bool = True, retain_graph: bool = False):
        try:
            initial_graphs = []
            yield
        finally:
            if not retain_graph:
                torch.cuda.synchronize() if torch.cuda.is_available() else None
                gc.collect()

    @staticmethod
    def compute_derivatives_safe(func: Callable, x: torch.Tensor,
                                order: int = 1, create_graph: bool = True) -> torch.Tensor:
        x = x.requires_grad_(True)
        y = func(x)

        grad1 = torch.autograd.grad(
            y, x,
            grad_outputs=torch.ones_like(y),
            create_graph=create_graph,
            retain_graph=False,
            only_inputs=True
        )[0]

        if order == 1:
            return grad1.detach() if not create_graph else grad1

        if order == 2:
            grad2 = torch.autograd.grad(
                grad1, x,
                grad_outputs=torch.ones_like(grad1),
                create_graph=False,
                retain_graph=False,
                only_inputs=True
            )[0]
            return grad2.detach()

        return grad1

class EnergyPINNMemoryWrapper:

    def __init__(self, pinn_model):
        self.model = pinn_model
        self.memory_guard = CUDAMemoryGuard()

    def compute_energy_safe(self, x_samples: torch.Tensor,
                          cleanup_intermediate: bool = True) -> Dict[str, torch.Tensor]:

        batch_size = min(1000, len(x_samples))
        total_energy = 0
        energy_components = {}

        for i in range(0, len(x_samples), batch_size):
            x_batch = x_samples[i:i+batch_size]

            with torch.no_grad() if cleanup_intermediate else torch.enable_grad():
                batch_energy = self._compute_batch_energy(x_batch)

            for key, value in batch_energy.items():
                if key not in energy_components:
                    energy_components[key] = 0
                energy_components[key] += value.detach() if cleanup_intermediate else value

            if cleanup_intermediate and i % (batch_size * 10) == 0:
                torch.cuda.synchronize() if torch.cuda.is_available() else None
                torch.cuda.empty_cache() if torch.cuda.is_available() else None

        return energy_components

    def _compute_batch_energy(self, x_batch):
        return {'Pi_str': torch.tensor(0.0), 'Pi_e': torch.tensor(0.0)}

def safe_training_iteration(train_func: Callable,
                           cleanup_every: int = 100,
                           force_cleanup_on_error: bool = True) -> Callable:
    def wrapper(*args, **kwargs):
        iteration = kwargs.get('epoch', 0)

        try:
            result = train_func(*args, **kwargs)

            if iteration % cleanup_every == 0:
                CUDAMemoryGuard.force_cleanup()

            return result

        except RuntimeError as e:
            if "out of memory" in str(e) or "illegal memory access" in str(e):
                print(f"[ERROR] GPU memory error at iteration {iteration}: {e}")

                if force_cleanup_on_error:
                    print("[INFO] Attempting memory recovery...")
                    CUDAMemoryGuard.force_cleanup()
                    CUDAMemoryGuard.reset_cuda_device()

                    if 'N_samples' in kwargs:
                        kwargs['N_samples'] = max(100, kwargs['N_samples'] // 2)
                        print(f"[INFO] Retrying with reduced samples: {kwargs['N_samples']}")
                        return train_func(*args, **kwargs)

                raise
            else:
                raise

    return wrapper

class SensitivityAnalysisMemoryFix:

    @staticmethod
    def run_case_with_cleanup(run_func: Callable, case_params: Dict,
                             verbose: bool = False) -> Dict:

        memory_guard = CUDAMemoryGuard(verbose=verbose)
        memory_guard.take_snapshot("Before case")

        try:
            result = run_func(case_params)

            clean_result = SensitivityAnalysisMemoryFix._extract_clean_results(result)

        except Exception as e:
            print(f"[ERROR] Case failed: {e}")
            clean_result = {'success': False, 'error': str(e)}

        finally:
            SensitivityAnalysisMemoryFix._complete_cleanup(result if 'result' in locals() else None)
            memory_guard.take_snapshot("After cleanup")

        return clean_result

    @staticmethod
    def _extract_clean_results(result: Dict) -> Dict:
        clean = {'success': result.get('success', False)}

        if 'results' in result:
            clean['results'] = {}
            for key, value in result['results'].items():
                if isinstance(value, tuple):
                    clean['results'][key] = tuple(
                        v.copy() if hasattr(v, 'copy') else v
                        for v in value
                    )
                elif torch.is_tensor(value):
                    clean['results'][key] = value.detach().cpu().numpy()

        if 'summary' in result:
            clean['summary'] = result['summary']

        return clean

    @staticmethod
    def _complete_cleanup(result: Optional[Dict]):

        if result:
            models_to_clean = []
            optimizers_to_clean = []

            for key in ['model_linear', 'model_nonlinear', 'model', 'lin_model', 'non_model']:
                if key in result and result[key] is not None:
                    models_to_clean.append(result[key])

            for key in ['optimizer', 'optimizer_lin', 'optimizer_non']:
                if key in result and result[key] is not None:
                    optimizers_to_clean.append(result[key])

            CUDAMemoryGuard.force_cleanup(models_to_clean, optimizers_to_clean)

            result.clear()

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.empty_cache()

def get_gpu_memory_info() -> Dict[str, float]:
    if not torch.cuda.is_available():
        return {}

    return {
        'allocated_gb': torch.cuda.memory_allocated() / 1e9,
        'reserved_gb': torch.cuda.memory_reserved() / 1e9,
        'free_gb': (torch.cuda.get_device_properties(0).total_memory -
                   torch.cuda.memory_reserved()) / 1e9
    }

def check_memory_health() -> bool:
    if not torch.cuda.is_available():
        return True

    info = get_gpu_memory_info()
    return info.get('free_gb', 0) > 1.0

def generate_memory_checklist() -> list:
    return [
        "✓ 使用 with torch.no_grad() 包装不需要梯度的计算",
        "✓ 定期调用 torch.cuda.empty_cache()",
        "✓ 在保存模型后立即删除模型引用",
        "✓ 使用 .detach() 断开不需要的计算图连接",
        "✓ LBFGS优化器使用后清理 optimizer.state",
        "✓ 批处理大型积分计算，避免一次性计算",
        "✓ 使用 create_graph=False 计算最终导数",
        "✓ 每个训练案例后完全清理GPU内存",
        "✓ 监控GPU内存使用，设置预警阈值",
        "✓ 实现错误恢复机制，处理CUDA错误"
    ]

if __name__ == "__main__":
    print("Energy-based PINN CUDA Memory Management Fix Module")
    print("=" * 50)
    print("\nBest Practices Checklist:")
    for item in generate_memory_checklist():
        print(item)
    print("\nGPU Memory Status:")
    print(get_gpu_memory_info())
