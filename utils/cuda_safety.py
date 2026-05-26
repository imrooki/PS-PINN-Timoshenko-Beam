
import torch
from contextlib import contextmanager
from typing import Optional, Dict, Any, Union

@contextmanager
def safe_autograd_context(enable_grad: bool = True,
                          cleanup_on_exit: bool = True):
    if enable_grad:
        ctx = torch.enable_grad()
    else:
        ctx = torch.no_grad()

    try:
        with ctx:
            yield
    finally:
        if cleanup_on_exit and torch.cuda.is_available():
            try:
                torch.cuda.synchronize()
                torch.cuda.empty_cache()
            except RuntimeError as e:
                print(f"[WARNING] CUDA cleanup failed in autograd context: {e}")

def safe_tensor_delete(*tensors):
    for tensor in tensors:
        if tensor is None:
            continue

        try:
            if isinstance(tensor, dict):
                for k in list(tensor.keys()):
                    item = tensor[k]
                    if hasattr(item, 'grad') and item.grad is not None:
                        item.grad = None
                    tensor[k] = None
                tensor.clear()

            elif isinstance(tensor, (list, tuple)):
                for item in tensor:
                    if hasattr(item, 'grad') and item.grad is not None:
                        item.grad = None

            elif hasattr(tensor, 'grad') and tensor.grad is not None:
                tensor.grad = None

        except Exception as e:
            print(f"[WARNING] Failed to clear tensor reference: {e}")

        try:
            del tensor
        except:
            pass

    import gc
    for _ in range(2):
        gc.collect()

    if torch.cuda.is_available():
        try:
            torch.cuda.empty_cache()
        except RuntimeError as e:
            print(f"[WARNING] CUDA cache cleanup failed: {e}")

def check_cuda_memory(device: torch.device,
                     threshold_mb: float = 100.0,
                     cleanup: bool = True) -> Dict[str, Any]:
    if device.type != "cuda":
        return {
            "available": True,
            "message": "CPU mode - no memory constraints"
        }

    try:
        allocated = torch.cuda.memory_allocated(device) / 1024**2
        reserved = torch.cuda.memory_reserved(device) / 1024**2
        max_allocated = torch.cuda.max_memory_allocated(device) / 1024**2

        props = torch.cuda.get_device_properties(device)
        total = props.total_memory / 1024**2
        available = total - allocated

        status = {
            "available": available > threshold_mb,
            "allocated_mb": allocated,
            "reserved_mb": reserved,
            "max_allocated_mb": max_allocated,
            "total_mb": total,
            "available_mb": available,
            "message": f"Available: {available:.1f}MB / Total: {total:.1f}MB"
        }

        if not status["available"] and cleanup:
            try:
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
                import gc
                gc.collect()

                allocated_after = torch.cuda.memory_allocated(device) / 1024**2
                available_after = total - allocated_after
                status["available"] = available_after > threshold_mb
                status["available_mb"] = available_after
                status["allocated_mb"] = allocated_after
                status["message"] += f" (After cleanup: {available_after:.1f}MB)"

            except RuntimeError as e:
                status["message"] += f" (Cleanup failed: {e})"

        return status

    except Exception as e:
        return {
            "available": False,
            "message": f"CUDA memory check failed: {e}"
        }

def cuda_health_check(device: torch.device,
                     reset_on_error: bool = True) -> bool:
    if device.type != "cuda":
        return True

    try:
        test_tensor = torch.zeros(10, device=device)
        result = test_tensor + 1
        _ = result.sum().item()

        del test_tensor, result
        torch.cuda.synchronize()

        return True

    except RuntimeError as e:
        error_msg = str(e).lower()

        if "illegal memory access" in error_msg:
            print(f"[ERROR] CUDA illegal memory access detected: {e}")

            if reset_on_error:
                print("[INFO] Attempting to reset CUDA state...")
                try:
                    torch.cuda.empty_cache()
                    torch.cuda.synchronize()
                    torch.cuda.reset_peak_memory_stats(device)
                    torch.cuda.reset_accumulated_memory_stats(device)

                    import gc
                    for _ in range(3):
                        gc.collect()

                    test_tensor = torch.zeros(10, device=device)
                    result = test_tensor + 1
                    _ = result.sum().item()
                    del test_tensor, result
                    torch.cuda.synchronize()

                    print("[OK] CUDA state recovered")
                    return True

                except Exception as reset_error:
                    print(f"[ERROR] CUDA reset failed: {reset_error}")
                    return False

            return False

        elif "cuda" in error_msg or "gpu" in error_msg:
            print(f"[ERROR] CUDA runtime error: {e}")
            return False

        else:
            print(f"[WARNING] Health check encountered unexpected error: {e}")
            return True

    except Exception as e:
        print(f"[WARNING] Health check exception: {e}")
        return True

def reset_cuda_device(device: torch.device) -> bool:
    if device.type != "cuda":
        return True

    try:
        print(f"[WARNING] Resetting CUDA device: {device}")

        torch.cuda.empty_cache()

        torch.cuda.synchronize()

        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.reset_accumulated_memory_stats(device)

        import gc
        for _ in range(5):
            gc.collect()

        torch.cuda.empty_cache()
        torch.cuda.synchronize()

        test = torch.tensor([1.0], device=device)
        _ = (test + 1).item()
        del test

        print(f"[OK] CUDA device reset successful")
        return True

    except Exception as e:
        print(f"[ERROR] CUDA device reset failed: {e}")
        return False

def get_cuda_memory_summary(device: Optional[torch.device] = None) -> str:
    if not torch.cuda.is_available():
        return "[INFO] CUDA not available"

    try:
        if device is None:
            device = torch.device("cuda")

        allocated = torch.cuda.memory_allocated(device) / 1024**3
        reserved = torch.cuda.memory_reserved(device) / 1024**3
        max_allocated = torch.cuda.max_memory_allocated(device) / 1024**3
        max_reserved = torch.cuda.max_memory_reserved(device) / 1024**3

        props = torch.cuda.get_device_properties(device)
        total = props.total_memory / 1024**3

        summary = f"""
CUDA Memory Summary ({torch.cuda.get_device_name(device)}):
  Current Allocated: {allocated:.3f} GB / {total:.3f} GB ({allocated/total*100:.1f}%)
  Current Reserved: {reserved:.3f} GB / {total:.3f} GB ({reserved/total*100:.1f}%)
  Peak Allocated: {max_allocated:.3f} GB
  Peak Reserved: {max_reserved:.3f} GB
  Available: {total-allocated:.3f} GB
"""
        return summary.strip()

    except Exception as e:
        return f"[ERROR] Unable to get memory summary: {e}"

__all__ = [
    'safe_autograd_context',
    'safe_tensor_delete',
    'check_cuda_memory',
    'cuda_health_check',
    'reset_cuda_device',
    'get_cuda_memory_summary',
]
