
import warnings
from typing import Union, Optional
import torch
import numpy as np

class NumericalSafetyError(Exception):
    pass

class NaNDetectedError(NumericalSafetyError):
    pass

class InfDetectedError(NumericalSafetyError):
    pass

class DivisionByZeroError(NumericalSafetyError):
    pass

class NumericalSafety:
    
    @staticmethod
    def safe_divide(numerator: torch.Tensor, 
                   denominator: Union[torch.Tensor, float], 
                   eps: float = 1e-10,
                   method: str = 'add_eps') -> torch.Tensor:
        if isinstance(denominator, (int, float)):
            if abs(denominator) < eps:
                if denominator == 0:
                    raise DivisionByZeroError(f"Division by exact zero: {denominator}")
                warnings.warn(f"Small denominator detected: {denominator}, using eps protection")
            denominator = torch.tensor(denominator, dtype=numerator.dtype, device=numerator.device)
        
        if method == 'add_eps':
            safe_denom = denominator + eps * torch.sign(denominator)
            zero_mask = torch.abs(denominator) < eps
            safe_denom = torch.where(zero_mask, eps, safe_denom)
            
        elif method == 'clamp':
            safe_denom = torch.clamp(torch.abs(denominator), min=eps) * torch.sign(denominator)
            safe_denom = torch.where(denominator == 0, eps, safe_denom)
            
        elif method == 'sign_preserving':
            abs_denom = torch.abs(denominator)
            sign_denom = torch.sign(denominator)
            safe_abs_denom = torch.maximum(abs_denom, torch.tensor(eps, device=denominator.device))
            safe_denom = safe_abs_denom * sign_denom
            
        else:
            raise ValueError(f"Unknown safe division method: {method}")
        
        result = numerator / safe_denom
        
        NumericalSafety.check_tensor_validity(result, f"safe_divide_result")
        
        return result
    
    @staticmethod
    def check_tensor_validity(tensor: torch.Tensor, 
                            name: str = "tensor",
                            raise_on_nan: bool = True,
                            raise_on_inf: bool = True,
                            max_abs_value: float = 1e10) -> bool:
        nan_count = torch.isnan(tensor).sum().item()
        if nan_count > 0:
            message = f"{name} contains {nan_count} NaN values out of {tensor.numel()} elements"
            if raise_on_nan:
                raise NaNDetectedError(message)
            else:
                warnings.warn(message)
                return False
        
        inf_count = torch.isinf(tensor).sum().item()
        if inf_count > 0:
            message = f"{name} contains {inf_count} Inf values out of {tensor.numel()} elements"
            if raise_on_inf:
                raise InfDetectedError(message)
            else:
                warnings.warn(message)
                return False
        
        max_val = torch.abs(tensor).max().item()
        if max_val > max_abs_value:
            message = f"{name} contains very large values (max: {max_val:.2e}), potential overflow risk"
            warnings.warn(message)
        
        return True
    
    @staticmethod
    def safe_sqrt(x: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
        safe_x = torch.clamp(x, min=eps)
        result = torch.sqrt(safe_x)
        NumericalSafety.check_tensor_validity(result, "safe_sqrt_result")
        return result
    
    @staticmethod
    def safe_log(x: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
        safe_x = torch.clamp(x, min=eps)
        result = torch.log(safe_x)
        NumericalSafety.check_tensor_validity(result, "safe_log_result")
        return result
    
    @staticmethod
    def check_gradient_health(model: torch.nn.Module, 
                            max_grad_norm: float = 10.0,
                            name: str = "model") -> dict:
        stats = {
            'total_params': 0,
            'params_with_grad': 0,
            'nan_grad_count': 0,
            'inf_grad_count': 0,
            'max_grad_norm': 0.0,
            'mean_grad_norm': 0.0,
            'is_healthy': True
        }
        
        grad_norms = []
        
        for param_name, param in model.named_parameters():
            stats['total_params'] += 1
            
            if param.grad is not None:
                stats['params_with_grad'] += 1
                grad = param.grad.data
                
                if torch.isnan(grad).any():
                    stats['nan_grad_count'] += 1
                    stats['is_healthy'] = False
                    warnings.warn(f"NaN gradient detected in {name}.{param_name}")
                
                if torch.isinf(grad).any():
                    stats['inf_grad_count'] += 1
                    stats['is_healthy'] = False
                    warnings.warn(f"Inf gradient detected in {name}.{param_name}")
                
                grad_norm = grad.norm().item()
                grad_norms.append(grad_norm)
                
                if grad_norm > max_grad_norm:
                    stats['is_healthy'] = False
                    warnings.warn(f"Large gradient norm {grad_norm:.2e} in {name}.{param_name}")
        
        if grad_norms:
            stats['max_grad_norm'] = max(grad_norms)
            stats['mean_grad_norm'] = np.mean(grad_norms)
        
        return stats
    
    @staticmethod  
    def safe_tensor_operation(func, *tensors, operation_name: str = "tensor_operation"):
        for i, tensor in enumerate(tensors):
            if isinstance(tensor, torch.Tensor):
                NumericalSafety.check_tensor_validity(tensor, f"{operation_name}_input_{i}")
        
        try:
            result = func(*tensors)
        except Exception as e:
            raise NumericalSafetyError(f"Error in {operation_name}: {str(e)}")
        
        if isinstance(result, torch.Tensor):
            NumericalSafety.check_tensor_validity(result, f"{operation_name}_output")
        elif isinstance(result, (tuple, list)):
            for i, out_tensor in enumerate(result):
                if isinstance(out_tensor, torch.Tensor):
                    NumericalSafety.check_tensor_validity(out_tensor, f"{operation_name}_output_{i}")
        
        return result

def check_nan_inf(tensor, name="tensor"):
    return NumericalSafety.check_tensor_validity(tensor, name)

if __name__ == "__main__":
    print("Numerical safety utilities test...")

    import sys, os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'modules'))
    from numerics import safe_divide

    a = torch.tensor([1.0, 2.0, 3.0])
    b = torch.tensor([2.0, 0.0, 1e-12])

    try:
        result = safe_divide(a, b)
        print(f"Safe division test passed: {result}")
    except Exception as e:
        print(f"Safe division test failed: {e}")

    nan_tensor = torch.tensor([1.0, float('nan'), 3.0])
    try:
        check_nan_inf(nan_tensor, "test_tensor")
    except NaNDetectedError as e:
        print(f"NaN detection working: {e}")

    print("Numerical safety utilities module test complete!")
