
import traceback
import logging
from typing import Optional, Dict, Any, List
import torch

class TimoshenkoError(Exception):
    
    def __init__(self, message: str, error_code: Optional[str] = None, 
                 suggestions: Optional[List[str]] = None, 
                 context: Optional[Dict[str, Any]] = None):
        super().__init__(message)
        self.message = message
        self.error_code = error_code or self.__class__.__name__
        self.suggestions = suggestions or []
        self.context = context or {}
        
    def __str__(self) -> str:
        result = f"[{self.error_code}] {self.message}"
        
        if self.context:
            result += f"\n上下文信息: {self.context}"
            
        if self.suggestions:
            result += "\n建议解决方案:"
            for i, suggestion in enumerate(self.suggestions, 1):
                result += f"\n  {i}. {suggestion}"
                
        return result

class ConfigurationError(TimoshenkoError):
    pass

class PhysicsError(TimoshenkoError):
    pass

class NumericalError(TimoshenkoError):
    pass

class NetworkError(TimoshenkoError):
    pass

class TrainingError(TimoshenkoError):
    pass

class DataError(TimoshenkoError):
    pass

class InvalidParameterError(ConfigurationError):
    
    def __init__(self, param_name: str, param_value: Any, 
                 valid_range: Optional[str] = None):
        suggestions = [
            f"检查参数 {param_name} 的取值范围",
            "参考documentation中的参数说明",
            "使用默认参数值进行测试"
        ]
        
        if valid_range:
            suggestions.insert(0, f"确保 {param_name} 在有效范围内: {valid_range}")
        
        context = {
            "parameter": param_name,
            "value": param_value,
            "valid_range": valid_range
        }
        
        message = f"参数 {param_name} 的值 {param_value} 无效"
        super().__init__(message, "INVALID_PARAM", suggestions, context)

class PhysicsViolationError(PhysicsError):
    
    def __init__(self, violation_type: str, details: Optional[str] = None):
        suggestions = [
            "检查材料参数是否合理",
            "验证边界条件设置",
            "确认载荷大小和方向",
            "检查几何参数的物理意义"
        ]
        
        context = {
            "violation_type": violation_type,
            "details": details
        }
        
        message = f"物理约束违反: {violation_type}"
        if details:
            message += f" - {details}"
            
        super().__init__(message, "PHYSICS_VIOLATION", suggestions, context)

class NumericalInstabilityError(NumericalError):
    
    def __init__(self, instability_type: str, location: Optional[str] = None):
        suggestions = [
            "减小学习率或时间步长",
            "增加数值稳定性检查",
            "使用更稳定的数值方法",
            "检查边界条件的实现"
        ]
        
        if "NaN" in instability_type:
            suggestions.insert(0, "检查除零操作和数值溢出")
        elif "divergence" in instability_type.lower():
            suggestions.insert(0, "减小训练参数或增加正则化")
        
        context = {
            "type": instability_type,
            "location": location
        }
        
        message = f"数值不稳定: {instability_type}"
        if location:
            message += f" (位置: {location})"
            
        super().__init__(message, "NUMERICAL_INSTABILITY", suggestions, context)

class ConvergenceError(TrainingError):
    
    def __init__(self, reason: str, epoch: Optional[int] = None, 
                 loss_value: Optional[float] = None):
        suggestions = [
            "调整学习率和优化器参数",
            "检查网络架构的合理性",
            "增加训练数据或改进数据质量",
            "使用不同的初始化策略",
            "添加正则化项"
        ]
        
        context = {
            "reason": reason,
            "epoch": epoch,
            "loss_value": loss_value
        }
        
        message = f"训练收敛失败: {reason}"
        if epoch is not None:
            message += f" (epoch: {epoch})"
        if loss_value is not None:
            message += f" (loss: {loss_value:.2e})"
            
        super().__init__(message, "CONVERGENCE_FAILED", suggestions, context)

class GPUMemoryError(NetworkError):
    
    def __init__(self, operation: str, required_memory: Optional[float] = None):
        suggestions = [
            "减少批处理大小(batch size)",
            "使用梯度累积",
            "启用混合精度训练(AMP)",
            "清理GPU缓存",
            "使用CPU进行计算"
        ]
        
        context = {
            "operation": operation,
            "required_memory": required_memory
        }
        
        message = f"GPU内存不足: {operation}"
        if required_memory:
            message += f" (需要: {required_memory:.2f}GB)"
            
        super().__init__(message, "GPU_MEMORY_ERROR", suggestions, context)

class ExceptionHandler:
    
    def __init__(self, logger: Optional[logging.Logger] = None):
        self.logger = logger or logging.getLogger(__name__)
        
    def handle_exception(self, exc: Exception, context: Optional[Dict] = None) -> Dict[str, Any]:
        
        exc_info = {
            "type": type(exc).__name__,
            "message": str(exc),
            "traceback": traceback.format_exc(),
            "context": context or {}
        }
        
        if isinstance(exc, TimoshenkoError):
            exc_info.update({
                "error_code": exc.error_code,
                "suggestions": exc.suggestions,
                "exception_context": exc.context
            })
        
        self.logger.error(f"异常发生: {exc_info['type']} - {exc_info['message']}")
        
        if isinstance(exc, TimoshenkoError) and exc.suggestions:
            self.logger.info("建议解决方案:")
            for suggestion in exc.suggestions:
                self.logger.info(f"  - {suggestion}")
        
        return exc_info
    
    def try_recovery(self, exc: Exception) -> Optional[str]:
        
        if isinstance(exc, torch.cuda.OutOfMemoryError):
            torch.cuda.empty_cache()
            return "已清理GPU缓存，请尝试减少批处理大小"
        
        elif isinstance(exc, NumericalInstabilityError):
            return "建议降低学习率并检查数值稳定性"
        
        elif "NaN" in str(exc) or "Inf" in str(exc):
            return "检测到数值异常，建议检查输入数据和计算过程"
        
        return None

def safe_execute(func, *args, exception_handler: Optional[ExceptionHandler] = None, **kwargs):
    
    handler = exception_handler or ExceptionHandler()
    
    try:
        return func(*args, **kwargs)
    except Exception as e:
        exc_info = handler.handle_exception(e)
        recovery_msg = handler.try_recovery(e)
        
        if recovery_msg:
            print(f"Auto recovery attempt: {recovery_msg}")
        
        raise e

def error_handler(logger: Optional[logging.Logger] = None):
    
    def decorator(func):
        def wrapper(*args, **kwargs):
            handler = ExceptionHandler(logger)
            try:
                return func(*args, **kwargs)
            except Exception as e:
                exc_info = handler.handle_exception(e, {"function": func.__name__})
                recovery_msg = handler.try_recovery(e)
                
                if recovery_msg:
                    print(f"Function {func.__name__} exception, attempting recovery: {recovery_msg}")
                
                raise e
        return wrapper
    return decorator

def validate_parameters(**validators):
    
    def decorator(func):
        def wrapper(*args, **kwargs):
            import inspect
            sig = inspect.signature(func)
            bound_args = sig.bind(*args, **kwargs)
            bound_args.apply_defaults()
            
            for param_name, validator in validators.items():
                if param_name in bound_args.arguments:
                    value = bound_args.arguments[param_name]
                    if not validator(value):
                        raise InvalidParameterError(
                            param_name, value, 
                            f"验证函数: {validator.__name__}"
                        )
            
            return func(*args, **kwargs)
        return wrapper
    return decorator

def is_positive(x):
    return x > 0

def is_non_negative(x):
    return x >= 0

def is_in_range(min_val, max_val):
    def validator(x):
        return min_val <= x <= max_val
    return validator

def is_valid_tensor(x):
    if not isinstance(x, torch.Tensor):
        return False
    return not (torch.isnan(x).any() or torch.isinf(x).any())

if __name__ == "__main__":
    print("Exception handling system test...")
    
    handler = ExceptionHandler()
    
    try:
        raise InvalidParameterError("learning_rate", -0.01, "must be positive")
    except TimoshenkoError as e:
        print(f"Caught custom exception:\n{e}")
        handler.handle_exception(e)
    
    @validate_parameters(x=is_positive, y=is_in_range(0, 1))
    def test_function(x, y):
        return x * y
    
    try:
        test_function(-1, 0.5)
    except InvalidParameterError as e:
        print(f"\nParameter validation exception:\n{e}")
    
    print("\nException handling system created!")
