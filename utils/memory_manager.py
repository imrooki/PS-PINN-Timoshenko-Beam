
import gc
import warnings
from typing import Optional, Dict, Any
import torch
import psutil
import os

class MemoryManager:
    
    def __init__(self, gpu_memory_threshold: float = 0.9, 
                 cpu_memory_threshold: float = 0.85):
        self.gpu_threshold = gpu_memory_threshold
        self.cpu_threshold = cpu_memory_threshold
        self.initial_memory = self.get_memory_stats()
        
    def get_memory_stats(self) -> Dict[str, Any]:
        stats = {}
        
        cpu_memory = psutil.virtual_memory()
        stats['cpu'] = {
            'total': cpu_memory.total / (1024**3),
            'used': cpu_memory.used / (1024**3),
            'available': cpu_memory.available / (1024**3),
            'percent': cpu_memory.percent
        }
        
        if torch.cuda.is_available():
            gpu_id = torch.cuda.current_device()
            gpu_memory = torch.cuda.get_device_properties(gpu_id).total_memory
            gpu_allocated = torch.cuda.memory_allocated(gpu_id)
            gpu_reserved = torch.cuda.memory_reserved(gpu_id)
            
            stats['gpu'] = {
                'total': gpu_memory / (1024**3),
                'allocated': gpu_allocated / (1024**3),
                'reserved': gpu_reserved / (1024**3),
                'free': (gpu_memory - gpu_reserved) / (1024**3),
                'utilization': gpu_reserved / gpu_memory
            }
        else:
            stats['gpu'] = None
            
        return stats
    
    def check_memory_health(self) -> Dict[str, Any]:
        stats = self.get_memory_stats()
        health = {
            'cpu_healthy': stats['cpu']['percent'] < self.cpu_threshold * 100,
            'gpu_healthy': True,
            'warnings': [],
            'critical': False
        }
        
        if not health['cpu_healthy']:
            health['warnings'].append(f"CPU memory usage too high: {stats['cpu']['percent']:.1f}%")
            if stats['cpu']['percent'] > 95:
                health['critical'] = True
        
        if stats['gpu'] is not None:
            gpu_util = stats['gpu']['utilization']
            health['gpu_healthy'] = gpu_util < self.gpu_threshold
            
            if not health['gpu_healthy']:
                health['warnings'].append(f"GPU memory usage too high: {gpu_util:.1f}")
                if gpu_util > 0.95:
                    health['critical'] = True
        
        return health
    
    def smart_cleanup(self, force_gc: bool = False) -> Dict[str, Any]:
        cleanup_stats = {
            'actions_taken': [],
            'memory_freed': {'cpu': 0, 'gpu': 0},
            'success': False
        }
        
        before_stats = self.get_memory_stats()
        
        try:
            if force_gc or before_stats['cpu']['percent'] > 80:
                collected = gc.collect()
                cleanup_stats['actions_taken'].append(f"Python GC: collected {collected} objects")
            
            if torch.cuda.is_available():
                if (before_stats['gpu'] and 
                    before_stats['gpu']['utilization'] > 0.8) or force_gc:
                    torch.cuda.empty_cache()
                    cleanup_stats['actions_taken'].append("GPU cache cleanup")
                
                if force_gc:
                    torch.cuda.synchronize()
                    cleanup_stats['actions_taken'].append("GPU sync")
            
            after_stats = self.get_memory_stats()
            
            cpu_freed = before_stats['cpu']['used'] - after_stats['cpu']['used']
            cleanup_stats['memory_freed']['cpu'] = max(0, cpu_freed)
            
            if before_stats['gpu'] and after_stats['gpu']:
                gpu_freed = (before_stats['gpu']['reserved'] - 
                           after_stats['gpu']['reserved'])
                cleanup_stats['memory_freed']['gpu'] = max(0, gpu_freed)
            
            cleanup_stats['success'] = True
            
        except Exception as e:
            cleanup_stats['actions_taken'].append(f"Cleanup failed: {str(e)}")
            warnings.warn(f"Error during memory cleanup: {e}")
        
        return cleanup_stats
    
    def monitor_training_memory(self, epoch: int, 
                              auto_cleanup: bool = True) -> Optional[Dict]:
        health = self.check_memory_health()
        
        if epoch % 100 == 0:
            stats = self.get_memory_stats()
            print(f"Epoch {epoch} - CPU: {stats['cpu']['percent']:.1f}%, "
                  f"GPU: {stats['gpu']['utilization']:.1f if stats['gpu'] else 'N/A'}")
        
        cleanup_result = None
        if auto_cleanup and (health['warnings'] or health['critical']):
            cleanup_result = self.smart_cleanup(force_gc=health['critical'])
            
            if cleanup_result['success']:
                print(f"Memory cleanup complete: {cleanup_result['actions_taken']}")
            else:
                warnings.warn("Memory cleanup failed")
        
        if health['critical']:
            warnings.warn("Critical memory shortage! Consider reducing batch size or increasing cleanup frequency")
        
        return cleanup_result
    
    def get_memory_recommendations(self) -> list:
        stats = self.get_memory_stats()
        recommendations = []
        
        if stats['cpu']['percent'] > 80:
            recommendations.append("Consider reducing training batch size")
            recommendations.append("Increase garbage collection frequency")

        if stats['gpu'] and stats['gpu']['utilization'] > 0.8:
            recommendations.append("Use gradient accumulation to reduce memory demand")
            recommendations.append("Consider using mixed precision training (AMP)")
            recommendations.append("Reduce AGQ integration nodes")

        if not recommendations:
            recommendations.append("Memory usage healthy, no optimization needed")
            
        return recommendations

class MemoryOptimizer:
    
    @staticmethod
    def optimize_for_training(model: torch.nn.Module) -> Dict[str, Any]:
        optimizations = {
            'applied': [],
            'recommendations': []
        }
        
        if torch.cuda.is_available() and hasattr(torch.cuda, 'amp'):
            optimizations['recommendations'].append("启用自动混合精度(AMP)训练")
        
        total_params = sum(p.numel() for p in model.parameters())
        if total_params > 1e6:
            optimizations['recommendations'].append("考虑使用梯度检查点")
        
        optimizations['recommendations'].append("使用动态batch size适应内存")
        
        return optimizations
    
    @staticmethod
    def adaptive_batch_size(initial_batch_size: int,
                          compute_fn,
                          min_batch_size: int = 32) -> int:
        batch_size = initial_batch_size
        
        while batch_size >= min_batch_size:
            try:
                torch.cuda.empty_cache()
                result = compute_fn(batch_size)
                return batch_size
            except torch.cuda.OutOfMemoryError:
                batch_size = max(batch_size // 2, min_batch_size)
                print(f"GPU out of memory, reducing batch size to {batch_size}")
        
        raise RuntimeError(f"Still out of memory even with minimum batch size {min_batch_size}")

_global_memory_manager = None

def get_memory_manager() -> MemoryManager:
    global _global_memory_manager
    if _global_memory_manager is None:
        _global_memory_manager = MemoryManager()
    return _global_memory_manager

def memory_cleanup():
    return get_memory_manager().smart_cleanup()

def check_memory():
    return get_memory_manager().get_memory_stats()

class MemoryContext:
    
    def __init__(self, auto_cleanup: bool = True):
        self.auto_cleanup = auto_cleanup
        self.manager = get_memory_manager()
        self.initial_stats = None
        
    def __enter__(self):
        self.initial_stats = self.manager.get_memory_stats()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.auto_cleanup:
            cleanup_result = self.manager.smart_cleanup()
            if cleanup_result['success']:
                print(f"Context cleanup: {cleanup_result['actions_taken']}")

if __name__ == "__main__":
    print("Memory management utilities test...")

    memory_mgr = MemoryManager()

    stats = memory_mgr.get_memory_stats()
    print(f"Memory stats: {stats}")

    health = memory_mgr.check_memory_health()
    print(f"Memory health: {health}")

    recommendations = memory_mgr.get_memory_recommendations()
    print(f"Optimization recommendations: {recommendations}")

    print("Memory management utilities module test complete!")
