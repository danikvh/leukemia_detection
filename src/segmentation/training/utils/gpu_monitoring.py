"""GPU memory monitoring utilities."""

import torch
import psutil
import logging
from typing import Dict, Optional
from dataclasses import dataclass


@dataclass
class GPUMemoryInfo:
    """Container for GPU memory information."""
    allocated_mb: float
    reserved_mb: float
    free_mb: float
    total_mb: float
    utilization_percent: float


@dataclass
class SystemMemoryInfo:
    """Container for system memory information."""
    used_mb: float
    available_mb: float
    total_mb: float
    utilization_percent: float


class GPUMonitor:
    """Monitor GPU and system memory usage."""
    
    def __init__(self, device: Optional[torch.device] = None):
        """
        Initialize GPU monitor.
        
        Args:
            device: Target device to monitor. If None, uses current device.
        """
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.logger = logging.getLogger(__name__)
        
    def get_gpu_memory_info(self) -> Optional[GPUMemoryInfo]:
        """
        Get current GPU memory information.
        
        Returns:
            GPUMemoryInfo object or None if CUDA not available.
        """
        if not torch.cuda.is_available():
            return None
            
        allocated = torch.cuda.memory_allocated(self.device) / (1024 ** 2)
        reserved = torch.cuda.memory_reserved(self.device) / (1024 ** 2)
        
        # Get total GPU memory
        total = torch.cuda.get_device_properties(self.device).total_memory / (1024 ** 2)
        free = total - allocated
        utilization = (allocated / total) * 100 if total > 0 else 0
        
        return GPUMemoryInfo(
            allocated_mb=allocated,
            reserved_mb=reserved,
            free_mb=free,
            total_mb=total,
            utilization_percent=utilization
        )
    
    def get_system_memory_info(self) -> SystemMemoryInfo:
        """
        Get current system memory information.
        
        Returns:
            SystemMemoryInfo object.
        """
        memory = psutil.virtual_memory()
        
        return SystemMemoryInfo(
            used_mb=memory.used / (1024 ** 2),
            available_mb=memory.available / (1024 ** 2),
            total_mb=memory.total / (1024 ** 2),
            utilization_percent=memory.percent
        )
    
    def print_memory_summary(self, prefix: str = "") -> None:
        """
        Print memory summary to console.
        
        Args:
            prefix: Prefix for the log message.
        """
        gpu_info = self.get_gpu_memory_info()
        sys_info = self.get_system_memory_info()
        
        if gpu_info:
            print(f"[{prefix}] GPU - Allocated: {gpu_info.allocated_mb:.2f} MB | "
                  f"Reserved: {gpu_info.reserved_mb:.2f} MB | "
                  f"Utilization: {gpu_info.utilization_percent:.1f}%")
        
        print(f"[{prefix}] System - Used: {sys_info.used_mb:.2f} MB | "
              f"Available: {sys_info.available_mb:.2f} MB | "
              f"Utilization: {sys_info.utilization_percent:.1f}%")
    
    def log_memory_summary(self, prefix: str = "", level: int = logging.INFO) -> None:
        """
        Log memory summary.
        
        Args:
            prefix: Prefix for the log message.
            level: Logging level.
        """
        gpu_info = self.get_gpu_memory_info()
        sys_info = self.get_system_memory_info()
        
        if gpu_info:
            self.logger.log(level, 
                f"[{prefix}] GPU Memory - Allocated: {gpu_info.allocated_mb:.2f} MB, "
                f"Reserved: {gpu_info.reserved_mb:.2f} MB, "
                f"Utilization: {gpu_info.utilization_percent:.1f}%")
        
        self.logger.log(level,
            f"[{prefix}] System Memory - Used: {sys_info.used_mb:.2f} MB, "
            f"Available: {sys_info.available_mb:.2f} MB, "
            f"Utilization: {sys_info.utilization_percent:.1f}%")
    
    def clear_gpu_cache(self) -> None:
        """Clear GPU cache if CUDA is available."""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            self.logger.debug("GPU cache cleared")
    
    def get_memory_stats(self) -> Dict:
        """
        Get comprehensive memory statistics.
        
        Returns:
            Dictionary containing memory statistics.
        """
        stats = {
            'system_memory': self.get_system_memory_info().__dict__
        }
        
        gpu_info = self.get_gpu_memory_info()
        if gpu_info:
            stats['gpu_memory'] = gpu_info.__dict__
            
        return stats
    
    def check_memory_threshold(self, gpu_threshold_percent: float = 90.0, 
                             sys_threshold_percent: float = 90.0) -> Dict[str, bool]:
        """
        Check if memory usage exceeds thresholds.
        
        Args:
            gpu_threshold_percent: GPU memory threshold percentage.
            sys_threshold_percent: System memory threshold percentage.
            
        Returns:
            Dictionary indicating if thresholds are exceeded.
        """
        gpu_info = self.get_gpu_memory_info()
        sys_info = self.get_system_memory_info()
        
        result = {
            'system_exceeded': sys_info.utilization_percent > sys_threshold_percent,
            'gpu_exceeded': False
        }
        
        if gpu_info:
            result['gpu_exceeded'] = gpu_info.utilization_percent > gpu_threshold_percent
            
        return result


class MemoryTracker:
    """Track memory usage over time during training."""
    
    def __init__(self, device: Optional[torch.device] = None):
        """
        Initialize memory tracker.
        
        Args:
            device: Target device to monitor.
        """
        self.monitor = GPUMonitor(device)
        self.history = []
        
    def record(self, step: int, phase: str = "train") -> None:
        """
        Record current memory usage.
        
        Args:
            step: Current training step.
            phase: Training phase (train, val, etc.).
        """
        stats = self.monitor.get_memory_stats()
        stats.update({
            'step': step,
            'phase': phase
        })
        self.history.append(stats)
    
    def get_peak_usage(self) -> Dict:
        """
        Get peak memory usage from history.
        
        Returns:
            Dictionary with peak usage statistics.
        """
        if not self.history:
            return {}
            
        peak_sys = max(self.history, key=lambda x: x['system_memory']['utilization_percent'])
        result = {'peak_system': peak_sys}
        
        # Check if GPU stats are available
        if 'gpu_memory' in self.history[0]:
            peak_gpu = max(self.history, key=lambda x: x['gpu_memory']['utilization_percent'])
            result['peak_gpu'] = peak_gpu
            
        return result
    
    def clear_history(self) -> None:
        """Clear tracking history."""
        self.history.clear()
    
    def save_history(self, filepath: str) -> None:
        """
        Save memory tracking history to file.
        
        Args:
            filepath: Path to save the history.
        """
        import json
        with open(filepath, 'w') as f:
            json.dump(self.history, f, indent=2)


# Convenience functions for backward compatibility
def print_gpu_memory(prefix: str = "") -> None:
    """Print GPU memory usage (backward compatibility)."""
    monitor = GPUMonitor()
    monitor.print_memory_summary(prefix)


def clear_gpu_cache() -> None:
    """Clear GPU cache (backward compatibility)."""
    monitor = GPUMonitor()
    monitor.clear_gpu_cache()