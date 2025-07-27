"""GPU memory monitoring utilities for training workflows."""

import torch
import psutil
import logging
import time
from typing import Dict, Optional, List, Tuple
from dataclasses import dataclass, asdict
from contextlib import contextmanager


@dataclass
class GPUMemoryInfo:
    """Container for GPU memory information."""
    allocated_mb: float
    reserved_mb: float
    free_mb: float
    total_mb: float
    utilization_percent: float
    max_allocated_mb: float
    max_reserved_mb: float


@dataclass
class SystemMemoryInfo:
    """Container for system memory information."""
    used_mb: float
    available_mb: float
    total_mb: float
    utilization_percent: float


@dataclass
class MemorySnapshot:
    """Snapshot of memory state at a specific time."""
    timestamp: float
    label: str
    gpu_info: Optional[GPUMemoryInfo]
    system_info: SystemMemoryInfo
    epoch: Optional[int] = None
    batch_idx: Optional[int] = None
    stage: Optional[str] = None


class GPUMonitor:
    """Monitor GPU and system memory usage during training."""
    
    def __init__(self, device: Optional[torch.device] = None, enable_logging: bool = True):
        """
        Initialize GPU monitor.
        
        Args:
            device: Target device to monitor. If None, uses current device.
            enable_logging: Whether to enable logging output.
        """
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.enable_logging = enable_logging
        self.logger = logging.getLogger(__name__)
        
        # Memory tracking
        self.snapshots: List[MemorySnapshot] = []
        self.peak_gpu_usage = 0.0
        self.peak_system_usage = 0.0
        
        # Reset peak stats if CUDA available
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats(self.device)
        
    def get_gpu_memory_info(self) -> Optional[GPUMemoryInfo]:
        """
        Get current GPU memory information.
        
        Returns:
            GPUMemoryInfo object or None if CUDA not available.
        """
        if not torch.cuda.is_available():
            return None
            
        try:
            allocated = torch.cuda.memory_allocated(self.device) / (1024 ** 2)
            reserved = torch.cuda.memory_reserved(self.device) / (1024 ** 2)
            max_allocated = torch.cuda.max_memory_allocated(self.device) / (1024 ** 2)
            max_reserved = torch.cuda.max_memory_reserved(self.device) / (1024 ** 2)
            
            # Get total GPU memory
            total = torch.cuda.get_device_properties(self.device).total_memory / (1024 ** 2)
            free = total - allocated
            utilization = (allocated / total) * 100 if total > 0 else 0
            
            return GPUMemoryInfo(
                allocated_mb=allocated,
                reserved_mb=reserved,
                free_mb=free,
                total_mb=total,
                utilization_percent=utilization,
                max_allocated_mb=max_allocated,
                max_reserved_mb=max_reserved
            )
        except Exception as e:
            self.logger.warning(f"Failed to get GPU memory info: {e}")
            return None
    
    def get_system_memory_info(self) -> SystemMemoryInfo:
        """
        Get current system memory information.
        
        Returns:
            SystemMemoryInfo object.
        """
        try:
            memory = psutil.virtual_memory()
            
            return SystemMemoryInfo(
                used_mb=memory.used / (1024 ** 2),
                available_mb=memory.available / (1024 ** 2),
                total_mb=memory.total / (1024 ** 2),
                utilization_percent=memory.percent
            )
        except Exception as e:
            self.logger.warning(f"Failed to get system memory info: {e}")
            # Return empty info if failed
            return SystemMemoryInfo(0.0, 0.0, 0.0, 0.0)
    
    def log_memory(self, label: str = "", level: int = logging.INFO, 
                   epoch: Optional[int] = None, batch_idx: Optional[int] = None,
                   stage: Optional[str] = None) -> None:
        """
        Log current memory usage with optional context.
        
        Args:
            label: Description label for the log entry.
            level: Logging level.
            epoch: Current epoch number (optional).
            batch_idx: Current batch index (optional).
            stage: Training stage (optional).
        """
        if not self.enable_logging:
            return
            
        gpu_info = self.get_gpu_memory_info()
        sys_info = self.get_system_memory_info()
        
        # Create snapshot
        snapshot = MemorySnapshot(
            timestamp=time.time(),
            label=label,
            gpu_info=gpu_info,
            system_info=sys_info,
            epoch=epoch,
            batch_idx=batch_idx,
            stage=stage
        )
        self.snapshots.append(snapshot)
        
        # Update peak usage
        if gpu_info:
            self.peak_gpu_usage = max(self.peak_gpu_usage, gpu_info.utilization_percent)
        self.peak_system_usage = max(self.peak_system_usage, sys_info.utilization_percent)
        
        # Format log message
        context_parts = []
        if epoch is not None:
            context_parts.append(f"Epoch {epoch}")
        if batch_idx is not None:
            context_parts.append(f"Batch {batch_idx}")
        if stage:
            context_parts.append(f"Stage {stage}")
        
        context_str = f"[{', '.join(context_parts)}] " if context_parts else ""
        label_str = f"[{label}] " if label else ""
        
        if gpu_info:
            self.logger.log(level, 
                f"{context_str}{label_str}GPU: {gpu_info.allocated_mb:.1f}MB allocated "
                f"({gpu_info.utilization_percent:.1f}%), "
                f"{gpu_info.reserved_mb:.1f}MB reserved, "
                f"{gpu_info.free_mb:.1f}MB free")
        
        self.logger.log(level,
            f"{context_str}{label_str}System: {sys_info.used_mb:.1f}MB used "
            f"({sys_info.utilization_percent:.1f}%), "
            f"{sys_info.available_mb:.1f}MB available")
    
    def log_memory_summary(self, prefix: str = "", level: int = logging.INFO) -> None:
        """
        Log memory summary (for backward compatibility).
        
        Args:
            prefix: Prefix for the log message.
            level: Logging level.
        """
        self.log_memory(label=prefix, level=level)
    
    def print_memory_summary(self, prefix: str = "") -> None:
        """
        Print memory summary to console.
        
        Args:
            prefix: Prefix for the message.
        """
        gpu_info = self.get_gpu_memory_info()
        sys_info = self.get_system_memory_info()
        
        prefix_str = f"[{prefix}] " if prefix else ""
        
        if gpu_info:
            print(f"{prefix_str}GPU - Allocated: {gpu_info.allocated_mb:.2f} MB | "
                  f"Reserved: {gpu_info.reserved_mb:.2f} MB | "
                  f"Utilization: {gpu_info.utilization_percent:.1f}% | "
                  f"Peak: {gpu_info.max_allocated_mb:.2f} MB")
        
        print(f"{prefix_str}System - Used: {sys_info.used_mb:.2f} MB | "
              f"Available: {sys_info.available_mb:.2f} MB | "
              f"Utilization: {sys_info.utilization_percent:.1f}%")
    
    def clear_gpu_cache(self) -> None:
        """Clear GPU cache if CUDA is available."""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            if self.enable_logging:
                self.logger.debug("GPU cache cleared")
    
    def reset_peak_stats(self) -> None:
        """Reset peak memory statistics."""
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats(self.device)
        self.peak_gpu_usage = 0.0
        self.peak_system_usage = 0.0
        if self.enable_logging:
            self.logger.debug("Peak memory statistics reset")
    
    def get_memory_stats(self) -> Dict:
        """
        Get comprehensive memory statistics.
        
        Returns:
            Dictionary containing memory statistics.
        """
        stats = {
            'system_memory': asdict(self.get_system_memory_info()),
            'peak_system_usage': self.peak_system_usage,
            'num_snapshots': len(self.snapshots)
        }
        
        gpu_info = self.get_gpu_memory_info()
        if gpu_info:
            stats['gpu_memory'] = asdict(gpu_info)
            stats['peak_gpu_usage'] = self.peak_gpu_usage
            
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
    
    def get_peak_usage(self) -> Dict[str, float]:
        """
        Get peak memory usage from monitoring session.
        
        Returns:
            Dictionary with peak usage statistics.
        """
        return {
            'peak_gpu_usage_percent': self.peak_gpu_usage,
            'peak_system_usage_percent': self.peak_system_usage,
            'peak_gpu_allocated_mb': max((s.gpu_info.allocated_mb for s in self.snapshots 
                                        if s.gpu_info), default=0.0),
            'peak_system_used_mb': max((s.system_info.used_mb for s in self.snapshots), default=0.0)
        }
    
    def get_snapshots_by_stage(self, stage: str) -> List[MemorySnapshot]:
        """Get all snapshots for a specific training stage."""
        return [s for s in self.snapshots if s.stage == stage]
    
    def get_snapshots_by_epoch(self, epoch: int) -> List[MemorySnapshot]:
        """Get all snapshots for a specific epoch."""
        return [s for s in self.snapshots if s.epoch == epoch]
    
    def clear_snapshots(self) -> None:
        """Clear all stored snapshots."""
        self.snapshots.clear()
    
    @contextmanager
    def monitor_context(self, label: str, epoch: Optional[int] = None, 
                       stage: Optional[str] = None):
        """
        Context manager for monitoring memory during a specific operation.
        
        Args:
            label: Label for the operation.
            epoch: Current epoch (optional).
            stage: Training stage (optional).
        """
        self.log_memory(f"Before {label}", epoch=epoch, stage=stage)
        try:
            yield
        finally:
            self.log_memory(f"After {label}", epoch=epoch, stage=stage)


class MemoryTracker:
    """Track memory usage over time during training with enhanced features."""
    
    def __init__(self, device: Optional[torch.device] = None, enable_logging: bool = True):
        """
        Initialize memory tracker.
        
        Args:
            device: Target device to monitor.
            enable_logging: Whether to enable logging.
        """
        self.monitor = GPUMonitor(device, enable_logging)
        self.history = []
        
    def record(self, step: int, phase: str = "train", epoch: Optional[int] = None,
               label: str = "") -> None:
        """
        Record current memory usage.
        
        Args:
            step: Current training step.
            phase: Training phase (train, val, etc.).
            epoch: Current epoch number.
            label: Additional label for this recording.
        """
        stats = self.monitor.get_memory_stats()
        stats.update({
            'step': step,
            'phase': phase,
            'epoch': epoch,
            'label': label,
            'timestamp': time.time()
        })
        self.history.append(stats)
        
        # Log if enabled
        if self.monitor.enable_logging:
            self.monitor.log_memory(
                f"Step {step} ({phase})", 
                epoch=epoch, 
                stage=phase
            )
    
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
        if self.history and 'gpu_memory' in self.history[0]:
            peak_gpu = max(self.history, key=lambda x: x['gpu_memory']['utilization_percent'])
            result['peak_gpu'] = peak_gpu
            
        return result
    
    def get_history_by_phase(self, phase: str) -> List[Dict]:
        """Get history entries for a specific phase."""
        return [h for h in self.history if h.get('phase') == phase]
    
    def get_history_by_epoch(self, epoch: int) -> List[Dict]:
        """Get history entries for a specific epoch."""
        return [h for h in self.history if h.get('epoch') == epoch]
    
    def clear_history(self) -> None:
        """Clear tracking history."""
        self.history.clear()
        self.monitor.clear_snapshots()
    
    def save_history(self, filepath: str) -> None:
        """
        Save memory tracking history to file.
        
        Args:
            filepath: Path to save the history.
        """
        import json
        
        # Make history JSON serializable
        serializable_history = []
        for entry in self.history:
            serializable_entry = {}
            for key, value in entry.items():
                if isinstance(value, (int, float, str, bool, type(None))):
                    serializable_entry[key] = value
                elif isinstance(value, dict):
                    serializable_entry[key] = value
                else:
                    serializable_entry[key] = str(value)
            serializable_history.append(serializable_entry)
        
        with open(filepath, 'w') as f:
            json.dump(serializable_history, f, indent=2)
    
    def print_summary(self) -> None:
        """Print a summary of memory tracking."""
        if not self.history:
            print("No memory tracking history available")
            return
        
        peak_stats = self.get_peak_usage()
        print("\n" + "="*60)
        print("MEMORY TRACKING SUMMARY")
        print("="*60)
        print(f"Total recordings: {len(self.history)}")
        
        if 'peak_system' in peak_stats:
            sys_peak = peak_stats['peak_system']['system_memory']
            print(f"Peak system memory: {sys_peak['used_mb']:.1f} MB "
                  f"({sys_peak['utilization_percent']:.1f}%)")
        
        if 'peak_gpu' in peak_stats:
            gpu_peak = peak_stats['peak_gpu']['gpu_memory']
            print(f"Peak GPU memory: {gpu_peak['allocated_mb']:.1f} MB "
                  f"({gpu_peak['utilization_percent']:.1f}%)")
        
        print("="*60)


# Convenience functions for backward compatibility
def print_gpu_memory(prefix: str = "") -> None:
    """Print GPU memory usage (backward compatibility)."""
    monitor = GPUMonitor()
    monitor.print_memory_summary(prefix)


def clear_gpu_cache() -> None:
    """Clear GPU cache (backward compatibility)."""
    monitor = GPUMonitor()
    monitor.clear_gpu_cache()


# Enhanced convenience functions
def log_memory_usage(label: str, device: Optional[torch.device] = None) -> None:
    """Log current memory usage with label."""
    monitor = GPUMonitor(device)
    monitor.log_memory(label)


def check_memory_usage(gpu_threshold: float = 90.0, sys_threshold: float = 90.0,
                      device: Optional[torch.device] = None) -> Dict[str, bool]:
    """Check if memory usage exceeds thresholds."""
    monitor = GPUMonitor(device)
    return monitor.check_memory_threshold(gpu_threshold, sys_threshold)