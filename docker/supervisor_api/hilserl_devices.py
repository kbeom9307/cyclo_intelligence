"""Explicit compute placement; a requested CUDA device never falls back to CPU."""
import subprocess
import time


def resolve_device(name):
    import torch
    device = torch.device(name)
    if device.type == 'cuda':
        index = device.index or 0
        if not torch.cuda.is_available() or index >= torch.cuda.device_count():
            raise RuntimeError(f'{name} is unavailable. Enable NVIDIA container GPU access or select another device.')
        torch.cuda.set_device(index)
    elif device.type != 'cpu':
        raise ValueError('Only cpu and cuda:N devices are supported')
    return device


def memory_status(device):
    import torch
    device = torch.device(device)
    if device.type != 'cuda':
        return {'device': str(device)}
    free, total = torch.cuda.mem_get_info(device)
    mib = 1024 ** 2
    return {'device': str(device), 'name': torch.cuda.get_device_name(device),
            'allocated_mib': round(torch.cuda.memory_allocated(device) / mib, 1),
            'reserved_mib': round(torch.cuda.memory_reserved(device) / mib, 1),
            'peak_allocated_mib': round(torch.cuda.max_memory_allocated(device) / mib, 1),
            'free_mib': round(free / mib, 1), 'total_mib': round(total / mib, 1)}


_cache = (0, {})


def hardware_status():
    """NVML via nvidia-smi: no CUDA context/VRAM allocation in the control plane."""
    global _cache
    if time.monotonic() - _cache[0] < 5:
        return _cache[1]
    result = {'gpus': [], 'error': None}
    try:
        output = subprocess.check_output(['nvidia-smi', '--query-gpu=index,name,memory.total,memory.used,memory.free',
                                          '--format=csv,noheader,nounits'], timeout=2, text=True, stderr=subprocess.PIPE)
        for line in output.strip().splitlines():
            index, name, total, used, free = [value.strip() for value in line.split(',')]
            result['gpus'].append({'device': f'cuda:{index}', 'name': name, 'total_mib': float(total),
                                   'used_mib': float(used), 'free_mib': float(free)})
    except (OSError, ValueError, subprocess.SubprocessError):
        result['error'] = 'NVIDIA GPU is unavailable to this container'
    _cache = (time.monotonic(), result)
    return result


def device_ready(name, hardware):
    return name == 'cpu' or any(gpu['device'] == name for gpu in hardware['gpus'])
