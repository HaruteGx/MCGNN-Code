import torch
import numpy as np

def setup_seed(seed: int = 666):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def get_device(mem_fraction: float = 0.7):
    if torch.cuda.is_available():

        torch.cuda.set_per_process_memory_fraction(mem_fraction)
        device = torch.device("cuda")
        print("CUDA available; using GPU; memory fraction limit:", mem_fraction)
    else:
        device = torch.device("cpu")
        print("CUDA unavailable; using CPU")
    return device
