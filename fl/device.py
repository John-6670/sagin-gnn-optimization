import os
import torch

# Allow forcing CUDA via environment variable `FORCE_CUDA` (1/true/yes)
_force_cuda = str(os.environ.get("FORCE_CUDA", "0")).lower() in ("1", "true", "yes")

if _force_cuda:
	try:
		# Try allocating a tiny tensor on CUDA to trigger initialization and detect driver issues
		torch.zeros(1, device="cuda")
		DEVICE = torch.device("cuda")
	except Exception as e:
		print(
			"WARNING: Requested CUDA via FORCE_CUDA but failed to initialize CUDA."
			f"\n  Error: {e}\n  Falling back to CPU."
		)
		DEVICE = torch.device("cpu")
else:
	DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"Using device: {DEVICE}")