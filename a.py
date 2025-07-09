# check_imports.py
try:
    import torch
    import transformers
    import deepspeed
    import vllm
    import launchpad as lp

    print("--- OAT Core Dependencies Check ---")
    print(f"PyTorch version: {torch.__version__}")
    print(f"Transformers version: {transformers.__version__}")
    print(f"DeepSpeed version: {deepspeed.__version__}")
    print(f"vLLM version: {vllm.__version__}")
    print(f"Launchpad version: {lp.__version__}")
    print(f"\n✅ SUCCESS: Your environment is correctly set up!")

except ImportError as e:
    print(f"\n❌ FAILED: A library is missing or could not be loaded.")
    print(f"Error details: {e}")
    print("\nThere may be an issue with the installation or system dependencies.")