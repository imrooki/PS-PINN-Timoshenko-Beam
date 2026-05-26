
import os
import sys

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from utils.training_core import run_training_core, validate_params_dict

def main():
    try:
        import params
    except ImportError as e:
        print(f"[ERROR] Failed to import params: {e}")
        sys.exit(1)

    params_dict = {}
    for attr in dir(params):
        if not attr.startswith('_'):
            params_dict[attr] = getattr(params, attr)

    if not validate_params_dict(params_dict):
        print("ERROR: Parameter validation failed, please check params.py")
        sys.exit(1)

    print("TARGET: Running Timoshenko beam PINNs training via main.py")

    result = run_training_core(
        params_dict=params_dict,
        script_name="main",
        verbose=True
    )

    if result['success']:
        print("[SUCCESS] Training completed successfully!")
    else:
        print(f"[ERROR] Training failed: {result.get('error', 'Unknown error')}")
        sys.exit(1)

if __name__ == "__main__":
    main()

