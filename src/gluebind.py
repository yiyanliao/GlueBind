"""GlueBind - Main module with GPU utilities"""

import subprocess
import sys

__version__ = "0.1.0"


def hello():
    """Return a greeting message."""
    return "Hello from GlueBind!"


def check_gpu():
    """Check if NVIDIA GPU is available."""
    try:
        result = subprocess.run(
            ["nvidia-smi"], 
            capture_output=True, 
            text=True,
            timeout=10
        )
        return result.returncode == 0
    except (subprocess.TimeoutExpired, FileNotFoundError, Exception):
        return False


def get_gpu_info():
    """Get GPU information using nvidia-smi."""
    if not check_gpu():
        return {}
    
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total,memory.used,temperature.gpu", 
             "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=10
        )
        if result.returncode == 0:
            lines = result.stdout.strip().split("\n")
            gpus = []
            for line in lines:
                parts = [p.strip() for p in line.split(",")]
                if len(parts) >= 4:
                    gpus.append({
                        "name": parts[0],
                        "memory_total": parts[1],
                        "memory_used": parts[2],
                        "temperature": parts[3]
                    })
            return {"gpus": gpus, "count": len(gpus)}
    except Exception:
        pass
    return {}


def main():
    """Main entry point for CLI."""
    print(hello())
    print(f"Version: {__version__}")
    
    if check_gpu():
        print("GPU is available!")
        gpu_info = get_gpu_info()
        if gpu_info:
            print(f"Found {gpu_info['count']} GPU(s):")
            for gpu in gpu_info["gpus"]:
                print(f"  - {gpu['name']}")
                print(f"    Memory: {gpu['memory_used']} / {gpu['memory_total']}")
                print(f"    Temperature: {gpu['temperature']}")
    else:
        print("GPU is not available")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
