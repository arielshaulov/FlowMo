# 
# import pynvml

def printGPUState_old():
    try:
        pynvml.nvmlInit()
        device_count = pynvml.nvmlDeviceGetCount()

        for i in range(device_count):
            handle = pynvml.nvmlDeviceGetHandleByIndex(i)
            name = pynvml.nvmlDeviceGetName(handle).decode()
            mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
            utilization = pynvml.nvmlDeviceGetUtilizationRates(handle)

            print(f"GPU {i}: {name}")
            print(f"  Memory Used: {mem_info.used / 1024**2:.2f} MB / {mem_info.total / 1024**2:.2f} MB")
            print(f"  GPU Utilization: {utilization.gpu}%")
            print(f"  Memory Utilization: {utilization.memory}%")
            print()
            
        pynvml.nvmlShutdown()

    except pynvml.NVMLError as error:
        print(f"Failed to access GPU info: {error}")

import subprocess

def printGPUState():
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=index,name,memory.total,memory.used,utilization.gpu,utilization.memory",
                "--format=csv,noheader,nounits"
            ],
            capture_output=True,
            text=True,
            check=True
        )

        lines = result.stdout.strip().split("\n")
        for line in lines:
            index, name, mem_total, mem_used, util_gpu, util_mem = [item.strip() for item in line.split(",")]
            print(f"GPU {index}: {name}")
            print(f"  Memory Used: {mem_used} MB / {mem_total} MB")
            print(f"  GPU Utilization: {util_gpu}%")
            print(f"  Memory Utilization: {util_mem}%")
            print()

    except subprocess.CalledProcessError as e:
        print("Failed to run nvidia-smi:", e)
    except FileNotFoundError:
        print("nvidia-smi not found. Make sure NVIDIA drivers are installed and accessible.")

if __name__ == "__main__":
    printGPUState()
    # printGPUState_old()