import subprocess

if __name__ == "__main__":
    experiment = "Bubble"
    subprocess.run(["bash", "-c", f'find "{experiment}" -type f -name "*.jpg" -delete'], check=True)