import subprocess

if __name__ == "__main__":
    experiment = "Processed_bubble_unpadded"
    subprocess.run(["bash", "-c", f'find "{experiment}" -type f -name "*.jpg" -delete'], check=True)
    subprocess.run(["bash", "-c", f'find "{experiment}" -type f -name "*.txt" -delete'], check=True)


    experiment = "Bubble"
    subprocess.run(["bash", "-c", f'find "{experiment}" -type f -name "*.jpg" -delete'], check=True)