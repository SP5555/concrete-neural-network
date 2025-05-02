import random
import kagglehub

path = kagglehub.dataset_download("maajdl/yeh-concret-data")
print("Path to dataset files:", path)

for i in range(10):
    print(f"Hello World_{random.randint(0,1000)}")
