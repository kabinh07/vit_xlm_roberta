import os, tarfile, math
from tqdm import tqdm

IMG_DIR = "images"
LBL_DIR = "labels"
OUT_DIR = "shards"
SAMPLES_PER_SHARD = 2000

os.makedirs(OUT_DIR, exist_ok=True)

files = sorted(os.listdir(IMG_DIR))
num_shards = math.ceil(len(files) / SAMPLES_PER_SHARD)

for shard_id in tqdm(range(num_shards)):
    start = shard_id * SAMPLES_PER_SHARD
    end = min((shard_id + 1) * SAMPLES_PER_SHARD, len(files))
    
    shard_path = f"{OUT_DIR}/shard-{shard_id:05d}.tar"
    
    with tarfile.open(shard_path, "w") as tar:
        for fname in files[start:end]:
            base = os.path.splitext(fname)[0]
            
            img_path = os.path.join(IMG_DIR, fname)
            lbl_path = os.path.join(LBL_DIR, base + ".txt")
            
            tar.add(img_path, arcname=f"{base}.jpg")
            tar.add(lbl_path, arcname=f"{base}.txt")
