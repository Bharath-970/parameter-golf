import modal
import modal.mount
import os

app = modal.App("parameter-golf")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git")
    .pip_install(
        "torch",
        "numpy",
        "tiktoken",
        "sentencepiece",
        "huggingface_hub",
        "brotli",
        "zstandard",
        extra_index_url="https://download.pytorch.org/whl/cu124",
    )
)

volume = modal.Volume.from_name("parameter-golf-data", create_if_missing=True)

REPO_URL = os.environ.get("REPO_URL", "https://github.com/Bharath-970/parameter-golf.git")
BRANCH = os.environ.get("BRANCH", "main")
DATA_VARIANT = os.environ.get("DATA_VARIANT", "sp8192")
DATA_REPO_ID = os.environ.get("MATCHED_FINEWEB_REPO_ID", "kevclark/parameter-golf")
TRAIN_SHARDS = int(os.environ.get("TRAIN_SHARDS", "5"))
SUBMISSION = os.environ.get(
    "SUBMISSION",
    "records/track_10min_16mb/2026-04-28_SP8192_SwiGLU_WarmdownFix_QBatch/train_gpt.py",
)
N_GPUS = 8


def dataset_dir_for_variant(variant: str) -> str:
    if variant == "byte260":
        return "fineweb10B_byte260"
    if variant.startswith("sp") and variant[2:].isdigit():
        return f"fineweb10B_{variant}"
    raise ValueError(f"Unsupported DATA_VARIANT={variant!r}")


def tokenizer_filename_for_variant(variant: str) -> str:
    if variant.startswith("sp") and variant[2:].isdigit():
        return f"fineweb_{variant[2:]}_bpe.model"
    if variant == "byte260":
        return "fineweb_byte260.model"
    raise ValueError(f"Unsupported DATA_VARIANT={variant!r}")


@app.function(
    image=image,
    cpu=4,
    timeout=600,
    volumes={"/data": volume},
)
def check_data():
    """CPU-only check of what's cached in the volume. Costs ~$0.02."""
    import os
    base = "/data/datasets"
    if os.path.exists(base):
        for d in sorted(os.listdir(base)):
            path = os.path.join(base, d)
            files = os.listdir(path) if os.path.isdir(path) else []
            sizes = [os.path.getsize(os.path.join(path, f)) for f in files]
            total_gb = sum(sizes) / 1e9
            print(f"  {d}: {len(files)} files, {total_gb:.1f} GB")
    else:
        print("No datasets dir found")
    tok = "/data/tokenizers"
    if os.path.exists(tok):
        print("tokenizers:", os.listdir(tok))
    else:
        print("No tokenizers dir found")


@app.function(
    image=image,
    cpu=4,
    timeout=3600,
    volumes={"/data": volume},
)
def download_data():
    """Download dataset/tokenizer into the volume once. CPU-only, no GPU cost.
    Points HF cache at /data so downloaded files persist across runs."""
    import subprocess
    import sys
    import os

    dataset_dir = dataset_dir_for_variant(DATA_VARIANT)
    tokenizer_name = tokenizer_filename_for_variant(DATA_VARIANT)

    subprocess.run([
        "git", "clone", "--branch", BRANCH,
        REPO_URL,
        "/repo"
    ], check=True)

    os.chdir("/repo")

    # Point HF cache to the volume so it persists
    os.makedirs("/data/hf_cache", exist_ok=True)
    env = {
        **os.environ,
        "HOME": "/root",
        "HF_HOME": "/data/hf_cache",
        "HUGGINGFACE_HUB_CACHE": "/data/hf_cache",
    }

    # Delete stale manifest in case repo/variant changed.
    for stale in ["data/manifest.json", "data/datasets/manifest.json"]:
        if os.path.exists(stale):
            os.remove(stale)

    print(
        f"Downloading {DATA_VARIANT} data from {DATA_REPO_ID} "
        f"(train_shards={TRAIN_SHARDS}, takes ~5-15 min)..."
    )
    env["MATCHED_FINEWEB_REPO_ID"] = DATA_REPO_ID
    subprocess.run([
        sys.executable, "data/cached_challenge_fineweb.py",
        "--variant", DATA_VARIANT, "--train-shards", str(TRAIN_SHARDS),
    ], check=True, env=env)

    # Copy downloaded files into volume for direct access
    import shutil
    src_datasets = f"/repo/data/datasets/{dataset_dir}"
    src_tokenizer_dir = "/repo/data/tokenizers"
    dst_datasets = f"/data/datasets/{dataset_dir}"
    dst_tokenizers = "/data/tokenizers"

    if os.path.exists(src_datasets):
        print(f"Copying {len(os.listdir(src_datasets))} dataset files to volume...")
        os.makedirs(dst_datasets, exist_ok=True)
        for f in os.listdir(src_datasets):
            shutil.copy2(os.path.join(src_datasets, f), os.path.join(dst_datasets, f))

    if os.path.exists(src_tokenizer_dir):
        os.makedirs(dst_tokenizers, exist_ok=True)
        for f in os.listdir(src_tokenizer_dir):
            shutil.copy2(os.path.join(src_tokenizer_dir, f), os.path.join(dst_tokenizers, f))
        print(f"Copied tokenizers (including {tokenizer_name}): {os.listdir(dst_tokenizers)}")

    volume.commit()
    print("Done — data committed to volume.")


@app.function(
    image=image,
    gpu=f"H100:{N_GPUS}",
    timeout=3600,
    volumes={"/data": volume},
)
def train(seed: int = 42, submission_code: str = None, submission_json: str = None):
    import subprocess
    import sys
    import os

    dataset_dir = dataset_dir_for_variant(DATA_VARIANT)
    tokenizer_name = tokenizer_filename_for_variant(DATA_VARIANT)
    dataset_cache_dir = f"/data/datasets/{dataset_dir}"
    tokenizer_cache_path = f"/data/tokenizers/{tokenizer_name}"

    subprocess.run([
        "git", "clone", "--branch", BRANCH,
        REPO_URL,
        "/repo"
    ], check=True)

    os.chdir("/repo")

    # Overwrite/create submission files from local source if provided
    if submission_code:
        sub_path = os.path.join("/repo", SUBMISSION)
        os.makedirs(os.path.dirname(sub_path), exist_ok=True)
        with open(sub_path, "w") as f:
            f.write(submission_code)
    
    if submission_json:
        json_path = os.path.join(os.path.dirname(os.path.join("/repo", SUBMISSION)), "submission.json")
        with open(json_path, "w") as f:
            f.write(submission_json)

    # Wire up cached data from volume
    local_dataset_dir = f"data/datasets/{dataset_dir}"
    local_tokenizer_path = f"data/tokenizers/{tokenizer_name}"

    if os.path.exists(dataset_cache_dir) and len(os.listdir(dataset_cache_dir)) >= 3:
        print(f"{DATA_VARIANT} data cached ({len(os.listdir(dataset_cache_dir))} files), symlinking...")
        os.makedirs("data/datasets", exist_ok=True)
        os.makedirs("data/tokenizers", exist_ok=True)
        if os.path.lexists(local_dataset_dir):
            os.unlink(local_dataset_dir)
        os.symlink(dataset_cache_dir, local_dataset_dir)
        if os.path.exists(tokenizer_cache_path):
            if os.path.lexists(local_tokenizer_path):
                os.unlink(local_tokenizer_path)
            os.symlink(tokenizer_cache_path, local_tokenizer_path)
    else:
        print(f"{DATA_VARIANT} data not in volume - downloading now (adds ~5-15 min)...")
        for stale in ["data/manifest.json", "data/datasets/manifest.json"]:
            if os.path.exists(stale):
                os.remove(stale)
        subprocess.run([
            sys.executable, "data/cached_challenge_fineweb.py",
            "--variant", DATA_VARIANT, "--train-shards", str(TRAIN_SHARDS),
        ], check=True, env={
            **os.environ,
            "HOME": "/root",
            "MATCHED_FINEWEB_REPO_ID": DATA_REPO_ID,
        })

    vocab_size = int(DATA_VARIANT[2:]) if DATA_VARIANT.startswith("sp") and DATA_VARIANT[2:].isdigit() else 260
    env = {
        **os.environ,
        "HOME": "/root",
        "SEED": str(seed),
        "DATA_PATH": f"./data/datasets/{dataset_dir}",
        "TOKENIZER_PATH": f"./data/tokenizers/{tokenizer_name}",
        "VOCAB_SIZE": str(vocab_size),
    }

    result = subprocess.run([
        "torchrun", "--standalone", f"--nproc_per_node={N_GPUS}",
        SUBMISSION,
    ], capture_output=False, env=env)

    return result.returncode


@app.local_entrypoint()
def check():
    """Check what data is cached in the volume (~$0.02)."""
    print("Checking cached data...")
    check_data.remote()


@app.local_entrypoint()
def download():
    """Download variant data into the volume. CPU-only, run once before training."""
    print(f"Downloading {DATA_VARIANT} data into volume (CPU, no GPU cost)...")
    download_data.remote()
    print("Done. Run 'python3 -m modal run run_modal.py' to train.")


@app.local_entrypoint()
def main(seed: int = 42):
    """Run training on 8xH100. Usage: python3 -m modal run run_modal.py --seed 42"""
    # Read the submission files locally
    submission_path = SUBMISSION
    submission_code = None
    if os.path.exists(submission_path):
        with open(submission_path, "r") as f:
            submission_code = f.read()
            
    submission_json_path = os.path.join(os.path.dirname(submission_path), "submission.json")
    submission_json = None
    if os.path.exists(submission_json_path):
        with open(submission_json_path, "r") as f:
            submission_json = f.read()

    print(
        f"Launching seed={seed} on {N_GPUS}xH100 branch={BRANCH} "
        f"variant={DATA_VARIANT} submission={SUBMISSION}"
    )
    returncode = train.remote(
        seed=seed, 
        submission_code=submission_code, 
        submission_json=submission_json
    )
    print(f"Training finished with return code: {returncode}")
