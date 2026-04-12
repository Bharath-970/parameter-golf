import modal
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
        "zstandard",
        extra_index_url="https://download.pytorch.org/whl/cu124",
    )
)

volume = modal.Volume.from_name("parameter-golf-data", create_if_missing=True)

BRANCH = "sp4096-depth-recur-parallel-resid"
SUBMISSION = "records/track_10min_16mb/2026-04-12_SP4096_DepthRecur_ParallelResid_MuonEqR/train_gpt.py"
SEED = int(os.environ.get("SEED", "42"))
N_GPUS = 8

@app.function(
    image=image,
    gpu=modal.gpu.H100(count=N_GPUS),
    timeout=3600,
    volumes={"/data": volume},
)
def train(seed: int = 42):
    import subprocess
    import sys
    import os

    # Clone repo on the submission branch
    subprocess.run([
        "git", "clone", "--branch", BRANCH,
        "https://github.com/Bharath-970/parameter-golf.git",
        "/repo"
    ], check=True)

    os.chdir("/repo")

    # Download SP4096 data: tokenizer + val shards + 5 train shards
    subprocess.run([
        sys.executable, "data/cached_challenge_fineweb.py",
        "--variant", "sp4096", "--train-shards", "5"
    ], check=True, env={**os.environ, "HOME": "/root"})

    env = {
        **os.environ,
        "HOME": "/root",
        "SEED": str(seed),
        "DATA_PATH": "./data/datasets/fineweb10B_sp4096",
        "TOKENIZER_PATH": "./data/tokenizers/fineweb_4096_bpe.model",
    }

    # Run training on 8×H100
    result = subprocess.run([
        "torchrun", "--standalone", f"--nproc_per_node={N_GPUS}",
        SUBMISSION,
    ], capture_output=False, env=env)

    return result.returncode


@app.local_entrypoint()
def main():
    import sys
    seed = int(sys.argv[1]) if len(sys.argv) > 1 else 42
    print(f"Launching seed={seed} on {N_GPUS}×H100, branch={BRANCH}")
    returncode = train.remote(seed=seed)
    print(f"Training finished with return code: {returncode}")
