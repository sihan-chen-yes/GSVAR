from huggingface_hub import hf_hub_download

repo_id = "qyp2000/VARSR"
filenames = ["VARSR.pth", "VARSR_C2I.pth", "VQVAE.pth"]

for filename in filenames:
    print(f"Downloading {filename}...")
    hf_hub_download(
        repo_id=repo_id,
        filename=filename,
        local_dir=".",
        local_dir_use_symlinks=False  
    )
print("All .pth files downloaded.")
