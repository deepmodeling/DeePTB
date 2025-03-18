import subprocess

try:
    import torch
    print("The torch module has been successfully imported!")

    torch_version = torch.__version__
    print(f"Current torch version: {torch_version}")

    if torch.cuda.is_available():
        cuda_version = torch.version.cuda
    else:
        cuda_version = "cpu"
    print(f"Versions of CUDA used by PyTorch: {cuda_version}")

    try:
        import torch_scatter
        print("You have already installed torch-scatter!")

    except ImportError:
        url = f"https://data.pyg.org/whl/torch-{torch_version}.html"
        print(f"torch-scatter will be installed from {url}...")
        subprocess.run(["pip", "install", "torch-scatter==2.1.2", "-f", url], check=True)
        print("Installation complete, please re-run the program.")

except ImportError:
    print("The torch module is not found, please install PyTorch first.")
    print("You can install PyTorch with the following command (version range: 2.0.0-2.5.1) :")
    print("pip install \"torch>=2.0.0,<=2.5.1\"")