import subprocess

try:
    import torch
    print("torch 模块已成功导入！")

    torch_version = torch.__version__
    print(f"当前 torch 版本: {torch_version}")

    if torch.cuda.is_available():
        cuda_version = torch.version.cuda
    else:
        cuda_version = "cpu"
    print(f"PyTorch 使用的 CUDA 版本: {cuda_version}")

    try:
        import torch_scatter
        print("您已安装过torch-scatter！")

    except ImportError:
        url = f"https://data.pyg.org/whl/torch-{torch_version}.html"
        print(f"将从 {url} 安装 torch-scatter...")
        subprocess.run(["pip", "install", "torch-scatter==2.1.2", "-f", url], check=True)
        print("torch-scatter 安装完成，请重新运行程序。")

except ImportError:
    print("未找到 torch 模块，请先安装 PyTorch。")
    print("你可以通过以下命令安装 PyTorch（版本范围：2.0.0 - 2.5.1）：")
    print("pip install \"torch>=2.0.0,<=2.5.1\"")