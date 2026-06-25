# 依赖版本升级测试计划

本文档用于追踪 DeePTB 在本地 macOS 与 RTX 5090 上测试 Python、PyTorch、
PyG 和 `torch-scatter` 升级兼容性的工作。目标是沉淀一个可复用的矩阵测试
流程，并把最终支持策略落到 `install.sh`、`pyproject.toml` 和维护文档中。

## 基本原则

- 每个版本组合都放在独立的 uv 环境中测试，不污染当前 `.venv`。
- 当前 `pyproject.toml` 的 pin 只作为 baseline 对照；本轮目标是探索未来可支持的新版本组合，可以在矩阵里显式放宽旧 pin。
- `torch-scatter` 以及其他 C 扩展依赖优先使用预编译 binary wheel；源码编译不作为正式支持路径。
- 本地 Mac 只负责 CPU 安装、import、基础 smoke 测试；CUDA 和 RTX 5090 留到远程 GPU 机器验证。
- 每个 job 都输出 JSON 结果，最终在本文档里写清楚结论。
- 安装兼容、import 兼容、pytest 测试和性能测试分阶段做，方便定位失败原因。
- 第一阶段的安装/import 快筛不跑 pytest；通过快筛的组合必须进入后续 pytest 分层测试。

## 本轮落地决策

- `install.sh` 是官方推荐的保守安装通道：面向新机器，按 CPU/GPU、CUDA/driver
  和 Python 版本选择已经测试过的 Torch / PyG / `torch-scatter` 组合。
- `pyproject.toml` 是开发者兼容范围：允许开发者使用 `uv sync`、
  `pip install -e .` 或已有兼容 Torch 环境，但不负责表达 CUDA/driver
  相关的精确 wheel 选择。
- 正式支持 Python 3.10-3.13；Python 3.14 因 TorchScript 风险暂不纳入本轮。
- 正式依赖范围更新为：`torch>=2.5.1,<=2.12.1`、`torch-scatter==2.1.2`、
  `torch-geometric>=2.8.0`、`numpy>=1.26,<3`、`scipy>=1.12`、
  `h5py>=3.11`、`lmdb>=1.4.1`。
- 其他普通依赖不再使用无必要的精确 pin：`dargs`、`torch-runstats`、
  `opt-einsum`、`pyfiglet` 改为带 major 上限的兼容范围。
- `pytest` 和 `pytest-order` 默认随 DeePTB 安装，方便新机器安装后立即运行测试；兼容性矩阵不再单独维护另一套测试依赖。
- Python 3.13 的推荐安装通道额外使用现代 C 扩展约束：
  `lmdb>=2.2.1`、`h5py>=3.16.0`、`numpy>=2.5.0,<3`、`scipy>=1.18.0`。
- 依赖放宽后已补做解析和安装/import 快筛：`uv lock --dry-run` 通过；
  `py312-torch2121-cpu` 和 `py313-torch2121-cpu-modern-cdeps` 均通过实际安装、
  `torch_scatter.scatter_add` 和 DeePTB import。
- CI 单元测试必须测试新版安装路径，不能因旧 glibc 自动降级到 Torch 2.5.1。
  因此 `unit_test.yml` 不再使用 Ubuntu 20.04-based 旧容器，`ut.sh` 改为
  `bash install.sh cpu --extra pythtb`，不再二次执行 `uv sync --extra pythtb`。

## 拟测试矩阵

| Job | Python | Torch | PyG wheel 页面 | 目的 | 状态 |
| --- | --- | --- | --- | --- | --- |
| `py310-torch251-cpu` | 3.10 | 2.5.1 | `torch-2.5.0+cpu` | 当前 baseline | 通过 smoke |
| `py312-torch251-cpu` | 3.12 | 2.5.1 | `torch-2.5.0+cpu` | 当前声明范围的 Python 上限 | 通过 smoke |
| `py312-torch210-cpu` | 3.12 | 2.10.0 | `torch-2.10.0+cpu` | 主要 Torch 升级探针 | 通过 not slow |
| `py313-torch210-cpu` | 3.13 | 2.10.0 | `torch-2.10.0+cpu` | 主要 Python + Torch 升级探针 | 旧 C 扩展 pin 失败 |
| `py313-torch210-cpu-lmdb221` | 3.13 | 2.10.0 | `torch-2.10.0+cpu` | 旧 pin 阻塞定位：只放开 `lmdb` | `h5py` 旧 pin 失败 |
| `py313-torch210-cpu-modern-cdeps` | 3.13 | 2.10.0 | `torch-2.10.0+cpu` | Python 3.13 新版依赖主探索组合 | 通过 not slow |
| `py312-torch2120-cpu` | 3.12 | 2.12.0 | `torch-2.12.0+cpu` | Torch 2.12.0 最新线探索 | 通过安装/import |
| `py312-torch2121-cpu` | 3.12 | 2.12.1 | `torch-2.12.1+cpu` | Torch 2.12.1 最新线探索 | 通过安装/import |
| `py313-torch2120-cpu-modern-cdeps` | 3.13 | 2.12.0 | `torch-2.12.0+cpu` | Python 3.13 + Torch 2.12.0 最新线探索 | 通过安装/import |
| `py313-torch2121-cpu-modern-cdeps` | 3.13 | 2.12.1 | `torch-2.12.1+cpu` | Python 3.13 + Torch 2.12.1 最新线探索 | 通过 not slow |
| `py314-torch211-cpu` | 3.14 | 2.11.0 | `torch-2.11.0+cpu` | 前沿实验项 | 旧 C 扩展 pin 失败 |
| `py314-torch211-cpu-modern-cdeps` | 3.14 | 2.11.0 | `torch-2.11.0+cpu` | 前沿实验项 + 现代 C 扩展依赖 | 通过 smoke；不纳入正式支持 |

## 测试分层

| 层级 | 目的 | 适用组合 | 命令 |
| --- | --- | --- | --- |
| 安装/import 快筛 | 快速排除无法安装、无法 import、`torch-scatter` 非 binary wheel 的组合 | 所有 job | `python tools/compat/test_matrix.py --job <job-name> --no-tests` |
| smoke pytest | 验证核心 import、配置生成、模型构建、OrbitalMapper 等低成本路径 | 所有快筛通过的 job | `python tools/compat/test_matrix.py --job <job-name>` |
| regression pytest | 验证数值语义、tensor shape/order、兼容性敏感路径 | 主候选组合 | `python tools/compat/test_matrix.py --job <job-name> --test-command python -m pytest dptb/tests -m regression -q` |
| not slow pytest | 覆盖除慢训练/重型集成外的大多数本地测试 | 最终推荐组合 | `python tools/compat/test_matrix.py --job <job-name> --test-command python -m pytest dptb/tests -m "not slow" -q` |
| full pytest | 合并/发布前最终确认 | 最终推荐组合，最好在 CI 或干净机器上 | `python tools/compat/test_matrix.py --job <job-name> --test-command python -m pytest dptb/tests -q` |

## 分步执行清单

- [x] 检查矩阵测试工具是否能正常解析。
  - 命令：`python3 -m py_compile tools/compat/test_matrix.py`
  - 预期：脚本语法检查通过。
  - 结果：已通过，本地 Python 为 3.12.12。

- [x] 确认本地可用的 Python 版本。
  - 命令：`uv python list`
  - 预期：本地至少有 Python 3.10 和 3.12；需要时再用 `uv python install` 安装 3.13/3.14。
  - 结果：本地已有 Python 3.10.19、3.12.12、3.14.0；Python 3.13.9 可由 uv 下载。

- [x] 修正并验证 macOS 上的 binary wheel 判断逻辑。
  - 命令：从 baseline 环境读取 `torch-scatter` wheel metadata。
  - 预期：只要 wheel metadata 里有 `Root-Is-Purelib: false` 和平台 `Tag:`，即使版本号只是 `2.1.2`，也判定为 binary wheel。
  - 结果：baseline 里 `torch-scatter==2.1.2` 的 wheel metadata 为 `Root-Is-Purelib: false`，tag 为 `cp310-cp310-macosx_10_9_universal2`，已被判定为 binary wheel。

- [x] 运行 baseline 的第一阶段安装/import 快筛。
  - 命令：`python tools/compat/test_matrix.py --job py310-torch251-cpu --test-command ''`
  - 预期：环境创建成功；`torch`、`torch_geometric`、`torch_scatter` 和 DeePTB 可 import；`torch_scatter.scatter_add` 通过。
  - 结果：通过。实际版本为 Python 3.10.19、Torch 2.5.1、torch-geometric 2.8.0、torch-scatter 2.1.2 binary wheel。由于旧跳过参数未生效，此 job 同时完成了 smoke pytest。

- [x] 运行当前声明 Python 上限的第一阶段安装/import 快筛。
  - 命令：`python tools/compat/test_matrix.py --job py312-torch251-cpu --no-tests`
  - 预期：Python 3.12 + Torch 2.5.1 组合安装和 import 均通过。
  - 结果：通过。实际 Python 为 3.12.12，超过当前 `<=3.12.9` 上限，uv 给出 requires-python warning；安装、import、`torch_scatter.scatter_add` 均通过。实际版本为 Torch 2.5.1、torch-geometric 2.8.0、torch-scatter 2.1.2 cp312 binary wheel。另有非阻塞 `SyntaxWarning: invalid escape sequence '\l'`。

- [x] 运行主要 Torch 升级组合的第一阶段安装/import 快筛。
  - 命令：`python tools/compat/test_matrix.py --job py312-torch210-cpu --no-tests`
  - 预期：Torch 2.10 能安装，并且 `torch-scatter` 来自匹配的 binary wheel。
  - 结果：通过。实际版本为 Python 3.12.12、Torch 2.10.0、torch-geometric 2.8.0、torch-scatter 2.1.2 cp312 macOS binary wheel；`torch_scatter.scatter_add` 和 DeePTB import 均通过。uv 仍提示 Python 3.12.12 超过当前 `<=3.12.9` 上限。

- [x] 运行主要 Python + Torch 升级组合的第一阶段安装/import 快筛。
  - 命令：`python tools/compat/test_matrix.py --job py313-torch210-cpu --no-tests`
  - 预期：Python 3.13 + Torch 2.10 组合安装和 import 均通过。
  - 结果：失败。失败点不是 Torch、PyG 或 `torch-scatter`，而是当前固定依赖 `lmdb==1.4.1` 在 Python 3.13.9 上需要源码构建并失败，报错为 `PyObject_AsReadBuffer` 未声明。结论：Python 3.13 支持需要先评估并升级 `lmdb` 版本。

- [x] 运行 Python 3.13 + Torch 2.10 + `lmdb==2.2.1` 的依赖 pin 探索。
  - 命令：`python tools/compat/test_matrix.py --job py313-torch210-cpu-lmdb221 --no-tests`
  - 预期：如果只被 `lmdb==1.4.1` 卡住，升级到 `lmdb==2.2.1` 后应能继续安装和 import。
  - 结果：失败。`lmdb` 阻塞解除后，新的失败点是 `h5py==3.11.0` 在 Python 3.13 上源码构建失败。结论：Python 3.13 还需要放开 `h5py`/`scipy`/NumPy 等 C 扩展依赖范围。

- [x] 运行 Python 3.13 + Torch 2.10 + 现代 C 扩展依赖探索。
  - 命令：`python tools/compat/test_matrix.py --job py313-torch210-cpu-modern-cdeps --no-tests`
  - 预期：使用 `lmdb==2.2.1`、`h5py==3.16.0`、`numpy==2.5.0`、`scipy==1.18.0` 后，安装和 import 继续推进。
  - 结果：通过。实际版本为 Python 3.13.9、Torch 2.10.0、torch-geometric 2.8.0、torch-scatter 2.1.2 cp313 binary wheel、lmdb 2.2.1、h5py 3.16.0、numpy 2.5.0、scipy 1.18.0；`torch_scatter.scatter_add` 和 DeePTB import 均通过。仍有非阻塞 `SyntaxWarning: invalid escape sequence '\l'`。

- [x] 运行前沿实验组合的第一阶段安装/import 快筛。
  - 命令：`python tools/compat/test_matrix.py --job py314-torch211-cpu --no-tests`
  - 预期：只作为实验结果记录；失败不阻塞主线升级。
  - 结果：失败。失败点为 `lmdb==1.4.1` 在 Python 3.14.0 上源码构建失败，错误同样是 `PyObject_AsReadBuffer` 未声明。解析阶段已找到 `torch-scatter` cp314 binary wheel，因此前沿组合的第一阻塞仍是旧 C 扩展 pin。

- [x] 运行前沿实验组合 + 现代 C 扩展依赖的第一阶段快筛。
  - 命令：`python tools/compat/test_matrix.py --job py314-torch211-cpu-modern-cdeps --no-tests`
  - 预期：只作为实验结果记录；判断 Python 3.14 + Torch 2.11 在现代 C 扩展依赖下能否安装和 import。
  - 结果：通过。实际版本为 Python 3.14.0、Torch 2.11.0、torch-geometric 2.8.0、torch-scatter 2.1.2 cp314 binary wheel、lmdb 2.2.1、h5py 3.16.0、numpy 2.5.0、scipy 1.18.0；`torch_scatter.scatter_add` 和 DeePTB import 均通过。Python 3.14 下同一个 `\lambda` docstring warning 更明确提示未来会出问题。

- [x] 运行 Torch 2.12 系列第一阶段安装/import 快筛。
  - 命令：
    - `python tools/compat/test_matrix.py --job py312-torch2120-cpu --no-tests`
    - `python tools/compat/test_matrix.py --job py312-torch2121-cpu --no-tests`
    - `python tools/compat/test_matrix.py --job py313-torch2120-cpu-modern-cdeps --no-tests`
    - `python tools/compat/test_matrix.py --job py313-torch2121-cpu-modern-cdeps --no-tests`
  - 预期：Torch 2.12.0/2.12.1 与对应 `torch-scatter` binary wheel 可以安装和 import。
  - 结果：四个组合均通过安装/import 和 `torch_scatter.scatter_add` 快筛。macOS 上 `torch-scatter==2.1.2` 使用 cp312/cp313 binary wheel；Linux/CUDA 后续仍需在 5090 上验证 `cu130/cu132`。

- [x] 对所有第一阶段快筛通过的 job 运行 smoke pytest。
  - 命令：`python tools/compat/test_matrix.py --job <job-name>`
  - 预期：`python -m pytest dptb/tests -m smoke -q` 通过。
  - 结果：`py310-torch251-cpu` 已通过，28 passed、3 skipped、478 deselected。`py312-torch251-cpu` 已通过，28 passed、3 skipped、478 deselected。`py312-torch210-cpu` 已通过，28 passed、3 skipped、478 deselected，并出现 Torch 2.10 的 `torch.jit.script` deprecation warning。`py313-torch210-cpu-modern-cdeps` 已通过，28 passed、3 skipped、478 deselected，警告主要来自 Torch JIT、ASE + NumPy 2.5、PyG inspector。`py314-torch211-cpu-modern-cdeps` 已通过，28 passed、3 skipped、478 deselected；Torch 明确警告 `torch.jit.script` 在 Python 3.14+ 不支持且可能会坏，因此 3.14 暂只作为实验项。`py313-torch2121-cpu-modern-cdeps` 已通过，28 passed、3 skipped、478 deselected。

- [x] 对主候选组合运行 regression pytest。
  - 命令：`python tools/compat/test_matrix.py --job <job-name> --test-command python -m pytest dptb/tests -m regression -q`
  - 预期：主候选组合的回归测试通过。
  - 结果：`py312-torch210-cpu` 已通过，29 passed、4 skipped、476 deselected。`py313-torch210-cpu-modern-cdeps` 已通过，29 passed、4 skipped、476 deselected。`py313-torch2121-cpu-modern-cdeps` 已通过，29 passed、4 skipped、476 deselected。

- [x] 对最终推荐组合运行 not slow pytest。
  - 命令：`python tools/compat/test_matrix.py --job <job-name> --test-command python -m pytest dptb/tests -m "not slow" -q`
  - 预期：最终推荐组合的非慢速测试通过。
  - 结果：`py312-torch210-cpu` 已通过，445 passed、30 skipped、34 deselected、58 warnings，用时 227.65s。主要警告来自 Torch JIT deprecation、nested tensor prototype、spglib error handling deprecation、occupy.py 的 log 数值 warning、非 writable NumPy array 转 tensor。`py313-torch210-cpu-modern-cdeps` 已通过，445 passed、30 skipped、34 deselected、248 warnings，用时 148.86s；新增/增加的警告主要来自 ASE 与 NumPy 2.5 的 deprecated shape/copy 行为、PyG inspector 对 Python 3.15 的未来兼容警告。`py313-torch2121-cpu-modern-cdeps` 已通过，445 passed、30 skipped、34 deselected、248 warnings，用时 98.00s。

- [x] 放宽剩余 runtime pin 后补做依赖解析与安装/import 快筛。
  - 命令：
    - `uv lock --dry-run`
    - `python tools/compat/test_matrix.py --job py312-torch2121-cpu --job py313-torch2121-cpu-modern-cdeps --no-tests --results-dir tools/compat/results_depcheck`
  - 结果：通过。实际解析到 `dargs 0.4.10`、`opt-einsum 3.4.0`、`pyfiglet 1.0.4`、`pytest 9.1.1`、`pytest-order 1.5.0`；`torch-scatter 2.1.2` 仍为 macOS binary wheel。该补测同时发现并修复了矩阵脚本对 CPU job 错误生成 `torch-scatter==2.1.2+pt212cpu` 的问题，CPU job 现在固定使用 plain `2.1.2`。

- [x] 放宽剩余 runtime pin 后补跑主组合 `not slow`。
  - 命令：`python tools/compat/test_matrix.py --job py313-torch2121-cpu-modern-cdeps --skip-install --results-dir tools/compat/results_depcheck --test-command python -m pytest dptb/tests -m "not slow" -q`
  - 结果：通过。`445 passed, 30 skipped, 34 deselected, 249 warnings in 109.67s`。主要 warning 仍是 TorchScript deprecation、ASE/NumPy 2.5 deprecation、PyG inspector 对 Python 3.15 的未来兼容警告。

## 当前本地结论

- 本地 macOS CPU 阶段，`Python 3.12.12 + Torch 2.10.0 + torch-geometric 2.8.0 + torch-scatter 2.1.2 binary wheel` 已通过安装/import、smoke、regression、not slow。
- `Python 3.13.9 + Torch 2.10.0 + torch-geometric 2.8.0 + torch-scatter 2.1.2 binary wheel` 也可行，但需要放宽旧 C 扩展依赖：当前验证组合为 `lmdb 2.2.1`、`h5py 3.16.0`、`numpy 2.5.0`、`scipy 1.18.0`，已通过安装/import、smoke、regression、not slow。
- 当前旧 pin 下，Python 3.13/3.14 会被 `lmdb==1.4.1` 和 `h5py==3.11.0` 卡住，不应直接把 Python 上限放到 3.13+ 而不更新这些依赖。
- `Python 3.14.0 + Torch 2.11.0 + 现代 C 扩展依赖` 已通过安装/import 和 smoke，但 Torch 明确警告 `torch.jit.script` 在 Python 3.14+ 不支持且可能会坏，因此暂时只作为实验项，不建议纳入正式支持范围。
- Torch 2.12.0/2.12.1 已补测安装/import 快筛，`Python 3.13.9 + Torch 2.12.1 + 现代 C 扩展依赖` 进一步通过 smoke、regression、not slow。因此本地 Mac 阶段可把 Torch 候选上限提升到 2.12.1。
- RTX 5090 验证见下一节；GPU 路径受 driver 和 PyG binary wheel 共同约束，不能直接沿用 macOS CPU 的 Torch 2.12.1 结论。

## RTX 5090 结论

- 远程机器：`node221`，driver `570.211.01`，`nvidia-smi` 显示 CUDA 12.8 能力。
- `torch 2.10.0+cu128` 有匹配的 `torch-scatter 2.1.2+pt210cu128` binary wheel，已验证 CUDA 可用。
- `torch 2.12.1+cu130` 和 `torch-scatter 2.1.2+pt212cu130` 可以安装到 binary wheel，但当前 driver 太旧，`torch.cuda.is_available()` 为 false，并提示需要更新 NVIDIA driver 或安装与当前 driver 匹配的 PyTorch。
- `torch 2.12.x+cu128` 没有对应 `torch-scatter` binary wheel，因此不能作为当前 5090 的正式支持路径。
- `gpu-py312-torch210-cu128` 已通过 install/import、`torch_scatter.scatter_add`、smoke、regression、修复后的 `not slow`。最终 `not slow` 结果：444 passed、31 skipped、34 deselected、58 warnings，用时 676.37s。
- `gpu-py313-torch210-cu128-modern-cdeps` 已通过 install/import、`torch_scatter.scatter_add`、smoke、regression、修复后的 `not slow`。最终 `not slow` 结果：444 passed、31 skipped、34 deselected、249 warnings，用时 742.96s。
- 5090 当前建议支持组合：`Python 3.12/3.13 + Torch 2.10.0+cu128 + torch-geometric 2.8.0 + torch-scatter 2.1.2+pt210cu128`。
- FEAST/MKL native solver 已从默认兼容性验证中隔离：默认跳过，只有设置 `DPTB_TEST_MKL_FEAST=1` 才运行。该问题应单独开任务排查 `ctypes`/MKL ABI，不作为本轮 Torch/PyG/CUDA 兼容性阻塞。

## 支持范围决策

本轮依赖升级的正式目标定到 Python 3.13，不把 Python 3.14 纳入正式支持范围。

建议支持层级：

| 层级 | Python | Torch | 依赖策略 | 说明 |
| --- | --- | --- | --- | --- |
| 当前稳定线 | 3.10-3.12 | 2.5.1 | 保持旧 C 扩展 pin | 对照当前发布环境 |
| CPU 新主线候选 | 3.12 | 2.12.1 | 可继续使用 NumPy 1.26 / SciPy 1.12 / h5py 3.11 / lmdb 1.4.1 | macOS CPU install/import 已通过；2.10.0 已通过 not slow |
| CPU 新扩展候选 | 3.13 | 2.12.1 | 需要现代 C 扩展依赖：lmdb 2.2.1、h5py 3.16、NumPy 2.5、SciPy 1.18 | macOS CPU not slow 已通过 |
| 5090 GPU 当前候选 | 3.12-3.13 | 2.10.0+cu128 | 3.13 需要现代 C 扩展依赖；必须使用 `torch-scatter 2.1.2+pt210cu128` binary wheel | driver 570/CUDA 12.8 下已通过 not slow |
| 实验项 | 3.14 | 2.11.0 | 需要现代 C 扩展依赖，并且要处理 TorchScript | 不进入本轮正式支持 |

`pyproject.toml` 已按本轮结论更新为开发者兼容范围。注意：这里表达的是源码层面的依赖范围，不表达 CUDA/driver 相关的精确 wheel 选择。

1. Python / Torch / PyG 范围：

   ```toml
   requires-python = ">=3.10,<3.14"
   dependencies = [
       "torch>=2.5.1,<=2.12.1",
       "torch_scatter==2.1.2",
       "torch_geometric>=2.8.0",
   ]
   ```

2. C 扩展依赖范围：

   ```toml
   dependencies = [
       "numpy>=1.26,<3",
       "scipy>=1.12",
       "h5py>=3.11",
       "lmdb>=1.4.1",
   ]
   ```

   Python 3.13 在 `install.sh` 推荐通道中会进一步收紧到已验证的现代 C 扩展下限，避免落回源码构建。

## 后续任务

- [x] 总结本地兼容性结论。
  - 输出：在本文档中写明每个组合的 pass/fail、失败原因，以及建议的 `pyproject.toml` 版本范围。
  - 结果：已完成，见“当前本地结论”和“支持范围决策”。

- [x] 在 RTX 5090 上验证 `Python 3.12/3.13 + Torch 2.10.0+cu128 + torch-scatter binary wheel`。
- [x] 准备 RTX 5090 后续测试矩阵。
  - 输出：拿到 SSH 访问后，添加使用 `cu128` 或更新 CUDA wheel 页面的 GPU jobs。
  - 结果：已添加并验证 `gpu-py312-torch210-cu128`、`gpu-py313-torch210-cu128-modern-cdeps`。`cu130` 安装可解析到 binary wheel，但当前 driver 不足以启用 CUDA；`cu132` 当前不是 uv/PyTorch 可选 Torch backend，留待上游支持后再测。

- [ ] 单独创建 FEAST/MKL native solver 修复任务。
  - 现象：5090 上系统 MKL 2020.4 可加载且 FEAST 符号存在，但 `dfeast_scsrev` native 调用会 segfault。
  - 范围：审计 `dptb/utils/feast_wrapper.py` 的 `ctypes` ABI、CSR 输入格式、integer width、MKL runtime 来源和线程库冲突。
  - 当前处理：默认跳过 FEAST native tests；显式设置 `DPTB_TEST_MKL_FEAST=1` 时才运行。

- [ ] 为 Python 3.14 单独创建 TorchScript 迁移任务。
  - 清点 `@torch.jit.script` 用法。
  - 设计 `maybe_script` 或 eager fallback。
  - 评估 `torch.compile` / `torch.export` 是否适合性能敏感路径。
  - 跑 full tests 和性能 benchmark 后再决定是否支持 Python 3.14。

## 预检查记录

一次被中断的 baseline 尝试已经创建了 `.compat-envs/py310-torch251-cpu`
临时环境和一份 partial result。该结果提示：macOS 上
`torch-scatter==2.1.2` 可以是 binary wheel，但版本号不一定带
`+pt...` local suffix。因此，矩阵工具应该依据 wheel metadata 判断
是否为 binary wheel，而不是只看 package version。
