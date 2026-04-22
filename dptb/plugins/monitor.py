import logging
import time
import collections
import re

import torch
from dptb.data import AtomicData
from dptb.plugins.base_plugin import Plugin
from torch.utils.tensorboard import SummaryWriter

log = logging.getLogger(__name__)

import csv
import os
from dptb.nn.norm import SeperableLayerNorm
import math
import torch.nn as nn

# 确保导入你的 SO2_Linear 类定义，以便 isinstance 判断
from dptb.nn.tensor_product import SO2_Linear
try:
    from dptb.nn.tensor_product_moe_v3 import SO2_Linear as MOE_SO2_Linear
except Exception:
    MOE_SO2_Linear = None

try:
    from dptb.nn.embedding.eqv3_grid_helpers import EqV3StyleNodeFFN, FlatSwiGLUS2Merge
except Exception:
    EqV3StyleNodeFFN = None
    FlatSwiGLUS2Merge = None


def _is_so2_linear_module(module):
    so2_types = tuple(
        cls for cls in (SO2_Linear, MOE_SO2_Linear)
        if isinstance(cls, type)
    )
    return isinstance(module, so2_types) or module.__class__.__name__ == "SO2_Linear"


def _is_eqv3_ffn_module(module):
    return (
        (isinstance(EqV3StyleNodeFFN, type) and isinstance(module, EqV3StyleNodeFFN))
        or module.__class__.__name__ == "EqV3StyleNodeFFN"
    )


def _is_s2_activation_module(module):
    return (
        (isinstance(FlatSwiGLUS2Merge, type) and isinstance(module, FlatSwiGLUS2Merge))
        or module.__class__.__name__ == "FlatSwiGLUS2Merge"
    )


class PreTPBlockMonitor(Plugin):
    def __init__(self, log_dir="monitor_logs", buffer_size=50):
        super(PreTPBlockMonitor, self).__init__([(1, 'iteration')])
        self.log_dir = log_dir
        self.buffer_size = buffer_size
        self.buffer = []

        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)

        self.csv_path = os.path.join(self.log_dir, "pre_tp_decay_analysis.csv")

        # 监控链条：
        # Grad Loss <--- Linear_Post <--- Activation <--- TP <--- [Concatenation] <--- SLN <--- Previous Layer
        # 我们重点看 Backward 梯度流向 (从左向右看)

        self.header = [
            'iter', 'block_name', 'component',
            'grad_in_std',  # 该组件输出端接收到的梯度 (Upstream)
            'grad_out_std',  # 该组件输入端传出的梯度 (Downstream, 向更浅层传)
            'decay_ratio'  # 衰减率 (Out/In), < 0.1 说明此处阻断
        ]

        with open(self.csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(self.header)

    def register(self, trainer):
        self.trainer = trainer
        self._register_hooks()
        log.info(f"🕵️ [Pre-TP Monitor] 已启动 | 正在追踪 TP 前序模块的梯度衰减情况")

    def _analyze_std(self, t):
        if t is None: return 0.0
        if isinstance(t, (tuple, list)):
            t = t[0] if len(t) > 0 else None
        if t is None or not isinstance(t, torch.Tensor): return 0.0
        if t.numel() == 0: return 0.0
        return t.detach().std().item()

    def _register_hooks(self):
        # 我们手动遍历模型，找到 UpdateNode 和 UpdateEdge 里的关键子模块
        count = 0
        for name, module in self.trainer.model.named_modules():

            # 识别 UpdateNode 和 UpdateEdge
            if "UpdateNode" in module.__class__.__name__ or "UpdateEdge" in module.__class__.__name__:

                # 1. 监控 SLN (TP 的输入归一化)
                if hasattr(module, 'sln'):
                    module.sln.register_full_backward_hook(self._make_hook(name, "SLN_Node"))
                if hasattr(module, 'sln_e'):
                    module.sln_e.register_full_backward_hook(self._make_hook(name, "SLN_Edge"))

                # 2. 监控 Activation (Gate)
                if hasattr(module, 'activation'):
                    module.activation.register_full_backward_hook(self._make_hook(name, "Gate_Act"))

                # 3. 监控 Linear Post (TP 后置线性层)
                if hasattr(module, 'lin_post'):
                    module.lin_post.register_full_backward_hook(self._make_hook(name, "Linear_Post"))

                # 4. 监控 Residual Linear (旁路)
                if hasattr(module, 'linear_res'):
                    module.linear_res.register_full_backward_hook(self._make_hook(name, "Residual_Linear"))

                count += 1

        if count == 0:
            log.warning("⚠️ 未找到 UpdateNode/UpdateEdge 模块，请检查命名。")
        else:
            log.info(f"✅ 已植入探针到 {count} 个 Update 模块内部。")

    def _make_hook(self, block_name, component_name):
        def hook(module, grad_input, grad_output):
            # grad_input: 传给该层输入的 (Downstream, 往 Input 方向)
            # grad_output: 该层输出接收到的 (Upstream, 往 Loss 方向)

            g_in = grad_input[0] if grad_input else None  # 这是我们要看的“传出去的梯度”
            g_out = grad_output[0] if grad_output else None  # 这是“接收到的梯度”

            # 临时保存，iteration 结束统一写
            if not hasattr(module, '_mon_stats'): module._mon_stats = {}

            # 我们这里只取 Std，因为我们只关心幅度
            module._mon_stats['grad_out'] = self._analyze_std(g_in)  # 注意 PyTorch hook 定义：grad_input 是关于输入的导数
            module._mon_stats['grad_in'] = self._analyze_std(g_out)  # grad_output 是关于输出的导数

            # 记录元数据
            module._mon_meta = (block_name, component_name)

        return hook

    def iteration(self, **kwargs):
        current_iter = self.trainer.iter

        for name, module in self.trainer.model.named_modules():
            # 检查是否有我们的监控数据
            if hasattr(module, '_mon_stats') and module._mon_stats:
                stats = module._mon_stats
                meta = getattr(module, '_mon_meta', ("Unknown", "Unknown"))

                g_in = stats.get('grad_in', 0.0)  # 接收到的 (来自 Loss)
                g_out = stats.get('grad_out', 0.0)  # 传出去的 (去往 Input)

                # Decay Ratio = 传出 / 接收
                # 理想情况 ~ 1.0
                # 如果 < 0.1 说明这个组件把梯度吃掉了
                decay = g_out / (g_in + 1e-12)

                # 过滤掉完全没梯度的
                if g_in > 1e-12:
                    self.buffer.append([
                        current_iter,
                        meta[0],  # Block Name (e.g., layers.2.node_update)
                        meta[1],  # Component (e.g., Gate_Act)
                        f"{g_in:.2e}",
                        f"{g_out:.2e}",
                        f"{decay:.3f}"
                    ])

                # 清空
                module._mon_stats = {}

        if len(self.buffer) >= self.buffer_size:
            self._flush()

    def _flush(self):
        if not self.buffer: return
        try:
            with open(self.csv_path, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerows(self.buffer)
            self.buffer = []
        except Exception as e:
            log.warning(f"PreTP Monitor Write Failed: {e}")



class SO2ModuleMonitor(Plugin):
    def __init__(self, log_dir="monitor_logs", buffer_size=50):
        super(SO2ModuleMonitor, self).__init__([(1, 'iteration')])
        self.log_dir = log_dir
        self.buffer_size = buffer_size
        self.buffer = []

        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)

        self.csv_path = os.path.join(self.log_dir, "so2_detailed_analysis.csv")

        # 表头说明：
        # [Forward]
        # fwd_in: 输入信号强度 (Std)
        # fwd_out: 输出信号强度 (Std)
        # fwd_gain: 信号放大倍数 (Out/In) -> 理想值 ~1.0
        #
        # [Backward]
        # bwd_in:  反向传给上一层的梯度 (Downstream Grad)
        # bwd_out: 反向接收到的梯度 (Upstream Grad)
        # bwd_gain: 梯度穿透率 (In/Out) -> 理想值 ~1.0，过低(<0.1)为阻断，过高(>5.0)为爆炸
        #
        # [Status]
        # blockage: 自动判定状态 (Normal, Fwd_Dead, Bwd_Block, Explode)

        self.header = [
            'iter', 'layer_name',
            'fwd_in', 'fwd_out', 'fwd_gain',
            'bwd_in', 'bwd_out', 'bwd_gain',
            'weight_norm', 'grad_weight_norm', 'update_ratio',
            'status'
        ]

        with open(self.csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(self.header)

    def register(self, trainer):
        self.trainer = trainer
        self._register_hooks()
        log.info(f"🔬 [SO2 Monitor V2] 深度显微镜已启动 | 双向(Forward+Backward)信号监控")

    def _analyze_std(self, t):
        if t is None: return 0.0
        if isinstance(t, (tuple, list)):
            t = t[0] if len(t) > 0 else None
        if t is None or not isinstance(t, torch.Tensor): return 0.0
        if t.numel() == 0: return 0.0
        return t.detach().std().item()

    def _register_hooks(self):
        count = 0
        for name, module in self.trainer.model.named_modules():
            if _is_so2_linear_module(module):
                # 注册双向 Hook
                module.register_forward_hook(self._make_forward_hook(name))
                module.register_full_backward_hook(self._make_backward_hook(name))

                module._is_monitored_so2 = True
                module._so2_stats = {}  # 初始化存储
                count += 1

        if count == 0:
            log.warning("⚠️ [SO2 Monitor] 未找到 SO2_Linear 模块！")
        else:
            log.info(f"✅ [SO2 Monitor] 已锁定 {count} 个核心模块，开始双向追踪。")

    def _make_forward_hook(self, name):
        def hook(module, inputs, output):
            # SO2_Linear forward(x, r, ...)
            # 我们主要关注第一个输入 x (通常是 feature)
            x = inputs[0] if inputs else None
            y = output[0] if isinstance(output, tuple) else output

            # 初始化 stats 字典 (防止backward先跑的极端情况，虽不可能)
            if not hasattr(module, '_so2_stats'): module._so2_stats = {}

            module._so2_stats['fwd_in'] = self._analyze_std(x)
            module._so2_stats['fwd_out'] = self._analyze_std(y)

        return hook

    def _make_backward_hook(self, name):
        def hook(module, grad_input, grad_output):
            # grad_input: 传给上一层 (Downstream)
            # grad_output: 来自下一层 (Upstream)
            g_in = grad_input[0] if grad_input else None
            g_out = grad_output[0] if grad_output else None

            if not hasattr(module, '_so2_stats'): module._so2_stats = {}

            module._so2_stats['bwd_in'] = self._analyze_std(g_in)
            module._so2_stats['bwd_out'] = self._analyze_std(g_out)

        return hook

    def _diagnose(self, fwd_in, fwd_out, bwd_in, bwd_out):
        """自动诊断状态"""
        eps = 1e-9
        status = []

        # 1. Forward Check
        if fwd_in > 1e-6 and fwd_out < 1e-9:
            status.append("FWD_DEAD")  # 正向传播猝死
        elif fwd_out > 1e3:
            status.append("FWD_EXPLODE")  # 正向发散

        # 2. Backward Check
        bwd_gain = bwd_in / (bwd_out + eps)

        if bwd_out > 1e-9:
            if bwd_gain < 0.1:
                status.append("BWD_BLOCK")  # 梯度阻断 (Gain < 10%)
            elif bwd_gain > 5.0:
                status.append("BWD_EXPLODE")  # 梯度爆炸 (Gain > 500%)
        elif bwd_out < 1e-9 and bwd_in < 1e-9:
            status.append("GRAD_VANISHED")  # 梯度已消失

        if not status:
            return "NORMAL"
        return "|".join(status)

    def iteration(self, **kwargs):
        current_iter = self.trainer.iter

        for name, module in self.trainer.model.named_modules():
            if getattr(module, '_is_monitored_so2', False):
                stats = getattr(module, '_so2_stats', {})

                # 获取四项核心指标
                f_in = stats.get('fwd_in', 0.0)
                f_out = stats.get('fwd_out', 0.0)
                b_in = stats.get('bwd_in', 0.0)
                b_out = stats.get('bwd_out', 0.0)

                # 计算 Gains
                f_gain = f_out / (f_in + 1e-9)
                b_gain = b_in / (b_out + 1e-9)

                # 计算 Weight Stats
                total_w_norm_sq = 0.0
                total_g_norm_sq = 0.0
                has_grad = False

                for param in module.parameters():
                    if param.requires_grad:
                        total_w_norm_sq += param.data.norm().item() ** 2
                        if param.grad is not None:
                            total_g_norm_sq += param.grad.norm().item() ** 2
                            has_grad = True

                w_norm = math.sqrt(total_w_norm_sq)
                g_norm = math.sqrt(total_g_norm_sq)
                up_ratio = g_norm / (w_norm + 1e-9)

                # 自动诊断
                status = self._diagnose(f_in, f_out, b_in, b_out)

                # 记录 (只要 Forward 活着或者有梯度就记录)
                if f_in > 0 or has_grad:
                    self.buffer.append([
                        current_iter, name,
                        f"{f_in:.2e}", f"{f_out:.2e}", f"{f_gain:.2f}",
                        f"{b_in:.2e}", f"{b_out:.2e}", f"{b_gain:.2f}",
                        f"{w_norm:.2e}", f"{g_norm:.2e}", f"{up_ratio:.2e}",
                        status
                    ])

                # 清空
                module._so2_stats = {}

        if len(self.buffer) >= self.buffer_size:
            self._flush()

    def _flush(self):
        if not self.buffer: return
        try:
            with open(self.csv_path, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerows(self.buffer)
            self.buffer = []
        except Exception as e:
            log.warning(f"Monitor Write Failed: {e}")


class CUDAModuleMemoryMonitor(Plugin):
    """
    Record CUDA allocator snapshots around selected module forward/backward hooks.

    This does not reset CUDA peak counters. It is meant to complement the regular
    iteration peak monitor: after the trainer resets the window peak, rows here
    show which SO2/S2-related module first observes a new global peak.
    """

    def __init__(self, log_dir="monitor_logs", buffer_size=20, cuda_sync=False):
        super(CUDAModuleMemoryMonitor, self).__init__([(1, 'iteration')])
        self.log_dir = log_dir
        self.buffer_size = int(buffer_size)
        self.cuda_sync = bool(cuda_sync)
        self.buffer = []

        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)

        self.csv_path = os.path.join(self.log_dir, "cuda_module_memory.csv")
        self.header = [
            "iter", "module_name", "module_kind", "phase",
            "alloc_before_mb", "alloc_after_mb", "alloc_delta_mb",
            "reserved_before_mb", "reserved_after_mb", "reserved_delta_mb",
            "peak_alloc_before_mb", "peak_alloc_after_mb", "peak_alloc_delta_mb",
            "peak_reserved_before_mb", "peak_reserved_after_mb", "peak_reserved_delta_mb",
            "global_peak_crossed",
        ]
        with open(self.csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(self.header)

    def register(self, trainer):
        self.trainer = trainer
        self._register_hooks()

    def _device(self):
        if hasattr(self.trainer, "_device_obj"):
            return self.trainer._device_obj()
        dev = getattr(self.trainer, "device", None)
        return dev if isinstance(dev, torch.device) else torch.device(dev)

    def _is_cuda(self):
        try:
            return torch.cuda.is_available() and self._device().type == "cuda"
        except Exception:
            return False

    def _snapshot(self):
        if not self._is_cuda():
            return None
        dev = self._device()
        if self.cuda_sync:
            torch.cuda.synchronize(dev)
        mb = 1024 ** 2
        free, total = torch.cuda.mem_get_info(dev)
        peak_reserved = (
            torch.cuda.max_memory_reserved(dev)
            if hasattr(torch.cuda, "max_memory_reserved")
            else torch.cuda.memory_reserved(dev)
        )
        return {
            "alloc": torch.cuda.memory_allocated(dev) / mb,
            "reserved": torch.cuda.memory_reserved(dev) / mb,
            "peak_alloc": torch.cuda.max_memory_allocated(dev) / mb,
            "peak_reserved": peak_reserved / mb,
            "free": free / mb,
            "total": total / mb,
        }

    def _module_kind(self, module):
        if _is_so2_linear_module(module):
            return "SO2_Linear"
        if _is_eqv3_ffn_module(module):
            return "EqV3StyleNodeFFN"
        if _is_s2_activation_module(module):
            return "FlatSwiGLUS2Merge"
        return module.__class__.__name__

    def _should_monitor(self, module):
        return (
            _is_so2_linear_module(module)
            or _is_eqv3_ffn_module(module)
            or _is_s2_activation_module(module)
        )

    def _record(self, module, name, kind, phase, pre):
        post = self._snapshot()
        if pre is None or post is None:
            return
        peak_alloc_delta = post["peak_alloc"] - pre["peak_alloc"]
        peak_reserved_delta = post["peak_reserved"] - pre["peak_reserved"]
        self.buffer.append([
            getattr(self.trainer, "iter", 0),
            name,
            kind,
            phase,
            f"{pre['alloc']:.1f}",
            f"{post['alloc']:.1f}",
            f"{post['alloc'] - pre['alloc']:.1f}",
            f"{pre['reserved']:.1f}",
            f"{post['reserved']:.1f}",
            f"{post['reserved'] - pre['reserved']:.1f}",
            f"{pre['peak_alloc']:.1f}",
            f"{post['peak_alloc']:.1f}",
            f"{peak_alloc_delta:.1f}",
            f"{pre['peak_reserved']:.1f}",
            f"{post['peak_reserved']:.1f}",
            f"{peak_reserved_delta:.1f}",
            int(peak_alloc_delta > 0.1 or peak_reserved_delta > 0.1),
        ])
        if len(self.buffer) >= self.buffer_size:
            self._flush()

    def _make_forward_pre_hook(self, name, kind):
        def hook(module, inputs):
            module._cuda_mem_fwd_pre = self._snapshot()
        return hook

    def _make_forward_hook(self, name, kind):
        def hook(module, inputs, output):
            pre = getattr(module, "_cuda_mem_fwd_pre", None)
            self._record(module, name, kind, "forward", pre)
            module._cuda_mem_fwd_pre = None
        return hook

    def _make_backward_pre_hook(self, name, kind):
        def hook(module, grad_output):
            module._cuda_mem_bwd_pre = self._snapshot()
        return hook

    def _make_backward_hook(self, name, kind):
        def hook(module, grad_input, grad_output):
            pre = getattr(module, "_cuda_mem_bwd_pre", None)
            self._record(module, name, kind, "backward", pre)
            module._cuda_mem_bwd_pre = None
        return hook

    def _register_hooks(self):
        count = 0
        for name, module in self.trainer.model.named_modules():
            if not self._should_monitor(module):
                continue
            kind = self._module_kind(module)
            module.register_forward_pre_hook(self._make_forward_pre_hook(name, kind))
            module.register_forward_hook(self._make_forward_hook(name, kind))
            if hasattr(module, "register_full_backward_pre_hook"):
                module.register_full_backward_pre_hook(self._make_backward_pre_hook(name, kind))
            module.register_full_backward_hook(self._make_backward_hook(name, kind))
            count += 1
        log.info(f"[CUDA Module Memory Monitor] hooked {count} SO2/S2 modules; csv={self.csv_path}")

    def iteration(self, **kwargs):
        self._flush()

    def _flush(self):
        if not self.buffer:
            return
        try:
            with open(self.csv_path, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerows(self.buffer)
            self.buffer = []
        except Exception as e:
            log.warning(f"CUDA module memory monitor write failed: {e}")


class DeepDoctorMonitor(Plugin):
    def __init__(self, log_dir="monitor_logs", verbose_freq=1, max_steps=1000):
        super(DeepDoctorMonitor, self).__init__([(1, 'iteration')])
        self.verbose_freq = verbose_freq
        self.log_dir = log_dir
        self.hooks_registered = False

        # [Logic Split]
        # 1. has_run_once: 控制细粒度的 Hook 分析 (forward/backward cliff)，因为太慢，只跑一次。
        # 2. total_grad 记录: 需要持续跑一段路程，统计分布。
        self.has_run_once = False

        self.records = {
            "forward": [],
            "backward": [],
            "param": [],
            "total_grad": []  # [新增]
        }
        self.buffer_size = 50

        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)

        self.files = {
            "forward": os.path.join(self.log_dir, "forward_act.csv"),
            "backward": os.path.join(self.log_dir, "backward_flow.csv"),
            "param": os.path.join(self.log_dir, "param_grad.csv"),
            "total_grad": os.path.join(self.log_dir, "total_grad_norm.csv")  # [新增]
        }

        self.headers = {
            "forward": ['iter', 'name', 'type', 'in_std', 'in_mean', 'out_std', 'out_mean', 'gain_ratio'],
            "backward": ['iter', 'name', 'type', 'grad_in_std', 'grad_in_mean', 'grad_out_std', 'grad_out_mean',
                         'grad_gain'],
            "param": ['iter', 'name', 'grad_std', 'grad_mean', 'grad_norm', 'weight_norm', 'update_ratio'],
            "total_grad": ['iter', 'total_norm', 'clipped']  # [新增]
        }

        # 初始化 CSV
        for key, filepath in self.files.items():
            with open(filepath, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(self.headers[key])

    def register(self, trainer):
        self.trainer = trainer
        self._register_hooks()
        log.info(f"🚑 [DeepDoctor] 启动监控 | One-Shot 诊断 + 持续 Total Norm 记录")

    def _analyze_tensor(self, t):
        if t is None: return 0.0, 0.0, 0.0
        if isinstance(t, (tuple, list)):
            if len(t) > 0:
                t = t[0]
            else:
                return 0.0, 0.0, 0.0
        if not isinstance(t, torch.Tensor): return 0.0, 0.0, 0.0
        if t.numel() == 0: return 0.0, 0.0, 0.0
        with torch.no_grad():
            t = t.detach().float()
            std = t.std().item()
            mean = t.mean().item()
            norm = t.norm().item()
        if math.isnan(std): std = 0.0
        if math.isnan(mean): mean = 0.0
        if math.isnan(norm): norm = 0.0
        return std, mean, norm

    # Forward Hook (保持不变)
    def _forward_hook_fn(self, name, layer_type):
        def hook(module, inputs, output):
            if self.has_run_once: return
            inp = inputs[0] if isinstance(inputs, tuple) and len(inputs) > 0 else inputs
            out = output[0] if isinstance(output, tuple) and len(output) > 0 else output
            i_std, i_mean, _ = self._analyze_tensor(inp)
            o_std, o_mean, _ = self._analyze_tensor(out)
            gain = o_std / (i_std + 1e-9)
            self.records["forward"].append(
                [self.trainer.iter, name, layer_type, f"{i_std:.2e}", f"{i_mean:.2e}", f"{o_std:.2e}", f"{o_mean:.2e}",
                 f"{gain:.2f}"])

        return hook

    # Backward Hook (保持不变)
    def _backward_hook_fn(self, name, layer_type):
        def hook(module, grad_input, grad_output):
            if self.has_run_once: return
            g_in = grad_input[0] if isinstance(grad_input, (tuple, list)) and len(grad_input) > 0 else grad_input
            g_out = grad_output[0] if isinstance(grad_output, (tuple, list)) and len(grad_output) > 0 else grad_output
            gi_std, gi_mean, _ = self._analyze_tensor(g_in)
            go_std, go_mean, _ = self._analyze_tensor(g_out)
            grad_gain = gi_std / (go_std + 1e-9)
            self.records["backward"].append(
                [self.trainer.iter, name, layer_type, f"{gi_std:.2e}", f"{gi_mean:.2e}", f"{go_std:.2e}",
                 f"{go_mean:.2e}", f"{grad_gain:.2f}"])

        return hook

    def _register_hooks(self):
        if self.hooks_registered: return
        # ... (Hook 注册逻辑保持不变) ...
        # 为了简洁省略，请使用你原有的代码，仅需注意 target_norm_types 的导入
        target_norm_types = (nn.LayerNorm, nn.GroupNorm, nn.BatchNorm1d)  # 确保有 import
        for name, module in self.trainer.model.named_modules():
            if isinstance(module, torch.jit.ScriptModule) or name == "": continue
            should_hook = False
            tag = module.__class__.__name__
            if isinstance(module, target_norm_types):
                should_hook = True;
                tag = "Norm"
            elif "att" in name.lower():
                should_hook = True;
                tag = "Attention"
            elif len(list(module.parameters(recurse=False))) > 0:
                should_hook = True
            if should_hook:
                try:
                    module.register_forward_hook(self._forward_hook_fn(name, tag))
                    module.register_full_backward_hook(self._backward_hook_fn(name, tag))
                except RuntimeError:
                    pass
        self.hooks_registered = True

    def iteration(self, **kwargs):
        """
        每次 iteration 都会被 Trainer 调用
        kwargs 包含: train_loss, lr, total_grad_norm
        """

        # 1. [持续记录] Total Gradient Norm
        if "total_grad_norm" in kwargs:
            total_norm = kwargs["total_grad_norm"]
            is_clipped = 1 if total_norm > self.trainer.clip_grad_norm else 0
            self.records["total_grad"].append([
                self.trainer.iter,
                f"{total_norm:.4e}",
                is_clipped
            ])

        # 2. [One-Shot] 详细的 Hook 分析
        if not self.has_run_once:
            self._log_param_grads()
            self.has_run_once = True
            log.info(f"🛑 [DeepDoctor] 详细诊断数据已采集完毕。Total Norm 将持续记录。")

        # 3. 刷盘 (每 buffer_size 次写入一次)
        if len(self.records["total_grad"]) >= self.buffer_size or len(self.records["forward"]) > 0:
            self._flush_to_disk()

    def _log_param_grads(self):
        # 保持你原有的逻辑
        for name, param in self.trainer.model.named_parameters():
            if param.grad is not None and param.requires_grad:
                g_std, g_mean, g_norm = self._analyze_tensor(param.grad)
                _, _, w_norm = self._analyze_tensor(param)
                update_ratio = g_norm / (w_norm + 1e-9)
                self.records["param"].append(
                    [self.trainer.iter, name, f"{g_std:.2e}", f"{g_mean:.2e}", f"{g_norm:.2e}", f"{w_norm:.2e}",
                     f"{update_ratio:.2e}"])

    def _flush_to_disk(self):
        for key in self.records:
            if not self.records[key]: continue
            try:
                with open(self.files[key], 'a', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerows(self.records[key])
                self.records[key] = []
            except Exception as e:
                log.warning(f"CSV Write Failed ({key}): {e}")

    def epoch(self, **kwargs):
        pass

class Monitor(Plugin):
    def __init__(self, running_average=True, epoch_average=True, smoothing=0.7,
                 precision=None, number_format=None, unit='', sliding_win_size=50, avg_per_iter=False):

        if precision is None:
            precision = 4
        if number_format is None:
            number_format = '.{}f'.format(precision)
        number_format = ':' + number_format

        super(Monitor, self).__init__([(1, 'iteration'), (1, 'epoch')])

        self.smoothing = smoothing
        self.with_running_average = running_average
        self.with_epoch_average = epoch_average

        self.log_format = number_format
        self.log_unit = unit
        self.log_epoch_fields = None
        self.log_iter_fields = ['{last' + number_format + '}' + unit]

        if self.with_running_average:
            self.log_iter_fields += [' ({running_avg' + number_format + '}' + unit + ')']
        if self.with_epoch_average:
            self.log_epoch_fields = ['{epoch_mean' + number_format + '}' + unit]
        if avg_per_iter:
            self.loss_queue = collections.deque(maxlen=sliding_win_size)
        self.avg_per_iter = avg_per_iter

    def _normalize_scalar(self, val):
        if val is None:
            return None
        if torch.is_tensor(val):
            val = val.detach()
            if val.ndim > 0:
                val = val.mean()
            return float(val.item())
        return float(val)

    def register(self, trainer):
        self.trainer = trainer
        stats = self.trainer.stats.setdefault(self.stat_name, {})

        stats['last'] = 0.0
        if self.with_running_average:
            stats['running_avg'] = 0.0
        if self.with_epoch_average:
            stats['epoch_mean'] = 0.0
            stats['epoch_stats'] = (0, 0)

        stats['log_format'] = self.log_format
        stats['log_unit'] = self.log_unit
        stats['log_iter_fields'] = self.log_iter_fields
        if self.with_epoch_average:
            stats['log_epoch_fields'] = self.log_epoch_fields

    def iteration(self, **kwargs):
        stats = self.trainer.stats.setdefault(self.stat_name, {})
        val = self._get_value(**kwargs)
        val = self._normalize_scalar(val)

        if val is None:
            return

        stats['last'] = val

        if self.with_epoch_average:
            curr_s, curr_c = stats.get('epoch_stats', (0, 0))
            stats['epoch_stats'] = (curr_s + val, curr_c + 1)

        if self.with_running_average:
            previous_avg = stats.get('running_avg', 0.0)
            stats['running_avg'] = previous_avg * self.smoothing + val * (1 - self.smoothing)

        if self.avg_per_iter:
            self.loss_queue.append(val)
            stats['latest_avg_iter_loss'] = sum(self.loss_queue) / len(self.loss_queue)

    def epoch(self, **kwargs):
        stats = self.trainer.stats.setdefault(self.stat_name, {})
        if self.with_epoch_average:
            epoch_stats = stats.get('epoch_stats', (0, 0))
            if epoch_stats[1] > 0:
                stats['epoch_mean'] = epoch_stats[0] / epoch_stats[1]
            else:
                stats['epoch_mean'] = stats.get('epoch_mean', stats.get('last', 0.0))
            stats['epoch_stats'] = (0, 0)


class TrainOnsiteLossMonitor(Monitor):
    stat_name = "train_onsite_loss"

    def __init__(self, interval=None, **kwargs):
        super().__init__(**kwargs)
        if interval is None:
            interval = [(1, 'iteration'), (1, 'epoch')]
        self.trigger_interval = interval

    def _get_value(self, **kwargs):
        return kwargs.get("train_onsite_loss", None)


class TrainHoppingLossMonitor(Monitor):
    stat_name = "train_hopping_loss"

    def __init__(self, interval=None, **kwargs):
        super().__init__(**kwargs)
        if interval is None:
            interval = [(1, 'iteration'), (1, 'epoch')]
        self.trigger_interval = interval

    def _get_value(self, **kwargs):
        return kwargs.get("train_hopping_loss", None)


class TrainZLossMonitor(Monitor):
    stat_name = "mean_max_prob"

    def __init__(self, interval=None, **kwargs):
        super().__init__(**kwargs)
        if interval is None:
            interval = [(1, 'iteration'), (1, 'epoch')]
        self.trigger_interval = interval

    def _get_value(self, **kwargs):
        return kwargs.get("mean_max_prob", None)


class ExpertLoadCVMonitor(Monitor):
    stat_name = "expert_load_cv"

    def __init__(self, interval=None, **kwargs):
        super().__init__(**kwargs)
        if interval is None:
            interval = [(1, 'iteration'), (1, 'epoch')]
        self.trigger_interval = interval

    def _get_value(self, **kwargs):
        return kwargs.get("expert_load_cv", None)


class ScalarFieldMonitor(Monitor):
    """
    通用标量监控器。
    只要 Trainer 在 call_plugins(..., **state) 里传了对应字段，
    就可以写进 trainer.stats，供 TensorBoard / Logger / epoch_mean 使用。
    """

    def __init__(self, stat_name, interval=None, **kwargs):
        self.stat_name = stat_name
        super().__init__(**kwargs)
        if interval is None:
            interval = [(1, 'iteration'), (1, 'epoch')]
        self.trigger_interval = interval

    def _get_value(self, **kwargs):
        return kwargs.get(self.stat_name, None)


class CUDAMemoryMonitor(Plugin):
    """
    Track CUDA memory scalar fields supplied by Trainer.

    The trainer owns CUDA allocator sampling because it can gather per-rank
    values before rank0-only logging plugins run. This monitor normalizes those
    fields into ``trainer.stats`` so Logger and TensorBoard can consume them.
    """

    _EXPERT_MEMORY_RE = re.compile(r"^expert_\d+_cuda_.*_mb$")

    def __init__(self, interval=None, precision=1):
        if interval is None:
            interval = [(1, 'iteration'), (1, 'epoch')]
        super(CUDAMemoryMonitor, self).__init__(interval)
        self.precision = int(precision)
        self._epoch_max = {}

    @classmethod
    def is_memory_field(cls, name):
        return isinstance(name, str) and (
            (name.startswith("cuda_") and name.endswith("_mb"))
            or cls._EXPERT_MEMORY_RE.match(name) is not None
        )

    def register(self, trainer):
        self.trainer = trainer
        for name in (
            "cuda_allocated_mb",
            "cuda_reserved_mb",
            "cuda_peak_allocated_mb",
            "cuda_peak_reserved_mb",
            "cuda_free_mb",
            "cuda_total_mb",
        ):
            self._ensure_stat(name)

    def _to_float(self, val):
        if val is None:
            return None
        if torch.is_tensor(val):
            val = val.detach()
            if val.ndim > 0:
                val = val.mean()
            return float(val.item())
        try:
            return float(val)
        except Exception:
            return None

    def _ensure_stat(self, name):
        stat = self.trainer.stats.setdefault(name, {})
        fmt = f":.{self.precision}f"
        stat.setdefault("log_name", name)
        stat.setdefault("log_format", fmt)
        stat.setdefault("log_unit", "MB")
        stat.setdefault("last", 0.0)
        stat.setdefault("max", 0.0)
        stat.setdefault("epoch_max", 0.0)
        stat.setdefault("log_iter_fields", [
            "{last" + fmt + "}MB",
            "(max {max" + fmt + "}MB)",
        ])
        stat.setdefault("log_epoch_fields", [
            "{epoch_max" + fmt + "}MB",
            "(max {max" + fmt + "}MB)",
        ])
        return stat

    def iteration(self, **kwargs):
        for name, raw_val in kwargs.items():
            if not self.is_memory_field(name):
                continue
            val = self._to_float(raw_val)
            if val is None:
                continue

            stat = self._ensure_stat(name)
            stat["last"] = val
            stat["max"] = max(float(stat.get("max", val)), val)

            prev_epoch = self._epoch_max.get(name)
            self._epoch_max[name] = val if prev_epoch is None else max(prev_epoch, val)
            stat["epoch_max"] = self._epoch_max[name]

    def epoch(self, **kwargs):
        memory_names = {
            name for name in self.trainer.stats
            if self.is_memory_field(name)
        }
        memory_names.update(
            name for name in kwargs
            if self.is_memory_field(name)
        )

        for name in memory_names:
            stat = self._ensure_stat(name)
            if name in kwargs:
                val = self._to_float(kwargs.get(name))
                if val is not None:
                    stat["last"] = val
                    stat["max"] = max(float(stat.get("max", val)), val)
                    prev_epoch = self._epoch_max.get(name)
                    self._epoch_max[name] = val if prev_epoch is None else max(prev_epoch, val)

            epoch_val = self._epoch_max.get(name)
            if epoch_val is None:
                epoch_val = stat.get("last", 0.0)
            stat["epoch_max"] = float(epoch_val)
            self._epoch_max[name] = None


class TrainLossMonitor(Monitor):
    stat_name = 'train_loss'

    def _get_value(self, **kwargs):
        return kwargs.get('train_loss', None)


class TestLossMonitor(Monitor):
    stat_name = 'test_loss'

    def __init__(self):
        super(TestLossMonitor, self).__init__(
            precision=6,
        )

    def _get_value(self, **kwargs):
        return kwargs.get('test_loss', None)


class LearningRateMonitor(Monitor):
    stat_name = 'lr'

    def __init__(self):
        super(LearningRateMonitor, self).__init__(
            running_average=False, epoch_average=False, smoothing=0.7,
            precision=6, number_format='.{}g'.format(4), unit=''
        )

    def _get_value(self, **kwargs):
        return kwargs.get('lr', None)


class Validationer(Monitor):
    stat_name = 'validation_loss'

    def __init__(self, interval, fast_mode=True):
        super(Validationer, self).__init__(
            precision=8,
        )
        self.trigger_interval = interval
        self.fast_mode = fast_mode

    def _get_value(self, **kwargs):
        if kwargs.get('field') == "iteration":
            return self.trainer.validation(fast=True)

    def epoch(self, **kwargs):
        stats = self.trainer.stats.setdefault(self.stat_name, {})
        val = self.trainer.validation(fast=self.fast_mode)
        if torch.is_tensor(val):
            val = val.detach()
            if val.ndim > 0:
                val = val.mean()
            val = float(val.item())
        else:
            val = float(val)
        stats['epoch_mean'] = val


class TensorBoardMonitor(Plugin):
    """
    单 event run 版本：
    - 只建议主进程注册
    - iteration / epoch 均支持 expert_i_lr
    - 优先使用 kwargs["time"] 作为 step
    """
    def __init__(self, interval, log_dir='./tensorboard_logs', flush_every=20):
        super(TensorBoardMonitor, self).__init__(interval=interval)
        self.writer = SummaryWriter(log_dir=log_dir)
        self.flush_every = flush_every

    def register(self, trainer):
        self.trainer = trainer

    def _to_float(self, val, default=None):
        if val is None:
            return default
        if torch.is_tensor(val):
            val = val.detach()
            if val.ndim > 0:
                val = val.mean()
            return float(val.item())
        try:
            return float(val)
        except Exception:
            return default

    def _get_stat(self, name, key, default=None):
        stat = self.trainer.stats.get(name, {})
        if stat is None:
            return default
        return self._to_float(stat.get(key, default), default=default)

    def _get_value(self, name, key='last', kwargs=None, default=None):
        if kwargs is not None and name in kwargs:
            return self._to_float(kwargs.get(name), default=default)
        return self._get_stat(name, key, default=default)

    def _memory_names(self, kwargs=None):
        names = {
            name for name in self.trainer.stats
            if CUDAMemoryMonitor.is_memory_field(name)
        }
        if kwargs:
            names.update(
                name for name in kwargs
                if CUDAMemoryMonitor.is_memory_field(name)
            )
        return sorted(names)

    def epoch(self, **kwargs):
        epoch = kwargs.get("time", self.trainer.ep)

        lr = self._get_stat('lr', 'last', None)
        train_loss_mean = self._get_stat('train_loss', 'epoch_mean', None)
        train_loss_opt_mean = self._get_stat('train_loss_opt', 'epoch_mean', None)
        validation_loss_mean = self._get_stat('validation_loss', 'epoch_mean', None)

        if lr is not None:
            self.writer.add_scalar('lr/epoch', lr, epoch)
        if train_loss_mean is not None:
            self.writer.add_scalar('train_loss_mean/epoch', train_loss_mean, epoch)
        if train_loss_opt_mean is not None:
            self.writer.add_scalar('train_loss_opt_mean/epoch', train_loss_opt_mean, epoch)
        if validation_loss_mean is not None:
            self.writer.add_scalar('validation_loss_mean/epoch', validation_loss_mean, epoch)

        if 'train_onsite_loss' in self.trainer.stats:
            self.writer.add_scalar(
                'train_onsite_loss_mean/epoch',
                self._get_stat('train_onsite_loss', 'epoch_mean', 0.0),
                epoch
            )
        if 'train_hopping_loss' in self.trainer.stats:
            self.writer.add_scalar(
                'train_hopping_loss_mean/epoch',
                self._get_stat('train_hopping_loss', 'epoch_mean', 0.0),
                epoch
            )
        if 'mean_max_prob' in self.trainer.stats:
            self.writer.add_scalar(
                'mean_max_prob_mean/epoch',
                self._get_stat('mean_max_prob', 'epoch_mean', 0.0),
                epoch
            )
        if 'expert_load_cv' in self.trainer.stats:
            self.writer.add_scalar(
                'expert_load_cv_mean/epoch',
                self._get_stat('expert_load_cv', 'epoch_mean', 0.0),
                epoch
            )

        num_experts = getattr(self.trainer, 'num_experts', 0)
        for i in range(num_experts):
            onsite_key = f'expert_{i}_onsite'
            hopping_key = f'expert_{i}_hopping'
            lr_key = f'expert_{i}_lr'

            if onsite_key in self.trainer.stats:
                self.writer.add_scalar(
                    f'Expert_Onsite_Epoch_Mean/Expert_{i}',
                    self._get_stat(onsite_key, 'epoch_mean', 0.0),
                    epoch
                )
            if hopping_key in self.trainer.stats:
                self.writer.add_scalar(
                    f'Expert_Hopping_Epoch_Mean/Expert_{i}',
                    self._get_stat(hopping_key, 'epoch_mean', 0.0),
                    epoch
                )
            if lr_key in self.trainer.stats:
                self.writer.add_scalar(
                    f'Expert_LR_Epoch/Expert_{i}',
                    self._get_stat(lr_key, 'last', 0.0),
                    epoch
                )

        for name in self._memory_names(kwargs):
            epoch_max = self._get_stat(name, 'epoch_max', None)
            if epoch_max is not None:
                self.writer.add_scalar(f'CUDA_Memory_Epoch_Max/{name}', epoch_max, epoch)
            overall_max = self._get_stat(name, 'max', None)
            if overall_max is not None:
                self.writer.add_scalar(f'CUDA_Memory_Overall_Max/{name}', overall_max, epoch)

        self.writer.flush()

    def iteration(self, **kwargs):
        iteration = kwargs.get("time", self.trainer.iter)

        lr = self._get_value('lr', 'last', kwargs, default=None)
        train_loss = self._get_value('train_loss', 'last', kwargs, default=None)
        train_loss_opt = self._get_value('train_loss_opt', 'last', kwargs, default=None)
        train_onsite = self._get_value('train_onsite_loss', 'last', kwargs, default=None)
        train_hopping = self._get_value('train_hopping_loss', 'last', kwargs, default=None)
        mean_max_prob = self._get_value('mean_max_prob', 'last', kwargs, default=None)
        expert_load_cv = self._get_value('expert_load_cv', 'last', kwargs, default=None)

        if lr is not None:
            self.writer.add_scalar('lr_iter/iteration', lr, iteration)
        if train_loss is not None:
            self.writer.add_scalar('train_loss_iter/iteration', train_loss, iteration)
        if train_loss_opt is not None:
            self.writer.add_scalar('train_loss_opt_iter/iteration', train_loss_opt, iteration)
        if train_onsite is not None:
            self.writer.add_scalar('train_onsite_loss_iter/iteration', train_onsite, iteration)
        if train_hopping is not None:
            self.writer.add_scalar('train_hopping_loss_iter/iteration', train_hopping, iteration)
        if mean_max_prob is not None:
            self.writer.add_scalar('mean_max_prob_iter/iteration', mean_max_prob, iteration)
        if expert_load_cv is not None:
            self.writer.add_scalar('expert_load_cv_iter/iteration', expert_load_cv, iteration)

        latest_avg_iter_loss = self._get_stat('train_loss', 'latest_avg_iter_loss', None)
        if latest_avg_iter_loss is not None:
            self.writer.add_scalar('latest_avg_loss/iteration', latest_avg_iter_loss, iteration)

        num_experts = getattr(self.trainer, 'num_experts', 0)
        for i in range(num_experts):
            onsite_key = f'expert_{i}_onsite'
            hopping_key = f'expert_{i}_hopping'
            lr_key = f'expert_{i}_lr'

            onsite_val = self._get_value(onsite_key, 'last', kwargs, default=None)
            hopping_val = self._get_value(hopping_key, 'last', kwargs, default=None)
            lr_val = self._get_value(lr_key, 'last', kwargs, default=None)

            if onsite_val is not None:
                self.writer.add_scalar(f'Expert_Onsite_Iter/Expert_{i}', onsite_val, iteration)
            if hopping_val is not None:
                self.writer.add_scalar(f'Expert_Hopping_Iter/Expert_{i}', hopping_val, iteration)
            if lr_val is not None:
                self.writer.add_scalar(f'Expert_LR_Iter/Expert_{i}', lr_val, iteration)

        for name in self._memory_names(kwargs):
            val = self._get_value(name, 'last', kwargs, default=None)
            if val is not None:
                self.writer.add_scalar(f'CUDA_Memory_Iter/{name}', val, iteration)
            max_val = self._get_stat(name, 'max', None)
            if max_val is not None:
                self.writer.add_scalar(f'CUDA_Memory_Overall_Max_Iter/{name}', max_val, iteration)

        if self.flush_every and iteration % self.flush_every == 0:
            self.writer.flush()
