# HybridADFWI

在保留 ADFWI 原生 PSV 正演内核的条件下，对比完整自动微分与混合离散伴随的梯度、耗时及峰值内存。ADFWI 本身已经可微；本仓库用于对照两种求导策略。SPECFEM 与 AxiSEM3D 的可微扩展另行实现，不由本仓库提供。

HybridADFWI compares native automatic differentiation with an external discrete adjoint for the same already-differentiable ADFWI solver. The supported path is **2-D isotropic elastic PSV, FD8, split PML, float64 state and first-order Vp/Vs/rho gradients**. This is a source-checkout research benchmark, with CPU correctness tests and archived GPU measurements; it does not implement all ADFWI physics or a complete inversion workflow.

安装、无数据下载的 CPU 测试、Marmousi-II 示例及硬件要求见 [Reproducibility guide](docs/REPRODUCIBILITY.md)。数据来源与再分发范围见 [Data provenance](docs/DATA_PROVENANCE.md)。

## 首次实验已完成

Marmousi-II / PSV / FD8 / float64 状态，9 炮、2500 步，两边均 50 段 checkpoint，RTX A5000，三次重复：

| | ADFWI 原生 AD | HybridADFWI |
|---|---:|---:|
| 总耗时，均值 ± 样本标准差 | 144.62 ± 21.41 s | 54.01 ± 8.98 s |
| CUDA allocated 峰值 | 2.11267 GiB | 1.89532 GiB |

波形逐位一致，三个参数的敏感核最大相对 L2 误差 **5.01e-12**。该次测试有 CPU 满载争用，时间比约 2.68×，应视为初次测量。完整配置、精度口径、原始数据和限制见 [结果报告](docs/RESULTS_20261106.md)。

### 补充：双方均关闭 checkpoint

原 9 炮直接 AD 在 A5000 上显存不足；Hybrid 可完成，峰值20.83 GiB。因此完整配对将双方同时改为 1 炮，保持原网格及2500步，三次重复：

| | ADFWI，无 checkpoint | HybridADFWI，无 checkpoint |
|---|---:|---:|
| 总耗时，均值 ± 样本标准差 | 149.00 ± 0.96 s | 62.64 ± 0.59 s |
| CUDA allocated 峰值 | 4.16962 GiB | 2.32287 GiB |

本组显存减少 **44.29%**，时间比 **2.38×**，波形逐位一致，最大敏感核相对 L2 误差 **9.06e-13**。新旧炮数与 CPU 条件不同，耗时不能直接横比。详见 [无 checkpoint 报告](docs/NO_CHECKPOINT_RESULTS_20261106.md)。

## 两条计算路径

| | ADFWI 原生 AD | HybridADFWI |
|---|---|---|
| 正演 | 原生 FD8 / split PML | 同一原生内核 |
| 时间传播的 AD 图 | 开启；启用 checkpoint 时在反传重建 | 关闭 |
| 目标函数 | PyTorch AD | PyTorch AD |
| 数据导数 / 伴随源 | 由反传隐式传入 | 显式计算 VJP：`torch.autograd.grad(loss, data)` |
| 模型梯度 | 对传播算子自动微分 | 显式离散伴随 + 静态材料映射 VJP |
| 参数 | Vp、Vs、rho | Vp、Vs、rho |

伴随不是把原正演简单倒放。扩展显式转置原生更新、接收采样、split PML 和自由表面赋值；按选项分段重算或保存完整波场历史，用于敏感核。上游源码保持不变。

## 复现

```bash
git clone --recurse-submodules https://github.com/Wangxi404/HybridADFWI-Open.git
cd HybridADFWI-Open
python -m pip install -e '.[test]'
python -m pytest -q
python scripts/prepare_marmousi.py --preset smoke
python scripts/benchmark.py run --preset smoke --device cpu --repeats 1 --output runs/smoke
```

CPU 单元测试使用现场生成的小模型，不下载 Marmousi-II。准备脚本从 AGL 官方站点下载三份校验和固定的文件；请先查看 [数据条款与来源](docs/DATA_PROVENANCE.md)。本仓库的 MIT 代码许可不授予数据的 MIT 许可。上游以 Git submodule 固定，已有克隆可执行 `git submodule update --init --recursive`。运行需要源码检出及上游子模块；独立安装 wheel 不包含运行所需的 vendor、协议及脚本。

Linux / NVIDIA GPU 正式测试（先运行 `python scripts/prepare_marmousi.py --preset paper`）：

```bash
python scripts/run_when_idle.py --min-gpu-gib 18 --min-ram-gib 12 \
  --evidence results/resource_gate.json -- \
  python scripts/benchmark.py run --preset paper --device cuda \
  --repeats 3 --threads 1 --output runs/marmousi2-fd8-f64
```

脚本仅在 GPU 连续三次满足空闲条件后启动，限制本实验 CPU 线程、降低优先级，不改动其他任务。已知桌面图形进程可忽略；未知计算进程阻止启动。CPU 并行任务仍可能影响 GPU wall-clock 耗时，资源证据随结果记录。

每次测量使用新进程，交替 AB/BA 顺序，固定输入和分段数，排除数据准备、预热和结果存储。输出 `REPORT.md`、`summary.json`、逐次 JSON/NPZ/日志，包含 Vp/Vs/rho 未处理敏感核、相对 L2/L∞ 误差、余弦、分段耗时、CPU RSS、CUDA allocated/reserved 峰值及源代码/数据版本。

### 真正关闭 checkpoint

`--no-checkpoint` 会绕过 PyTorch checkpoint 回调；仅设置上游 `checkpoint_segments=1` 仍会进行重算，不能替代这个开关。

```bash
python scripts/run_when_idle.py --min-gpu-gib 18 --min-ram-gib 12 \
  --evidence results/no-checkpoint-gate.json -- \
  python scripts/benchmark.py run --preset paper --shots 1 --device cuda \
  --no-checkpoint --repeats 3 --output runs/no-checkpoint-1shot
```

ADFWI 保留完整传播 AD 图；HybridADFWI 在原始正演中将伴随所需的五个波场写入连续 GPU 历史缓冲区，反向直接使用，零重算、无 CPU offload。历史记录采用对固定原生时间循环的外部插桩副本，数值更新不变，上游文件不改动。缓冲区分配及历史写入均计入正演时间。

对于显存不足的配置，`scripts/check_no_checkpoint_capacity.py --reference-protocol <已有协议> --output <新目录>` 可在相同模型、炮数、时间步和观测下分别尝试两边，保留 OOM 阶段与显存信息。失败耗时不作为完整梯度时间，缺失梯度不填充误差。

## 固定配置与范围

- 上游：[ADFWI](https://github.com/liufeng2317/ADFWI)，commit `559f581149343b1ed3dd2168caae595690bd601d`；MIT 许可证归原作者。
- PSV、八阶交错网格、split PML、自由表面；模型、波场、观测及梯度为 float64。
- 上游 FD 常数由 float32 计算，两条路径共享其原值。上游 padding 有默认精度分配，两个分支都使用同一个保留输入 dtype 的外部适配器，避免材料参数被降为 float32。该适配器不修改 vendor 文件。
- 真实 Marmousi-II；采用 ADFWI 示例的约 5–14 km 横向截取。`smoke` 为 39×101 / 90 m，仅用于启动与正确性验证；`paper` 为 78×200 / 45 m。更大规模与空间收敛需另行实验。
- 初始速度为固体区平滑结果，保持水层；默认密度 2450 kg/m³，沿用上游示例，准备脚本也支持原始密度。
- 默认目标为观测 RMS 固定归一化的波形 L2，使用 `vz`；可选时间差分复合目标。无梯度平滑、裁剪、照明预条件及优化器。
- 仅一阶模型梯度；源、几何和阻尼固定，不支持高阶 AD。模块 hook 在进程内串行使用。

本次对比保持**相同离散正演**。不能把其他求解器与 ADFWI 的差异算作 AD 策略的误差或收益。性能改善由实际测量决定，不预设加速比。

详见 [实验协议](docs/EXPERIMENT.md)。本项目代码采用 [MIT License](LICENSE)；ADFWI 的原始 MIT 版权声明与数据说明见 [Third-party notices](THIRD_PARTY_NOTICES.md)。
