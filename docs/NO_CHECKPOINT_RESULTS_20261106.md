# 实验 02：双方均关闭 checkpoint

2026-11-06，Sid。新增对比完成，代码版本 `35ba90a`；ADFWI 固定版本不变。

## 为什么此前显存只减少约 10%

此前双方均使用 50 段 checkpoint，原生 AD 已避免保存完整传播图，Hybrid 也通过分段重算减少历史存储。这已将双方显存显著压低，公共的分段状态缓存占据了较大比例。关闭 checkpoint 后，AD 保存完整运算中间量，Hybrid 只保存伴随敏感核所需的五个波场历史，差距更明显。

此次确实绕过 PyTorch checkpoint 回调；仅设上游 `checkpoint_segments=1` 仍会重算，不能算关闭。测试检查 AD 图没有 CheckpointFunction 节点。Hybrid 使用对原生编译时间循环的外部历史记录插桩，数值更新不变；首次正演预分配五个连续 GPU 历史张量并写入，反向重算为零，无 CPU offload。历史分配及写入计入总耗时。

## 原 9 炮规模：容量测试

保持原模型、78×200 网格、45 m、2500 步、dt=0.003 s、100 接收点、PSV / FD8 / float64 状态及梯度，复用上一组完全相同的观测与目标。

| 方法，无 checkpoint | 完整梯度耗时 | CUDA allocated 峰值 | 结果 |
|---|---:|---:|---|
| ADFWI 原生 AD | N/A | 23.35055 GiB，失败前峰值 | 正演阶段 CUDA OOM |
| HybridADFWI | 62.35 s | 20.82992 GiB | 完成 |

AD 在失败前运行 24.53 秒，该值不是完整梯度耗时，不能用来计算加速比。24 GiB A5000 无法完成本配置的直接 AD，因此也没有双方均无 checkpoint 的 9 炮敏感核误差。

成功的 9 炮 Hybrid 与此前 **50 段 checkpoint 的 AD 参考**波形逐位一致，敏感核相对 L2 误差：Vp=5.0091e-12，Vs=3.5763e-12，rho=3.5683e-12。这是物理梯度参考检查，不能标成两边都无 checkpoint 的完整配对结果。

## 双方均为 1 炮：完整配对比较

为取得双方都可完成的结果，仅将两边同时改为 1 炮，保留原网格、2500 步、时间步长、精度、边界、接收点和目标定义；使用该共同单炮真模型观测，三次重复，交替 AB/BA，新进程测量。

| 指标 | ADFWI，无 checkpoint | HybridADFWI，无 checkpoint |
|---|---:|---:|
| 完整耗时，均值 ± 样本标准差 | 149.00 ± 0.96 s | 62.64 ± 0.59 s |
| 三次耗时 | 149.23 / 147.94 / 149.82 s | 63.22 / 62.03 / 62.68 s |
| 正演平均耗时 | 31.38 s | 21.17 s |
| 反向平均耗时 | 117.62 s | 41.48 s |
| CUDA allocated 峰值，三次相同 | 4.16962 GiB | 2.32287 GiB |
| CPU RSS 最大采样峰值 | 2787.04 MiB | 829.43 MiB |

本组平均耗时之比 **2.379×**，显存减少 **44.29%**。两个分支均无 checkpoint 和正演重放；Hybrid 五场历史完整保留 2500 步。CUDA allocated 不含全部驱动及进程上下文内存；reserved、baseline、increment 和各阶段原始记录均随结果提供。

三轮正演均逐位一致；各核为有限 float64、形状 78×200，无梯度处理。最大相对 L2 误差：

| 参数 | 三次中的最大相对 L2 误差 |
|---|---:|
| Vp | 9.0580 × 10⁻¹³ |
| Vs | 5.8161 × 10⁻¹⁴ |
| rho | 2.6906 × 10⁻¹³ |

全部小于预先设置的 1e-8 阈值；余弦接近 1。完整波形 SHA256 在服务器另行复核，敏感核、原始 JSON 和源码哈希保留在归档中。

## 条件与解释范围

GPU 仅在空闲时启动；每个 worker 前再次检查，运行期间每 3 秒采样计算进程。保留其他 CPU 任务，本组单线程、nice10、绑定 CPU30，load 约21–22。逐 worker 审计全部通过，所有实验 GPU worker 已退出。新增四项 no-checkpoint 验证与原测试合计 21 项通过。

上一组为 9 炮、CPU load 约40；本组为 1 炮且 CPU 条件不同，**不可将新旧总耗时直接横比来判断 checkpoint 的时间收益**。显存不足的原 9 炮测试和单炮配对提供不同证据，分别保留。float64 指模型、波场、数据和梯度，上游 float32 生成的 FD 常数仍原样共享。

无 checkpoint 的 Hybrid 仍需要保存完整正演历史，显存随炮数、网格和时间步增长；本次结果不表示常数内存或所有模拟器均有同样收益。

## 材料与复现

- [9 炮容量记录](../evidence/no-checkpoint-capacity-9shot-20261106/)
- [1 炮完整对比、原始敏感核与资源审计](../evidence/no-checkpoint-1shot-20261106/)
- 完整波形与观测分别保留在服务器 `/home/xiwang/HybridADFWI/runs/no-checkpoint-capacity-9shot-20261106` 与 `/home/xiwang/HybridADFWI/runs/no-checkpoint-1shot-20261106`。

```bash
python scripts/run_when_idle.py --min-gpu-gib 18 --min-ram-gib 12 \
  --evidence results/no-checkpoint-reproduction-gate.json -- \
  python scripts/benchmark.py run --preset paper --shots 1 --device cuda \
  --no-checkpoint --repeats 3 \
  --model evidence/no-checkpoint-1shot-20261106/model_input.npz \
  --output runs/no-checkpoint-reproduction
```
