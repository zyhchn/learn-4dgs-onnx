# onnx-demo

4DGS 渲染教学 Demo（路线 B）：用 WebGPU compute 手写 `mutant.onnx`（HexPlane + MLP）的神经形变场前向，产物直接写进渲染器读的 storage buffer，不回读 CPU，也不用 onnxruntime。参考自 [Visionary](https://github.com/Visionary-Laboratory/visionary)。

## 简介

打开页面自动加载清单与权重，时间轴循环播放。前向在 `src/shaders/deform_mutant.wgsl` 里，一个线程算一个高斯：

```
coord    = 1 - 2*(p - aabbMin)/(aabbMax - aabbMin)，第 4 维补 time
HexPlane  两层 × 6 个平面，逐通道相乘 → 拼成 64 维
MLP       64 → 64 + ReLU
三个头    pos / scales / rotations：64→64 + ReLU → 64→3/3/4，输出残差
装配      pos += 残差；scale = exp(base + 残差)；quat = normalize(base + 残差)；opacity = sigmoid(base)
协方差    Σ = R·diag(s²)·Rᵀ
```

网络只吃归一化位置和时间，scale / rotation / opacity 是静态基准叠加残差。CPU 跑一遍 38599 个高斯要 ~1 秒，放进 compute 并行后单帧是毫秒级。

数据就两个文件：`mutant_manifest.json`（各张量 offset / 形状、包围盒、12 个平面）+ `mutant_weights.bin`（initializer 首尾相接的 f32 流，14.9 MB）；`deform_onnx.ts` 只负责装载。排序仍是 GPU 4-bit 基数排序，键为「深度 + 索引」。

| 文件 | 作用 |
| --- | --- |
| `src/deform_onnx.ts` | 清单 + 权重的数据容器 |
| `src/deform_mutant_gpu.ts` | compute 前向管线 |
| `src/shaders/deform_mutant.wgsl` | HexPlane + MLP 前向 |
| `src/Renderer.ts` / `src/ViewerApp.ts` | 渲染器与主循环 |

## 安装与使用

Node 18+，以及支持 WebGPU 的浏览器（Chrome / Edge 113+）。

```bash
pnpm install
pnpm dev
```

打开 http://localhost:5174，自动加载并播放：拖拽旋转、滚轮缩放，时间以 0.15 倍速在 `[0, 1)` 循环。

```bash
pnpm build        # 打包
pnpm preview      # 预览打包产物
pnpm typecheck    # tsc --noEmit
```

## 来源

神经形变场结构与 ONNX 语义参考 [Visionary](https://github.com/Visionary-Laboratory/visionary)（Apache-2.0），模型取自其 `public/models/dyn/mutant.onnx`，离线导出成这里的清单 + 权重。区别是 Visionary 把 ONNX 图交给 onnxruntime-web 跑，本 Demo 用纯 WebGPU 手写前向。
