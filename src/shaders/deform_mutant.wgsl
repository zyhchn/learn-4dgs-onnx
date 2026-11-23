// deform_mutant.wgsl —— 路线 B：HexPlane + MLP 神经形变场的 WebGPU compute 前向
//
// 一个线程负责一个高斯（`global_invocation_id.x`），把 HexPlane + MLP 的前向并行跑完。
// 管线与 ONNX 图逐层对应：
//
//   1. 归一化 4D 坐标    coord = 1 - 2*(p - aabbMin)/(aabbMax - aabbMin)，第 4 维补 time
//   2. HexPlane          两层 × 6 个平面，每层把 6 个平面 GridSample 结果逐通道相乘
//   3. 特征拼接          level0(32) + level1(32) → 64 维
//   4. 共享 MLP          feature_out.0(64→64) + ReLU
//   5. 三个头            pos/scales/rotations 各自 (64→64)+ReLU → (64→3/3/4) 得「残差」
//   6. 装配              pos = base + 残差；scale = exp(base + 残差)；
//                        quat = normalize(base + 残差)；opacity = sigmoid(base)
//   7. 协方差            Σ = R·diag(s²)·Rᵀ → 6 个独立分量
//
// 两个约定与 ONNX 图保持一致（踏错就会整体偏掉）：
//   - 四元数在权重里是 **(w, x, y, z)**；
//   - 所有权重都是 Gemm 的 `transB=1` 形式：y[o] = Σ_k x[k]·W[o][k] + b[o]，
//     即 W 的形状是 [out, in]、行优先。
//
// 为什么必须在 GPU 上跑？放在 CPU 上一次要 ~1 秒（38599 个高斯），每帧重算就是 1 fps；
// 这里是 38599 个线程同时跑，单帧降到毫秒量级 —— 这正是「神经形变场」能实时化的关键。

// 权重清单里的固定值（见 src/deform_mutant_gpu.ts 的 buildMeta 布局）
struct Meta {
    posOff: u32,        // positions        [N, 3]
    scaleOff: u32,      // scales           [N, 3]
    rotOff: u32,        // rotations        [N, 4]（w 在前）
    opaOff: u32,        // opacity          [N, 1]
    dcOff: u32,         // features_dc      [N, 3, 1]
    count: u32,         // 高斯总数 N（线程数上限）
    featW: u32,         // feature_out.0.weight  [64, 64]
    featB: u32,         // feature_out.0.bias    [64]
    posW1: u32,         // pos_deform.1 / .3
    posB1: u32,         // pos 头第一层偏置
    posW2: u32,         // pos 头第二层权重
    posB2: u32,         // pos 头第二层偏置
    scaW1: u32,         // scales_deform.1 / .3
    scaB1: u32,         // scale 头第一层偏置
    scaW2: u32,         // scale 头第二层权重
    scaB2: u32,         // scale 头第二层偏置
    rotW1: u32,         // rotations_deform.1 / .3
    rotB1: u32,         // rot 头第一层偏置
    rotW2: u32,         // rot 头第二层权重
    rotB2: u32,         // rot 头第二层偏置
    aabbMinX: f32,      // 归一化用包围盒下界 x
    aabbMinY: f32,      // 归一化用包围盒下界 y
    aabbMinZ: f32,      // 归一化用包围盒下界 z
    aabbMinPad: f32,    // 对齐填充（使 vec4 满足 16 字节对齐）
    aabbMaxX: f32,      // 归一化用包围盒上界 x
    aabbMaxY: f32,      // 归一化用包围盒上界 y
    aabbMaxZ: f32,      // 归一化用包围盒上界 z
    aabbMaxPad: f32,    // 对齐填充
};

struct Params {
    time: f32,      // 当前动画时间（形变场第 4 维输入）
    pad0: f32,      // 填充，保持 16 字节对齐
    pad1: f32,      // 填充
    pad2: f32,      // 填充
};

@group(0) @binding(0) var<uniform> params: Params;   // 每帧更新的 time
// 整个 mutant_weights.bin：所有平面 + 所有 MLP 权重都在里面，用 offset 索引
@group(0) @binding(1) var<storage, read> weights: array<f32>;   // 权重/平面数据大数组
@group(0) @binding(2) var<storage, read> cfg: Meta;             // 上面那组 offset 清单
// 12 个平面，每个占 2 个 vec4：[offset, dim0, dim1, channels] + [height, width, 0, 0]
@group(0) @binding(3) var<storage, read> planes: array<vec4<u32>>;   // 平面元数据（Offset/H/W/坐标轴对）
// 前向产物：与渲染器直接读取的 buffers 是同一批（零回读）
@group(0) @binding(4) var<storage, read_write> outCenters: array<vec4<f32>>;      // 输出：形变后中心
@group(0) @binding(5) var<storage, read_write> outColorOpacity: array<vec4<f32>>; // 输出：颜色 + 不透明度
@group(0) @binding(6) var<storage, read_write> outCovA: array<vec4<f32>>;         // 输出：协方差上半
@group(0) @binding(7) var<storage, read_write> outCovB: array<vec4<f32>>;         // 输出：协方差下半

// 3DGS 的 SH 直流分量基线：RGB = 0.5 + SH_C0 · dc
const SH_C0: f32 = 0.28209479177387814;   // 球谐 0 阶基函数值 1/(2√π)
// 整体缩放 2.5 倍；绕 X 轴旋转 -90° 让模型空间的 +z（人物头顶）对齐世界 +y（向上）。
const MODEL_SCALE: f32 = 2.5;             // 世界坐标整体缩放系数

// 复刻 PyTorch 的 grid_sample(mode='bilinear', padding_mode='border', align_corners=True)：
// 归一化坐标 -1 / +1 分别对齐到像素 0 / (size-1)，越界钳到边缘。
// 平面在权重里是 [C, H, W] 行优先，所以通道 c 的起点是 planeOff + c*H*W。
fn samplePlane(planeOff: u32, H: u32, W: u32, cx: f32, cy: f32, c: u32) -> f32 {
    let Wm1 = f32(W - 1u);              // 宽度减一（align_corners=True 的映射端点）
    let Hm1 = f32(H - 1u);              // 高度减一
    var ix = ((cx + 1.0) * Wm1) * 0.5;  // 归一化 x∈[-1,1] → 浮点像素坐标 ix∈[0, W-1]
    var iy = ((cy + 1.0) * Hm1) * 0.5;  // 归一化 y → 浮点像素坐标 iy
    ix = clamp(ix, 0.0, Wm1);           // padding_mode='border'：越界钳到边缘
    iy = clamp(iy, 0.0, Hm1);           // 同上
    let x0 = u32(floor(ix));            // 左列整数坐标
    let y0 = u32(floor(iy));            // 下行整数坐标
    let x1 = min(x0 + 1u, W - 1u);      // 右列（越界则钳到 W-1）
    let y1 = min(y0 + 1u, H - 1u);      // 上行（越界则钳到 H-1）
    let wx = ix - f32(x0);              // x 方向插值权重（小数部分）
    let wy = iy - f32(y0);              // y 方向插值权重
    let base = planeOff + c * H * W;    // 通道 c 在该平面数据里的起始偏移
    let r0 = base + y0 * W;             // 第 y0 行的起始偏移
    let r1 = base + y1 * W;             // 第 y1 行的起始偏移
    let v00 = weights[r0 + x0];         // 左下角采样值
    let v01 = weights[r0 + x1];         // 右下角采样值
    let v10 = weights[r1 + x0];         // 左上角采样值
    let v11 = weights[r1 + x1];         // 右上角采样值
    let top = v00 + (v01 - v00) * wx;   // 下边沿水平线性插值
    let bot = v10 + (v11 - v10) * wx;   // 上边沿水平线性插值
    return top + (bot - top) * wy;      // 两行再按 wy 做垂直线性插值
}

// 一个「头」的前半段：h = relu(act · W1ᵀ + b1)，64 → 64
fn head64(act: array<f32, 64>, w1: u32, b1: u32, out: ptr<function, array<f32, 64>>) {
    for (var o = 0u; o < 64u; o = o + 1u) {          // 遍历输出维度 o（64 个神经元）
        var s = weights[b1 + o];                      // 累加器初值 = 偏置 b1[o]
        let wo = w1 + o * 64u;                        // 第 o 行权重起点（W 为 [out, in] 行优先）
        for (var k = 0u; k < 64u; k = k + 1u) {       // 遍历输入维度 k（64）
            s = s + act[k] * weights[wo + k];         // s += act[k] * W[o][k]
        }
        (*out)[o] = max(s, 0.0);                      // ReLU：负值截断为 0，写回 out[o]
    }
}

@compute @workgroup_size(256)  // compute 入口，每个工作组 256 个线程
fn cs_forward(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;          // 本线程负责的高斯索引
    if (i >= cfg.count) {   // 超出高斯总数（最后一组可能越界）
        return;             // 直接返回，不产生输出
    }

    // ---------- 1. 归一化 4D 坐标 ----------
    let p3 = i * 3u;        // 第 i 个高斯位置在 [N,3] 数据中的起始下标
    let base = vec3f(       // 原始（未形变）位置
        weights[cfg.posOff + p3],        // x
        weights[cfg.posOff + p3 + 1u],   // y
        weights[cfg.posOff + p3 + 2u]);  // z
    let amin = vec3f(cfg.aabbMinX, cfg.aabbMinY, cfg.aabbMinZ);  // 包围盒下界
    let amax = vec3f(cfg.aabbMaxX, cfg.aabbMaxY, cfg.aabbMaxZ);  // 包围盒上界
    let norm = vec3f(1.0) - 2.0 * (base - amin) / (amax - amin); // 归一化到 [-1,1]（min→+1，max→-1）
    var coord: array<f32, 4>;   // 4D 采样坐标
    coord[0] = norm.x;          // 维度 0
    coord[1] = norm.y;          // 维度 1
    coord[2] = norm.z;          // 维度 2
    coord[3] = params.time;     // 维度 3 = 时间（第 4 维）

    // ---------- 2. HexPlane：每层 6 个平面逐通道相乘 ----------
    var feature: array<f32, 64>;   // 两层各 32 通道拼成的 64 维特征
    for (var level = 0u; level < 2u; level = level + 1u) {   // 遍历两个 HexPlane 层级
        var acc: array<f32, 32>;   // 本层 6 平面逐通道相乘的累加结果（32 通道）
        for (var pl = 0u; pl < 6u; pl = pl + 1u) {           // 遍历 6 个平面（XY,XZ,YZ,XT,YT,ZT）
            let di = (level * 6u + pl) * 2u;                 // 该平面元数据在 planes[] 中占 2 个 vec4
            let d = planes[di];                              // 元数据 A：[offset, dim0, dim1, channels]
            let e = planes[di + 1u];                         // 元数据 B：[height, width, 0, 0]
            let planeOff = d.x;                              // 该平面数据在 weights 里的偏移
            let H = e.x;                                     // 平面高度
            let W = e.y;                                     // 平面宽度
            let cx = coord[d.y];                             // 采样 x 坐标：取 dim0 指定的那一维
            let cy = coord[d.z];                             // 采样 y 坐标：取 dim1 指定的那一维
            for (var c = 0u; c < 32u; c = c + 1u) {          // 遍历 32 个通道
                let v = samplePlane(planeOff, H, W, cx, cy, c);  // 双线性采样该通道
                if (pl == 0u) {                              // 第一个平面：直接赋值（初始化累加器）
                    acc[c] = v;
                } else {                                     // 其余平面：逐通道相乘
                    acc[c] = acc[c] * v;
                }
            }
        }
        for (var c = 0u; c < 32u; c = c + 1u) {              // 把本层结果写入 feature 的对应区间
            feature[level * 32u + c] = acc[c];               // level 0 → [0,32)，level 1 → [32,64)
        }
    }

    // ---------- 3 / 4. 共享 MLP + ReLU ----------
    var act: array<f32, 64>;    // 共享 MLP 的输出（激活值），三个头共用
    for (var o = 0u; o < 64u; o = o + 1u) {              // 遍历 64 个输出神经元
        var s = weights[cfg.featB + o];                  // 累加器初值 = feature_out.0.bias[o]
        let wo = cfg.featW + o * 64u;                    // 第 o 行权重起点（[64,64] 行优先）
        for (var k = 0u; k < 64u; k = k + 1u) {          // 遍历 64 个输入
            s = s + feature[k] * weights[wo + k];        // s += feature[k] * W[o][k]
        }
        act[o] = max(s, 0.0);                            // ReLU 激活
    }

    // ---------- 5. 三个头（残差） ----------
    var h1: array<f32, 64>;     // 三个头各自的隐藏层复用同一块缓冲
    var posRes: array<f32, 3>;  // 位置残差（3 维）
    var scaRes: array<f32, 3>;  // 尺度残差（3 维）
    var rotRes: array<f32, 4>;  // 四元数残差（4 维）

    head64(act, cfg.posW1, cfg.posB1, &h1);              // 位置头第一层：64→64 + ReLU，结果写入 h1
    for (var o = 0u; o < 3u; o = o + 1u) {               // 位置头第二层输出 3 维
        var s = weights[cfg.posB2 + o];                  // 初值 = pos 第二层偏置
        let wo = cfg.posW2 + o * 64u;                    // 第 o 行权重起点
        for (var k = 0u; k < 64u; k = k + 1u) {          // 64 个输入
            s = s + h1[k] * weights[wo + k];             // s += h1[k] * W[o][k]
        }
        posRes[o] = s;                                   // 位置残差（第二层无激活）
    }

    head64(act, cfg.scaW1, cfg.scaB1, &h1);              // 尺度头第一层：64→64 + ReLU
    for (var o = 0u; o < 3u; o = o + 1u) {               // 尺度头第二层输出 3 维
        var s = weights[cfg.scaB2 + o];                  // 初值 = scale 第二层偏置
        let wo = cfg.scaW2 + o * 64u;                    // 第 o 行权重起点
        for (var k = 0u; k < 64u; k = k + 1u) {          // 64 个输入
            s = s + h1[k] * weights[wo + k];             // s += h1[k] * W[o][k]
        }
        scaRes[o] = s;                                   // 尺度残差
    }

    head64(act, cfg.rotW1, cfg.rotB1, &h1);              // 旋转头第一层：64→64 + ReLU
    for (var o = 0u; o < 4u; o = o + 1u) {               // 旋转头第二层输出 4 维
        var s = weights[cfg.rotB2 + o];                  // 初值 = rot 第二层偏置
        let wo = cfg.rotW2 + o * 64u;                    // 第 o 行权重起点
        for (var k = 0u; k < 64u; k = k + 1u) {          // 64 个输入
            s = s + h1[k] * weights[wo + k];             // s += h1[k] * W[o][k]
        }
        rotRes[o] = s;                                   // 四元数残差
    }

    // ---------- 6. 装配 ----------
    var opos: array<f32, 3>;    // 最终位置
    var oscale: array<f32, 3>;  // 最终尺度
    for (var a = 0u; a < 3u; a = a + 1u) {                                              // 逐分量装配
        opos[a] = weights[cfg.posOff + p3 + a] + posRes[a];            // 位置 = 基准位置 + 残差
        oscale[a] = exp(weights[cfg.scaleOff + p3 + a] + scaRes[a]);   // 尺度 = exp(基准尺度 + 残差)，保证为正
    }
    var qraw: array<f32, 4>;    // 未归一化的四元数
    var norm2 = 0.0;            // 四元数模长的平方
    for (var a = 0u; a < 4u; a = a + 1u) {                        // 逐分量装配四元数
        let r = weights[cfg.rotOff + i * 4u + a] + rotRes[a];     // 基准四元数 + 残差
        qraw[a] = r;                                             // 暂存
        norm2 = norm2 + r * r;                                   // 累加各分量平方和
    }
    let inv = 1.0 / max(sqrt(norm2), 1e-12);   // 模长倒数（加下限防除零）
    let qw = qraw[0] * inv;     // 归一化后的 w（权重里 w 在前）
    let qx = qraw[1] * inv;     // 归一化后的 x
    let qy = qraw[2] * inv;     // 归一化后的 y
    let qz = qraw[3] * inv;     // 归一化后的 z
    let opa = 1.0 / (1.0 + exp(-weights[cfg.opaOff + i]));   // 不透明度 = sigmoid（基准值，无形变残差）

    // ---------- 7. 协方差 Σ = R·diag(s²)·Rᵀ ----------
    let xx = qx * qx;   // 预先算好四元数各分量乘积，供旋转矩阵复用
    let yy = qy * qy;   // y*y
    let zz = qz * qz;   // z*z
    let xy = qx * qy;   // x*y
    let xz = qx * qz;   // x*z
    let yz = qy * qz;   // y*z
    let wx = qw * qx;   // w*x
    let wy = qw * qy;   // w*y
    let wz = qw * qz;   // w*z
    let r00 = 1.0 - 2.0 * (yy + zz);   // 四元数 → 旋转矩阵 R 的第 0 行第 0 列
    let r01 = 2.0 * (xy - wz);         // R[0][1]
    let r02 = 2.0 * (xz + wy);         // R[0][2]
    let r10 = 2.0 * (xy + wz);         // R[1][0]
    let r11 = 1.0 - 2.0 * (xx + zz);   // R[1][1]
    let r12 = 2.0 * (yz - wx);         // R[1][2]
    let r20 = 2.0 * (xz - wy);         // R[2][0]
    let r21 = 2.0 * (yz + wx);         // R[2][1]
    let r22 = 1.0 - 2.0 * (xx + yy);   // R[2][2]
    // M = R·diag(s)：R 的第 j 列按 s_j 缩放
    let s0 = oscale[0];   // 第 0 轴尺度
    let s1 = oscale[1];   // 第 1 轴尺度
    let s2 = oscale[2];   // 第 2 轴尺度
    let m00 = r00 * s0;   // M 第 0 行
    let m01 = r01 * s1;   // M[0][1]
    let m02 = r02 * s2;   // M[0][2]
    let m10 = r10 * s0;   // M 第 1 行
    let m11 = r11 * s1;   // M[1][1]
    let m12 = r12 * s2;   // M[1][2]
    let m20 = r20 * s0;   // M 第 2 行
    let m21 = r21 * s1;   // M[2][1]
    let m22 = r22 * s2;   // M[2][2]
    let c00 = m00 * m00 + m01 * m01 + m02 * m02;   // Σ = M·Mᵀ 的 [0][0]
    let c01 = m00 * m10 + m01 * m11 + m02 * m12;   // Σ[0][1]
    let c02 = m00 * m20 + m01 * m21 + m02 * m22;   // Σ[0][2]
    let c11 = m10 * m10 + m11 * m11 + m12 * m12;   // Σ[1][1]
    let c12 = m10 * m20 + m11 * m21 + m12 * m22;   // Σ[1][2]
    let c22 = m20 * m20 + m21 * m21 + m22 * m22;   // Σ[2][2]

    // ---------- 输出 ----------
    var rgb: array<f32, 3>;     // 由 SH 直流分量得到的颜色
    for (var a = 0u; a < 3u; a = a + 1u) {               // 逐通道计算颜色
        rgb[a] = clamp(0.5 + SH_C0 * weights[cfg.dcOff + p3 + a], 0.0, 1.0);  // RGB = 0.5 + C0*dc，钳到 [0,1]
    }
    // Rx(-90°)：(x, y, z) → (x, z, -y)，使模型 +z 映射到世界 +y（y 轴朝上）。
    let worldPos = MODEL_SCALE * vec3f(opos[0], opos[2], -opos[1]);   // 缩放并绕 X 轴 -90° 旋转到世界空间
    let scale2 = MODEL_SCALE * MODEL_SCALE;   // 协方差是二次型，缩放系数要平方
    outCenters[i] = vec4f(worldPos, 1.0);     // 写出形变后的中心（w=1 占位）
    outColorOpacity[i] = vec4f(rgb[0], rgb[1], rgb[2], opa);   // 写出颜色 + 不透明度
    // Σ' = s² R Σ Rᵀ，R 为 X 轴 -90° 旋转矩阵。
    outCovA[i] = vec4f(c00 * scale2, c02 * scale2, -c01 * scale2, 0.0);   // 打包协方差上半 (σ'00, σ'01, σ'02)
    outCovB[i] = vec4f(c22 * scale2, -c12 * scale2, c11 * scale2, 0.0);   // 打包协方差下半 (σ'11, σ'12, σ'22)
}
