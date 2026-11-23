/**
 * deform_onnx.ts —— 路线 B：mutant.onnx 导出的「权重清单 + 扁平权重」数据层
 *
 * mutant.onnx（HexPlane + MLP 神经形变场）被离线导出成两个静态文件：
 *
 *   - `mutant_manifest.json`：各张量在扁平权重里的 offset / 形状、包围盒、
 *     12 个 HexPlane 平面的描述……也就是下面这些接口；
 *   - `mutant_weights.bin`：把 ONNX 图里**所有** initializer 首尾相接拼成的
 *     一段连续 f32（约 15.6 MB）。
 *
 * 本文件只负责「把这两个文件装进内存」，**不含任何前向计算**。
 * 真正的前向是 WebGPU compute：
 *
 *   - 管线代码：`src/deform_mutant_gpu.ts`
 *   - 着色器：  `shaders/deform_mutant.wgsl`（一个线程算一个高斯）
 *
 * 之所以不在 CPU 上做前向：38599 个高斯跑一遍 HexPlane + MLP 实测约 1 秒，
 * 每帧重算就是 1 fps；只有搬进 compute shader 才能实时。
 */

/** 一个张量在扁平权重里的位置：offset 以 f32 为单位 */
export interface MutantTensorInfo {
    offset: number;   // 该张量首元素在整段 f32 权重里的下标
    shape: number[];  // 张量形状；元素个数 = 各维相乘
}

/** HexPlane 的一个平面（形状 [1, C, H, W]） */
export interface MutantPlaneInfo extends MutantTensorInfo {
    name: string;     // 该平面在清单 tensors 里的键名
    level: number;    // 属于第几层 HexPlane（0 或 1）
    plane: number;    // 该层里的第几个平面（0..5）
    /** 该平面取 4D 坐标里的哪两维；第 0 个映射到 width(x)，第 1 个映射到 height(y) */
    dims: number[];
    channels: number; // 通道数 C
    height: number;   // 平面高 H
    width: number;    // 平面宽 W
}

/** mutant_manifest.json 的结构 */
export interface MutantManifest {
    source: string;              // 来源说明（通常指向导出的 onnx 文件名）
    count: number;               // 高斯点数量
    numPoints: number;           // 点数（与 count 对应）
    floatCount: number;          // 权重的 f32 总个数，用于校验二进制长度
    aabbMax: number[];           // 包围盒上界（3 个数）
    aabbMin: number[];           // 包围盒下界（3 个数）
    coords: { note: string; timeIndex: number }; // 归一化坐标说明：哪一维是时间
    planes: MutantPlaneInfo[];   // 12 个平面的描述（平铺）
    tensors: Record<string, MutantTensorInfo>;   // 所有张量名 → offset/形状
    mlp: {
        feature: { weight: string; bias: string };                          // 共享 MLP 的权重/偏置名
        pos: { w1: string; b1: string; w2: string; b2: string };            // 位置头两层
        scales: { w1: string; b1: string; w2: string; b2: string };         // 尺度头两层
        rotations: { w1: string; b1: string; w2: string; b2: string };      // 旋转头两层
    };
}

/**
 * 清单 + 扁平权重的数据容器。
 *
 * 它本身不算任何东西，只是把「两个静态文件」与「清单里的元信息」绑在一起，
 * 供 GPU 前向（`deform_mutant_gpu.ts`）在构造时读取：
 *   - `weights` 整段上传成 weightsBuffer；
 *   - `info`   用来拼 metaBuffer / planesBuffer（各张量的 offset、包围盒、平面描述）。
 */
export class MutantField {
    readonly count: number;          // 高斯数量
    readonly info: MutantManifest;   // 原始清单
    /** 扁平权重视图。GPU 前向要整段上传它，故为 public */
    readonly weights: Float32Array;

    constructor(manifest: MutantManifest, weights: Float32Array) {
        // 长度不够时，后面按 offset 索引会越界，尽早报错。
        if (weights.length < manifest.floatCount) {
            throw new Error(`权重长度不足：需要 ${manifest.floatCount} 个 f32，实际 ${weights.length}`);
        }
        this.info = manifest;        // 保存清单
        this.count = manifest.count; // 保存点数
        this.weights = weights;      // 保存权重视图
    }
}

/** 便捷构造：manifest + 权重二进制视图 */
export function createMutantField(manifest: MutantManifest, weights: Float32Array): MutantField {
    return new MutantField(manifest, weights); // 直接 new 出实例
}
