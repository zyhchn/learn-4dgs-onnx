/**
 * deform_mutant_gpu.ts —— 路线 B 的 WebGPU compute 前向（HexPlane + MLP 神经形变场）
 *
 * 为什么必须是 GPU：把 mutant.onnx 的前向放在 CPU 上跑，38599 个高斯一次要 ~1 秒，
 * 每帧重算就是 1 fps。这里把同一套前向写成 `shaders/deform_mutant.wgsl` 的
 * compute shader：一个线程算一个高斯，单帧降到毫秒量级，而且产物**直接写进渲染器
 * 读的那几个 storage buffer**，不经过 CPU 回读 —— 这才是「神经形变场实时渲染」的完整形态。
 *
 * 与 Visionary 的区别值得点一句：它把整张 ONNX 图交给 onnxruntime-web 的 WebGPU
 * execution provider 去跑；我们从这里开始**不依赖 onnxruntime**，用纯 WebGPU 手写前向。
 *
 * 资源清单：
 *   - weightsBuffer  ：整个 mutant_weights.bin（15.6 MB），平面与 MLP 权重都靠 offset 索引；
 *   - metaBuffer     ：各张量在 weights 里的 offset + aabb，见 WGSL 的 Meta 结构；
 *   - planesBuffer   ：12 个平面的描述（offset / dims / h / w）；
 *   - paramsBuffer   ：每帧写入的 time；
 *   - 输出缓冲 4 个   ：centers / colorOpacity / covA / covB，渲染器直接读。
 */

import type { MutantField } from "./deform_onnx";
import type { SplatBuffers } from "./packSplats";
import deformShaderStr from "./shaders/deform_mutant.wgsl?raw";

/** 与 deform_mutant.wgsl 的 @workgroup_size 一致 */
const WORKGROUP_SIZE = 256;
/** HexPlane 每层的平面数 */
const PLANES_PER_LEVEL = 6;
/** 每个平面在 planesBuffer 里占 2 个 vec4 */
const PLANE_VEC4_STRIDE = 2;
/** Meta 结构：20 个 u32 + 8 个 f32（WGSL 里全是 4 字节标量，所以就是顺序排布） */
const META_BYTES = (20 + 8) * 4; // 28 个标量 × 4 字节 = 112 字节

export class MutantForwardGpu {
    readonly count: number; // 高斯数量

    private readonly device: GPUDevice;              // WebGPU 设备
    private readonly weightsBuffer: GPUBuffer;       // 整个 mutant_weights.bin（平面 + MLP 权重）
    private readonly metaBuffer: GPUBuffer;          // 张量 offset 与 aabb 的 Meta 块
    private readonly planesBuffer: GPUBuffer;        // 12 个平面描述
    private readonly paramsBuffer: GPUBuffer;        // 每帧写入的 time
    private readonly centersBuffer: GPUBuffer;       // 输出：中心（渲染器直接读）
    private readonly colorOpacityBuffer: GPUBuffer;  // 输出：颜色+不透明度（渲染器直接读）
    private readonly covABuffer: GPUBuffer;          // 输出：协方差上半（渲染器直接读）
    private readonly covBBuffer: GPUBuffer;          // 输出：协方差下半（渲染器直接读）
    private readonly bindGroup: GPUBindGroup;        // 8 个绑定的组合
    private readonly pipeline: GPUComputePipeline;   // compute 前向管线
    private readonly paramsData = new Float32Array(4); // params 的 CPU 暂存（time + 3 个 padding）

    private constructor(
        device: GPUDevice,
        field: MutantField,
        pipeline: GPUComputePipeline,
        bindGroup: GPUBindGroup,
        buffers: {
            weights: GPUBuffer;
            meta: GPUBuffer;
            planes: GPUBuffer;
            params: GPUBuffer;
            centers: GPUBuffer;
            colorOpacity: GPUBuffer;
            covA: GPUBuffer;
            covB: GPUBuffer;
        },
    ) {
        this.device = device;                        // 保存设备
        this.count = field.count;                    // 保存点数
        this.pipeline = pipeline;                    // 保存管线
        this.bindGroup = bindGroup;                  // 保存绑定组
        this.weightsBuffer = buffers.weights;        // 保存权重缓冲
        this.metaBuffer = buffers.meta;              // 保存 Meta 缓冲
        this.planesBuffer = buffers.planes;          // 保存平面表缓冲
        this.paramsBuffer = buffers.params;          // 保存 params 缓冲
        this.centersBuffer = buffers.centers;        // 保存输出：中心
        this.colorOpacityBuffer = buffers.colorOpacity; // 保存输出：颜色+不透明度
        this.covABuffer = buffers.covA;              // 保存输出：协方差 A
        this.covBBuffer = buffers.covB;              // 保存输出：协方差 B
    }

    static create(device: GPUDevice, field: MutantField): MutantForwardGpu {
        // 着色器由 Vite `?raw` 在构建时静态导入，无需等待；这里同步建好模块与管线。
        const module = device.createShaderModule({ label: "deform-mutant", code: deformShaderStr }); // 创建着色器模块

        const bindGroupLayout = device.createBindGroupLayout({
            label: "deform-mutant-bgl",
            entries: [
                { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: "uniform" } },           // params（time）
                { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } }, // weights
                { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } }, // meta
                { binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } }, // planes
                { binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },           // 输出 centers（可写）
                { binding: 5, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },           // 输出 colorOpacity
                { binding: 6, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },           // 输出 covA
                { binding: 7, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },           // 输出 covB
            ],
        });
        const pipeline = device.createComputePipeline({
            label: "deform-mutant-forward",
            layout: device.createPipelineLayout({ bindGroupLayouts: [bindGroupLayout] }), // 只有这一组绑定
            compute: { module, entryPoint: "cs_forward" }, // compute 入口：一个线程算一个高斯
        });

        const n = field.count; // 点数

        // 整个权重二进制一段上传，shader 用 offset 索引（平面 12 张 + MLP 全部）
        const weights = device.createBuffer({
            label: "mutant-weights",
            size: field.weights.byteLength, // 约 15.6 MB
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST, // 作 storage 读，且允许上传
        });
        device.queue.writeBuffer(weights, 0, field.weights as Float32Array<ArrayBuffer>); // 一次性上传全部权重

        const metaData = buildMeta(field); // 组装 Meta 块（各张量 offset + aabb）
        const meta = device.createBuffer({
            label: "mutant-meta",
            size: metaData.byteLength, // 112 字节
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
        });
        device.queue.writeBuffer(meta, 0, metaData as Uint8Array<ArrayBuffer>); // 上传 Meta

        const planeData = buildPlanes(field); // 组装 12 个平面的描述
        const planes = device.createBuffer({
            label: "mutant-planes",
            size: planeData.byteLength, // 12 × 2 × 4 × 4 字节
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
        });
        device.queue.writeBuffer(planes, 0, planeData as Uint32Array<ArrayBuffer>); // 上传平面表

        const params = device.createBuffer({
            label: "mutant-params",
            size: 16, // 1 个 f32（time）+ 3 个 padding = 16 字节
            usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST, // uniform 且可写
        });

        // 输出缓冲：compute 写（storage）、渲染器读（read-only-storage）。
        // 注意：shader 在写出前已把结果做 SH_C0 颜色还原、MODEL_SCALE(2.5) 缩放与 Rx(-90°) 世界旋转，
        // 所以渲染器拿到的就是最终世界空间的高斯参数。
        const outUsage = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST;
        const centers = device.createBuffer({ label: "mutant-centers", size: n * 16, usage: outUsage });            // 每个高斯 vec4(中心,1)
        const colorOpacity = device.createBuffer({ label: "mutant-color-opacity", size: n * 16, usage: outUsage }); // 每个高斯 vec4(rgb,opacity)
        const covA = device.createBuffer({ label: "mutant-cov-a", size: n * 16, usage: outUsage });                 // 协方差上半
        const covB = device.createBuffer({ label: "mutant-cov-b", size: n * 16, usage: outUsage });                 // 协方差下半

        const bindGroup = device.createBindGroup({
            label: "deform-mutant-bindgroup",
            layout: bindGroupLayout, // 与上面定义的布局一致
            entries: [
                { binding: 0, resource: { buffer: params } },        // params
                { binding: 1, resource: { buffer: weights } },       // weights
                { binding: 2, resource: { buffer: meta } },          // meta
                { binding: 3, resource: { buffer: planes } },        // planes
                { binding: 4, resource: { buffer: centers } },       // 输出 centers
                { binding: 5, resource: { buffer: colorOpacity } },  // 输出 colorOpacity
                { binding: 6, resource: { buffer: covA } },          // 输出 covA
                { binding: 7, resource: { buffer: covB } },          // 输出 covB
            ],
        });

        return new MutantForwardGpu(device, field, pipeline, bindGroup, { // 构造实例并回填所有资源
            weights,
            meta,
            planes,
            params,
            centers,
            colorOpacity,
            covA,
            covB,
        });
    }

    /** 交给 renderer.setScene 的缓冲集合（渲染器只持引用，不负责销毁） */
    externalBuffers(): SplatBuffers {
        return {
            count: this.count,                            // 点数
            centers: this.centersBuffer,                  // 直接暴露 centers 缓冲引用
            colorOpacity: this.colorOpacityBuffer,        // 直接暴露 colorOpacity 引用
            covA: this.covABuffer,                        // 直接暴露 covA 引用
            covB: this.covBBuffer,                        // 直接暴露 covB 引用
        };
    }

    /**
     * 在时刻 `time` 上跑一遍 compute 前向，把形变后的高斯写进输出缓冲。
     *
     * 结果全部留在 GPU 上（渲染器直接读那几个 storage buffer），不做任何 CPU 回读。
     */
    run(time: number): void {
        this.paramsData[0] = time; // 第 0 位写当前时间
        this.paramsData[1] = 0;    // 其余 3 位是 padding（保持 16 字节对齐）
        this.device.queue.writeBuffer(this.paramsBuffer, 0, this.paramsData as Float32Array<ArrayBuffer>); // 上传 time

        const encoder = this.device.createCommandEncoder({ label: "mutant-forward" }); // 本帧命令编码器
        const pass = encoder.beginComputePass({ label: "mutant-forward-pass" });       // 开始 compute pass
        pass.setPipeline(this.pipeline);              // 选 compute 管线
        pass.setBindGroup(0, this.bindGroup);         // 绑定第 0 组
        pass.dispatchWorkgroups(Math.ceil(this.count / WORKGROUP_SIZE)); // 按每个高斯一个线程派发（向上取整补齐）
        pass.end();                                   // 结束 compute pass

        this.device.queue.submit([encoder.finish()]); // 提交命令
    }

}

/**
 * 把各张量在 `mutant_weights.bin` 里的 offset 按 WGSL 的 Meta 布局排进一个 112 字节块。
 *
 * 布局（全部是 4 字节标量，顺序排布，与 WGSL 的 Meta 结构逐字段对应）：
 *   u32[0..20) 各张量 offset 与 count；f32[20..28) aabbMin / aabbMax（各 3 个，后补 1 个 pad）
 */
function buildMeta(field: MutantField): Uint8Array {
    const m = field.info; // 清单
    const off = (name: string): number => { // 小工具：取某张量的起始 offset
        const info = m.tensors[name];       // 查张量元信息
        if (!info) throw new Error(`清单里找不到张量：${name}`); // 名字写错尽早暴露
        return info.offset;                 // 返回以 f32 为单位的起始下标
    };

    const buf = new ArrayBuffer(META_BYTES); // 112 字节的底层缓冲
    const u32 = new Uint32Array(buf);        // 前 20 个位置按 u32 解释（offset / count）
    const f32 = new Float32Array(buf);       // 后 8 个位置按 f32 解释（aabb）
    u32[0] = off("positions");   // positions offset
    u32[1] = off("scales");      // scales offset
    u32[2] = off("rotations");   // rotations offset
    u32[3] = off("opacity");     // opacity offset
    u32[4] = off("features_dc"); // features_dc offset（SH 直流分量）
    u32[5] = m.count;            // 高斯总数（线程数上限）
    u32[6] = off(m.mlp.feature.weight); // 共享 MLP 权重 offset
    u32[7] = off(m.mlp.feature.bias);   // 共享 MLP 偏置 offset
    u32[8] = off(m.mlp.pos.w1);         // pos 头第一层权重
    u32[9] = off(m.mlp.pos.b1);         // pos 头第一层偏置
    u32[10] = off(m.mlp.pos.w2);        // pos 头第二层权重
    u32[11] = off(m.mlp.pos.b2);        // pos 头第二层偏置
    u32[12] = off(m.mlp.scales.w1);     // scale 头第一层权重
    u32[13] = off(m.mlp.scales.b1);     // scale 头第一层偏置
    u32[14] = off(m.mlp.scales.w2);     // scale 头第二层权重
    u32[15] = off(m.mlp.scales.b2);     // scale 头第二层偏置
    u32[16] = off(m.mlp.rotations.w1);  // rot 头第一层权重
    u32[17] = off(m.mlp.rotations.b1);  // rot 头第一层偏置
    u32[18] = off(m.mlp.rotations.w2);  // rot 头第二层权重
    u32[19] = off(m.mlp.rotations.b2);  // rot 头第二层偏置
    f32[20] = m.aabbMin[0]; // 包围盒下界 x（对应 WGSL 的 aabbMinX）
    f32[21] = m.aabbMin[1]; // 包围盒下界 y
    f32[22] = m.aabbMin[2]; // 包围盒下界 z
    // f32[23] 是 aabbMinPad：对齐填充，故意留 0（让 vec4 满足 16 字节对齐）
    f32[24] = m.aabbMax[0]; // 包围盒上界 x（对应 WGSL 的 aabbMaxX）
    f32[25] = m.aabbMax[1]; // 包围盒上界 y
    f32[26] = m.aabbMax[2]; // 包围盒上界 z
    // f32[27] 是 aabbMaxPad：对齐填充，故意留 0
    return new Uint8Array(buf); // 以字节视图返回，交给 writeBuffer
}

/** 12 个平面 → 24 个 vec4：[offset, dim0, dim1, channels] + [height, width, 0, 0] */
function buildPlanes(field: MutantField): Uint32Array {
    // 尺寸直接由平面条数推导：每平面 2 个 vec4、每 vec4 4 个 u32，避免与手写常量脱节
    const u32PerPlane = PLANE_VEC4_STRIDE * 4; // 每个平面占 8 个 u32（2×4）
    const data = new Uint32Array(field.info.planes.length * u32PerPlane); // 12 × 8 = 96 个 u32
    for (const p of field.info.planes) { // 遍历清单里的每个平面
        const v = (p.level * PLANES_PER_LEVEL + p.plane) * PLANE_VEC4_STRIDE; // 该平面首 vec4 的序号（按 level/plane 排序）
        data[v * 4] = p.offset;     // vec4 A：平面数据 offset
        data[v * 4 + 1] = p.dims[0]; //        取坐标的 dim0（映射到 width）
        data[v * 4 + 2] = p.dims[1]; //        取坐标的 dim1（映射到 height）
        data[v * 4 + 3] = p.channels; //       通道数 C
        const w = (v + 1) * 4;      // 第二个 vec4 的起始下标
        data[w] = p.height;         // vec4 B：高度 H
        data[w + 1] = p.width;      //        宽度 W
        data[w + 2] = 0;            //        占位 0
        data[w + 3] = 0;            //        占位 0
    }
    return data; // 返回平面表
}
