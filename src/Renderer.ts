/**
 * Renderer.ts —— WebGPU 管线组装与绘制
 *
 * 渲染部分与路线 A（splatv-demo）完全一样：顶点扩张 + 片元椭圆高斯核 + 按深度排序。
 * 唯一的差别在**数据来源**：这里不接受 CPU 打包好的数组，而是直接引用 compute
 * 前向写好的 storage buffer（setScene），零回读。
 *
 * 因为着色器端不做时间形变（时间形变已经在 compute 前向里做完了），
 * 所以 uniform 里没有时间字段，也不需要 timeParams 绑定。
 */

import type { SplatBuffers } from "./packSplats";
import splatShaderStr from "./shaders/splat.wgsl?raw";

// 每帧清屏用的背景色（深蓝灰色），clearValue 直接用它
const BACKGROUND: GPUColor = { r: 0.055, g: 0.062, b: 0.078, a: 1 };

// 标准「over」混合：新片元按自己的 α 覆盖在已画内容之上
const BLEND_OVER: GPUBlendState = {
    // 颜色通道：src 乘自身 α，dst 乘 (1 - src α)，再相加 —— 即 src*α + dst*(1-α)
    color: { srcFactor: "src-alpha", dstFactor: "one-minus-src-alpha", operation: "add" },
    // alpha 通道：src 全量、dst 乘 (1 - src α)，让最终 α 累积正确
    alpha: { srcFactor: "one", dstFactor: "one-minus-src-alpha", operation: "add" },
};

// 高斯的mesh顶点数据：只提供单位四边形的角点，真正的几何都在 storage buffer 里
const QUAD_VERTICES = new Float32Array([
    -2, -2, 0,
    2, -2, 0,
    -2, 2, 0,
    2, 2, 0,
]);

// uniform 段的总 f32 个数：44 个 f32 = 176 字节（viewProj 16 + view 16 + focal 4 + viewport 4 + params 4）
const COMMON_UNIFORM_DATA_SIZE = 44;

/** 每帧交给 renderer.render(state) 的状态，由 ViewerApp 组装 */
export interface RenderCameraState {
    /** view * projection（列主序 16 个数） */
    viewProj: Float32Array;
    /** view 矩阵（列主序 16 个数） */
    view: Float32Array;
    /** 相机世界位置 */
    camPos: Float32Array;
    /** 相机世界前向（用于 GPU 排序的深度键） */
    forward: Float32Array;
    /** 像素单位焦距 [fx, fy] */
    focal: [number, number];
    /** 相机像素宽 */
    width: number;
    /** 相机像素高 */
    height: number;
    /** 低通滤波强度，把小于一个像素的高斯撑大 */
    kernelSize: number;
    /** 长轴直径小于该像素数的高斯直接丢弃 */
    minPixelSize: number;
    /** 高斯整体缩放倍率 */
    splatScale: number;
}

export class Renderer {
    device: GPUDevice;          // WebGPU 逻辑设备，所有资源/命令都从它创建
    context: GPUCanvasContext;  // 画布上下文，渲染时从它取当前帧纹理
    canvas: HTMLCanvasElement;  // 承载画面的 DOM 画布
    format: GPUTextureFormat;   // 画布纹理格式（由 getPreferredCanvasFormat 决定）

    count = 0;   // 当前高斯数量（= 绘制实例数），由 setScene 设定
    width = 1;   // 画布像素宽度（已乘 dpr）
    height = 1;  // 画布像素高度（已乘 dpr）

    /** 公共统一变量数据（CPU 侧复用同一块，避免每帧分配）
     * viewProj: Mat4x4<f32>
     * view: Mat4x4<f32>
     * focal: vec4<f32>[fx, fy, 0, 0]
     * viewport: vec4<f32>[width, height, 2/width, 2/height]
     * params: vec4<f32>[kernelSize, minPixelSize, splatScale, 0]
     */
    commonUniformData: Float32Array;

    // 以下字段在 init() / setScene() 里赋值，用 ! 断言
    shaderModule!: GPUShaderModule;            // 顶点 + 片元合一的 WGSL 模块
    bindGroupLayout!: GPUBindGroupLayout;      // 6 个绑定的布局（1 uniform + 5 storage）
    bindGroup!: GPUBindGroup;                  // 把上面这些缓冲绑到管线上的 bind group
    verticesBuffer!: GPUBuffer;                // 单位四边形顶点缓冲
    commonUniformBuffer!: GPUBuffer;           // uniform 缓冲（每帧写相机/视口参数）
    pipeline!: GPURenderPipeline;              // 高斯绘制的渲染管线

    // 高斯场景资源（除 sortedBuffer 外，都只是 compute 前向缓冲的引用）
    centersBuffer!: GPUBuffer;                 // 高斯中心，外部 compute 前向提供
    colorOpacityBuffer!: GPUBuffer;            // 颜色+不透明度，外部提供
    covABuffer!: GPUBuffer;                    // 协方差分量 A，外部提供
    covBBuffer!: GPUBuffer;                    // 协方差分量 B，外部提供
    // 排序缓冲区：唯一由渲染器自己创建/销毁的场景缓冲
    sortedBuffer!: GPUBuffer;                  // 排序索引缓冲（渲染器自己管理）

    constructor(
        device: GPUDevice,
        context: GPUCanvasContext,
        canvas: HTMLCanvasElement,
        format: GPUTextureFormat,
    ) {
        this.device = device;   // 保存设备
        this.context = context; // 保存画布上下文
        this.canvas = canvas;   // 保存画布
        this.format = format;   // 保存纹理格式
        this.commonUniformData = new Float32Array(COMMON_UNIFORM_DATA_SIZE); // 分配 CPU 侧 uniform 暂存
    }

    static async create(canvas: HTMLCanvasElement): Promise<Renderer> {
        if (!navigator.gpu) {
            throw new Error("当前浏览器没有 navigator.gpu —— WebGPU 未启用。"); // 环境不支持 WebGPU 直接失败
        }
        const adapter = await navigator.gpu.requestAdapter({ powerPreference: "high-performance" }); // 申请高性能适配器
        if (!adapter) {
            throw new Error("requestAdapter() 返回空 —— 没有可用的 GPU 适配器。"); // 没拿到适配器则中断
        }
        const device = await adapter.requestDevice();  // 从适配器拿到逻辑设备
        const context = canvas.getContext("webgpu");   // 取画布的 webgpu 上下文
        if (!context) {
            throw new Error("canvas.getContext('webgpu') 返回空。"); // 上下文创建失败则中断
        }
        const format = navigator.gpu.getPreferredCanvasFormat(); // 用浏览器推荐格式，避免额外转换开销
        context.configure({ device, format, alphaMode: "opaque" }); // 用设备/格式配置上下文；不透明输出

        const renderer = new Renderer(device, context, canvas, format); // 构造实例
        renderer.init(); // 同步组建管线（着色器由 Vite `?raw` 静态导入，无需等待）
        return renderer;
    }

    private init(): void {
        this.shaderModule = this.device.createShaderModule({
            label: "splat-shader",
            code: splatShaderStr, // Vite `?raw` 在构建时就把 WGSL 源码内联进来
        });

        // 6 个绑定：0 uniform，1..5 都是顶点阶段读的只读 storage
        this.bindGroupLayout = this.device.createBindGroupLayout({
            label: "splat-bind-group-layout",
            entries: [
                { binding: 0, visibility: GPUShaderStage.VERTEX, buffer: { type: "uniform" } },           // 相机/视口参数
                { binding: 1, visibility: GPUShaderStage.VERTEX, buffer: { type: "read-only-storage" } }, // centers
                { binding: 2, visibility: GPUShaderStage.VERTEX, buffer: { type: "read-only-storage" } }, // colorOpacity
                { binding: 3, visibility: GPUShaderStage.VERTEX, buffer: { type: "read-only-storage" } }, // covA
                { binding: 4, visibility: GPUShaderStage.VERTEX, buffer: { type: "read-only-storage" } }, // covB
                { binding: 5, visibility: GPUShaderStage.VERTEX, buffer: { type: "read-only-storage" } }, // 排序后的索引
            ],
        });

        const pipelineLayout = this.device.createPipelineLayout({
            label: "splat-pipeline-layout",
            bindGroupLayouts: [this.bindGroupLayout], // 管线只用到这一组绑定
        });

        this.pipeline = this.device.createRenderPipeline({
            label: "splat-pipeline",
            layout: pipelineLayout,
            vertex: {
                module: this.shaderModule,
                entryPoint: "vs_main",              // 顶点阶段：扩张四边形
                buffers: [{
                    //顶点，一个顶点3个float32，12字节
                    arrayStride: 12,
                    attributes: [{
                        shaderLocation: 0,
                        offset: 0,
                        format: "float32x3",
                    }]
                }]
            },
            fragment: {
                module: this.shaderModule,
                entryPoint: "fs_main",              // 片元阶段：算椭圆高斯核与颜色
                targets: [{ format: this.format, blend: BLEND_OVER }], // 输出到画布并按 over 混合
            },
            primitive: { topology: "triangle-strip", cullMode: "none" }, // 三角带；不剔除背面（billboard 双面可见）
        });

        this.verticesBuffer = this.device.createBuffer({
            label: "splat-vertices",
            size: QUAD_VERTICES.length * 4, // 4 个顶点 × 12 字节
            usage: GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST, // 作为顶点缓冲且允许 CPU 写入
        });
        this.device.queue.writeBuffer(this.verticesBuffer, 0, QUAD_VERTICES); // 把固定的四边形顶点上传一次

        this.commonUniformBuffer = this.device.createBuffer({
            label: "splat-common-uniform",
            size: COMMON_UNIFORM_DATA_SIZE * 4, // 44 个 f32 × 4 字节 = 176 字节
            usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST, // 作为 uniform 且允许 CPU 写入
        });
    }

    /* ------------------------------------------------------------------ *
     * 场景数据
     * ------------------------------------------------------------------ */

    /**
     * 接入 compute 前向直接产出的 storage buffer。
     *
     * 渲染器只持有引用、不负责销毁这些缓冲（它们归 MutantForwardGpu 所有）；
     * 只有排序索引缓冲始终由渲染器自己管理。
     */
    setScene(buffers: SplatBuffers): void {
        this.count = buffers.count;                          // 记录高斯数量
        this.centersBuffer = buffers.centers;                // 直接引用 compute 前向写好的 centers 缓冲
        this.colorOpacityBuffer = buffers.colorOpacity;      // 直接引用颜色+不透明度缓冲
        this.covABuffer = buffers.covA;                      // 直接引用协方差 A 缓冲
        this.covBBuffer = buffers.covB;                      // 直接引用协方差 B 缓冲
        this.sortedBuffer = this.createStorageBuffer("splat-sorted", new Uint32Array(buffers.count)); // 初始全 0；排序后再写真实顺序
        this.rebuildBindGroup();                             // 用新缓冲重建 bind group
    }

    private createStorageBuffer(label: string, data: Float32Array | Uint32Array): GPUBuffer {
        const size = Math.max(data.byteLength, 16); // 至少 16 字节，避免空缓冲非法
        const buffer = this.device.createBuffer({
            label,
            size,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST, // storage + 可写
        });
        // TypedArray 默认泛型是 ArrayBufferLike；这些数组都由 ArrayBuffer 支撑，
        // 断言为 ArrayBuffer 视图以匹配 writeBuffer 的签名
        this.device.queue.writeBuffer(buffer, 0, data as Float32Array<ArrayBuffer> | Uint32Array<ArrayBuffer>);
        return buffer;
    }

    private rebuildBindGroup(): void {
        this.bindGroup = this.device.createBindGroup({
            label: "splat-bindgroup",
            layout: this.bindGroupLayout, // 必须与 init() 里建的同名布局一致
            entries: [
                { binding: 0, resource: { buffer: this.commonUniformBuffer } }, // uniform
                { binding: 1, resource: { buffer: this.centersBuffer } },       // centers
                { binding: 2, resource: { buffer: this.colorOpacityBuffer } },  // colorOpacity
                { binding: 3, resource: { buffer: this.covABuffer } },          // covA
                { binding: 4, resource: { buffer: this.covBBuffer } },          // covB
                { binding: 5, resource: { buffer: this.sortedBuffer } },        // 排序索引
            ],
        });
    }

    /* ------------------------------------------------------------------ *
     * 每帧
     * ------------------------------------------------------------------ */

    resize(width: number, height: number): void {
        const dpr = Math.min(window.devicePixelRatio || 1, 2); // 设备像素比，封顶 2 以免高分屏开销过大
        const w = Math.max(1, Math.floor(width * dpr));  // CSS 宽 × dpr，至少 1 像素
        const h = Math.max(1, Math.floor(height * dpr)); // CSS 高 × dpr，至少 1 像素
        if (this.canvas.width !== w || this.canvas.height !== h) { // 尺寸变了才改画布（改尺寸会重建后缓冲）
            this.canvas.width = w;
            this.canvas.height = h;
        }
        this.width = w;  // 记录当前像素宽
        this.height = h; // 记录当前像素高
    }

    /**
     * 把相机状态写进 uniform 缓冲，字段顺序必须与 WGSL 的 CommonUniforms 结构一致。
     */
    private writeUniforms(state: RenderCameraState): void {
        // 数组下标与 WGSL CommonUniforms 结构按 16-byte 对齐：
        //   [0..16)   viewProj
        //   [16..32)  view
        //   [32..36)  focal (fx, fy, -, -)
        //   [36..40)  viewport (W, H, 2/W, 2/H)
        //   [40..44)  params (kernelSize, minPixelSize, splatScale, -)
        const u = this.commonUniformData; // 复用暂存数组
        u.set(state.viewProj, 0);         // 写入 viewProj（16 个）
        u.set(state.view, 16);            // 写入 view（16 个）
        // focal：像素单位焦距，第 3/4 分量补 0。
        u[32] = state.focal[0];           // fx
        u[33] = state.focal[1];           // fy
        u[34] = 0;                        // 占位
        u[35] = 0;                        // 占位
        // viewport：前两个是像素宽高，后两个是「像素 -> NDC」的换算系数 2/width、2/height。
        u[36] = this.width;               // 视口宽（像素）
        u[37] = this.height;              // 视口高（像素）
        u[38] = 2 / this.width;           // 2/W：clip 到像素坐标换算用
        u[39] = 2 / this.height;          // 2/H
        // params：低通核、最小像素尺寸、全局缩放；第 4 个分量留空。
        u[40] = state.kernelSize;         // 低通滤波强度
        u[41] = state.minPixelSize;       // 最小像素尺寸阈值
        u[42] = state.splatScale;         // 整体缩放
        u[43] = 0;                        // 占位
        this.device.queue.writeBuffer(this.commonUniformBuffer, 0, toGpuBufferSource(u)); // 整块上传到 GPU
    }

    render(state: RenderCameraState): void {
        this.writeUniforms(state); // 先把相机/视口参数写进 uniform 缓冲

        const encoder = this.device.createCommandEncoder({ label: "frame" }); // 本帧的命令编码器
        const view = this.context.getCurrentTexture().createView();          // 取画布当前帧纹理的视图
        const pass = encoder.beginRenderPass({
            label: "splat-pass",
            colorAttachments: [{
                view,
                clearValue: BACKGROUND, // 用背景色清屏
                loadOp: "clear",        // 每帧清一次
                storeOp: "store",       // 结果保留下来（要显示）
            }],
        });

        if (this.count > 0 && this.bindGroup) { // 有高斯才绘制
            pass.setPipeline(this.pipeline);               // 选管线
            pass.setBindGroup(0, this.bindGroup);          // 绑定第 0 组（全部缓冲）
            pass.setVertexBuffer(0, this.verticesBuffer);  // 槽 0 = 四边形顶点
            // triangle-strip 用 4 个顶点画一个四边形，实例数 = 高斯数
            pass.draw(4, this.count);
        }

        pass.end();                                    // 结束渲染 pass
        this.device.queue.submit([encoder.finish()]);  // 提交命令缓冲，GPU 开始执行
    }

    dispose(): void {
        // 只销毁渲染器自己创建的资源；外部 compute 前向的缓冲由它自己管理。
        this.sortedBuffer?.destroy();
        this.commonUniformBuffer?.destroy();
        this.verticesBuffer?.destroy();
    }
}

function toGpuBufferSource(data: Float32Array | Uint32Array): ArrayBufferView<ArrayBuffer> {
    // 类型转换只满足 WebGPU TypeScript 定义；底层仍是原来的连续 TypedArray 字节。
    return data as Float32Array<ArrayBuffer> | Uint32Array<ArrayBuffer>;
}
