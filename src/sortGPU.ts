/**
 * sortGPU.ts —— WebGPU compute 版 4-bit 基数排序（第 11 章）
 *
 * 每条记录是一个 32 位「组合键」——高位是深度、低位是原始索引——
 * 每轮处理 RADIX_BITS 位，一轮 = histogram → scan → scatter 三个 dispatch，
 * 共 TOTAL_BITS / RADIX_BITS 轮。
 *
 * 为什么 scatter 必须「稳定」？（这是不闪烁的关键）
 *   LSD 基数排序要求每一趟都稳定：同一个桶里，原本靠前的元素排完还得靠前。
 *   如果某一趟靠「原子加抢占桶内空位」来分位置，谁先抢到就由 GPU 线程调度决定，
 *   每帧都可能不一样；LSD 会把这个错乱放大到最终结果 —— 画面每帧都在跳（闪烁）。
 *   所以这里采用分块直方图 + 两级前缀和的做法：先把每个「块 × 桶」的写出起点算死，
 *   scatter 再按原索引顺序逐块写出，全程与线程调度无关，每帧结果一致。
 *
 * 为什么键里要塞原始索引？
 *   让所有键互不相等 —— 深度相同的元素也有确定的先后，进一步消除不确定性。
 *
 * 深度范围（minD/maxD）由 GPU 归约（cs_depth_range）算出，全程不回读 CPU ——
 * 这正是 WebGPU 相对 WebGL2 的最大好处（storage buffer + compute）。
 */

import keygenShaderStr from "./shaders/sort_keygen.wgsl?raw";
import histogramShaderStr from "./shaders/sort_histogram.wgsl?raw";
import scanShaderStr from "./shaders/sort_scan.wgsl?raw";
import scatterShaderStr from "./shaders/sort_scatter.wgsl?raw";

// 每个 block 负责 256 个键（= 一个工作组的线程数，对应 WGSL 里的 @workgroup_size(256)）。
// 排序被切成固定大小的块，直方图、前缀和、散射三阶段共享同一块划分。
const BLOCK_KEYS = 256;
// 每一轮处理的二进制位数（4 位 = 恰好能表示 0~15 共 16 个桶）。
const RADIX_BITS = 4;
// 桶的个数 = 2^RADIX_BITS = 16。
const RADIX_BINS = 1 << RADIX_BITS;
// 组合键总位数固定为 32；其中高位 depthBits 存深度、低位 indexBits 存原始索引。
const TOTAL_BITS = 32;
// 需要跑的总轮数 = 32/4 = 8 轮（每轮消化 4 位，8 轮刚好覆盖 32 位）。
const ROUNDS = TOTAL_BITS / RADIX_BITS; // 8
// 公共参数 uniform（keygen 与各轮共用）：按 16-byte 对齐，总长 64 字节。
const COMMON_BYTES = 64;
// 每轮 shift 独占一块 uniform：shader 里就是一个 u32，缓冲区 4 字节即可。
const SHIFT_BYTES = 4;

/** 返回容纳索引 0..count-1 所需的最少 bit 数，即 ceil(log2(count))。 */
function indexBitsForCount(count: number): number {
    // 至少 1 位（count<=1 时也得有位可用）。
    let bits = 1;
    // 逐步倍增直到 2^bits 足以覆盖 count（最多 31 位，留 1 位安全余量）。
    while (bits < 31 && 2 ** bits < count) {
        bits += 1;
    }
    return bits;
}

/** CPU 侧的一次排序参数；发送给 GPU 时会拆成「公共 uniform + 每轮 shift uniform」两部分。 */
interface SortParams {
    // 元素总数。
    count: number;
    // 本轮从组合键的第几位开始取 4 位（0/4/…/28）。
    shift: number;
    /** 组合键高位（深度）占的 bit 数。 */
    depthBits: number;
    /** 组合键低位（原始索引）占的 bit 数。 */
    indexBits: number;
    // 相机位置（keygen 用它算深度）。
    camPos: ArrayLike<number>;
    // 相机朝向（keygen 用它算深度）。
    forward: ArrayLike<number>;
}

/** 一次排序调用所需的外部资源与相机参数。 */
export interface SortArgs {
    /** 相机位置。 */
    camPos: ArrayLike<number>;
    /** 相机朝向。 */
    forward: ArrayLike<number>;
    /** 元素总数。 */
    count: number;
}

/** 一个 uniform 槽：CPU 侧镜像 data 用于写参数，buffer 才是 GPU 读取的对象。 */
interface SortUniform {
    // data 是 CPU 侧写入镜像；buffer 是 GPU dispatch 实际读取的 uniform。
    buffer: GPUBuffer;
    data: ArrayBuffer;
}

// GPU 4-bit 基数排序器：把一堆「组合键 + 原始索引」按深度从远到近（键从小到大）稳定排序。
export class GPUSorter {
    // 创建缓冲/管线等资源用的逻辑设备。
    device: GPUDevice;
    // 要排序的元素个数，构造时固定。
    count: number;
    /** 组合键低位（原始索引）占的 bit 数 = ceil(log2(count))。 */
    indexBits: number;
    /** 组合键高位（深度）占的 bit 数 = 32 - indexBits。 */
    depthBits: number;
    /** 键数组被划分成的块数（每块 BLOCK_KEYS 个键）。 */
    numBlocks: number;

    // 以下字段在 init() / bindCenters() 里赋值，用 ! 断言
    // 排序用到的 4 段着色器模块。
    keygenModule!: GPUShaderModule;
    histogramModule!: GPUShaderModule;
    scanModule!: GPUShaderModule;
    scatterModule!: GPUShaderModule;
    // 5 条计算管线（keygen / 深度范围 / histogram / scan / scatter）。
    keygenPipeline!: GPUComputePipeline;
    depthRangePipeline!: GPUComputePipeline;
    histogramPipeline!: GPUComputePipeline;
    scanPipeline!: GPUComputePipeline;
    scatterPipeline!: GPUComputePipeline;
    // A/B 两组键缓冲：乒乓交替，一轮的输入是上一轮输出。
    keysA!: GPUBuffer;
    keysB!: GPUBuffer;
    // 与键缓冲配对的「原始索引」值缓冲。
    valuesA!: GPUBuffer;
    valuesB!: GPUBuffer;
    /** 分块直方图/前缀和缓冲：numBlocks × 16 个 u32。 */
    blockHistBuffer!: GPUBuffer;
    /** 深度范围 [minD, maxD]（2 个 f32），由 GPU 归约写入、keygen 读取。 */
    depthRangeBuffer!: GPUBuffer;
    // 公共参数（相机/块数）在 keygen 与 8 轮里完全相同，只写一份。
    commonUniform!: SortUniform;
    /** 每轮独占的 shift uniform；每轮一块，避免 write-before-submit 读到同一份。 */
    shiftUniforms!: SortUniform[];

    // 渲染端 centers 缓冲的引用（binding 1 要指向它，bindCenters 之后才有）。
    centersBufferRef!: GPUBuffer;
    // 渲染端要用的排序结果缓冲（最后一轮直接写它）。
    sortedBufferRef!: GPUBuffer;
    // 工厂函数：给定某轮的 shift 下标与输入/输出缓冲，造出一个 bind group。
    bindGroupFactory!: (
        shiftIndex: number,
        keysIn: GPUBuffer,
        valuesIn: GPUBuffer,
        keysOut: GPUBuffer,
        valuesOut: GPUBuffer,
    ) => GPUBindGroup;
    // keygen 阶段用的 bind group（读 B 写 A，保证 key 写入 keysA）。
    keygenBindGroup!: GPUBindGroup;
    // 每一轮各自的 bind group（奇偶轮交替 A/B 乒乓）。
    roundBindGroups!: GPUBindGroup[];

    // 构造：只记录基本参数并算出 indexBits/depthBits/numBlocks，不碰 GPU 资源。
    constructor(device: GPUDevice, count: number) {
        this.device = device;
        this.count = count;
        // 低位留出足够多的 bit 容纳全部原始索引，使每个高斯拿到互不相同的键。
        // 这是「排序结果确定、不闪烁」的关键：键唯一 ⇒ 不存在并列项。
        this.indexBits = indexBitsForCount(count);
        // 高位剩下的位数都给深度（32 - 索引位数）。
        this.depthBits = TOTAL_BITS - this.indexBits;
        // 块数 = 向上取整(count/256)，至少 1 块（count 很小时也保证有 1 块）。
        this.numBlocks = Math.max(1, Math.ceil(count / BLOCK_KEYS));
    }

    // 工厂：先构造，再同步把 GPU 资源全部建好（着色器由 `?raw` 静态导入）。
    static create(device: GPUDevice, count: number): GPUSorter {
        const sorter = new GPUSorter(device, count);
        sorter.init();
        return sorter;
    }

    // 初始化：建 4 个模块、布局与 5 条管线、乒乓缓冲、每轮 uniform、bind group 工厂。
    private init(): void {
        this.keygenModule = this.device.createShaderModule({ label: "sort-keygen", code: keygenShaderStr });
        this.histogramModule = this.device.createShaderModule({ label: "sort-histogram", code: histogramShaderStr });
        this.scanModule = this.device.createShaderModule({ label: "sort-scan", code: scanShaderStr });
        this.scatterModule = this.device.createShaderModule({ label: "sort-scatter", code: scatterShaderStr });

        // bind group 布局：9 个 binding（公共 uniform / centers / keysIn / valuesIn / keysOut / valuesOut / blockHist / shift / depthRange）。
        const bindGroupLayout = this.device.createBindGroupLayout({
            label: "sort-bindgroup-layout",
            entries: [
                { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: "uniform" } },           // 公共参数
                { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } }, // 中心点
                { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } }, // keysIn
                { binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } }, // valuesIn
                { binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },           // keysOut
                { binding: 5, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },           // valuesOut
                { binding: 6, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },           // blockHist
                { binding: 7, visibility: GPUShaderStage.COMPUTE, buffer: { type: "uniform" } },           // 每轮 shift
                { binding: 8, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },           // 深度范围（归约写、keygen 读）
            ],
        });
        const pipelineLayout = this.device.createPipelineLayout({
            label: "sort-pipeline-layout",
            bindGroupLayouts: [bindGroupLayout], // 5 条计算管线都复用它，保证 bind group 通用
        });

        // 小工具：给定标签、模块、入口函数名，造一条共用该布局的计算管线。
        const compute = (label: string, module: GPUShaderModule, entryPoint: string): GPUComputePipeline =>
            this.device.createComputePipeline({ label, layout: pipelineLayout, compute: { module, entryPoint } });
        this.keygenPipeline = compute("sort-keygen-pipeline", this.keygenModule, "cs_keygen");
        this.depthRangePipeline = compute("sort-depth-range-pipeline", this.keygenModule, "cs_depth_range");
        this.histogramPipeline = compute("sort-histogram-pipeline", this.histogramModule, "cs_histogram");
        this.scanPipeline = compute("sort-scan-pipeline", this.scanModule, "cs_scan");
        this.scatterPipeline = compute("sort-scatter-pipeline", this.scatterModule, "cs_scatter");

        // 单缓冲所需字节数：n 个 u32；至少 16 字节（WebGPU 缓冲最小尺寸限制）。
        const bytes = Math.max(this.count * 4, 16);
        // 乒乓缓冲的用途：存取用的 storage，同时允许拷贝。
        const pairUsage = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST;
        // A/B 两组键和值交替读写，避免同一轮同时读写同一个 storage buffer。
        this.keysA = this.device.createBuffer({ label: "sort-keys-a", size: bytes, usage: pairUsage });
        this.keysB = this.device.createBuffer({ label: "sort-keys-b", size: bytes, usage: pairUsage });
        this.valuesA = this.device.createBuffer({ label: "sort-values-a", size: bytes, usage: pairUsage });
        this.valuesB = this.device.createBuffer({ label: "sort-values-b", size: bytes, usage: pairUsage });

        // 分块直方图缓冲：numBlocks × 16 个桶，每个 u32 4 字节。
        this.blockHistBuffer = this.device.createBuffer({
            label: "sort-block-hist",
            size: this.numBlocks * RADIX_BINS * 4,
            usage: pairUsage,
        });
        // 深度范围 [minD, maxD]（2 个 f32）。归约阶段每帧完整重写，无需 CPU 初始化。
        this.depthRangeBuffer = this.device.createBuffer({
            label: "sort-depth-range-buffer",
            size: 16,
            usage: GPUBufferUsage.STORAGE,
        });

        // 公共参数（相机/块数）在 keygen 与 8 轮里完全相同，只写一份。
        this.commonUniform = {
            buffer: this.device.createBuffer({
                label: "sort-common-uniform",
                size: COMMON_BYTES,
                usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
            }),
            data: new ArrayBuffer(COMMON_BYTES),
        };

        // 每轮一个 shift uniform：因为 queue.writeBuffer 是「立即提交」的，
        // 没法在同一个 encoder 里给 8 轮写入不同的 shift，所以必须每轮独占一个缓冲。
        this.shiftUniforms = [];
        for (let r = 0; r < ROUNDS; r++) {
            this.shiftUniforms.push({
                buffer: this.device.createBuffer({
                    label: `sort-shift-uniform-${r}`,
                    size: SHIFT_BYTES,
                    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
                }),
                data: new ArrayBuffer(SHIFT_BYTES),
            });
        }

        // 工厂函数：把公共 uniform、该轮 shift 与输入/输出缓冲组装成一个 bind group。
        this.bindGroupFactory = (shiftIndex, keysIn, valuesIn, keysOut, valuesOut) => this.device.createBindGroup({
            layout: bindGroupLayout,
            entries: [
                { binding: 0, resource: { buffer: this.commonUniform.buffer } },
                { binding: 1, resource: { buffer: this.centersBufferRef } },  // 此时可能尚未赋值，故延后到 bindCenters 时才会真正建 bind group
                { binding: 2, resource: { buffer: keysIn } },
                { binding: 3, resource: { buffer: valuesIn } },
                { binding: 4, resource: { buffer: keysOut } },
                { binding: 5, resource: { buffer: valuesOut } },
                { binding: 6, resource: { buffer: this.blockHistBuffer } },
                { binding: 7, resource: { buffer: this.shiftUniforms[shiftIndex].buffer } },
                { binding: 8, resource: { buffer: this.depthRangeBuffer } },
            ],
        });
    }

    /** 渲染器在 setScene 之后调用，把 centers / sorted 缓冲接进来并建好全部 bind group */
    bindCenters(centerBuffer: GPUBuffer, sortedBuffer: GPUBuffer): void {
        this.centersBufferRef = centerBuffer;
        this.sortedBufferRef = sortedBuffer;

        // keygen 的 bind group：读 B（当时为空）、写 A —— 保证 key 最终落在 A 里，方便后面的轮次从 A 开始。
        this.keygenBindGroup = this.bindGroupFactory(0, this.keysB, this.valuesB, this.keysA, this.valuesA);
        // 给 8 轮各建一个 bind group。
        this.roundBindGroups = [];
        for (let r = 0; r < ROUNDS; r++) {
            // 偶数轮输入取 A，奇数轮取 B。
            const even = r % 2 === 0;
            // 最后一轮的值不再写回内部 buffer，而是直接落到渲染器的 sortedBuffer，
            // 省去排序末尾一次整段 copyBufferToBuffer。
            const last = r === ROUNDS - 1;
            this.roundBindGroups.push(this.bindGroupFactory(
                r,
                even ? this.keysA : this.keysB,
                even ? this.valuesA : this.valuesB,
                even ? this.keysB : this.keysA,
                last ? this.sortedBufferRef : (even ? this.valuesB : this.valuesA),
            ));
        }
    }

    // 把公共参数写进 uniform staging 内存并上传到 GPU。
    private writeCommon(params: SortParams): void {
        const entry = this.commonUniform;
        // 用同一块 ArrayBuffer 开两个视图：u32 视图写整数字段，f32 视图写浮点字段。
        const u32 = new Uint32Array(entry.data);
        const f32 = new Float32Array(entry.data);
        u32[0] = params.count;   // 元素数量
        u32[1] = params.depthBits; // 深度位数
        u32[2] = params.indexBits; // 索引位数
        u32[3] = 0;              // 补齐（使 camPos 落在 16 字节边界）
        f32[4] = params.camPos[0]; // 相机位置 xyz
        f32[5] = params.camPos[1];
        f32[6] = params.camPos[2];
        f32[7] = 1;              // 补齐 vec4 的第 4 分量
        f32[8] = params.forward[0]; // 相机朝向 xyz
        f32[9] = params.forward[1];
        f32[10] = params.forward[2];
        f32[11] = 0;             // 补齐 vec4
        u32[12] = this.numBlocks; // 块数
        u32[13] = 0;             // 补齐
        this.device.queue.writeBuffer(entry.buffer, 0, entry.data);
    }

    // 把本轮 shift 写进第 slot 份 uniform 并上传。
    private writeShift(slot: number, shift: number): void {
        const entry = this.shiftUniforms[slot];
        // shader 侧 sh 是单个 u32，只写这 4 字节。
        new Uint32Array(entry.data)[0] = shift;
        this.device.queue.writeBuffer(entry.buffer, 0, entry.data);
    }

    /**
     * 一趟完整的 GPU 排序。
     * @returns 是否成功（失败时调用方应回退到原始顺序）
     */
    sort(args: SortArgs): boolean {
        // 元素个数对不上、或还没 bindCenters（roundBindGroups 为空）就不能跑，返回失败让上层回退。
        if (args.count !== this.count || !this.roundBindGroups) {
            return false;
        }

        // 写公共参数（相机/块数）；深度范围不在这里算，改由 GPU 归约。
        this.writeCommon({
            count: args.count,
            shift: 0,
            depthBits: this.depthBits,
            indexBits: this.indexBits,
            camPos: args.camPos,
            forward: args.forward,
        });
        // 每轮的 shift 依次为 0,4,8,…28；每轮写自己的那块 uniform。
        for (let r = 0; r < ROUNDS; r++) {
            this.writeShift(r, r * RADIX_BITS);
        }

        const blocks = this.numBlocks;
        // 建一个命令编码器，把这一整趟排序的所有 GPU 命令先录进去，最后一次性提交。
        const encoder = this.device.createCommandEncoder({ label: "sort" });

        // —— 深度范围归约：只跑 1 个工作组，把 minD/maxD 写进 depthRangeBuffer ——
        const rangePass = encoder.beginComputePass({ label: "depth-range-pass" });
        rangePass.setPipeline(this.depthRangePipeline);
        rangePass.setBindGroup(0, this.keygenBindGroup); // 归约只用到 centers / 公共 uniform / depthRange
        rangePass.dispatchWorkgroups(1);
        rangePass.end();

        // —— keygen：只跑一次，算组合键并把原始索引写进 valuesA ——
        const keygenPass = encoder.beginComputePass({ label: "keygen-pass" });
        keygenPass.setPipeline(this.keygenPipeline);
        keygenPass.setBindGroup(0, this.keygenBindGroup);
        keygenPass.dispatchWorkgroups(blocks);
        keygenPass.end();

        // —— 8 轮 histogram + scan + scatter ——
        for (let r = 0; r < ROUNDS; r++) {
            const pass = encoder.beginComputePass({ label: `round-${r}` });
            const bindGroup = this.roundBindGroups[r];

            // 第 1 小步：histogram，数出每块每桶的个数。
            pass.setPipeline(this.histogramPipeline);
            pass.setBindGroup(0, bindGroup);
            pass.dispatchWorkgroups(blocks);

            // 第 2 小步：scan，把个数变成每块每桶的起点（只需 1 个工作组）。
            pass.setPipeline(this.scanPipeline);
            pass.setBindGroup(0, bindGroup);
            pass.dispatchWorkgroups(1);

            // 第 3 小步：scatter，按起点稳定写出到输出缓冲。
            pass.setPipeline(this.scatterPipeline);
            pass.setBindGroup(0, bindGroup);
            pass.dispatchWorkgroups(blocks);

            pass.end();
        }

        // 8 轮结束后结果已直接落在渲染器的 sortedBuffer（最后一轮绑定），无需再回读/拷贝。
        this.device.queue.submit([encoder.finish()]);
        return true;
    }

    dispose(): void {
        // GPUBuffer 必须显式销毁；页面关闭或重建排序器时调用。
        this.keysA.destroy();
        this.keysB.destroy();
        this.valuesA.destroy();
        this.valuesB.destroy();
        this.blockHistBuffer.destroy();
        this.depthRangeBuffer.destroy();
        this.commonUniform.buffer.destroy();
        for (const shift of this.shiftUniforms) {
            shift.buffer.destroy();
        }
    }
}
