/**
 * ViewerApp.ts —— 应用的「外壳」：把渲染器、GPU 前向、排序器、轨道相机串起来
 *
 * 与 gs-webgpu 的 ViewerApp 结构对齐，但这里没有标注/拾取/pointer 交互：
 * 本 Demo 只用一套固定流程 —— 每帧推进时间 → 跑一遍 compute 前向 → GPU 排序 → 渲染。
 *
 * 各角色的职责：
 *   - Renderer      ：只管画高斯（读 compute 前向写好的 storage buffer）；
 *   - MutantForwardGpu：每帧把形变后的高斯写进那几个 storage buffer（零回读）；
 *   - GPUSorter     ：按相机深度把高斯从远到近排好，决定绘制顺序；
 *   - OrbitCamera   ：three 的透视相机 + OrbitControls，提供渲染器要的矩阵与参数。
 */

import { Renderer, type RenderCameraState } from "./Renderer";
import { GPUSorter } from "./sortGPU";
import { MutantForwardGpu } from "./deform_mutant_gpu";
import { OrbitCamera } from "./camera";
import { createMutantField, type MutantField, type MutantManifest } from "./deform_onnx";

/**
 * 拉取清单与权重，构造 MutantField（GPU 前向与权重都从这里取）。
 * 与旧 main.ts 里的同名函数逻辑一致，只是搬到了 ViewerApp 里。
 */
async function loadMutantField(): Promise<MutantField> {
    const [manifestRes, weightRes] = await Promise.all([
        fetch("./models/mutant_manifest.json"), // 清单：各张量 offset、aabb、平面描述
        fetch("./models/mutant_weights.bin"),   // 权重二进制（约 15.6 MB 的 f32 流）
    ]);
    if (!manifestRes.ok) {
        throw new Error(`加载 mutant_manifest.json 失败（HTTP ${manifestRes.status}）`); // 清单请求失败
    }
    if (!weightRes.ok) {
        throw new Error(`加载 mutant_weights.bin 失败（HTTP ${weightRes.status}）`); // 权重请求失败
    }
    const manifest = (await manifestRes.json()) as MutantManifest;              // 解析清单 JSON
    const weights = new Float32Array(await weightRes.arrayBuffer());           // 权重按 f32 视图读取
    return createMutantField(manifest, weights);                               // 组装成 MutantField
}

export class ViewerApp {
    canvas: HTMLCanvasElement;        // 承载画面的 DOM 画布
    renderer: Renderer | null;        // WebGPU 渲染器（异步创建）
    sorter: GPUSorter | null;         // GPU 深度排序器（异步创建）
    forward: MutantForwardGpu | null; // GPU compute 前向（每帧刷新高斯参数）
    camera: OrbitCamera;              // 轨道相机（拖拽旋转 / 滚轮缩放）
    ready: Promise<void>;             // 异步初始化的句柄，外部可 await

    frameId: number | null;           // 当前 requestAnimationFrame 句柄
    resizeHandler: () => void;        // 窗口尺寸变化的监听器（保存引用以便解绑）
    disposed: boolean;                // 幂等释放标志

    time: number;                     // 归一化时间 [0, 1]，每帧自动推进
    lastFrame: number;                // 上一帧时间戳，用来算 dt

    constructor(canvas: HTMLCanvasElement) {
        this.canvas = canvas;                           // 保存画布
        this.renderer = null;                           // 渲染器待异步创建
        this.sorter = null;                             // 排序器待异步创建
        this.forward = null;                            // 前向待异步创建

        // 相机对准这个模型：mutant 模型按 2.5 倍缩放，从正面（phi = 90°）观看
        this.camera = new OrbitCamera(canvas);          // 建轨道相机，并接管画布上的指针事件
        this.camera.setOrbit(5.3, 0, Math.PI * 0.5);    // 距离 5.3、俯仰 0、方位 90°（正对模型正面）

        this.frameId = null;                            // 尚无帧循环
        this.disposed = false;                          // 尚未释放
        this.time = 0;                                  // 时间从 0 开始
        this.lastFrame = performance.now();             // 记录起始时间戳

        // 用箭头函数包一层，保留 this 绑定以便 removeEventListener 时引用一致。
        this.resizeHandler = () => this.handleResize();
        window.addEventListener("resize", this.resizeHandler);

        // 构造完成即开始异步初始化；外部可 await this.ready。
        this.ready = this.init();
    }

    private async init(): Promise<void> {
        const renderer = await Renderer.create(this.canvas); // 初始化 WebGPU 渲染器（含建管线）
        this.renderer = renderer;

        // 加载权重 → 建立 GPU 前向 → 把它的输出缓冲交给渲染器（零回读）
        const field = await loadMutantField();                              // 拉取清单+权重，构造 MutantField
        const forward = MutantForwardGpu.create(renderer.device, field);    // 用同一个 device 建 compute 前向
        this.forward = forward;
        renderer.setScene(forward.externalBuffers());                       // 把前向的输出缓冲直接交给渲染器
        this.handleResize();                                                // 按窗口尺寸初始化画布

        // 建 GPU 排序器：深度范围与排序全部在 GPU 上完成，全程不回读 CPU。
        this.sorter = GPUSorter.create(renderer.device, renderer.count);
        this.sorter.bindCenters(renderer.centersBuffer, renderer.sortedBuffer);

        this.startLoop();                                                   // 启动主循环
    }

    private startLoop(): void {
        this.handleResize();                    // 启动前先对齐一次尺寸
        this.lastFrame = performance.now();     // 重置计时基准，避免首帧 dt 过大

        const tick = (now: number): void => {
            if (this.disposed) return;                              // 已释放就不再排下一帧
            this.frameId = requestAnimationFrame(tick);             // 先排下一帧
            const renderer = this.renderer;
            const forward = this.forward;
            if (!renderer || !forward) return;                      // 资源未就绪则跳过本帧

            const dt = Math.min(0.1, (now - this.lastFrame) / 1000); // 帧间隔（秒），封顶 0.1 防止卡顿后跳变过大
            this.lastFrame = now;

            this.camera.update();                                   // 更新相机（阻尼/惯性）
            this.time = (this.time + dt * 0.15) % 1;                // 时间自动推进并循环到 [0,1)

            // 跑一遍 compute 前向：形变结果直接写进渲染器要读的那几个 storage buffer，
            // 不重建 buffer、不回读 CPU，所以每帧调用也几乎无成本。
            forward.run(this.time);

            const state = this.getRenderCameraState();              // 组装本帧相机状态
            if (renderer.count > 0 && this.sorter) {                // 有高斯才排序
                this.sorter.sort({
                    camPos: state.camPos,                           // 相机位置（深度键要用）
                    forward: state.forward,                         // 相机前向（深度键要用）
                    count: renderer.count,                          // 元素数
                });
            }
            renderer.render(state);                                 // 绘制
        };

        this.frameId = requestAnimationFrame(tick);                 // 启动主循环
    }

    /** 把 OrbitCamera 的矩阵 / 焦距 / 位置打包成渲染器要的 RenderCameraState。 */
    private getRenderCameraState(): RenderCameraState {
        const renderer = this.renderer!;                            // 调用前已确保非空
        const width = renderer.width;                               // 画布像素宽
        const height = renderer.height;                             // 画布像素高
        const aspect = width / height;                              // 视口宽高比，供投影矩阵用
        const focal = this.camera.getFocal(width, height);          // 像素焦距 [fx, fy]

        return {
            viewProj: this.camera.getViewProjection(aspect),        // 视图×投影矩阵
            view: this.camera.getView(),                            // 视图矩阵
            camPos: this.camera.getEye(),                           // 相机世界位置
            forward: this.camera.getForward(),                      // 相机前向（排序深度键）
            focal: [focal[0], focal[1]],                            // 像素焦距
            width,
            height,
            kernelSize: 0.3,                                        // 低通滤波强度（把过小高斯撑开）
            minPixelSize: 0,                                        // 不按最小像素丢弃
            splatScale: 1,                                          // 高斯整体缩放倍率
        };
    }

    private handleResize(): void {
        if (!this.renderer) return;                                 // 渲染器未就绪就先不管
        // 相机/Renderer 都用 CSS 像素尺寸，Renderer 内部再按 devicePixelRatio 设实际分辨率。
        const width = this.canvas.clientWidth || window.innerWidth;  // 画布 CSS 宽
        const height = this.canvas.clientHeight || window.innerHeight; // 画布 CSS 高
        this.renderer.resize(width, height);                        // 同步尺寸
    }

    dispose(): void {
        if (this.disposed) return;                                  // 幂等：重复调用直接返回
        this.disposed = true;

        if (this.frameId !== null) {                                // 取消未执行的下一帧
            cancelAnimationFrame(this.frameId);
            this.frameId = null;
        }
        window.removeEventListener("resize", this.resizeHandler);   // 解绑窗口监听
        this.camera.controls.dispose();                             // 释放 OrbitControls
        this.sorter?.dispose();                                     // 释放排序器内部缓冲
        this.renderer?.dispose();                                   // 释放渲染器自建资源
    }
}
