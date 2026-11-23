/**
 * camera.ts —— 轨道相机：three.js 透视相机 + OrbitControls
 *
 * 这是整个 Demo 里唯一「不手写」的一块，原因很务实：
 *   - 拖拽旋转 / 滚轮缩放 / 右键平移这些交互，OrbitControls 已经做得很完善，
 *     教学时没必要重复造轮子；
 *   - 我们真正要讲的是「相机矩阵怎么喂给高斯渲染器」，相机状态交给 three 就好。
 *
 * 与渲染器之间只差一个约定要处理：
 *   three 的相机沿用 WebGL 约定 —— NDC 深度范围是 [-1, 1]；
 *   而 WebGPU 的 NDC 深度范围是 [0, 1]。
 *   所以不能直接把 `camera.projectionMatrix` 丢给着色器，要先左乘一个 z 重映射矩阵
 *   （见下方 NDC_Z_TO_ZERO_ONE），否则近远平面的裁剪会错位。
 *
 * 对外只需要三样东西（构造 RenderState 时用）：
 *   1. view / viewProj 矩阵（第 03 章）；
 *   2. focal（像素焦距），EWA 投影里的 Jacobian 要用（第 05 章）；
 *   3. 眼睛位置 + 前向，用来算每个高斯的深度键（第 07 章）。
 */

// 从 three 引入透视相机与三维向量工具。
import { PerspectiveCamera, Vector3 } from "three";
// 引入 three 官方示例里的轨道控制器（负责鼠标/触摸的旋转、缩放、平移）。
import { OrbitControls } from "three/examples/jsm/controls/OrbitControls.js";
// 复用本地数学工具：角度→弧度常量 DEG，以及 4x4 矩阵乘法。
import { DEG, mat4Multiply } from "./math";

/**
 * 把 WebGL 裁剪空间的 z（NDC ∈ [-1, 1]）重映射到 WebGPU 的 [0, 1]：
 *     z_clip' = 0.5 * z_clip + 0.5 * w
 * 列主序存储，与 WGSL 的 mat4x4<f32> 内存布局一致。
 */
// 预计算好的 z 重映射矩阵常量，把它左乘到 viewProj 上即可完成深度区间转换。
const NDC_Z_TO_ZERO_ONE = new Float32Array([
    // 第 0 列：x 不变。
    1, 0, 0, 0,
    // 第 1 列：y 不变。
    0, 1, 0, 0,
    // 第 2 列：z 缩放 0.5（第 4 行分量留 0，齐次部分由下一列提供）。
    0, 0, 0.5, 0,
    // 第 3 列：z 偏移 0.5·w —— 列主序下 m[14]=0.5 会把 w 的 0.5 倍叠加到 z 上。
    0, 0, 0.5, 1,
]);

// 轨道相机封装：内部持有 three 相机与控制器，对外暴露渲染器所需的矩阵与参数。
export class OrbitCamera {
    /** three 的透视相机：位置 / 朝向 / 视锥由它和 OrbitControls 共同维护 */
    readonly camera: PerspectiveCamera;
    /** 轨道控制器：接管画布上的鼠标 / 触摸输入 */
    readonly controls: OrbitControls;

    // 构造函数：创建透视相机并配置控制器与默认机位。
    constructor(canvas: HTMLCanvasElement) {
        // 创建 three 透视相机。
        this.camera = new PerspectiveCamera(
            55,     // fov：竖直方向视角度数
            1,      // aspect：占位，每帧按画布尺寸更新
            0.02,   // near
            100,    // far
        );

        // 把控制器绑定到相机与画布上（此后由它接管画布交互）。
        this.controls = new OrbitControls(this.camera, canvas);
        // 环绕中心设在原点。
        this.controls.target.set(0, 0, 0);
        this.controls.enableDamping = true;    // 阻尼把拖拽的突兀感抹平
        // 阻尼系数：越大跟随越慢、越平滑。
        this.controls.dampingFactor = 0.08;
        // 最近可缩放距离（防止穿进模型内部）。
        this.controls.minDistance = 0.35;
        // 最远可缩放距离。
        this.controls.maxDistance = 40;

        // 默认机位（球坐标，等价于旧版 radius=3.2 / theta=0.6 / phi=1.25）
        // 设置默认环绕机位。
        this.setOrbit(3.2, 0.6, 1.25);
    }

    /**
     * 把相机摆到球坐标 (radius, theta, phi)，始终环绕 controls.target：
     *   theta：水平角（绕 y 轴）；phi：天顶角，0 = 正上方。
     */
    // 按球坐标摆放相机。
    setOrbit(radius: number, theta: number, phi: number): void {
        // 天顶角的正弦，用于把半径投影到水平面。
        const sp = Math.sin(phi);
        // 由球坐标换算成相对 target 的笛卡尔偏移向量。
        const offset = new Vector3(
            radius * sp * Math.sin(theta),
            radius * Math.cos(phi),
            radius * sp * Math.cos(theta),
        );
        // 相机位置 = 目标点 + 偏移。
        this.camera.position.copy(this.controls.target).add(offset);
        // 让相机朝向目标点。
        this.camera.lookAt(this.controls.target);
        // 立即刷新控制器内部状态，使其球坐标缓存与新位置一致。
        this.controls.update();
    }

    /** 每帧调用一次：推进 OrbitControls 的阻尼动画 */
    // 每帧推进一步阻尼动画。
    update(): void {
        this.controls.update();
    }

    /** 相机在世界空间的位置 */
    // 取相机世界坐标，打包成裸数组 [x,y,z]。
    getEye(): Float32Array {
        // three 的向量对象。
        const p = this.camera.position;
        // 拷贝成分量数组，避免外部拿到的引用被 three 后续修改。
        return new Float32Array([p.x, p.y, p.z]);
    }

    /** 相机世界前向（眼睛指向目标），用于 GPU 排序的深度键 */
    // 计算相机前向单位向量。
    getForward(): Float32Array {
        // target - position 再归一化，即从眼睛指向目标的视线方向。
        const d = new Vector3().subVectors(this.controls.target, this.camera.position).normalize();
        // 拷贝成分量数组。
        return new Float32Array([d.x, d.y, d.z]);
    }

    /** view 矩阵（列主序）；three 的 matrixWorldInverse 就是我们要的 world → camera */
    // 取视图（世界→相机）矩阵。
    getView(): Float32Array {
        // 先确保世界矩阵是最新的，否则其逆矩阵可能滞后一帧。
        this.camera.updateMatrixWorld();
        // 直接拷贝 three 的 world→camera 矩阵元素（已是列主序）。
        return new Float32Array(this.camera.matrixWorldInverse.elements);
    }

    /**
     * 像素单位焦距。对称透视下 focal.x == focal.y，这里仍分别给出，
     * 方便第 05 章对照非对称视锥的写法。
     * 推导：x_ndc = (f / aspect) · x_cam / z_cam，像素 x = x_ndc · W / 2，
     * 故 focal.x = (W / 2) · f / aspect = (H / 2) · f。
     */
    // 计算像素焦距 [fx, fy]。
    getFocal(width: number, height: number): number[] {
        // f = 1/tan(fovY/2)，与投影矩阵里的 f 一致。
        const f = 1 / Math.tan((this.camera.fov * DEG) * 0.5);
        // 竖直像素焦距 = 半高 × f。
        const fy = 0.5 * height * f;
        // 水平像素焦距 = 半宽 × f / aspect（此处用 width/height 现算 aspect）。
        const fx = 0.5 * width * (f / (width / height));
        // 返回 [fx, fy]。
        return [fx, fy];
    }

    /** view * projection（列主序，且已把深度改造成 WebGPU 的 [0, 1]） */
    // 组合出 viewProj 矩阵。
    getViewProjection(aspect: number): Float32Array {
        // 更新相机的宽高比。
        this.camera.aspect = aspect;
        // 依据新宽高比重算投影矩阵。
        this.camera.updateProjectionMatrix();
        // 更新世界矩阵，保证其逆矩阵正确。
        this.camera.updateMatrixWorld();

        // three 的投影矩阵是 WebGL 约定，先组合 view，再统一做一次 z 重映射
        // 先算 projection × view，得到 viewProj（列主序）。
        const viewProj = mat4Multiply(
            this.camera.projectionMatrix.elements,
            this.camera.matrixWorldInverse.elements,
        );
        // 再左乘 z 重映射矩阵，把深度区间从 [-1,1] 变成 [0,1]。
        return mat4Multiply(NDC_Z_TO_ZERO_ONE, viewProj);
    }
}
