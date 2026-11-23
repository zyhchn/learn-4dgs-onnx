/**
 * math.js —— 手写线性代数小工具（零依赖）
 *
 * 约定（与《3DGS 实时渲染教学系列》第 03~05 章一致）：
 *   1. 右手坐标系，y 轴朝上，与现有教学系列一致。
 *   2. 所有矩阵为「列主序」（column-major），直接对应 WGSL 的 mat4x4<f32> 内存布局，
 *      也就是直接对应 GPUBuffer 里 16 个连续 f32 的顺序。
 *   3. 投影矩阵把视锥深度映射到 WebGPU 的 NDC 区间 [0, 1]（WebGL 是 [-1, 1]，这是
 *      WebGPU 与 WebGL 最容易踩坑的差异之一，见第 03 章）。
 */

/** 向量/矩阵入参：既支持按下标读取，也支持解构，调用方可传 Float32Array 或普通 number[] */
// 类型别名：所有接收向量的函数都用它当形参类型，从而兼容 Float32Array 与普通数组两种实参。
type Vec = Float32Array | number[];

// 角度制 → 弧度制 的系数：把「度」乘以它就得到「弧度」（π/180）。
export const DEG = Math.PI / 180;
// 数值容差：用于判断「长度约等于 0」「两向量几乎同向/反向」等退化情形，避免除零与方向未定义。
export const EPS = 1e-8;

/* ------------------------------------------------------------------ *
 * vec3
 * ------------------------------------------------------------------ */

// 构造一个 3 分量向量，默认 (0, 0, 0)，返回底层恰好 3 个 f32 的 Float32Array。
export function vec3(x = 0, y = 0, z = 0): Float32Array {
    // 返回新数组，不共享任何外部内存。
    return new Float32Array([x, y, z]);
}

// 原地写入 3 个分量到 out，并返回同一个数组，常用作「输出参数」以避免频繁分配。
export function vec3Set(out: Float32Array, x: number, y: number, z: number): Float32Array {
    // 写入 x 分量。
    out[0] = x;
    // 写入 y 分量。
    out[1] = y;
    // 写入 z 分量。
    out[2] = z;
    // 返回入参本身，方便链式调用。
    return out;
}

// 向量加法 a + b（逐分量），返回新数组。
export function vec3Add(a: Vec, b: Vec): Float32Array {
    // 三个分量分别相加。
    return vec3(a[0] + b[0], a[1] + b[1], a[2] + b[2]);
}

// 向量减法 a - b（逐分量），返回新数组。
export function vec3Sub(a: Vec, b: Vec): Float32Array {
    // 三个分量分别相减。
    return vec3(a[0] - b[0], a[1] - b[1], a[2] - b[2]);
}

// 向量数乘 a * s（逐分量），返回新数组。
export function vec3Scale(a: Vec, s: number): Float32Array {
    // 三个分量分别乘以标量 s。
    return vec3(a[0] * s, a[1] * s, a[2] * s);
}

// 点积 a·b：结果是一个标量，等于 |a||b|cosθ，也用于「把向量投影到某个方向上」。
export function vec3Dot(a: Vec, b: Vec): number {
    // 三个分量逐项相乘再求和。
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
}

// 叉积 a × b：结果垂直于 a、b 所在平面，长度等于 |a||b|sinθ（右手法则）。
export function vec3Cross(a: Vec, b: Vec): Float32Array {
    // 按右手法则展开成三个分量。
    return vec3(
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    );
}

// 向量长度（欧几里得模长），用 Math.hypot 兼顾数值稳定性。
export function vec3Length(a: Vec): number {
    // hypot(x,y,z) = sqrt(x² + y² + z²)。
    return Math.hypot(a[0], a[1], a[2]);
}

// 归一化：把向量缩放到长度 1，保持方向不变。
export function vec3Normalize(a: Vec): Float32Array {
    // 先求当前长度。
    const len = vec3Length(a);
    // 长度过小（近似零向量）时方向未定义，返回零向量；否则按 1/len 缩放成单位长度。
    return len < EPS ? vec3(0, 0, 0) : vec3Scale(a, 1 / len);
}

/* ------------------------------------------------------------------ *
 * quaternion —— 四元数以 (x, y, z, w) 存储，w 为实部
 * ------------------------------------------------------------------ */

// 构造四元数，默认是单位四元数 (0,0,0,1)；存储顺序 (x, y, z, w)，w 为实部。
export function quat(x = 0, y = 0, z = 0, w = 1): Float32Array {
    // 返回 4 个 f32 的新数组。
    return new Float32Array([x, y, z, w]);
}

// 由「旋转轴 + 旋转角（弧度）」构造四元数：q = (sin(θ/2)·axis, cos(θ/2))。
export function quatFromAxisAngle(axis: Vec, angleRad: number): Float32Array {
    // 轴先归一化，使向量部分只体现方向。
    const a = vec3Normalize(axis);
    // 半角 θ/2。
    const h = angleRad * 0.5;
    // 半角正弦，向量部分要乘它。
    const s = Math.sin(h);
    // 组合成 (axis·sin(θ/2), cos(θ/2))。
    return quat(a[0] * s, a[1] * s, a[2] * s, Math.cos(h));
}

/**
 * 求「把单位向量 from 转到单位向量 to」的最短弧四元数。
 * 4D 合成器用它给每片扁平高斯一个朝向（+z 是扁平面法线）。
 */
// 这是一个通用数学工具（构造两方向间的最短旋转），供各处复用。
export function quatFromUnitVectors(from: Vec, to: Vec): Float32Array {
    // 起点方向归一化。
    const a = vec3Normalize(from);
    // 终点方向归一化。
    const b = vec3Normalize(to);
    // 点积 = cos(两向量夹角)。
    const d = vec3Dot(a, b);
    // 几乎同向：夹角为 0，最短旋转就是恒等四元数。
    if (d >= 1 - EPS) return quat(0, 0, 0, 1); // 同向
    // 几乎反向：夹角 180°，需要绕一条与 a 正交的轴转 π。
    if (d <= -1 + EPS) {
        // 反向：取一条与 a 正交的轴转 180°
        // 先试 x 轴与 a 叉乘；若 a 恰与 x 轴平行则结果退化，改用 y 轴再试。
        let axis = vec3Cross(vec3(1, 0, 0), a);
        if (vec3Length(axis) < EPS) axis = vec3Cross(vec3(0, 1, 0), a);
        // 绕这条正交轴转 180°。
        return quatFromAxisAngle(axis, Math.PI);
    }
    // 一般情况：q = normalize((a×b, 1 + a·b)) 即为最短弧旋转。
    const c = vec3Cross(a, b);
    // 归一化后返回。
    return quatNormalize(quat(c[0], c[1], c[2], 1 + d));
}

// 四元数归一化：除以四分量模长，得到只表示旋转的单位四元数。
export function quatNormalize(q: Vec): Float32Array {
    // 4D 模长。
    const len = Math.hypot(q[0], q[1], q[2], q[3]);
    // 模长过小说明是退化输入，回退到单位四元数。
    if (len < EPS) return quat(0, 0, 0, 1);
    // 逐分量除以模长。
    return quat(q[0] / len, q[1] / len, q[2] / len, q[3] / len);
}

// 四元数乘法 a⊗b：对应「先施加 b 的旋转、再施加 a 的旋转」的旋转复合。
export function quatMultiply(a: Vec, b: Vec): Float32Array {
    // 解构出 a 的四个分量。
    const [ax, ay, az, aw] = a;
    // 解构出 b 的四个分量。
    const [bx, by, bz, bw] = b;
    // 按四元数乘法公式（Hamilton 积）逐分量计算。
    return quat(
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
        aw * bw - ax * bx - ay * by - az * bz,
    );
}

/**
 * 四元数 → 3x3 旋转矩阵，列主序（9 个 f32）。
 * 与 grammarly 无关：这里用的就是标准 Rodrigues 展开，见第 04 章公式 (4.3)。
 */
export function quatToMat3(q: Vec): Float32Array {
    // 先归一化，保证得到的是纯旋转矩阵（不含缩放）。
    const n = quatNormalize(q);
    // 取出四元数分量 (x, y, z, w)。
    const [x, y, z, w] = n;
    // 预计算各种二阶项，避免重复乘法。
    const xx = x * x, yy = y * y, zz = z * z;
    const xy = x * y, xz = x * z, yz = y * z;
    const wx = w * x, wy = w * y, wz = w * z;

    // 列主序：[c0, c1, c2]
    // 按标准公式展开成 9 个数，按列优先（每 3 个是一列）排列。
    return new Float32Array([
        1 - 2 * (yy + zz), 2 * (xy + wz), 2 * (xz - wy),
        2 * (xy - wz), 1 - 2 * (xx + zz), 2 * (yz + wx),
        2 * (xz + wy), 2 * (yz - wx), 1 - 2 * (xx + yy),
    ]);
}

/* ------------------------------------------------------------------ *
 * mat4 —— 列主序，m[col * 4 + row]
 * ------------------------------------------------------------------ */

// 单位 4x4 矩阵（列主序），用作恒等变换的初值。
export function mat4(): Float32Array {
    // 16 个 f32，对角线位置是下标 0、5、10、15。
    return new Float32Array([1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1]);
}

// 4x4 矩阵乘法，返回 a·b（顺序敏感：结果等价于先施加 b 再施加 a 的组合变换）。
export function mat4Multiply(a: Vec, b: Vec): Float32Array {
    // 返回 a * b
    // 结果缓冲区（16 个 f32）。
    const out = new Float32Array(16);
    // 遍历结果的每一列 c。
    for (let c = 0; c < 4; c++) {
        // 遍历结果的每一行 r。
        for (let r = 0; r < 4; r++) {
            // 累加求和变量。
            let s = 0;
            // 沿公共维度 k 求和：a 取第 r 行（列主序 a[k*4+r]），b 取第 c 列（b[c*4+k]）。
            for (let k = 0; k < 4; k++) {
                s += a[k * 4 + r] * b[c * 4 + k];
            }
            // 写入结果 (行 r, 列 c) 位置。
            out[c * 4 + r] = s;
        }
    }
    // 返回乘积矩阵。
    return out;
}

// 4x4 转置：行列互换（列主序下即把下标 c*4+r 与 r*4+c 对调）。
export function mat4Transpose(m: Vec): Float32Array {
    // 结果缓冲区。
    const out = new Float32Array(16);
    // 遍历原矩阵的列。
    for (let c = 0; c < 4; c++) {
        // 遍历原矩阵的行。
        for (let r = 0; r < 4; r++) {
            // 交换行列下标后写入。
            out[r * 4 + c] = m[c * 4 + r];
        }
    }
    // 返回转置结果。
    return out;
}

// 由平移向量 t 构造 4x4 平移矩阵（列主序下平移量落在第 4 列的后三位）。
export function mat4FromTranslation(t: Vec): Float32Array {
    // 从单位矩阵出发（左上 3x3 是单位旋转）。
    const m = mat4();
    // 写入平移 x 分量（列 3、行 0）。
    m[12] = t[0];
    // 写入平移 y 分量（列 3、行 1）。
    m[13] = t[1];
    // 写入平移 z 分量（列 3、行 2）。
    m[14] = t[2];
    // 返回平移矩阵。
    return m;
}

// 由四元数 q 与缩放 s 构造 4x4 矩阵：M = R·S（先缩放后旋转），第 4 列为 0 平移。
export function mat4FromQuatScale(q: Vec, s: Vec): Float32Array {
    // 取出旋转部分（3x3 列主序）。
    const r = quatToMat3(q);
    // 结果 4x4 矩阵。
    const m = new Float32Array(16);
    // 列 j 乘上 s[j]
    // 第 0 列：旋转矩阵第 0 列整体乘以 s[0]，第 4 行补 0。
    m[0] = r[0] * s[0]; m[1] = r[1] * s[0]; m[2] = r[2] * s[0]; m[3] = 0;
    // 第 1 列：旋转矩阵第 1 列整体乘以 s[1]。
    m[4] = r[3] * s[1]; m[5] = r[4] * s[1]; m[6] = r[5] * s[1]; m[7] = 0;
    // 第 2 列：旋转矩阵第 2 列整体乘以 s[2]。
    m[8] = r[6] * s[2]; m[9] = r[7] * s[2]; m[10] = r[8] * s[2]; m[11] = 0;
    // 第 3 列：平移为 0，齐次分量为 1。
    m[12] = 0; m[13] = 0; m[14] = 0; m[15] = 1;
    // 返回组合矩阵。
    return m;
}

/**
 * 透视投影矩阵，右手系，深度映射到 [0, 1]（WebGPU 约定）。
 * near 处 z_ndc = 0，far 处 z_ndc = 1。
 */
export function mat4Perspective(fovYRad: number, aspect: number, near: number, far: number): Float32Array {
    // f = 1/tan(fovY/2)，即竖直方向的焦距倍率（相当于把视锥边缘映射到 ±1）。
    const f = 1 / Math.tan(fovYRad * 0.5);
    // 提前算好 1/(near-far)，后面两项复用。
    const nf = 1 / (near - far);
    // 结果缓冲区。
    const m = new Float32Array(16);
    // 第 0 列：x 方向缩放 = f/aspect（宽高比越宽，x 缩放越小）。
    m[0] = f / aspect;
    // y 方向缩放 = f。
    m[5] = f;
    // z 缩放项：把深度压缩到 [0,1] 区间的缩放部分。
    m[10] = far * nf;          // = far / (near - far)
    // 使 w' = -z_cam，这是后续透视除法（除以 w）的来源，也是右手系相机的标准写法。
    m[11] = -1;
    // z 平移项：与 m[10] 配合保证 near→0、far→1。
    m[14] = far * near * nf;   // = far * near / (near - far)
    // 返回透视投影矩阵。
    return m;
}

/**
 * 正交投影矩阵，右手系，深度 [0, 1]。
 * 用途：第 05 章里把「正交相机」当作 EWA 退化成精确边缘化的特例来做对照。
 */
export function mat4Ortho(left: number, right: number, bottom: number, top: number, near: number, far: number): Float32Array {
    // 从单位矩阵出发（正交投影不做透视除法，w 恒为 1）。
    const m = mat4();
    // x 方向缩放：把 [left,right] 映射到 [-1,1]。
    m[0] = 2 / (right - left);
    // y 方向缩放：把 [bottom,top] 映射到 [-1,1]。
    m[5] = 2 / (top - bottom);
    // z 方向缩放：把 [near,far] 映射到 [0,1]（WebGPU 约定）。
    m[10] = 1 / (near - far);
    // x 平移：把区间中心对齐到 0。
    m[12] = -(right + left) / (right - left);
    // y 平移。
    m[13] = -(top + bottom) / (top - bottom);
    // z 平移。
    m[14] = near / (near - far);
    // 返回正交投影矩阵。
    return m;
}

// 构造视图矩阵（世界 → 相机）：给定眼睛位置 eye、注视目标 target、上方向 up。
export function mat4LookAt(eye: Vec, target: Vec, up: Vec): Float32Array {
    // 相机 +z 轴 = 由 target 指向 eye 的单位向量（右手系相机看向 -z，故取 eye-target）。
    const zAxis = vec3Normalize(vec3Sub(eye, target));
    // 相机 +x 轴 = up 与 z 轴的叉积（同时垂直于两者）。
    let xAxis = vec3Cross(up, zAxis);
    // 退化判断：视线与 up 平行时叉积为零。
    if (vec3Length(xAxis) < EPS) {
        // 视线与 up 平行时的退化处理
        // 改用一个不与视线平行的参考轴重新求 x 轴。
        xAxis = vec3Cross(vec3(0, 0, 1), zAxis);
    }
    // 单位化 x 轴。
    xAxis = vec3Normalize(xAxis);
    // 相机 +y 轴 = z 与 x 的叉积，保证三轴互相正交。
    const yAxis = vec3Cross(zAxis, xAxis);

    // 视图矩阵缓冲区。
    const m = new Float32Array(16);
    // 第 0 列：相机 x 轴在世界各轴上的分量，第 4 行补 0。
    m[0] = xAxis[0]; m[1] = yAxis[0]; m[2] = zAxis[0]; m[3] = 0;
    // 第 1 列：相机 y 轴分量。
    m[4] = xAxis[1]; m[5] = yAxis[1]; m[6] = zAxis[1]; m[7] = 0;
    // 第 2 列：相机 z 轴分量。
    m[8] = xAxis[2]; m[9] = yAxis[2]; m[10] = zAxis[2]; m[11] = 0;
    // 第 3 列：平移分量 = -R·eye，把世界原点搬到相机空间。
    m[12] = -vec3Dot(xAxis, eye);
    m[13] = -vec3Dot(yAxis, eye);
    m[14] = -vec3Dot(zAxis, eye);
    // 齐次分量为 1。
    m[15] = 1;
    // 返回视图矩阵。
    return m;
}

/** 4x4 矩阵变换一个点（w = 1），返回 4 分量齐次坐标 */
export function mat4TransformPoint4(m: Vec, p: Vec): Float32Array {
    // 取出点的三个坐标。
    const x = p[0], y = p[1], z = p[2];
    // 逐行做「矩阵行 · 点」的点积（列主序下第 r 行的元素是 m[r]、m[4+r]、m[8+r]、m[12+r]）。
    return new Float32Array([
        m[0] * x + m[4] * y + m[8] * z + m[12],
        m[1] * x + m[5] * y + m[9] * z + m[13],
        m[2] * x + m[6] * y + m[10] * z + m[14],
        m[3] * x + m[7] * y + m[11] * z + m[15],
    ]);
}

/* ------------------------------------------------------------------ *
 * 对称 3x3 小工具（CPU 侧，用于第 02 / 04 / 05 章的数值对照）
 * 3x3 矩阵按列主序存 9 个 f32：[c0, c1, c2]
 * ------------------------------------------------------------------ */

// 3x3 单位矩阵（列主序）。
export function mat3(): Float32Array {
    // 对角线元素为 1。
    return new Float32Array([1, 0, 0, 0, 1, 0, 0, 0, 1]);
}

// 3x3 矩阵乘法 out = a·b（列主序）。
export function mat3Multiply(a: Vec, b: Vec): Float32Array {
    // 结果缓冲区（9 个 f32）。
    const out = new Float32Array(9);
    // 遍历结果的列 c。
    for (let c = 0; c < 3; c++) {
        // 遍历结果的行 r。
        for (let r = 0; r < 3; r++) {
            // a 的第 r 行（列主序 a[r]、a[3+r]、a[6+r]）与 b 的第 c 列做点积。
            out[c * 3 + r] = a[r] * b[c * 3] + a[3 + r] * b[c * 3 + 1] + a[6 + r] * b[c * 3 + 2];
        }
    }
    // 返回乘积。
    return out;
}

// 3x3 转置：行列互换。
export function mat3Transpose(a: Vec): Float32Array {
    // 按列主序把 0/3/6、1/4/7、2/5/8 三组重新排列，即完成转置。
    return new Float32Array([a[0], a[3], a[6], a[1], a[4], a[7], a[2], a[5], a[8]]);
}

/**
 * 由缩放 s 与旋转四元数 q 构造三维协方差 Σ = R S Sᵀ Rᵀ。
 * 返回 6 个独立分量 [σ00, σ01, σ02, σ11, σ12, σ22]，
 * 顺序与第 04 章的 covA/covB 打包一致。
 */
export function covarianceFromScaleRotation(scale: Vec, q: Vec): number[] {
    // 旋转部分的 3x3 列主序矩阵。
    const r = quatToMat3(q);
    // M = R * S（列缩放）
    // M 的缓冲区（列主序）。
    const m = new Float32Array(9);
    // 遍历 3 列。
    for (let c = 0; c < 3; c++) {
        // 遍历每列的 3 行。
        for (let i = 0; i < 3; i++) {
            // 第 c 列整体乘以 scale[c]，等价于右乘对角缩放矩阵 S。
            m[c * 3 + i] = r[c * 3 + i] * scale[c];
        }
    }
    // 取 M 的转置，用于计算 M·Mᵀ。
    const mt = mat3Transpose(m);
    // Σ = M Mᵀ，列主序
    const sigma = mat3Multiply(m, mt); // Σ = M Mᵀ，列主序
    // sigma[c * 3 + r] = Σ[r][c]
    // 协方差是对称矩阵，只需 6 个独立分量，按 [σ00,σ01,σ02,σ11,σ12,σ22] 顺序取出。
    return [
        sigma[0], sigma[3], sigma[6], // σ00, σ01, σ02
        sigma[4], sigma[7], sigma[8], // σ11, σ12, σ22
    ];
}
