// splat.wgsl —— 高斯绘制：顶点扩张（EWA 投影）+ 片元椭圆高斯核
//
// 顶点输入只有一个「单位四边形」的角点，坐标范围 [-2, 2]²：
//     (-2,-2)  (2,-2)  (-2,2)  (2,2)
// 真正的几何信息全部来自 storage buffer 里的第 instance_index 个高斯。
//
// 与路线 A（splatv-demo）的差别：这里的高斯已经是「当前时刻」的静态高斯
// —— 时间形变在上游的 compute 前向里做完了 —— 所以顶点着色器里没有时间维，
// 直接使用打包阶段算好的预计算协方差。
//
// 顶点阶段顺序：
//   1. 齐次裁剪坐标 + 视锥粗剔除
//   2. 三维协方差 → 二维屏幕协方差（EWA）
//   3. 低通滤波（把过小的高斯撑到至少一个像素）
//   4. 特征分解 → 长轴 / 短轴
//   5. 按主轴把四边形展开成屏幕上的椭圆

// 所有相机/视口相关的常量打包在一个 uniform 缓冲里，供整个 draw call 共享
struct CommonUniforms {
    // view * projection（列主序）
    viewProj: mat4x4f,
    // 只有 view（列主序），EWA 投影要用它的左上 3x3
    view: mat4x4f,
    // 像素焦距 (fx, fy, -, -)
    focal: vec4f,
    // (width, height, 2/width, 2/height)
    viewport: vec4f,
    // (kernelSize, minPixelSize, splatScale, -)
    params: vec4f,
};

@group(0) @binding(0) var<uniform> u: CommonUniforms;                // 相机参数（uniform 常量缓冲）
@group(0) @binding(1) var<storage, read> centers: array<vec4f>;      // 每个高斯的世界中心 (x,y,z,1)
@group(0) @binding(2) var<storage, read> colorOpacity: array<vec4f>; // 每个高斯的 (r,g,b,opacity)
@group(0) @binding(3) var<storage, read> covA: array<vec4f>;         // 协方差上半：(σ00,σ01,σ02,0)
@group(0) @binding(4) var<storage, read> covB: array<vec4f>;         // 协方差下半：(σ11,σ12,σ22,0)
@group(0) @binding(5) var<storage, read> sortedIndices: array<u32>;  // 深度排序后的高斯索引（instance 顺序）

// 顶点着色器输出，会被光栅化插值后送进片元着色器
struct VSOut {
    @builtin(position) clipPos: vec4f,   // 裁剪空间位置（WebGPU 内置输出，片元里可读）
    // 四边形内的归一化坐标，直接来自顶点缓冲，范围 [-2, 2]
    @location(0) vPosition: vec2f,       // 传给片元用于算高斯核的角点坐标
    // rgb + 不透明度（已含低通补偿）
    @location(1) vColor: vec4f,          // 传给片元的颜色与不透明度
};

// 被剔除的 splat：输出一个必定被裁剪掉的顶点
fn culled() -> VSOut {
    var out: VSOut;                          // 构造一个 VSOut 作为返回值
    out.clipPos = vec4f(0.0, 0.0, 2.0, 1.0); // z=2 > w=1，落在近/远裁剪面之外，必定被丢弃
    out.vPosition = vec2f(0.0, 0.0);         // 占位坐标，永远不会被真正使用
    out.vColor = vec4f(0.0, 0.0, 0.0, 0.0);  // 透明黑，与占位保持一致
    return out;
}

@vertex  // 声明这是顶点阶段的入口函数
fn vs_main(@location(0) corner: vec3f, @builtin(instance_index) instanceIndex: u32) -> VSOut {
    let splatIndex = sortedIndices[instanceIndex];  // 用排序表把 instance 顺序映射回真实高斯索引
    let center = centers[splatIndex].xyz;           // 取出该高斯的世界空间中心 (x,y,z)

    // 1. 齐次裁剪坐标
    let clip = u.viewProj * vec4f(center, 1.0);      // 世界坐标 → 裁剪空间（透视除法前的四维）
    let viewPos = (u.view * vec4f(center, 1.0)).xyz; // 世界坐标 → 相机空间（EWA 雅可比需要它）

    // 视锥粗剔除：留 1.2 倍余量，避免边缘 splat 被过早切掉
    let bounds = 1.2 * clip.w;   // 以 w 为尺度的横向/纵向边界，w>0 时 |x|,|y| ≤ w 大致即视口内
    if (clip.w <= 0.0 || clip.z < 0.0 || clip.z > clip.w      // 在相机后方（w≤0）或深度越界
        || clip.x < -bounds || clip.x > bounds               // 横向越界
        || clip.y < -bounds || clip.y > bounds) {            // 纵向越界
        return culled();         // 直接输出被裁剪的顶点，跳过后面的重计算
    }

    // 2. 三维协方差 → 二维屏幕协方差（EWA 投影）
    //    预计算协方差按 6 分量打包：covA = (σ00, σ01, σ02)，covB = (σ11, σ12, σ22)
    let k = u.params.z;              // splatScale：对协方差做整体缩放
    let kk = k * k;                  // 协方差是二次型，缩放系数要平方
    let ca = covA[splatIndex].xyz * kk;  // 上半分量按 k² 缩放
    let cb = covB[splatIndex].xyz * kk;  // 下半分量按 k² 缩放
    let Vrk = mat3x3f(               // 由 6 个分量还原成对称的 3x3 世界协方差 Σ
        ca.x, ca.y, ca.z,            // 第 0 行：σ00 σ01 σ02
        ca.y, cb.x, cb.y,            // 第 1 行：σ01 σ11 σ12
        ca.z, cb.y, cb.z);           // 第 2 行：σ02 σ12 σ22

    // 透视投影的雅可比含 z 依赖，这是「局部仿射近似」的由来
    let J = mat3x3f(                 // 相机空间 → 屏幕的投影雅可比（真实是 2x3，第三行补 0 凑方阵）
        u.focal.x / viewPos.z, 0.0, -(u.focal.x * viewPos.x) / (viewPos.z * viewPos.z),  // ∂u/∂x, ∂u/∂y, ∂u/∂z
        0.0, u.focal.y / viewPos.z, -(u.focal.y * viewPos.y) / (viewPos.z * viewPos.z),  // ∂v/∂x, ∂v/∂y, ∂v/∂z
        0.0, 0.0, 0.0);             // 第三行全 0，仅为矩阵乘法对齐

    // view 只有旋转平移，左上 3x3 即世界→相机的旋转
    let mv3 = mat3x3f(u.view[0].xyz, u.view[1].xyz, u.view[2].xyz);  // 从 view 中取出旋转部分
    let T = transpose(mv3) * J;      // 用 mv3 的转置组装出作用在世界协方差上的映射 T
    var cov2d = transpose(T) * Vrk * T;  // Σ2D = Tᵀ Σ T：把 3x3 世界协方差线性变换到屏幕 2x2

    // 3. 低通滤波：把小于一个像素的 splat 撑到至少一个像素
    let kernelSize = u.params.x;     // 低通补偿量（像素²），0 表示关闭
    var compensation = 1.0;          // 不透明度的能量补偿系数，默认不补偿
    if (kernelSize > 0.0) {          // 只有开启低通时才计算
        let detOrig = cov2d[0][0] * cov2d[1][1] - cov2d[0][1] * cov2d[0][1];  // 膨胀前的行列式（代表面积）
        cov2d[0][0] += kernelSize;   // 对角线加上模糊项：撑大主轴方差
        cov2d[1][1] += kernelSize;   // 另一个对角线同样加上模糊项
        let detBlur = cov2d[0][0] * cov2d[1][1] - cov2d[0][1] * cov2d[0][1];  // 膨胀后的行列式
        // 膨胀会放大支撑域，用「面积比的开方」把总能量补回去
        if (detOrig > 0.0) {         // 只有原协方差可逆（面积为正）时才有意义
            compensation = sqrt(max(0.0, detOrig / detBlur));  // 面积比开方 = 线性尺度比，用于缩放不透明度
        }
    }

    // 4. 二维协方差的特征分解 → 长轴 / 短轴
    let mid = (cov2d[0][0] + cov2d[1][1]) * 0.5;  // 两个特征值的均值（即矩阵迹的一半）
    let rad = length(vec2f((cov2d[0][0] - cov2d[1][1]) * 0.5, cov2d[0][1]));  // 特征值半差（2x2 对称阵闭式解）
    let eps = 0.0001;                // 极小量，避免特征值恰为 0 导致退化
    let lambda1 = mid + rad + eps;   // 较大特征值（对应长轴方向的方差）
    let lambda2 = mid - rad + eps;   // 较小特征值（对应短轴方向的方差）
    if (lambda2 < 0.0) {             // 出现负特征值说明协方差非半正定
        return culled();             // 视为退化，丢弃该 splat
    }

    // minPixelSize：长轴上连 minPixelSize 个像素都不到的高斯直接丢弃
    let majorLength = 2.0 * min(sqrt(2.0 * lambda1), 1024.0);  // 长轴直径 ≈ 2√(2λ)，并对超大值封顶
    let minorLength = 2.0 * min(sqrt(2.0 * lambda2), 1024.0);  // 短轴直径，同样封顶
    if (u.params.y > 0.0 && max(majorLength, minorLength) < u.params.y) {  // 尺寸太小（< minPixelSize）
        return culled();             // 直接丢弃，避免屏幕上出现噪点
    }

    // 主轴方向（单位向量）
    let diagonalVector = normalize(vec2f(cov2d[0][1], lambda1 - cov2d[0][0]));  // 对应 λ1 的特征向量方向
    // 半轴长度 = sqrt(2λ)，这样片元里的马氏距离恰好等于 2|meshPos|²
    let majorAxis = min(sqrt(2.0 * lambda1), 1024.0) * diagonalVector;         // 长半轴向量（含方向与长度）
    let minorAxis = min(sqrt(2.0 * lambda2), 1024.0) * vec2f(diagonalVector.y, -diagonalVector.x);  // 与长轴垂直的短半轴

    // 5. 把 [-2, 2]² 的四边形按主轴展开成屏幕上的椭圆（单位：像素）
    let offsetPixels = corner.x * majorAxis + corner.y * minorAxis;  // 角点沿两个半轴线性组合 → 像素偏移
    // 像素 → NDC：乘 2/尺寸；再乘 clip.w 补回透视除法
    let offsetNdc = offsetPixels * u.viewport.zw;                    // 像素偏移换算为 NDC 偏移（zw = 2/w, 2/h）
    let clipXY = clip.xy + offsetNdc * clip.w;                       // 乘 w 抵消随后的透视除法，再加到中心

    var out: VSOut;                                  // 组装最终顶点输出
    out.clipPos = vec4f(clipXY, clip.z, clip.w);     // 保留原深度与 w，仅替换 xy
    out.vColor = vec4f(colorOpacity[splatIndex].rgb, colorOpacity[splatIndex].a * compensation);  // 颜色 + 补偿后的不透明度
    out.vPosition = corner.xy;                       // 原样把角点坐标传给片元（范围 [-2,2]）
    return out;
}

// 片元阶段：在椭圆支撑域里画高斯核
@fragment  // 声明这是片元阶段的入口函数
fn fs_main(in: VSOut) -> @location(0) vec4f {  // in 为插值后的顶点输出，返回颜色 + alpha
    // 顶点把 [-2, 2]² 的角点原样传过来（注意：没有乘 2），
    // 于是马氏距离 d² = 2|vPosition|²，故 A = -d²/2。
    let A = -dot(in.vPosition, in.vPosition);  // A = -|v|²，即高斯指数项 -d²/2
    if (A < -4.0) {   // 指数项太小，意味着离高斯中心过远
        // A < -4 等价于 d > 2√2 σ ≈ 2.83σ（核值已低于峰值的 1.83%），
        // 直接 discard，比继续算 exp 更划算。3DGS 官方 CUDA 光栅器用的是 3σ。
        discard;      // 丢弃该片元，不写入颜色与深度
    }
    // 论文里的 o · exp(-½ dᵀ Σ₂D⁻¹ d)，在这里恰好就是 exp(A) · 不透明度
    return vec4f(in.vColor.rgb, exp(A) * in.vColor.a);  // 输出高斯核值乘以不透明度
}
