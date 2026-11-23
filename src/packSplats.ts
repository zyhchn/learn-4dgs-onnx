/**
 * packSplats.ts —— 渲染器需要的高斯缓冲描述
 *
 * 与路线 A（splatv-demo）不同，路线 B 的高斯数据**不是**在 CPU 上打包好再上传的：
 * 神经形变场的 compute 前向每帧把形变后的结果直接写进这几个 storage buffer，
 * 渲染器只是「读」它们，全程不经过 CPU 回读 —— 这是实时播放的关键。
 *
 * 因此本文件不含 CPU 打包逻辑（打包/前向都已在 GPU 上完成），
 * 只保留渲染器要用的那份「缓冲描述」。
 */

/**
 * 由 GPU compute 直接产出的 splat 缓冲（Renderer.setScene 的入参）。
 *
 * 这里给的是 **GPUBuffer 而不是 Float32Array**：缓冲归 compute 前向所有，
 * 渲染器只持有引用、不负责销毁。
 */
export interface SplatBuffers {
    count: number;        // 高斯总数（= 顶点实例数）
    centers: GPUBuffer;   // 每个高斯的世界坐标中心，[N*4]（vec4，末位补 1 让矩阵乘法带上平移）
    colorOpacity: GPUBuffer; // 颜色 + 不透明度，[N*4]（前 3 个是 RGB，第 4 个是 opacity）
    covA: GPUBuffer;      // 协方差第一组分量，[N*4]（2D 投影椭圆的 2×2 矩阵上半部分）
    covB: GPUBuffer;      // 协方差第二组分量，[N*4]
}
