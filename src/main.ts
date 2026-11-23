/**
 * main.ts —— 路线 B 教学 Demo 的入口（最小入口）
 *
 * 这里只做两件事：取到画布、创建 ViewerApp 并把初始化错误写到页面上。
 * 真正的链路（加载权重 → GPU 前向 → GPU 排序 → 每帧渲染）都在 ViewerApp 里，
 * 这样入口文件与 gs-webgpu 的 main 保持同样的「薄」。
 *
 * 页面只有一块画布：拖拽旋转、滚轮缩放，时间自动播放，没有任何按钮或面板。
 */

import { ViewerApp } from "./ViewerApp"; // 应用外壳：串起渲染器、前向、排序器与相机

const canvas = document.getElementById("gpu-canvas") as HTMLCanvasElement; // 取页面上的主画布

/** 加载失败时把提示写进隐藏的 #error 层 */
function showError(message: string): void {
    const box = document.getElementById("error"); // 找到错误提示层
    if (!box) return;                             // 找不到就静默返回（不影响主流程）
    box.textContent = message;                    // 填入错误文本
    box.style.display = "block";                  // 把隐藏的层显示出来
}

const viewer = new ViewerApp(canvas); // 创建应用（内部异步初始化，帧循环也在其中启动）

viewer.ready.catch((err: unknown) => {
    console.error(err);                                                                            // 打印到控制台
    showError(`初始化失败：${err instanceof Error ? err.message : String(err)}`);                  // 把错误写进页面提示层
});
