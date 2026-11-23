import { defineConfig } from "vite"; // 从 Vite 引入配置辅助函数，获得类型提示

// 教学 Demo 的 Vite 配置：本目录即根目录，index.html 是入口。
//   pnpm dev        启动开发服务器
//   pnpm typecheck  只做类型检查（tsc --noEmit）
// 着色器用 `new URL("../shaders/x.wgsl", import.meta.url)` + fetch 读取，
// dev 下由 Vite 直接从项目目录提供，无需额外配置。
export default defineConfig({ // 导出默认配置对象
    server: { // 开发服务器配置
        host: "127.0.0.1", // 绑定本机回环地址，仅本机可访问
        port: 5174, // 监听端口
        open: true, // 启动后自动打开浏览器
        strictPort: false, // 端口被占用时自动换端口，而非直接报错退出
    },
    build: { // 生产构建配置
        target: "esnext", // 构建目标为最新 ES 语法（WebGPU 运行环境通常较新）
        sourcemap: true, // 生成 sourcemap，便于调试
    },
});
