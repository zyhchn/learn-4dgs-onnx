
const BLOCK_KEYS: u32 = 256u;

// 公共参数布局必须与 sortGpu.ts 的 writeCommon 一致；vec4 保证 uniform 按 16 字节对齐。
struct Common {
    count: u32,
    depthBits: u32,
    indexBits: u32,
    pad0: u32,
    camPos: vec4f,
    forward: vec4f,
    numBlocks: u32,
    pad1: u32,
};

@group(0) @binding(0) var<uniform> p: Common;
// 下面的通用绑定有些阶段不会读取，但保留同一布局可减少 pipeline/layout 管理。
@group(0) @binding(1) var<storage, read> centers: array<vec4f>;
@group(0) @binding(2) var<storage, read> keysIn: array<u32>;
@group(0) @binding(3) var<storage, read> valuesIn: array<u32>;
@group(0) @binding(4) var<storage, read_write> keysOut: array<u32>;
@group(0) @binding(5) var<storage, read_write> valuesOut: array<u32>;
@group(0) @binding(6) var<storage, read_write> blockHist: array<u32>;
@group(0) @binding(7) var<uniform> sh: u32;

var<workgroup> hist: array<atomic<u32>, 16>;

@compute @workgroup_size(256)
fn cs_histogram(
    @builtin(workgroup_id) wid: vec3u,
    @builtin(local_invocation_id) lid: vec3u,
) {
    let t = lid.x;
    let block = wid.x;
    
    if(t < 16u) {
        //重置直方图
        atomicStore(&hist[t], 0u);
    }

    workgroupBarrier();

    let idx = block * BLOCK_KEYS + t; //全局索引
    if(idx < p.count) {
        let digit = (keysIn[idx] >> sh) & 15u; //0-16之间的桶
        atomicAdd(&hist[digit], 1u); //在对应桶中增加1
    }

    workgroupBarrier();
    
    if(t < 16u) {
        //将每个workgroup的直方图写入blockHist中
        blockHist[block * 16u + t] = atomicLoad(&hist[t]);
    }

}
