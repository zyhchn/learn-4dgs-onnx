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
@group(0) @binding(1) var<storage, read> centers: array<vec4f>;
@group(0) @binding(2) var<storage, read> keysIn: array<u32>;
@group(0) @binding(3) var<storage, read> valuesIn: array<u32>;
@group(0) @binding(4) var<storage, read_write> keysOut: array<u32>;
@group(0) @binding(5) var<storage, read_write> valuesOut: array<u32>;
// scan 已把这里的计数改写成“该 block、该 digit”的全局起点。
@group(0) @binding(6) var<storage, read_write> blockHist: array<u32>;
// 本趟读取的 bit 起始偏移。
@group(0) @binding(7) var<uniform> sh: u32;


//根据上一轮的桶和偏移量，将每个深度键放入对应的位置，实现排序
@compute @workgroup_size(256)
fn cs_scatter(
    @builtin(workgroup_id) wid: vec3u,
    @builtin(local_invocation_id) lid: vec3u,
) {
    let t = lid.x;
    let block = wid.x;

    //每个workgroup只使用16个线程
    if(t >= 16u) {
        return;
    }

    //每个workgroup中各自d桶的偏移量
    var cursor = blockHist[block * 16u + t];

    var idx = block * BLOCK_KEYS; //从本workgroup的第一个key开始
    let blockEnd = min(idx + BLOCK_KEYS, p.count); //到本workgroup的最后一个key结束
    loop {
        if(idx >= blockEnd) {
            break;
        }
        //只计算当前线程负责的key桶
        let key = keysIn[idx];
        if (((key >> sh) & 15u) == t) {
            //将当前key和value放入对应位置
            keysOut[cursor] = key; //按顺序放入深度键，最好一轮，得到的结果是升序
            valuesOut[cursor] = valuesIn[idx]; //记录当前key对应的索引
            cursor++;
        }
        idx++;
    }
}
