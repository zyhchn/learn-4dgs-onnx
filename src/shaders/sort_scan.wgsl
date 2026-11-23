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
@group(0) @binding(6) var<storage, read_write> blockHist: array<u32>;
// 每趟的 shift（scan 用不到，仅为保持四个 shader 的 binding 集合一致）。
@group(0) @binding(7) var<uniform> sh: u32;

var<workgroup> binTotal: array<u32, 16>;

// 扫描直方图，计算每个桶的累计和，只有一个 workgroup
@compute @workgroup_size(256)
fn cs_scan(@builtin(local_invocation_id) lid: vec3u) {
    let d = lid.x;
    //将上一个直方图的0-16个桶的直方图累计和写入binTotal中
    if(d < 16u) {
        var total = 0u;
        for(var block = 0u; block < p.numBlocks; block++) {
            total += blockHist[block * 16u + d];
        }
        binTotal[d] = total;
    }
    workgroupBarrier();

    if(d < 16u) {
        //计算在当前d桶之前的总数，该数就是当前d桶的偏移量
        var offset = 0u;
        for(var lowBin = 0u; lowBin < d; lowBin++) {
            offset += binTotal[lowBin];
        }
        //计算每个workgroup中各自d桶的偏移量
        for(var block = 0u; block < p.numBlocks; block++) {
            let i = block * 16u + d;
            let count = blockHist[i];
            blockHist[i] = offset;
            offset += count;
        }
    }
}
