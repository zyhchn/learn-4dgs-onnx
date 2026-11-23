struct Common {
    count: u32,
    depthBits: u32,
    indexBits: u32,
    pad0: u32, //补位
    camPos: vec4f,
    forward: vec4f,
    numBlocks: u32,
    pad1: u32,
}

@group(0) @binding(0) var<uniform> p: Common;
@group(0) @binding(1) var<storage, read> centers: array<vec4f>;
@group(0) @binding(2) var<storage, read> keysIn: array<u32>;
@group(0) @binding(3) var<storage, read> valuesIn: array<u32>;
@group(0) @binding(4) var<storage, read_write> keysOut: array<u32>;
@group(0) @binding(5) var<storage, read_write> valuesOut: array<u32>;
@group(0) @binding(6) var<storage, read_write> blockHist: array<u32>;
// 每趟变化的 shift 直接用一个 u32 uniform；keygen 用不到，仅为保持各 shader 的 binding 一致。
@group(0) @binding(7) var<uniform> sh: u32;
@group(0) @binding(8) var<storage, read_write> depthRange: array<f32>;

var<workgroup> sharedMin: array<f32, 256>;
var<workgroup> sharedMax: array<f32, 256>;

//计算深度范围归约
@compute @workgroup_size(256)
fn cs_depth_range(@builtin(local_invocation_id) lid: vec3u) {
    let t = lid.x;

    // WGSL 不允许把 inf（0x7f800000）常量求值为 f32，改用有限的最大/最小规格化浮点数（±FLT_MAX）
    // 作为归约的单位元，对 min/max 结果没有影响。
    var low = 3.402823466e38; //最小深度
    var high = -3.402823466e38; //最大深度
    var i = t;

    loop {
        if(i >= p.count) {
            break;
        }
        let d = dot(centers[i].xyz - p.camPos.xyz, p.forward.xyz);
        low = min(low, d);
        high = max(high, d);
        i = i + 256u;
    }
    sharedMin[t] = low;
    sharedMax[t] = high;
    workgroupBarrier();

    var stride = 128u;
    loop {
        if(stride == 0u){
            break;
        }

        if(t < stride) {
            sharedMin[t] = min(sharedMin[t], sharedMin[t + stride]);
            sharedMax[t] = max(sharedMax[t], sharedMax[t + stride]);
        }
        workgroupBarrier();
        stride = stride / 2u;
    }

    if(t == 0u) {
        depthRange[0] = sharedMin[0];
        depthRange[1] = sharedMax[0];
    }
}
//讲深度量化为整数
@compute @workgroup_size(256)
fn cs_keygen(@builtin(global_invocation_id) gid: vec3u) {
    let i = gid.x;
    if(i >= p.count) {
        return;
    }

    let center = centers[i].xyz;
    let depth = dot(center - p.camPos.xyz, p.forward.xyz);
    let minD = depthRange[0];
    let maxD = depthRange[1];
    let range = max(maxD - minD, 1e-9);
    let t = clamp((depth - minD) / range, 0.0, 1.0);
    let maxDepth = (1u << p.depthBits) - 1u; //量化后的最大深度
    var depthKey = u32(round(t * f32(maxDepth)));
    depthKey = maxDepth - depthKey; //深度倒序

    keysOut[i] = (depthKey << p.indexBits) | i; //深度和索引拼接，深度在高位，索引在低位
    valuesOut[i] = i; //values 里存的是原始索引；keygen 不读 valuesIn（那份此时还未初始化）
}
