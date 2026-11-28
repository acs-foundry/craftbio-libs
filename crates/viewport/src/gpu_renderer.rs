#![forbid(unsafe_code)]

//! Pure-Rust `wgpu 30 + WGSL` Offscreen `Depth32Float` Biomolecular 3D Renderer (`egui_wgpu::CallbackTrait`).
//! - Pass 0 (Directional `1024×1024` `Depth32Float` Shadow Map in `prepare()`):
//!   - Depth-only opaque meshes + analytical depth spheres from the key directional light (`light_view_proj`)
//! - Pass 1 (Offscreen `Bgra8Unorm + Depth32Float` in `prepare()`):
//!   - Indexed `MeshVertex` Opaque Pipeline with **3×3 PCF Directional Soft Shadows**
//!   - Analytical Ray-Cast Sphere & Capsule Pipelines with exact `@builtin(frag_depth)`, Near/Far Clip & **3×3 PCF Soft Shadows**
//!   - Transparent Mesh + Wireframe `LineList` Pipeline & Translucent Overlay Spheres (depth test ON, write OFF)
//! - Pass 2 (Screen-Space Post-Process in `paint()`):
//!   - 12-tap Poisson-disk **SSAO (Ambient Occlusion)** + ChimeraX-style **Depth-Discontinuity Silhouettes**
//!     reading `(offscreen_color, offscreen_depth)` via `textureLoad` and blitting directly into `egui`.

use std::sync::{Arc, RwLock};

use bytemuck::{Pod, Zeroable};
use egui_wgpu::{Callback, CallbackResources, CallbackTrait, ScreenDescriptor, wgpu};
use wgpu::util::DeviceExt;

use crate::camera::Camera3D;
use crate::geometry::{CapsuleInstance, RenderSceneData, SphereInstance};
use crate::mesh::MeshVertex;

const SHADOW_MAP_SIZE: u32 = 1024;

#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
struct Uniforms {
    view_proj: [[f32; 4]; 4],
    light_view_proj: [[f32; 4]; 4],
    cam_right: [f32; 4],
    cam_up: [f32; 4],
    cam_fwd: [f32; 4],
    cam_eye: [f32; 4],
    /// `[clip_near, clip_far, z_ref, ssao_strength]`
    clip_params: [f32; 4],
    /// `[lighting_preset_id, show_silhouettes, viewport_w, viewport_h]`
    light_params: [f32; 4],
    /// `[shadow_strength, shadow_bias, shadow_tex_size, 0.0]`
    shadow_params: [f32; 4],
}

struct OffscreenTarget {
    width: u32,
    height: u32,
    color_tex: wgpu::Texture,
    color_view: wgpu::TextureView,
    depth_tex: wgpu::Texture,
    depth_view: wgpu::TextureView,
    post_bg: wgpu::BindGroup,
}

struct GpuBiomolResources {
    shadow_mesh_pipeline: wgpu::RenderPipeline,
    shadow_sphere_pipeline: wgpu::RenderPipeline,
    shadow_tex: wgpu::Texture,
    shadow_view: wgpu::TextureView,
    shadow_bg: wgpu::BindGroup,

    mesh_opaque_pipeline: wgpu::RenderPipeline,
    mesh_transp_pipeline: wgpu::RenderPipeline,
    mesh_wire_pipeline: wgpu::RenderPipeline,
    sphere_opaque_pipeline: wgpu::RenderPipeline,
    sphere_transp_pipeline: wgpu::RenderPipeline,
    capsule_pipeline: wgpu::RenderPipeline,
    post_pipeline: wgpu::RenderPipeline,
    post_bgl: wgpu::BindGroupLayout,
    uniform_buf: wgpu::Buffer,
    uniform_bg: wgpu::BindGroup,

    // Dynamic vertex/index/instance buffers
    opaque_vbuf: wgpu::Buffer,
    opaque_vcap: usize,
    opaque_ibuf: wgpu::Buffer,
    opaque_icap: usize,
    opaque_icount: u32,

    transp_vbuf: wgpu::Buffer,
    transp_vcap: usize,
    transp_ibuf: wgpu::Buffer,
    transp_icap: usize,
    transp_icount: u32,

    wire_vbuf: wgpu::Buffer,
    wire_vcap: usize,
    wire_ibuf: wgpu::Buffer,
    wire_icap: usize,
    wire_icount: u32,

    sphere_opaque_buf: wgpu::Buffer,
    sphere_opaque_cap: usize,
    sphere_opaque_count: u32,

    sphere_transp_buf: wgpu::Buffer,
    sphere_transp_cap: usize,
    sphere_transp_count: u32,

    capsule_buf: wgpu::Buffer,
    capsule_cap: usize,
    capsule_count: u32,

    offscreen: Option<OffscreenTarget>,
}

const SHADOW_MESH_WGSL: &str = r#"
struct Uniforms {
    view_proj: mat4x4<f32>,
    light_view_proj: mat4x4<f32>,
    cam_right: vec4<f32>,
    cam_up: vec4<f32>,
    cam_fwd: vec4<f32>,
    cam_eye: vec4<f32>,
    clip_params: vec4<f32>,
    light_params: vec4<f32>,
    shadow_params: vec4<f32>,
};
@group(0) @binding(0) var<uniform> u: Uniforms;

@vertex
fn vs_main(@location(0) pos: vec3<f32>) -> @builtin(position) vec4<f32> {
    return u.light_view_proj * vec4<f32>(pos, 1.0);
}
"#;

const SHADOW_SPHERE_WGSL: &str = r#"
struct Uniforms {
    view_proj: mat4x4<f32>,
    light_view_proj: mat4x4<f32>,
    cam_right: vec4<f32>,
    cam_up: vec4<f32>,
    cam_fwd: vec4<f32>,
    cam_eye: vec4<f32>,
    clip_params: vec4<f32>,
    light_params: vec4<f32>,
    shadow_params: vec4<f32>,
};
@group(0) @binding(0) var<uniform> u: Uniforms;

struct VsOut {
    @builtin(position) clip_pos: vec4<f32>,
    @location(0) local_uv: vec2<f32>,
    @location(1) center_z: f32,
    @location(2) radius_z: f32,
};

@vertex
fn vs_main(
    @builtin(vertex_index) vid: u32,
    @location(0) center: vec3<f32>,
    @location(1) radius: f32,
) -> VsOut {
    var corners = array<vec2<f32>, 6>(
        vec2<f32>(-1.0, -1.0),
        vec2<f32>( 1.0, -1.0),
        vec2<f32>( 1.0,  1.0),
        vec2<f32>(-1.0, -1.0),
        vec2<f32>( 1.0,  1.0),
        vec2<f32>(-1.0,  1.0),
    );
    let uv = corners[vid];
    let c_clip = u.light_view_proj * vec4<f32>(center, 1.0);
    let scale_x = length(vec3<f32>(u.light_view_proj[0].x, u.light_view_proj[1].x, u.light_view_proj[2].x));
    let scale_y = length(vec3<f32>(u.light_view_proj[0].y, u.light_view_proj[1].y, u.light_view_proj[2].y));
    let scale_z = length(vec3<f32>(u.light_view_proj[0].z, u.light_view_proj[1].z, u.light_view_proj[2].z));
    var out: VsOut;
    out.clip_pos = vec4<f32>(
        c_clip.x + uv.x * radius * scale_x,
        c_clip.y + uv.y * radius * scale_y,
        c_clip.z,
        1.0
    );
    out.local_uv = uv;
    out.center_z = c_clip.z;
    out.radius_z = radius * scale_z;
    return out;
}

@fragment
fn fs_main(in: VsOut) -> @builtin(frag_depth) f32 {
    let r2 = dot(in.local_uv, in.local_uv);
    if (r2 > 1.0) { discard; }
    let nz = sqrt(max(0.0, 1.0 - r2));
    return clamp(in.center_z - nz * in.radius_z, 0.0, 1.0);
}
"#;

const MESH_WGSL: &str = r#"
struct Uniforms {
    view_proj: mat4x4<f32>,
    light_view_proj: mat4x4<f32>,
    cam_right: vec4<f32>,
    cam_up: vec4<f32>,
    cam_fwd: vec4<f32>,
    cam_eye: vec4<f32>,
    clip_params: vec4<f32>,
    light_params: vec4<f32>,
    shadow_params: vec4<f32>,
};
@group(0) @binding(0) var<uniform> u: Uniforms;
@group(1) @binding(0) var shadow_tex: texture_depth_2d;

struct VsOut {
    @builtin(position) clip_pos: vec4<f32>,
    @location(0) world_normal: vec3<f32>,
    @location(1) color: vec4<f32>,
    @location(2) view_z: f32,
    @location(3) world_pos: vec3<f32>,
};

fn sample_shadow_pcf(world_pos: vec3<f32>, ndotl: f32) -> f32 {
    let str = clamp(u.shadow_params.x, 0.0, 1.0);
    if (str <= 0.01) {
        return 0.0;
    }
    let lp = u.light_view_proj * vec4<f32>(world_pos, 1.0);
    let ndc = lp.xyz / max(lp.w, 1e-5);
    let uv = vec2<f32>(ndc.x * 0.5 + 0.5, 0.5 - ndc.y * 0.5);
    if (uv.x <= 0.002 || uv.x >= 0.998 || uv.y <= 0.002 || uv.y >= 0.998 || ndc.z <= 0.0 || ndc.z >= 1.0) {
        return 0.0;
    }
    let dims = vec2<i32>(textureDimensions(shadow_tex));
    let base_xy = vec2<i32>(uv * vec2<f32>(dims));
    let bias = u.shadow_params.y + 0.0025 * (1.0 - clamp(ndotl, 0.0, 1.0));
    var occ = 0.0;
    for (var dy = -1; dy <= 1; dy = dy + 1) {
        for (var dx = -1; dx <= 1; dx = dx + 1) {
            let tc = clamp(base_xy + vec2<i32>(dx, dy), vec2<i32>(0, 0), dims - vec2<i32>(1, 1));
            let sd = textureLoad(shadow_tex, tc, 0);
            if (ndc.z - bias > sd) {
                occ = occ + 1.0;
            }
        }
    }
    return (occ / 9.0) * str;
}

@vertex
fn vs_main(
    @location(0) pos: vec3<f32>,
    @location(1) normal: vec3<f32>,
    @location(2) color: vec4<f32>,
) -> VsOut {
    var out: VsOut;
    out.clip_pos = u.view_proj * vec4<f32>(pos, 1.0);
    out.world_normal = normal;
    out.color = color;
    out.view_z = dot(pos - u.cam_eye.xyz, u.cam_fwd.xyz);
    out.world_pos = pos;
    return out;
}

@fragment
fn fs_main(in: VsOut, @builtin(front_facing) is_front: bool) -> @location(0) vec4<f32> {
    if (in.view_z < u.clip_params.x || in.view_z > u.clip_params.y) {
        discard;
    }
    var wn = normalize(in.world_normal);
    if (!is_front) {
        wn = -wn;
    }
    let n = normalize(vec3<f32>(
        dot(wn, u.cam_right.xyz),
        dot(wn, u.cam_up.xyz),
        max(-dot(wn, u.cam_fwd.xyz), 0.02)
    ));
    let l1 = normalize(vec3<f32>(0.56, 0.62, 0.55));
    let l2 = normalize(vec3<f32>(-0.58, -0.22, 0.78));
    let d1 = max(dot(n, l1), 0.0);
    let d2 = max(dot(n, l2), 0.0);
    let h1 = normalize(l1 + vec3<f32>(0.0, 0.0, 1.0));
    let preset = u.light_params.x;

    var amb = 0.28;
    var diff_k = 0.64;
    var fill_k = 0.22;
    var spec_k = 0.24;
    var shininess = 28.0;
    if (preset > 0.5 && preset < 1.5) {
        amb = 0.70; diff_k = 0.28; fill_k = 0.08; spec_k = 0.02;
    } else if (preset >= 1.5 && preset < 2.5) {
        amb = 0.22; diff_k = 0.72; fill_k = 0.20; spec_k = 0.48; shininess = 46.0;
    } else if (preset >= 2.5 && preset < 3.5) {
        amb = 0.18; diff_k = 0.70; fill_k = 0.26; spec_k = 0.34; shininess = 36.0;
    } else if (preset >= 3.5) {
        amb = 0.34; diff_k = 0.62; fill_k = 0.22; spec_k = 0.18; shininess = 24.0;
    }

    let sh_occ = select(0.0, sample_shadow_pcf(in.world_pos, d1), in.color.a >= 0.95);
    let sh_key = 1.0 - 0.85 * sh_occ;
    let sh_amb = 1.0 - 0.38 * sh_occ;
    let spec = pow(max(dot(n, h1), 0.0), shininess) * spec_k * sh_key;
    let rim = pow(clamp(1.0 - n.z, 0.0, 1.0), 3.0) * select(0.08, 0.22, preset >= 2.5 && preset < 3.5);
    let z_ref = max(u.clip_params.z, 10.0);
    let fog = clamp(1.0 - clamp((in.view_z - z_ref * 0.80) / (z_ref * 2.5), 0.0, 0.36), 0.64, 1.0);
    let lit = (amb * sh_amb + diff_k * d1 * sh_key + fill_k * d2 * sh_amb) * fog;
    let rgb = clamp(in.color.rgb * lit + vec3<f32>(spec + rim), vec3<f32>(0.0), vec3<f32>(1.0));
    return vec4<f32>(rgb, in.color.a);
}
"#;

const SPHERE_WGSL: &str = r#"
struct Uniforms {
    view_proj: mat4x4<f32>,
    light_view_proj: mat4x4<f32>,
    cam_right: vec4<f32>,
    cam_up: vec4<f32>,
    cam_fwd: vec4<f32>,
    cam_eye: vec4<f32>,
    clip_params: vec4<f32>,
    light_params: vec4<f32>,
    shadow_params: vec4<f32>,
};
@group(0) @binding(0) var<uniform> u: Uniforms;
@group(1) @binding(0) var shadow_tex: texture_depth_2d;

struct VsOut {
    @builtin(position) clip_pos: vec4<f32>,
    @location(0) local_uv: vec2<f32>,
    @location(1) color: vec4<f32>,
    @location(2) center_world: vec3<f32>,
    @location(3) radius: f32,
};

struct FsOut {
    @location(0) color: vec4<f32>,
    @builtin(frag_depth) depth: f32,
};

fn sample_shadow_pcf(world_pos: vec3<f32>, ndotl: f32) -> f32 {
    let str = clamp(u.shadow_params.x, 0.0, 1.0);
    if (str <= 0.01) {
        return 0.0;
    }
    let lp = u.light_view_proj * vec4<f32>(world_pos, 1.0);
    let ndc = lp.xyz / max(lp.w, 1e-5);
    let uv = vec2<f32>(ndc.x * 0.5 + 0.5, 0.5 - ndc.y * 0.5);
    if (uv.x <= 0.002 || uv.x >= 0.998 || uv.y <= 0.002 || uv.y >= 0.998 || ndc.z <= 0.0 || ndc.z >= 1.0) {
        return 0.0;
    }
    let dims = vec2<i32>(textureDimensions(shadow_tex));
    let base_xy = vec2<i32>(uv * vec2<f32>(dims));
    let bias = u.shadow_params.y + 0.0025 * (1.0 - clamp(ndotl, 0.0, 1.0));
    var occ = 0.0;
    for (var dy = -1; dy <= 1; dy = dy + 1) {
        for (var dx = -1; dx <= 1; dx = dx + 1) {
            let tc = clamp(base_xy + vec2<i32>(dx, dy), vec2<i32>(0, 0), dims - vec2<i32>(1, 1));
            let sd = textureLoad(shadow_tex, tc, 0);
            if (ndc.z - bias > sd) {
                occ = occ + 1.0;
            }
        }
    }
    return (occ / 9.0) * str;
}

@vertex
fn vs_main(
    @builtin(vertex_index) vid: u32,
    @location(0) center: vec3<f32>,
    @location(1) radius: f32,
    @location(2) color: vec4<f32>,
) -> VsOut {
    var corners = array<vec2<f32>, 6>(
        vec2<f32>(-1.0, -1.0),
        vec2<f32>( 1.0, -1.0),
        vec2<f32>( 1.0,  1.0),
        vec2<f32>(-1.0, -1.0),
        vec2<f32>( 1.0,  1.0),
        vec2<f32>(-1.0,  1.0),
    );
    let uv = corners[vid];
    let world = center + (u.cam_right.xyz * uv.x + u.cam_up.xyz * uv.y) * (radius * 1.04);
    var out: VsOut;
    out.clip_pos = u.view_proj * vec4<f32>(world, 1.0);
    out.local_uv = uv;
    out.color = color;
    out.center_world = center;
    out.radius = radius;
    return out;
}

@fragment
fn fs_main(in: VsOut) -> FsOut {
    let r2 = dot(in.local_uv, in.local_uv);
    if (r2 > 1.0) { discard; }
    let nz = sqrt(max(0.0, 1.0 - r2));
    let n = vec3<f32>(in.local_uv.x, in.local_uv.y, nz);
    let hit_world = in.center_world
        + u.cam_right.xyz * (in.local_uv.x * in.radius)
        + u.cam_up.xyz * (in.local_uv.y * in.radius)
        - u.cam_fwd.xyz * (nz * in.radius);
    let view_z = dot(hit_world - u.cam_eye.xyz, u.cam_fwd.xyz);
    if (view_z < u.clip_params.x || view_z > u.clip_params.y) {
        discard;
    }
    let proj = u.view_proj * vec4<f32>(hit_world, 1.0);
    let frag_z = clamp(proj.z / max(proj.w, 1e-5), 0.0, 1.0);

    let l1 = normalize(vec3<f32>(0.56, 0.62, 0.55));
    let l2 = normalize(vec3<f32>(-0.58, -0.22, 0.78));
    let d1 = max(dot(n, l1), 0.0);
    let d2 = max(dot(n, l2), 0.0);
    let h1 = normalize(l1 + vec3<f32>(0.0, 0.0, 1.0));
    let sh_occ = select(0.0, sample_shadow_pcf(hit_world, d1), in.color.a >= 0.95);
    let sh_key = 1.0 - 0.85 * sh_occ;
    let sh_amb = 1.0 - 0.38 * sh_occ;
    let spec = pow(max(dot(n, h1), 0.0), 30.0) * 0.30 * sh_key;
    let z_ref = max(u.clip_params.z, 10.0);
    let fog = clamp(1.0 - clamp((view_z - z_ref * 0.80) / (z_ref * 2.5), 0.0, 0.36), 0.64, 1.0);
    let lit = (0.26 * sh_amb + 0.66 * d1 * sh_key + 0.20 * d2 * sh_amb) * fog;
    let rgb = clamp(in.color.rgb * lit + vec3<f32>(spec), vec3<f32>(0.0), vec3<f32>(1.0));
    var out: FsOut;
    out.color = vec4<f32>(rgb, in.color.a);
    out.depth = frag_z;
    return out;
}
"#;

const CAPSULE_WGSL: &str = r#"
struct Uniforms {
    view_proj: mat4x4<f32>,
    light_view_proj: mat4x4<f32>,
    cam_right: vec4<f32>,
    cam_up: vec4<f32>,
    cam_fwd: vec4<f32>,
    cam_eye: vec4<f32>,
    clip_params: vec4<f32>,
    light_params: vec4<f32>,
    shadow_params: vec4<f32>,
};
@group(0) @binding(0) var<uniform> u: Uniforms;
@group(1) @binding(0) var shadow_tex: texture_depth_2d;

struct VsOut {
    @builtin(position) clip_pos: vec4<f32>,
    @location(0) local_uv: vec2<f32>,
    @location(1) color: vec4<f32>,
    @location(2) dashed: f32,
    @location(3) world_pos: vec3<f32>,
    @location(4) radius: f32,
};

struct FsOut {
    @location(0) color: vec4<f32>,
    @builtin(frag_depth) depth: f32,
};

fn sample_shadow_pcf(world_pos: vec3<f32>, ndotl: f32) -> f32 {
    let str = clamp(u.shadow_params.x, 0.0, 1.0);
    if (str <= 0.01) {
        return 0.0;
    }
    let lp = u.light_view_proj * vec4<f32>(world_pos, 1.0);
    let ndc = lp.xyz / max(lp.w, 1e-5);
    let uv = vec2<f32>(ndc.x * 0.5 + 0.5, 0.5 - ndc.y * 0.5);
    if (uv.x <= 0.002 || uv.x >= 0.998 || uv.y <= 0.002 || uv.y >= 0.998 || ndc.z <= 0.0 || ndc.z >= 1.0) {
        return 0.0;
    }
    let dims = vec2<i32>(textureDimensions(shadow_tex));
    let base_xy = vec2<i32>(uv * vec2<f32>(dims));
    let bias = u.shadow_params.y + 0.0025 * (1.0 - clamp(ndotl, 0.0, 1.0));
    var occ = 0.0;
    for (var dy = -1; dy <= 1; dy = dy + 1) {
        for (var dx = -1; dx <= 1; dx = dx + 1) {
            let tc = clamp(base_xy + vec2<i32>(dx, dy), vec2<i32>(0, 0), dims - vec2<i32>(1, 1));
            let sd = textureLoad(shadow_tex, tc, 0);
            if (ndc.z - bias > sd) {
                occ = occ + 1.0;
            }
        }
    }
    return (occ / 9.0) * str;
}

@vertex
fn vs_main(
    @builtin(vertex_index) vid: u32,
    @location(0) pa: vec3<f32>,
    @location(1) radius: f32,
    @location(2) pb: vec3<f32>,
    @location(3) dashed: f32,
    @location(4) color: vec4<f32>,
) -> VsOut {
    var quad = array<vec2<f32>, 6>(
        vec2<f32>(0.0, -1.0),
        vec2<f32>(1.0, -1.0),
        vec2<f32>(1.0,  1.0),
        vec2<f32>(0.0, -1.0),
        vec2<f32>(1.0,  1.0),
        vec2<f32>(0.0,  1.0),
    );
    let q = quad[vid];
    let axis = pb - pa;
    let mid = 0.5 * (pa + pb);
    let view_dir = normalize(u.cam_eye.xyz - mid);
    var side = cross(axis, view_dir);
    let slen = length(side);
    if (slen > 1e-5) {
        side = side / slen;
    } else {
        side = u.cam_right.xyz;
    }
    let ext = select(vec3<f32>(0.0), normalize(axis) * (radius * 0.30), length(axis) > 1e-4);
    let base_pt = mix(pa - ext, pb + ext, q.x);
    let world = base_pt + side * (q.y * radius);
    var out: VsOut;
    out.clip_pos = u.view_proj * vec4<f32>(world, 1.0);
    out.local_uv = q;
    out.color = color;
    out.dashed = dashed;
    out.world_pos = world;
    out.radius = radius;
    return out;
}

@fragment
fn fs_main(in: VsOut) -> FsOut {
    if (in.dashed > 0.5 && (i32(floor(in.local_uv.x * 9.0)) % 2) == 1) {
        discard;
    }
    let y = in.local_uv.y;
    let nz = sqrt(max(0.0, 1.0 - y * y));
    let hit_world = in.world_pos - u.cam_fwd.xyz * (nz * in.radius);
    let view_z = dot(hit_world - u.cam_eye.xyz, u.cam_fwd.xyz);
    if (view_z < u.clip_params.x || view_z > u.clip_params.y) {
        discard;
    }
    let proj = u.view_proj * vec4<f32>(hit_world, 1.0);
    let frag_z = clamp(proj.z / max(proj.w, 1e-5), 0.0, 1.0);

    let n = vec3<f32>(0.0, y, nz);
    let l1 = normalize(vec3<f32>(0.56, 0.62, 0.55));
    let l2 = normalize(vec3<f32>(-0.58, -0.22, 0.78));
    let d1 = max(dot(n, l1), 0.0);
    let d2 = max(dot(n, l2), 0.0);
    let sh_occ = sample_shadow_pcf(hit_world, d1);
    let sh_key = 1.0 - 0.85 * sh_occ;
    let sh_amb = 1.0 - 0.38 * sh_occ;
    let z_ref = max(u.clip_params.z, 10.0);
    let fog = clamp(1.0 - clamp((view_z - z_ref * 0.80) / (z_ref * 2.5), 0.0, 0.36), 0.64, 1.0);
    let lit = (0.28 * sh_amb + 0.62 * d1 * sh_key + 0.20 * d2 * sh_amb) * fog;
    var out: FsOut;
    out.color = vec4<f32>(clamp(in.color.rgb * lit, vec3<f32>(0.0), vec3<f32>(1.0)), in.color.a);
    out.depth = frag_z;
    return out;
}
"#;

/// Full-screen post-process shader: 12-tap Poisson-disk Screen-Space Ambient Occlusion (SSAO)
/// + ChimeraX-style Depth-Discontinuity Silhouette Outlines directly over the offscreen Depth32Float buffer.
const POST_SSAO_WGSL: &str = r#"
struct Uniforms {
    view_proj: mat4x4<f32>,
    light_view_proj: mat4x4<f32>,
    cam_right: vec4<f32>,
    cam_up: vec4<f32>,
    cam_fwd: vec4<f32>,
    cam_eye: vec4<f32>,
    clip_params: vec4<f32>,
    light_params: vec4<f32>,
    shadow_params: vec4<f32>,
};
@group(0) @binding(0) var<uniform> u: Uniforms;
@group(1) @binding(0) var color_tex: texture_2d<f32>;
@group(1) @binding(1) var depth_tex: texture_depth_2d;

struct VsOut {
    @builtin(position) pos: vec4<f32>,
    @location(0) uv: vec2<f32>,
};

@vertex
fn vs_main(@builtin(vertex_index) vid: u32) -> VsOut {
    var pts = array<vec2<f32>, 6>(
        vec2<f32>(-1.0, -1.0),
        vec2<f32>( 1.0, -1.0),
        vec2<f32>( 1.0,  1.0),
        vec2<f32>(-1.0, -1.0),
        vec2<f32>( 1.0,  1.0),
        vec2<f32>(-1.0,  1.0),
    );
    let p = pts[vid];
    var out: VsOut;
    out.pos = vec4<f32>(p, 0.0, 1.0);
    out.uv = vec2<f32>(p.x * 0.5 + 0.5, 0.5 - p.y * 0.5);
    return out;
}

@fragment
fn fs_main(in: VsOut) -> @location(0) vec4<f32> {
    let dims = vec2<i32>(textureDimensions(color_tex));
    let xy = clamp(vec2<i32>(in.uv * vec2<f32>(dims)), vec2<i32>(0, 0), dims - vec2<i32>(1, 1));
    let base = textureLoad(color_tex, xy, 0);
    let d0 = textureLoad(depth_tex, xy, 0);
    if (d0 >= 0.99995) {
        return base;
    }

    var factor = 1.0;
    let ssao_str = clamp(u.clip_params.w, 0.0, 1.0);
    if (ssao_str > 0.01) {
        var offsets = array<vec2<i32>, 12>(
            vec2<i32>( 3,  0), vec2<i32>(-3,  0), vec2<i32>( 0,  3), vec2<i32>( 0, -3),
            vec2<i32>( 5,  4), vec2<i32>(-5,  4), vec2<i32>( 5, -4), vec2<i32>(-5, -4),
            vec2<i32>( 9,  2), vec2<i32>(-9, -2), vec2<i32>( 2,  9), vec2<i32>(-2, -9),
        );
        var occ = 0.0;
        for (var i = 0; i < 12; i = i + 1) {
            let sc = clamp(xy + offsets[i], vec2<i32>(0, 0), dims - vec2<i32>(1, 1));
            let ds = textureLoad(depth_tex, sc, 0);
            let diff = d0 - ds;
            if (diff > 0.0004 && diff < 0.045) {
                occ = occ + 1.0;
            }
        }
        factor = factor * (1.0 - (occ / 12.0) * 0.42 * ssao_str);
    }

    if (u.light_params.y > 0.5) {
        let dl = textureLoad(depth_tex, clamp(xy + vec2<i32>(-1, 0), vec2<i32>(0, 0), dims - vec2<i32>(1, 1)), 0);
        let dr = textureLoad(depth_tex, clamp(xy + vec2<i32>( 1, 0), vec2<i32>(0, 0), dims - vec2<i32>(1, 1)), 0);
        let du = textureLoad(depth_tex, clamp(xy + vec2<i32>( 0,-1), vec2<i32>(0, 0), dims - vec2<i32>(1, 1)), 0);
        let dd = textureLoad(depth_tex, clamp(xy + vec2<i32>( 0, 1), vec2<i32>(0, 0), dims - vec2<i32>(1, 1)), 0);
        let max_jump = max(max(dl - d0, dr - d0), max(du - d0, dd - d0));
        if (max_jump > 0.0065) {
            factor = factor * 0.42;
        }
    }

    return vec4<f32>(clamp(base.rgb * factor, vec3<f32>(0.0), vec3<f32>(1.0)), 1.0);
}
"#;

/// Shared thread-safe handle holding the latest built biomolecular scene for the `wgpu` viewport.
#[derive(Clone, Default)]
pub struct SharedViewportScene {
    pub data: Arc<RwLock<RenderSceneData>>,
}

impl SharedViewportScene {
    pub fn update(&self, scene: RenderSceneData) {
        if let Ok(mut guard) = self.data.write() {
            *guard = scene;
        }
    }
}

struct BiomolPaintCallback {
    camera: Camera3D,
    aspect: f32,
    px_size: [u32; 2],
    show_silhouettes: bool,
    scene: SharedViewportScene,
}

fn ensure_offscreen(
    device: &wgpu::Device,
    post_bgl: &wgpu::BindGroupLayout,
    slot: &mut Option<OffscreenTarget>,
    w: u32,
    h: u32,
) {
    let w = w.clamp(64, 4096);
    let h = h.clamp(64, 4096);
    if let Some(cur) = slot.as_ref()
        && cur.width == w
        && cur.height == h
    {
        return;
    }
    let color_tex = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("biomol_offscreen_color"),
        size: wgpu::Extent3d {
            width: w,
            height: h,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::Bgra8Unorm,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
        view_formats: &[],
    });
    let color_view = color_tex.create_view(&wgpu::TextureViewDescriptor::default());
    let depth_tex = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("biomol_offscreen_depth"),
        size: wgpu::Extent3d {
            width: w,
            height: h,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::Depth32Float,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
        view_formats: &[],
    });
    let depth_view = depth_tex.create_view(&wgpu::TextureViewDescriptor::default());
    let post_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("biomol_post_bg"),
        layout: post_bgl,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: wgpu::BindingResource::TextureView(&color_view),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: wgpu::BindingResource::TextureView(&depth_view),
            },
        ],
    });
    *slot = Some(OffscreenTarget {
        width: w,
        height: h,
        color_tex,
        color_view,
        depth_tex,
        depth_view,
        post_bg,
    });
}

fn upload_buffer<T: Pod>(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    buf: &mut wgpu::Buffer,
    cap: &mut usize,
    usage: wgpu::BufferUsages,
    label: &str,
    items: &[T],
) -> u32 {
    if items.len() > *cap {
        let new_cap = items.len().next_power_of_two().max(1024);
        *buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some(label),
            size: (new_cap * std::mem::size_of::<T>()) as u64,
            usage: usage | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        *cap = new_cap;
    }
    if !items.is_empty() {
        queue.write_buffer(buf, 0, bytemuck::cast_slice(items));
    }
    items.len() as u32
}

impl CallbackTrait for BiomolPaintCallback {
    fn prepare(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        _screen_descriptor: &ScreenDescriptor,
        _egui_encoder: &mut wgpu::CommandEncoder,
        resources: &mut CallbackResources,
    ) -> Vec<wgpu::CommandBuffer> {
        let target_format = wgpu::TextureFormat::Bgra8Unorm;
        if resources.get::<GpuBiomolResources>().is_none() {
            let gpu = create_gpu_resources(device, target_format);
            resources.insert(gpu);
        }
        let Some(gpu) = resources.get_mut::<GpuBiomolResources>() else {
            return Vec::new();
        };

        ensure_offscreen(
            device,
            &gpu.post_bgl,
            &mut gpu.offscreen,
            self.px_size[0],
            self.px_size[1],
        );

        let (right, up, fwd) = self.camera.basis();
        let eye = self.camera.eye();
        let (clip_near, clip_far) = self.camera.clip_planes();
        let mut clear_bg = [0.06_f64, 0.08, 0.12, 1.0];
        let mut light_id = 0.0_f32;
        let mut ssao_str = 0.55_f32;
        let mut shadow_str = 0.55_f32;
        let mut shadow_rad = (self.camera.distance * 0.65).clamp(14.0, 220.0);
        let mut sil_on = self.show_silhouettes;

        if let Ok(sc) = self.scene.data.read() {
            clear_bg = [
                f64::from(sc.background[0]),
                f64::from(sc.background[1]),
                f64::from(sc.background[2]),
                1.0,
            ];
            light_id = sc.lighting.shader_id();
            ssao_str = sc.ssao_strength;
            shadow_str = sc.shadow_strength.clamp(0.0, 1.0);
            shadow_rad = sc.shadow_radius_around(self.camera.target, shadow_rad);
            sil_on = self.show_silhouettes && sc.show_silhouettes;

            let mut op_verts: Vec<MeshVertex> = Vec::new();
            let mut op_idxs: Vec<u32> = Vec::new();
            for m in &sc.opaque_meshes {
                let base = op_verts.len() as u32;
                op_verts.extend_from_slice(&m.vertices);
                op_idxs.extend(m.indices.iter().map(|&i| base + i));
            }
            let _ = upload_buffer(
                device,
                queue,
                &mut gpu.opaque_vbuf,
                &mut gpu.opaque_vcap,
                wgpu::BufferUsages::VERTEX,
                "biomol_opaque_vbuf",
                &op_verts,
            );
            gpu.opaque_icount = upload_buffer(
                device,
                queue,
                &mut gpu.opaque_ibuf,
                &mut gpu.opaque_icap,
                wgpu::BufferUsages::INDEX,
                "biomol_opaque_ibuf",
                &op_idxs,
            );

            let mut tr_verts: Vec<MeshVertex> = Vec::new();
            let mut tr_idxs: Vec<u32> = Vec::new();
            let mut wr_verts: Vec<MeshVertex> = Vec::new();
            let mut wr_idxs: Vec<u32> = Vec::new();
            for m in &sc.transparent_meshes {
                if m.wireframe {
                    let base = wr_verts.len() as u32;
                    wr_verts.extend_from_slice(&m.vertices);
                    for tri in m.indices.as_chunks::<3>().0 {
                        wr_idxs.extend_from_slice(&[
                            base + tri[0],
                            base + tri[1],
                            base + tri[1],
                            base + tri[2],
                            base + tri[2],
                            base + tri[0],
                        ]);
                    }
                } else {
                    let base = tr_verts.len() as u32;
                    tr_verts.extend_from_slice(&m.vertices);
                    tr_idxs.extend(m.indices.iter().map(|&i| base + i));
                }
            }
            let _ = upload_buffer(
                device,
                queue,
                &mut gpu.transp_vbuf,
                &mut gpu.transp_vcap,
                wgpu::BufferUsages::VERTEX,
                "biomol_transp_vbuf",
                &tr_verts,
            );
            gpu.transp_icount = upload_buffer(
                device,
                queue,
                &mut gpu.transp_ibuf,
                &mut gpu.transp_icap,
                wgpu::BufferUsages::INDEX,
                "biomol_transp_ibuf",
                &tr_idxs,
            );
            let _ = upload_buffer(
                device,
                queue,
                &mut gpu.wire_vbuf,
                &mut gpu.wire_vcap,
                wgpu::BufferUsages::VERTEX,
                "biomol_wire_vbuf",
                &wr_verts,
            );
            gpu.wire_icount = upload_buffer(
                device,
                queue,
                &mut gpu.wire_ibuf,
                &mut gpu.wire_icap,
                wgpu::BufferUsages::INDEX,
                "biomol_wire_ibuf",
                &wr_idxs,
            );

            let mut sp_op = Vec::new();
            let mut sp_tr = Vec::new();
            for &sp in &sc.spheres {
                if sp.color[3] >= 0.95 {
                    sp_op.push(sp);
                } else {
                    sp_tr.push(sp);
                }
            }
            gpu.sphere_opaque_count = upload_buffer(
                device,
                queue,
                &mut gpu.sphere_opaque_buf,
                &mut gpu.sphere_opaque_cap,
                wgpu::BufferUsages::VERTEX,
                "biomol_spheres_op",
                &sp_op,
            );
            gpu.sphere_transp_count = upload_buffer(
                device,
                queue,
                &mut gpu.sphere_transp_buf,
                &mut gpu.sphere_transp_cap,
                wgpu::BufferUsages::VERTEX,
                "biomol_spheres_tr",
                &sp_tr,
            );
            gpu.capsule_count = upload_buffer(
                device,
                queue,
                &mut gpu.capsule_buf,
                &mut gpu.capsule_cap,
                wgpu::BufferUsages::VERTEX,
                "biomol_capsules",
                &sc.capsules,
            );
        }

        let uniforms = Uniforms {
            view_proj: self.camera.view_proj_matrix(self.aspect),
            light_view_proj: self.camera.light_view_proj_matrix(shadow_rad),
            cam_right: [right[0], right[1], right[2], 0.0],
            cam_up: [up[0], up[1], up[2], 0.0],
            cam_fwd: [fwd[0], fwd[1], fwd[2], 0.0],
            cam_eye: [eye[0], eye[1], eye[2], 1.0],
            clip_params: [clip_near, clip_far, self.camera.distance, ssao_str],
            light_params: [
                light_id,
                if sil_on { 1.0 } else { 0.0 },
                self.px_size[0] as f32,
                self.px_size[1] as f32,
            ],
            shadow_params: [shadow_str, 0.0022, SHADOW_MAP_SIZE as f32, 0.0],
        };
        queue.write_buffer(&gpu.uniform_buf, 0, bytemuck::bytes_of(&uniforms));

        let Some(ref target) = gpu.offscreen else {
            return Vec::new();
        };
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("biomol_offscreen_encoder"),
        });

        // Pass 0: Directional Shadow Map Depth Pass (1024×1024 Depth32Float)
        {
            let mut sp = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("biomol_shadow_pass"),
                color_attachments: &[],
                depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                    view: &gpu.shadow_view,
                    depth_ops: Some(wgpu::Operations {
                        load: wgpu::LoadOp::Clear(1.0),
                        store: wgpu::StoreOp::Store,
                    }),
                    stencil_ops: None,
                }),
                timestamp_writes: None,
                occlusion_query_set: None,
                multiview_mask: None,
            });
            if shadow_str > 0.01 {
                if gpu.opaque_icount > 0 {
                    sp.set_pipeline(&gpu.shadow_mesh_pipeline);
                    sp.set_bind_group(0, &gpu.uniform_bg, &[]);
                    sp.set_vertex_buffer(0, gpu.opaque_vbuf.slice(..));
                    sp.set_index_buffer(gpu.opaque_ibuf.slice(..), wgpu::IndexFormat::Uint32);
                    sp.draw_indexed(0..gpu.opaque_icount, 0, 0..1);
                }
                if gpu.sphere_opaque_count > 0 {
                    sp.set_pipeline(&gpu.shadow_sphere_pipeline);
                    sp.set_bind_group(0, &gpu.uniform_bg, &[]);
                    sp.set_vertex_buffer(0, gpu.sphere_opaque_buf.slice(..));
                    sp.draw(0..6, 0..gpu.sphere_opaque_count);
                }
            }
        }

        // Pass 1: Main Offscreen Color + Depth Pass (sampling Shadow Map PCF 3×3)
        {
            let mut rp = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("biomol_offscreen_pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &target.color_view,
                    depth_slice: None,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color {
                            r: clear_bg[0],
                            g: clear_bg[1],
                            b: clear_bg[2],
                            a: clear_bg[3],
                        }),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                    view: &target.depth_view,
                    depth_ops: Some(wgpu::Operations {
                        load: wgpu::LoadOp::Clear(1.0),
                        store: wgpu::StoreOp::Store,
                    }),
                    stencil_ops: None,
                }),
                timestamp_writes: None,
                occlusion_query_set: None,
                multiview_mask: None,
            });

            if gpu.opaque_icount > 0 {
                rp.set_pipeline(&gpu.mesh_opaque_pipeline);
                rp.set_bind_group(0, &gpu.uniform_bg, &[]);
                rp.set_bind_group(1, &gpu.shadow_bg, &[]);
                rp.set_vertex_buffer(0, gpu.opaque_vbuf.slice(..));
                rp.set_index_buffer(gpu.opaque_ibuf.slice(..), wgpu::IndexFormat::Uint32);
                rp.draw_indexed(0..gpu.opaque_icount, 0, 0..1);
            }
            if gpu.capsule_count > 0 {
                rp.set_pipeline(&gpu.capsule_pipeline);
                rp.set_bind_group(0, &gpu.uniform_bg, &[]);
                rp.set_bind_group(1, &gpu.shadow_bg, &[]);
                rp.set_vertex_buffer(0, gpu.capsule_buf.slice(..));
                rp.draw(0..6, 0..gpu.capsule_count);
            }
            if gpu.sphere_opaque_count > 0 {
                rp.set_pipeline(&gpu.sphere_opaque_pipeline);
                rp.set_bind_group(0, &gpu.uniform_bg, &[]);
                rp.set_bind_group(1, &gpu.shadow_bg, &[]);
                rp.set_vertex_buffer(0, gpu.sphere_opaque_buf.slice(..));
                rp.draw(0..6, 0..gpu.sphere_opaque_count);
            }
            if gpu.transp_icount > 0 {
                rp.set_pipeline(&gpu.mesh_transp_pipeline);
                rp.set_bind_group(0, &gpu.uniform_bg, &[]);
                rp.set_bind_group(1, &gpu.shadow_bg, &[]);
                rp.set_vertex_buffer(0, gpu.transp_vbuf.slice(..));
                rp.set_index_buffer(gpu.transp_ibuf.slice(..), wgpu::IndexFormat::Uint32);
                rp.draw_indexed(0..gpu.transp_icount, 0, 0..1);
            }
            if gpu.wire_icount > 0 {
                rp.set_pipeline(&gpu.mesh_wire_pipeline);
                rp.set_bind_group(0, &gpu.uniform_bg, &[]);
                rp.set_bind_group(1, &gpu.shadow_bg, &[]);
                rp.set_vertex_buffer(0, gpu.wire_vbuf.slice(..));
                rp.set_index_buffer(gpu.wire_ibuf.slice(..), wgpu::IndexFormat::Uint32);
                rp.draw_indexed(0..gpu.wire_icount, 0, 0..1);
            }
            if gpu.sphere_transp_count > 0 {
                rp.set_pipeline(&gpu.sphere_transp_pipeline);
                rp.set_bind_group(0, &gpu.uniform_bg, &[]);
                rp.set_bind_group(1, &gpu.shadow_bg, &[]);
                rp.set_vertex_buffer(0, gpu.sphere_transp_buf.slice(..));
                rp.draw(0..6, 0..gpu.sphere_transp_count);
            }
        }
        let _ = (&target.color_tex, &target.depth_tex, &gpu.shadow_tex);
        vec![encoder.finish()]
    }

    fn paint(
        &self,
        _info: egui::PaintCallbackInfo,
        render_pass: &mut wgpu::RenderPass<'static>,
        resources: &CallbackResources,
    ) {
        let Some(gpu) = resources.get::<GpuBiomolResources>() else {
            return;
        };
        let Some(ref target) = gpu.offscreen else {
            return;
        };
        render_pass.set_pipeline(&gpu.post_pipeline);
        render_pass.set_bind_group(0, &gpu.uniform_bg, &[]);
        render_pass.set_bind_group(1, &target.post_bg, &[]);
        render_pass.draw(0..6, 0..1);
    }
}

fn depth_state(write_enabled: bool) -> Option<wgpu::DepthStencilState> {
    Some(wgpu::DepthStencilState {
        format: wgpu::TextureFormat::Depth32Float,
        depth_write_enabled: Some(write_enabled),
        depth_compare: Some(wgpu::CompareFunction::LessEqual),
        stencil: wgpu::StencilState::default(),
        bias: wgpu::DepthBiasState::default(),
    })
}

fn create_gpu_resources(
    device: &wgpu::Device,
    target_format: wgpu::TextureFormat,
) -> GpuBiomolResources {
    let uniform_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("biomol_uniforms"),
        contents: bytemuck::bytes_of(&Uniforms::zeroed()),
        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
    });
    let bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("biomol_bgl"),
        entries: &[wgpu::BindGroupLayoutEntry {
            binding: 0,
            visibility: wgpu::ShaderStages::VERTEX_FRAGMENT,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Uniform,
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        }],
    });
    let uniform_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("biomol_bg"),
        layout: &bgl,
        entries: &[wgpu::BindGroupEntry {
            binding: 0,
            resource: uniform_buf.as_entire_binding(),
        }],
    });

    // Dedicated 1024×1024 Depth32Float shadow map + group 1 bind group for scene passes
    let shadow_tex = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("biomol_shadow_depth"),
        size: wgpu::Extent3d {
            width: SHADOW_MAP_SIZE,
            height: SHADOW_MAP_SIZE,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::Depth32Float,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
        view_formats: &[],
    });
    let shadow_view = shadow_tex.create_view(&wgpu::TextureViewDescriptor::default());
    let shadow_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("biomol_shadow_bgl"),
        entries: &[wgpu::BindGroupLayoutEntry {
            binding: 0,
            visibility: wgpu::ShaderStages::FRAGMENT,
            ty: wgpu::BindingType::Texture {
                sample_type: wgpu::TextureSampleType::Depth,
                view_dimension: wgpu::TextureViewDimension::D2,
                multisampled: false,
            },
            count: None,
        }],
    });
    let shadow_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("biomol_shadow_bg"),
        layout: &shadow_bgl,
        entries: &[wgpu::BindGroupEntry {
            binding: 0,
            resource: wgpu::BindingResource::TextureView(&shadow_view),
        }],
    });

    let shadow_pl_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("biomol_shadow_pl_layout"),
        bind_group_layouts: &[Some(&bgl)],
        immediate_size: 0,
    });
    let pl_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("biomol_pl_layout"),
        bind_group_layouts: &[Some(&bgl), Some(&shadow_bgl)],
        immediate_size: 0,
    });

    // Post-process bind group layout (color_tex + depth_tex via textureLoad)
    let post_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("biomol_post_bgl"),
        entries: &[
            wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Texture {
                    sample_type: wgpu::TextureSampleType::Float { filterable: false },
                    view_dimension: wgpu::TextureViewDimension::D2,
                    multisampled: false,
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 1,
                visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Texture {
                    sample_type: wgpu::TextureSampleType::Depth,
                    view_dimension: wgpu::TextureViewDimension::D2,
                    multisampled: false,
                },
                count: None,
            },
        ],
    });
    let post_pl_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("biomol_post_pl_layout"),
        bind_group_layouts: &[Some(&bgl), Some(&post_bgl)],
        immediate_size: 0,
    });

    // 0. Shadow Map Depth Pipelines (Opaque Meshes + Analytical Spheres)
    let shadow_mesh_mod = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("biomol_shadow_mesh_wgsl"),
        source: wgpu::ShaderSource::Wgsl(SHADOW_MESH_WGSL.into()),
    });
    let shadow_mesh_attr = [wgpu::VertexAttribute {
        offset: 0,
        shader_location: 0,
        format: wgpu::VertexFormat::Float32x3,
    }];
    let shadow_mesh_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: Some("biomol_shadow_mesh_pipeline"),
        layout: Some(&shadow_pl_layout),
        vertex: wgpu::VertexState {
            module: &shadow_mesh_mod,
            entry_point: Some("vs_main"),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            buffers: &[Some(wgpu::VertexBufferLayout {
                array_stride: std::mem::size_of::<MeshVertex>() as u64,
                step_mode: wgpu::VertexStepMode::Vertex,
                attributes: &shadow_mesh_attr,
            })],
        },
        fragment: None,
        primitive: wgpu::PrimitiveState {
            topology: wgpu::PrimitiveTopology::TriangleList,
            cull_mode: None,
            ..Default::default()
        },
        depth_stencil: depth_state(true),
        multisample: wgpu::MultisampleState::default(),
        multiview_mask: None,
        cache: None,
    });

    let shadow_sphere_mod = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("biomol_shadow_sphere_wgsl"),
        source: wgpu::ShaderSource::Wgsl(SHADOW_SPHERE_WGSL.into()),
    });
    let shadow_sphere_attrs = [
        wgpu::VertexAttribute {
            offset: 0,
            shader_location: 0,
            format: wgpu::VertexFormat::Float32x3,
        },
        wgpu::VertexAttribute {
            offset: 12,
            shader_location: 1,
            format: wgpu::VertexFormat::Float32,
        },
    ];
    let shadow_sphere_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: Some("biomol_shadow_sphere_pipeline"),
        layout: Some(&shadow_pl_layout),
        vertex: wgpu::VertexState {
            module: &shadow_sphere_mod,
            entry_point: Some("vs_main"),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            buffers: &[Some(wgpu::VertexBufferLayout {
                array_stride: std::mem::size_of::<SphereInstance>() as u64,
                step_mode: wgpu::VertexStepMode::Instance,
                attributes: &shadow_sphere_attrs,
            })],
        },
        fragment: Some(wgpu::FragmentState {
            module: &shadow_sphere_mod,
            entry_point: Some("fs_main"),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            targets: &[],
        }),
        primitive: wgpu::PrimitiveState::default(),
        depth_stencil: depth_state(true),
        multisample: wgpu::MultisampleState::default(),
        multiview_mask: None,
        cache: None,
    });

    // 1. Indexed Triangle Mesh Pipelines (Opaque, Transparent, Wireframe LineList)
    let mesh_mod = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("biomol_mesh_wgsl"),
        source: wgpu::ShaderSource::Wgsl(MESH_WGSL.into()),
    });
    let mesh_attrs = [
        wgpu::VertexAttribute {
            offset: 0,
            shader_location: 0,
            format: wgpu::VertexFormat::Float32x3,
        },
        wgpu::VertexAttribute {
            offset: 12,
            shader_location: 1,
            format: wgpu::VertexFormat::Float32x3,
        },
        wgpu::VertexAttribute {
            offset: 24,
            shader_location: 2,
            format: wgpu::VertexFormat::Float32x4,
        },
    ];
    let make_mesh_pipeline = |label: &str,
                              topology: wgpu::PrimitiveTopology,
                              blend: Option<wgpu::BlendState>,
                              depth_write: bool| {
        device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some(label),
            layout: Some(&pl_layout),
            vertex: wgpu::VertexState {
                module: &mesh_mod,
                entry_point: Some("vs_main"),
                compilation_options: wgpu::PipelineCompilationOptions::default(),
                buffers: &[Some(wgpu::VertexBufferLayout {
                    array_stride: std::mem::size_of::<MeshVertex>() as u64,
                    step_mode: wgpu::VertexStepMode::Vertex,
                    attributes: &mesh_attrs,
                })],
            },
            fragment: Some(wgpu::FragmentState {
                module: &mesh_mod,
                entry_point: Some("fs_main"),
                compilation_options: wgpu::PipelineCompilationOptions::default(),
                targets: &[Some(wgpu::ColorTargetState {
                    format: target_format,
                    blend,
                    write_mask: wgpu::ColorWrites::ALL,
                })],
            }),
            primitive: wgpu::PrimitiveState {
                topology,
                cull_mode: None,
                ..Default::default()
            },
            depth_stencil: depth_state(depth_write),
            multisample: wgpu::MultisampleState::default(),
            multiview_mask: None,
            cache: None,
        })
    };

    let mesh_opaque_pipeline = make_mesh_pipeline(
        "biomol_mesh_opaque",
        wgpu::PrimitiveTopology::TriangleList,
        None,
        true,
    );
    let mesh_transp_pipeline = make_mesh_pipeline(
        "biomol_mesh_transp",
        wgpu::PrimitiveTopology::TriangleList,
        Some(wgpu::BlendState::ALPHA_BLENDING),
        false,
    );
    let mesh_wire_pipeline = make_mesh_pipeline(
        "biomol_mesh_wire",
        wgpu::PrimitiveTopology::LineList,
        Some(wgpu::BlendState::ALPHA_BLENDING),
        false,
    );

    // 2. Sphere Impostor Pipelines (Opaque + Transparent)
    let sphere_mod = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("biomol_sphere_wgsl"),
        source: wgpu::ShaderSource::Wgsl(SPHERE_WGSL.into()),
    });
    let sphere_attrs = [
        wgpu::VertexAttribute {
            offset: 0,
            shader_location: 0,
            format: wgpu::VertexFormat::Float32x3,
        },
        wgpu::VertexAttribute {
            offset: 12,
            shader_location: 1,
            format: wgpu::VertexFormat::Float32,
        },
        wgpu::VertexAttribute {
            offset: 16,
            shader_location: 2,
            format: wgpu::VertexFormat::Float32x4,
        },
    ];
    let make_sphere_pipeline = |label: &str, blend: Option<wgpu::BlendState>, depth_write: bool| {
        device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some(label),
            layout: Some(&pl_layout),
            vertex: wgpu::VertexState {
                module: &sphere_mod,
                entry_point: Some("vs_main"),
                compilation_options: wgpu::PipelineCompilationOptions::default(),
                buffers: &[Some(wgpu::VertexBufferLayout {
                    array_stride: std::mem::size_of::<SphereInstance>() as u64,
                    step_mode: wgpu::VertexStepMode::Instance,
                    attributes: &sphere_attrs,
                })],
            },
            fragment: Some(wgpu::FragmentState {
                module: &sphere_mod,
                entry_point: Some("fs_main"),
                compilation_options: wgpu::PipelineCompilationOptions::default(),
                targets: &[Some(wgpu::ColorTargetState {
                    format: target_format,
                    blend,
                    write_mask: wgpu::ColorWrites::ALL,
                })],
            }),
            primitive: wgpu::PrimitiveState::default(),
            depth_stencil: depth_state(depth_write),
            multisample: wgpu::MultisampleState::default(),
            multiview_mask: None,
            cache: None,
        })
    };
    let sphere_opaque_pipeline = make_sphere_pipeline("biomol_sphere_opaque", None, true);
    let sphere_transp_pipeline = make_sphere_pipeline(
        "biomol_sphere_transp",
        Some(wgpu::BlendState::ALPHA_BLENDING),
        false,
    );

    // 3. Capsule Pipeline
    let cap_mod = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("biomol_capsule_wgsl"),
        source: wgpu::ShaderSource::Wgsl(CAPSULE_WGSL.into()),
    });
    let cap_attrs = [
        wgpu::VertexAttribute {
            offset: 0,
            shader_location: 0,
            format: wgpu::VertexFormat::Float32x3,
        },
        wgpu::VertexAttribute {
            offset: 12,
            shader_location: 1,
            format: wgpu::VertexFormat::Float32,
        },
        wgpu::VertexAttribute {
            offset: 16,
            shader_location: 2,
            format: wgpu::VertexFormat::Float32x3,
        },
        wgpu::VertexAttribute {
            offset: 28,
            shader_location: 3,
            format: wgpu::VertexFormat::Float32,
        },
        wgpu::VertexAttribute {
            offset: 32,
            shader_location: 4,
            format: wgpu::VertexFormat::Float32x4,
        },
    ];
    let capsule_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: Some("biomol_capsule_pipeline"),
        layout: Some(&pl_layout),
        vertex: wgpu::VertexState {
            module: &cap_mod,
            entry_point: Some("vs_main"),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            buffers: &[Some(wgpu::VertexBufferLayout {
                array_stride: std::mem::size_of::<CapsuleInstance>() as u64,
                step_mode: wgpu::VertexStepMode::Instance,
                attributes: &cap_attrs,
            })],
        },
        fragment: Some(wgpu::FragmentState {
            module: &cap_mod,
            entry_point: Some("fs_main"),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            targets: &[Some(wgpu::ColorTargetState {
                format: target_format,
                blend: Some(wgpu::BlendState::ALPHA_BLENDING),
                write_mask: wgpu::ColorWrites::ALL,
            })],
        }),
        primitive: wgpu::PrimitiveState::default(),
        depth_stencil: depth_state(true),
        multisample: wgpu::MultisampleState::default(),
        multiview_mask: None,
        cache: None,
    });

    // 4. Post-process SSAO + Depth Silhouettes Blit Pipeline
    let post_mod = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("biomol_post_ssao_wgsl"),
        source: wgpu::ShaderSource::Wgsl(POST_SSAO_WGSL.into()),
    });
    let post_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: Some("biomol_post_ssao_pipeline"),
        layout: Some(&post_pl_layout),
        vertex: wgpu::VertexState {
            module: &post_mod,
            entry_point: Some("vs_main"),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            buffers: &[],
        },
        fragment: Some(wgpu::FragmentState {
            module: &post_mod,
            entry_point: Some("fs_main"),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            targets: &[Some(wgpu::ColorTargetState {
                format: target_format,
                blend: Some(wgpu::BlendState::REPLACE),
                write_mask: wgpu::ColorWrites::ALL,
            })],
        }),
        primitive: wgpu::PrimitiveState::default(),
        depth_stencil: None,
        multisample: wgpu::MultisampleState::default(),
        multiview_mask: None,
        cache: None,
    });

    let mk_buf = |label: &str, bytes: u64, usage: wgpu::BufferUsages| {
        device.create_buffer(&wgpu::BufferDescriptor {
            label: Some(label),
            size: bytes.max(64),
            usage: usage | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        })
    };

    let init_cap = 1024usize;
    GpuBiomolResources {
        shadow_mesh_pipeline,
        shadow_sphere_pipeline,
        shadow_tex,
        shadow_view,
        shadow_bg,
        mesh_opaque_pipeline,
        mesh_transp_pipeline,
        mesh_wire_pipeline,
        sphere_opaque_pipeline,
        sphere_transp_pipeline,
        capsule_pipeline,
        post_pipeline,
        post_bgl,
        uniform_buf,
        uniform_bg,
        opaque_vbuf: mk_buf(
            "biomol_opaque_vbuf",
            (init_cap * std::mem::size_of::<MeshVertex>()) as u64,
            wgpu::BufferUsages::VERTEX,
        ),
        opaque_vcap: init_cap,
        opaque_ibuf: mk_buf(
            "biomol_opaque_ibuf",
            (init_cap * 4) as u64,
            wgpu::BufferUsages::INDEX,
        ),
        opaque_icap: init_cap,
        opaque_icount: 0,
        transp_vbuf: mk_buf(
            "biomol_transp_vbuf",
            (init_cap * std::mem::size_of::<MeshVertex>()) as u64,
            wgpu::BufferUsages::VERTEX,
        ),
        transp_vcap: init_cap,
        transp_ibuf: mk_buf(
            "biomol_transp_ibuf",
            (init_cap * 4) as u64,
            wgpu::BufferUsages::INDEX,
        ),
        transp_icap: init_cap,
        transp_icount: 0,
        wire_vbuf: mk_buf(
            "biomol_wire_vbuf",
            (init_cap * std::mem::size_of::<MeshVertex>()) as u64,
            wgpu::BufferUsages::VERTEX,
        ),
        wire_vcap: init_cap,
        wire_ibuf: mk_buf(
            "biomol_wire_ibuf",
            (init_cap * 4) as u64,
            wgpu::BufferUsages::INDEX,
        ),
        wire_icap: init_cap,
        wire_icount: 0,
        sphere_opaque_buf: mk_buf(
            "biomol_spheres_op",
            (init_cap * std::mem::size_of::<SphereInstance>()) as u64,
            wgpu::BufferUsages::VERTEX,
        ),
        sphere_opaque_cap: init_cap,
        sphere_opaque_count: 0,
        sphere_transp_buf: mk_buf(
            "biomol_spheres_tr",
            (init_cap * std::mem::size_of::<SphereInstance>()) as u64,
            wgpu::BufferUsages::VERTEX,
        ),
        sphere_transp_cap: init_cap,
        sphere_transp_count: 0,
        capsule_buf: mk_buf(
            "biomol_capsules",
            (init_cap * std::mem::size_of::<CapsuleInstance>()) as u64,
            wgpu::BufferUsages::VERTEX,
        ),
        capsule_cap: init_cap,
        capsule_count: 0,
        offscreen: None,
    }
}

/// Interactive embeddable `egui` 3D biomolecular viewport widget (backed by `wgpu 30` offscreen
/// `Depth32Float` + Directional PCF 3×3 Soft Shadows + GPU SSAO/Silhouettes).
pub fn show_biomol_viewport(
    ui: &mut egui::Ui,
    camera: &mut Camera3D,
    scene: &SharedViewportScene,
    show_silhouettes: bool,
    use_wgpu: bool,
) -> egui::Response {
    show_biomol_viewport_ex(ui, camera, scene, show_silhouettes, use_wgpu, false)
}

/// Extended viewport widget with `suppress_primary_orbit` so callers can intercept primary mouse
/// drag for interactive 3D tools (e.g. live mouse `Drag-to-Sculpt` in `BiomolCraft`).
pub fn show_biomol_viewport_ex(
    ui: &mut egui::Ui,
    camera: &mut Camera3D,
    scene: &SharedViewportScene,
    show_silhouettes: bool,
    use_wgpu: bool,
    suppress_primary_orbit: bool,
) -> egui::Response {
    let avail = ui.available_size();
    let (rect, response) = ui.allocate_exact_size(
        egui::vec2(avail.x.max(200.0), avail.y.max(200.0)),
        egui::Sense::click_and_drag(),
    );

    if !suppress_primary_orbit && response.dragged_by(egui::PointerButton::Primary) {
        let d = response.drag_delta();
        camera.orbit(d.x, d.y);
    }
    if response.dragged_by(egui::PointerButton::Secondary)
        || response.dragged_by(egui::PointerButton::Middle)
    {
        let d = response.drag_delta();
        camera.pan(d.x, d.y);
    }
    if response.hovered() {
        let scroll = ui.input(|i| i.smooth_scroll_delta.y);
        if scroll.abs() > 0.1 {
            camera.zoom(scroll);
        }
    }

    let painter = ui.painter_at(rect);
    let bg = scene
        .data
        .read()
        .map(|s| s.background)
        .unwrap_or([0.06, 0.08, 0.12, 1.0]);
    painter.rect_filled(
        rect,
        0.0,
        egui::Color32::from_rgb(
            (bg[0] * 255.0) as u8,
            (bg[1] * 255.0) as u8,
            (bg[2] * 255.0) as u8,
        ),
    );

    let aspect = (rect.width() / rect.height().max(1.0)).max(0.1);
    let ppp = ui.ctx().pixels_per_point();
    let px_w = ((rect.width() * ppp).round() as u32).clamp(64, 4096);
    let px_h = ((rect.height() * ppp).round() as u32).clamp(64, 4096);

    if use_wgpu {
        let cb = Callback::new_paint_callback(
            rect,
            BiomolPaintCallback {
                camera: camera.clone(),
                aspect,
                px_size: [px_w, px_h],
                show_silhouettes,
                scene: scene.clone(),
            },
        );
        painter.add(cb);
    } else if let Ok(sc) = scene.data.read() {
        let eye = camera.eye();
        let (right, up, fwd) = camera.basis();
        let half_h = (0.5 * camera.fov_y_rad).tan();
        let proj = |p: [f32; 3]| -> Option<(egui::Pos2, f32, f32)> {
            let rel = [p[0] - eye[0], p[1] - eye[1], p[2] - eye[2]];
            let z = rel[0] * fwd[0] + rel[1] * fwd[1] + rel[2] * fwd[2];
            if z <= 0.5 {
                return None;
            }
            let scale = z * half_h;
            let sx = rect.left()
                + ((rel[0] * right[0] + rel[1] * right[1] + rel[2] * right[2]) / (scale * aspect)
                    * 0.5
                    + 0.5)
                    * rect.width();
            let sy = rect.top()
                + (0.5 - (rel[0] * up[0] + rel[1] * up[1] + rel[2] * up[2]) / scale * 0.5)
                    * rect.height();
            Some((egui::pos2(sx, sy), z, rect.height() / (2.0 * scale)))
        };
        for cap in &sc.capsules {
            if let (Some((pa, _, ppa)), Some((pb, _, _))) = (proj(cap.a), proj(cap.b)) {
                let col = egui::Color32::from_rgba_unmultiplied(
                    (cap.color[0] * 255.0) as u8,
                    (cap.color[1] * 255.0) as u8,
                    (cap.color[2] * 255.0) as u8,
                    (cap.color[3] * 255.0) as u8,
                );
                painter.line_segment(
                    [pa, pb],
                    egui::Stroke::new((cap.radius * ppa * 1.4).clamp(1.0, 12.0), col),
                );
            }
        }
        for sp in &sc.spheres {
            if let Some((p, _, ppa)) = proj(sp.center) {
                let col = egui::Color32::from_rgba_unmultiplied(
                    (sp.color[0] * 255.0) as u8,
                    (sp.color[1] * 255.0) as u8,
                    (sp.color[2] * 255.0) as u8,
                    (sp.color[3] * 255.0) as u8,
                );
                painter.circle_filled(p, (sp.radius * ppa).clamp(1.5, 24.0), col);
            }
        }
    }

    response
}
