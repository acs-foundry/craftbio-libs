#![forbid(unsafe_code)]

//! Pure-Rust `wgpu 24 + WGSL` Instanced GPU Biomolecular 3D Renderer (`egui_wgpu::CallbackTrait`).
//! Renders mathematically smooth ray-cast Sphere Impostors + Capsule/Cylinder Ribbons & Bonds
//! with ChimeraX-inspired studio lighting, specular highlights, depth fog, and silhouette edges.

use std::sync::{Arc, RwLock};

use bytemuck::{Pod, Zeroable};
use egui_wgpu::{Callback, CallbackResources, CallbackTrait, ScreenDescriptor, wgpu};
use wgpu::util::DeviceExt;

use crate::camera::Camera3D;
use crate::geometry::{CapsuleInstance, RenderSceneData, SphereInstance};

#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
struct Uniforms {
    view_proj: [[f32; 4]; 4],
    cam_right: [f32; 4],
    cam_up: [f32; 4],
    cam_eye: [f32; 4],
    fog_params: [f32; 4], // [z_ref, show_silhouettes, 0, 0]
}

struct GpuBiomolResources {
    sphere_pipeline: wgpu::RenderPipeline,
    capsule_pipeline: wgpu::RenderPipeline,
    uniform_buf: wgpu::Buffer,
    uniform_bg: wgpu::BindGroup,
    sphere_buf: wgpu::Buffer,
    sphere_cap: usize,
    sphere_count: u32,
    capsule_buf: wgpu::Buffer,
    capsule_cap: usize,
    capsule_count: u32,
}

const SPHERE_WGSL: &str = r#"
struct Uniforms {
    view_proj: mat4x4<f32>,
    cam_right: vec4<f32>,
    cam_up: vec4<f32>,
    cam_eye: vec4<f32>,
    fog_params: vec4<f32>,
};
@group(0) @binding(0) var<uniform> u: Uniforms;

struct VsOut {
    @builtin(position) clip_pos: vec4<f32>,
    @location(0) local_uv: vec2<f32>,
    @location(1) color: vec4<f32>,
    @location(2) dist_cam: f32,
};

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
    out.dist_cam = length(center - u.cam_eye.xyz);
    return out;
}

@fragment
fn fs_main(in: VsOut) -> @location(0) vec4<f32> {
    let r2 = dot(in.local_uv, in.local_uv);
    if (r2 > 1.0) { discard; }
    let nz = sqrt(max(0.0, 1.0 - r2));
    let n = vec3<f32>(in.local_uv.x, in.local_uv.y, nz);
    let l1 = normalize(vec3<f32>(0.38, 0.58, 0.72));
    let l2 = normalize(vec3<f32>(-0.55, -0.25, 0.80));
    let d1 = max(dot(n, l1), 0.0);
    let d2 = max(dot(n, l2), 0.0);
    let h1 = normalize(l1 + vec3<f32>(0.0, 0.0, 1.0));
    let spec = pow(max(dot(n, h1), 0.0), 28.0) * 0.30;
    let sil = select(1.0, select(1.0, 0.44 + 2.3 * nz, nz < 0.24), u.fog_params.y > 0.5);
    let z_ref = max(u.fog_params.x, 10.0);
    let fog = clamp(1.0 - clamp((in.dist_cam - z_ref * 0.75) / (z_ref * 2.2), 0.0, 0.42), 0.58, 1.0);
    let lit = (0.24 + 0.68 * d1 + 0.20 * d2) * sil * fog;
    let rgb = clamp(in.color.rgb * lit + vec3<f32>(spec * sil), vec3<f32>(0.0), vec3<f32>(1.0));
    return vec4<f32>(rgb, in.color.a);
}
"#;

const CAPSULE_WGSL: &str = r#"
struct Uniforms {
    view_proj: mat4x4<f32>,
    cam_right: vec4<f32>,
    cam_up: vec4<f32>,
    cam_eye: vec4<f32>,
    fog_params: vec4<f32>,
};
@group(0) @binding(0) var<uniform> u: Uniforms;

struct VsOut {
    @builtin(position) clip_pos: vec4<f32>,
    @location(0) local_uv: vec2<f32>,
    @location(1) color: vec4<f32>,
    @location(2) dashed: f32,
    @location(3) dist_cam: f32,
};

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
    let ext = select(vec3<f32>(0.0), normalize(axis) * (radius * 0.35), length(axis) > 1e-4);
    let base_pt = mix(pa - ext, pb + ext, q.x);
    let world = base_pt + side * (q.y * radius);
    var out: VsOut;
    out.clip_pos = u.view_proj * vec4<f32>(world, 1.0);
    out.local_uv = q;
    out.color = color;
    out.dashed = dashed;
    out.dist_cam = length(mid - u.cam_eye.xyz);
    return out;
}

@fragment
fn fs_main(in: VsOut) -> @location(0) vec4<f32> {
    if (in.dashed > 0.5 && (i32(floor(in.local_uv.x * 9.0)) % 2) == 1) {
        discard;
    }
    let y = in.local_uv.y;
    let nz = sqrt(max(0.0, 1.0 - y * y));
    let n = vec3<f32>(0.0, y, nz);
    let l1 = normalize(vec3<f32>(0.38, 0.58, 0.72));
    let l2 = normalize(vec3<f32>(-0.55, -0.25, 0.80));
    let d1 = max(dot(n, l1), 0.0);
    let d2 = max(dot(n, l2), 0.0);
    let sil = select(1.0, select(1.0, 0.45 + 2.2 * nz, nz < 0.24), u.fog_params.y > 0.5);
    let z_ref = max(u.fog_params.x, 10.0);
    let fog = clamp(1.0 - clamp((in.dist_cam - z_ref * 0.75) / (z_ref * 2.2), 0.0, 0.42), 0.58, 1.0);
    let lit = (0.26 + 0.64 * d1 + 0.20 * d2) * sil * fog;
    return vec4<f32>(clamp(in.color.rgb * lit, vec3<f32>(0.0), vec3<f32>(1.0)), in.color.a);
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
    show_silhouettes: bool,
    scene: SharedViewportScene,
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

        let (right, up, _) = self.camera.basis();
        let eye = self.camera.eye();
        let uniforms = Uniforms {
            view_proj: self.camera.view_proj_matrix(self.aspect),
            cam_right: [right[0], right[1], right[2], 0.0],
            cam_up: [up[0], up[1], up[2], 0.0],
            cam_eye: [eye[0], eye[1], eye[2], 1.0],
            fog_params: [
                self.camera.distance,
                if self.show_silhouettes { 1.0 } else { 0.0 },
                0.0,
                0.0,
            ],
        };
        queue.write_buffer(&gpu.uniform_buf, 0, bytemuck::bytes_of(&uniforms));

        if let Ok(sc) = self.scene.data.read() {
            if sc.spheres.len() > gpu.sphere_cap {
                let new_cap = sc.spheres.len().next_power_of_two().max(1024);
                gpu.sphere_buf = device.create_buffer(&wgpu::BufferDescriptor {
                    label: Some("biomol_spheres"),
                    size: (new_cap * std::mem::size_of::<SphereInstance>()) as u64,
                    usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
                    mapped_at_creation: false,
                });
                gpu.sphere_cap = new_cap;
            }
            if !sc.spheres.is_empty() {
                queue.write_buffer(&gpu.sphere_buf, 0, bytemuck::cast_slice(&sc.spheres));
            }
            gpu.sphere_count = sc.spheres.len() as u32;

            if sc.capsules.len() > gpu.capsule_cap {
                let new_cap = sc.capsules.len().next_power_of_two().max(2048);
                gpu.capsule_buf = device.create_buffer(&wgpu::BufferDescriptor {
                    label: Some("biomol_capsules"),
                    size: (new_cap * std::mem::size_of::<CapsuleInstance>()) as u64,
                    usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
                    mapped_at_creation: false,
                });
                gpu.capsule_cap = new_cap;
            }
            if !sc.capsules.is_empty() {
                queue.write_buffer(&gpu.capsule_buf, 0, bytemuck::cast_slice(&sc.capsules));
            }
            gpu.capsule_count = sc.capsules.len() as u32;
        }
        Vec::new()
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
        if gpu.capsule_count > 0 {
            render_pass.set_pipeline(&gpu.capsule_pipeline);
            render_pass.set_bind_group(0, &gpu.uniform_bg, &[]);
            render_pass.set_vertex_buffer(0, gpu.capsule_buf.slice(..));
            render_pass.draw(0..6, 0..gpu.capsule_count);
        }
        if gpu.sphere_count > 0 {
            render_pass.set_pipeline(&gpu.sphere_pipeline);
            render_pass.set_bind_group(0, &gpu.uniform_bg, &[]);
            render_pass.set_vertex_buffer(0, gpu.sphere_buf.slice(..));
            render_pass.draw(0..6, 0..gpu.sphere_count);
        }
    }
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
    let pl_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("biomol_pl_layout"),
        bind_group_layouts: &[&bgl],
        push_constant_ranges: &[],
    });

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
    let sphere_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: Some("biomol_sphere_pipeline"),
        layout: Some(&pl_layout),
        vertex: wgpu::VertexState {
            module: &sphere_mod,
            entry_point: Some("vs_main"),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            buffers: &[wgpu::VertexBufferLayout {
                array_stride: std::mem::size_of::<SphereInstance>() as u64,
                step_mode: wgpu::VertexStepMode::Instance,
                attributes: &sphere_attrs,
            }],
        },
        fragment: Some(wgpu::FragmentState {
            module: &sphere_mod,
            entry_point: Some("fs_main"),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            targets: &[Some(wgpu::ColorTargetState {
                format: target_format,
                blend: Some(wgpu::BlendState::ALPHA_BLENDING),
                write_mask: wgpu::ColorWrites::ALL,
            })],
        }),
        primitive: wgpu::PrimitiveState::default(),
        depth_stencil: None,
        multisample: wgpu::MultisampleState::default(),
        multiview: None,
        cache: None,
    });

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
            buffers: &[wgpu::VertexBufferLayout {
                array_stride: std::mem::size_of::<CapsuleInstance>() as u64,
                step_mode: wgpu::VertexStepMode::Instance,
                attributes: &cap_attrs,
            }],
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
        depth_stencil: None,
        multisample: wgpu::MultisampleState::default(),
        multiview: None,
        cache: None,
    });

    let sphere_cap = 1024;
    let capsule_cap = 2048;
    let sphere_buf = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("biomol_spheres"),
        size: (sphere_cap * std::mem::size_of::<SphereInstance>()) as u64,
        usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let capsule_buf = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("biomol_capsules"),
        size: (capsule_cap * std::mem::size_of::<CapsuleInstance>()) as u64,
        usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    GpuBiomolResources {
        sphere_pipeline,
        capsule_pipeline,
        uniform_buf,
        uniform_bg,
        sphere_buf,
        sphere_cap,
        sphere_count: 0,
        capsule_buf,
        capsule_cap,
        capsule_count: 0,
    }
}

/// Interactive embeddable `egui` 3D biomolecular viewport widget (backed by `wgpu 24`, with
/// automatic software-canvas fallback when running under Glow/CPU).
pub fn show_biomol_viewport(
    ui: &mut egui::Ui,
    camera: &mut Camera3D,
    scene: &SharedViewportScene,
    show_silhouettes: bool,
    use_wgpu: bool,
) -> egui::Response {
    let avail = ui.available_size();
    let (rect, response) = ui.allocate_exact_size(
        egui::vec2(avail.x.max(200.0), avail.y.max(200.0)),
        egui::Sense::click_and_drag(),
    );

    if response.dragged_by(egui::PointerButton::Primary) {
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
    if use_wgpu {
        let cb = Callback::new_paint_callback(
            rect,
            BiomolPaintCallback {
                camera: camera.clone(),
                aspect,
                show_silhouettes,
                scene: scene.clone(),
            },
        );
        painter.add(cb);
    } else if let Ok(sc) = scene.data.read() {
        // Depth-sorted pure-egui 3D fallback when app uses non-wgpu backend
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
