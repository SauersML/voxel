use std::iter;
use std::mem;

use wgpu::util::DeviceExt;
use winit::{
    dpi::PhysicalSize,
    event::*,
    event_loop::{ControlFlow, EventLoop},
    window::WindowBuilder,
};

use glam::{Mat4, Vec3};
use rand::Rng;
use noise::{NoiseFn, Perlin};

//
// ======================== CUBE GEOMETRY ========================
//
// We'll reuse the same cube geometry for planet voxels,
// atmosphere voxels, cloud voxels, and disk-like particles.
// The disk-likes are just squashed via non-uniform scale.
//

#[rustfmt::skip]
const VERTICES: &[f32] = &[
    // pos (x,y,z), normal (nx,ny,nz), uv (u,v)
    // +X
    0.5,  0.5, -0.5,    1.0,  0.0,  0.0,    1.0, 0.0,
    0.5, -0.5, -0.5,    1.0,  0.0,  0.0,    0.0, 0.0,
    0.5, -0.5,  0.5,    1.0,  0.0,  0.0,    0.0, 1.0,
    0.5,  0.5,  0.5,    1.0,  0.0,  0.0,    1.0, 1.0,

    // -X
   -0.5,  0.5,  0.5,   -1.0,  0.0,  0.0,    1.0, 0.0,
   -0.5, -0.5,  0.5,   -1.0,  0.0,  0.0,    0.0, 0.0,
   -0.5, -0.5, -0.5,   -1.0,  0.0,  0.0,    0.0, 1.0,
   -0.5,  0.5, -0.5,   -1.0,  0.0,  0.0,    1.0, 1.0,

    // +Y
   -0.5,  0.5,  0.5,    0.0,  1.0,  0.0,    1.0, 0.0,
   -0.5,  0.5, -0.5,    0.0,  1.0,  0.0,    0.0, 0.0,
    0.5,  0.5, -0.5,    0.0,  1.0,  0.0,    0.0, 1.0,
    0.5,  0.5,  0.5,    0.0,  1.0,  0.0,    1.0, 1.0,

    // -Y
    0.5, -0.5,  0.5,    0.0, -1.0,  0.0,    1.0, 0.0,
    0.5, -0.5, -0.5,    0.0, -1.0,  0.0,    0.0, 0.0,
   -0.5, -0.5, -0.5,    0.0, -1.0,  0.0,    0.0, 1.0,
   -0.5, -0.5,  0.5,    0.0, -1.0,  0.0,    1.0, 1.0,

    // +Z
    0.5,  0.5,  0.5,    0.0,  0.0,  1.0,    1.0, 0.0,
    0.5, -0.5,  0.5,    0.0,  0.0,  1.0,    0.0, 0.0,
   -0.5, -0.5,  0.5,    0.0,  0.0,  1.0,    0.0, 1.0,
   -0.5,  0.5,  0.5,    0.0,  0.0,  1.0,    1.0, 1.0,

    // -Z
   -0.5,  0.5, -0.5,    0.0,  0.0, -1.0,    1.0, 0.0,
   -0.5, -0.5, -0.5,    0.0,  0.0, -1.0,    0.0, 0.0,
    0.5, -0.5, -0.5,    0.0,  0.0, -1.0,    0.0, 1.0,
    0.5,  0.5, -0.5,    0.0,  0.0, -1.0,    1.0, 1.0,
];

#[rustfmt::skip]
const INDICES: &[u16] = &[
    0, 1, 2,  0, 2, 3,   
    4, 5, 6,  4, 6, 7,   
    8, 9, 10, 8, 10,11,  
    12,13,14, 12,14,15, 
    16,17,18, 16,18,19, 
    20,21,22, 20,22,23, 
];

//
// =============== INSTANCE DATA: position, scale3, color, alpha ===============
//

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct InstanceData {
    position: [f32; 3],
    // Non-uniform scale for disk-likes => scale3 instead of single float
    scale: [f32; 3],
    color: [f32; 3],
    alpha: f32,
}

impl InstanceData {
    fn desc<'a>() -> wgpu::VertexBufferLayout<'a> {
        use std::mem::size_of;
        wgpu::VertexBufferLayout {
            array_stride: size_of::<InstanceData>() as wgpu::BufferAddress,
            step_mode: wgpu::VertexStepMode::Instance,
            attributes: &[
                // position
                wgpu::VertexAttribute {
                    offset: 0,
                    shader_location: 5,
                    format: wgpu::VertexFormat::Float32x3,
                },
                // scale
                wgpu::VertexAttribute {
                    offset: (4 * 3) as wgpu::BufferAddress,
                    shader_location: 6,
                    format: wgpu::VertexFormat::Float32x3,
                },
                // color
                wgpu::VertexAttribute {
                    offset: (4 * 6) as wgpu::BufferAddress,
                    shader_location: 7,
                    format: wgpu::VertexFormat::Float32x3,
                },
                // alpha
                wgpu::VertexAttribute {
                    offset: (4 * 9) as wgpu::BufferAddress,
                    shader_location: 8,
                    format: wgpu::VertexFormat::Float32,
                },
            ],
        }
    }
}

//
// ====================== CAMERA UNIFORM ======================
//

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct CameraUniform {
    view_proj: [f32; 16],
}

//
// ========== Orbit Info for Disk Particles ==========
//

#[derive(Clone, Copy)]
struct DustOrbit {
    radius: f32,
    angle: f32,
    speed: f32,
    height: f32,
}

//
// ====================== APP STATE ======================
//

struct State {
    // WGPU
    surface: wgpu::Surface,
    device: wgpu::Device,
    queue: wgpu::Queue,
    config: wgpu::SurfaceConfiguration,
    size: PhysicalSize<u32>,

    // Pipeline
    render_pipeline: wgpu::RenderPipeline,
    depth_texture_view: wgpu::TextureView,

    // Geometry
    vertex_buffer: wgpu::Buffer,
    index_buffer: wgpu::Buffer,
    num_indices: u32,

    // Instances
    planet_voxels: Vec<InstanceData>,
    atmosphere_voxels: Vec<InstanceData>,
    cloud_voxels: Vec<InstanceData>,

    dust_orbits: Vec<DustOrbit>,
    dust_instances: Vec<InstanceData>,

    instance_buffer: wgpu::Buffer,
    num_instances_total: u32,

    // Noise
    perlin1: Perlin, 
    perlin2: Perlin,

    // Camera
    camera_uniform: CameraUniform,
    camera_buffer: wgpu::Buffer,
    camera_bind_group: wgpu::BindGroup,

    time: f32,

    // Camera orbit
    camera_yaw: f32,
    camera_pitch: f32,
    camera_dist: f32,

    // Mouse state
    last_mouse_pos: Option<(f64, f64)>,
    mouse_pressed: bool,
}

impl State {
    async fn new(window: &winit::window::Window) -> Self {
        // Instance + surface + adapter
        let size = window.inner_size();
        let instance = wgpu::Instance::default();
        let surface = unsafe { instance.create_surface(window) }.unwrap();
        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                compatible_surface: Some(&surface),
                force_fallback_adapter: false,
            })
            .await
            .expect("No GPU adapter found!");

        // Device + queue
        let (device, queue) = adapter
            .request_device(
                &wgpu::DeviceDescriptor {
                    label: Some("Device"),
                    features: wgpu::Features::empty(),
                    limits: wgpu::Limits::default(),
                },
                None,
            )
            .await
            .unwrap();

        // Surface config
        let surface_caps = surface.get_capabilities(&adapter);
        let format = surface_caps.formats[0];
        let config = wgpu::SurfaceConfiguration {
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            format,
            width: size.width,
            height: size.height,
            present_mode: surface_caps.present_modes[0],
            alpha_mode: surface_caps.alpha_modes[0],
            view_formats: vec![],
        };
        surface.configure(&device, &config);

        // Depth
        let depth_texture_desc = &wgpu::TextureDescriptor {
            label: Some("Depth Texture"),
            size: wgpu::Extent3d {
                width: config.width,
                height: config.height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Depth24Plus,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            view_formats: &[],
        };
        let depth_texture = device.create_texture(depth_texture_desc);
        let depth_texture_view = depth_texture.create_view(&wgpu::TextureViewDescriptor::default());

        // Cube buffers
        let vertex_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Cube Vertex Buffer"),
            contents: bytemuck::cast_slice(VERTICES),
            usage: wgpu::BufferUsages::VERTEX,
        });
        let index_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Cube Index Buffer"),
            contents: bytemuck::cast_slice(INDICES),
            usage: wgpu::BufferUsages::INDEX,
        });
        let num_indices = INDICES.len() as u32;

        // Build planet
        let mut planet_voxels = Vec::new();
        let radius = 12.0;
        let mut rng = rand::thread_rng();
        for x in -16..16 {
            for y in -16..16 {
                for z in -16..16 {
                    let fx = x as f32 + 0.5;
                    let fy = y as f32 + 0.5;
                    let fz = z as f32 + 0.5;
                    let dist = (fx * fx + fy * fy + fz * fz).sqrt();
                    let noise = rng.gen_range(0.0..2.0); 
                    if dist < radius + noise {
                        planet_voxels.push(InstanceData {
                            position: [fx, fy, fz],
                            scale: [1.0, 1.0, 1.0],
                            color: [0.5, 0.5, 0.5],
                            alpha: 1.0,
                        });
                    }
                }
            }
        }

        // Build atmosphere (slightly larger sphere with partial alpha)
        // We'll just do a second shell of voxels ~ 1.05x bigger radius
        let mut atmosphere_voxels = Vec::new();
        let atmosphere_factor = 1.05; 
        for voxel in &planet_voxels {
            let dist = (voxel.position[0]*voxel.position[0] +
                        voxel.position[1]*voxel.position[1] +
                        voxel.position[2]*voxel.position[2]).sqrt();
            if dist > radius - 1.0 {
                // near the "surface"
                let scale = [1.0, 1.0, 1.0];
                // Just offset outward slightly
                let outward = 1.0 + (atmosphere_factor - 1.0);
                let pos = [
                    voxel.position[0] * outward / dist.max(0.0001),
                    voxel.position[1] * outward / dist.max(0.0001),
                    voxel.position[2] * outward / dist.max(0.0001),
                ];
                atmosphere_voxels.push(InstanceData {
                    position: pos,
                    scale,
                    color: [0.3, 0.5, 0.8],
                    alpha: 0.2,
                });
            }
        }

        // Build clouds (a second shell even bigger, partial alpha)
        // We'll do ~1.1 radius and sparser
        let mut cloud_voxels = Vec::new();
        let cloud_factor = 1.1;
        for voxel in &planet_voxels {
            let dist = (voxel.position[0]*voxel.position[0] +
                        voxel.position[1]*voxel.position[1] +
                        voxel.position[2]*voxel.position[2]).sqrt();
            if dist > radius - 1.0 && rng.gen_bool(0.3) {
                let outward = cloud_factor;
                let pos = [
                    voxel.position[0] * outward / dist.max(0.0001),
                    voxel.position[1] * outward / dist.max(0.0001),
                    voxel.position[2] * outward / dist.max(0.0001),
                ];
                cloud_voxels.push(InstanceData {
                    position: pos,
                    scale: [1.0, 1.0, 1.0],
                    color: [1.0, 1.0, 1.0],
                    alpha: 0.15,
                });
            }
        }

        // Disk-like particles
        // We'll store orbits and update them each frame
        let mut dust_orbits = Vec::new();
        let mut dust_instances = Vec::new();
        for _ in 0..300 {
            let r = rng.gen_range(15.0..40.0);   // bigger than planet radius
            let angle = rng.gen_range(0.0..std::f32::consts::TAU);
            let speed = rng.gen_range(0.005..0.02);
            let height = rng.gen_range(-6.0..6.0);

            dust_orbits.push(DustOrbit {
                radius: r,
                angle,
                speed,
                height,
            });
            // The actual instance: flattened scale in Y => disk-like
            dust_instances.push(InstanceData {
                position: [0.0, 0.0, 0.0],
                scale: [0.2, 0.02, 0.2],
                color: [1.0, 1.0, 1.0],
                alpha: 0.2,
            });
        }

        // Combine everything into one big instance buffer
        let total_count = planet_voxels.len() 
                        + atmosphere_voxels.len() 
                        + cloud_voxels.len() 
                        + dust_instances.len();
        let num_instances_total = total_count as u32;

        let combined_data = [
            planet_voxels.as_slice(),
            atmosphere_voxels.as_slice(),
            cloud_voxels.as_slice(),
            dust_instances.as_slice(),
        ]
        .concat();

        let instance_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Instance Buffer"),
            contents: bytemuck::cast_slice(&combined_data),
            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
        });

        // Camera
        let camera_uniform = CameraUniform {
            view_proj: Mat4::IDENTITY.to_cols_array(),
        };
        let camera_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Camera Buffer"),
            contents: bytemuck::cast_slice(&camera_uniform.view_proj),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });
        let camera_bg_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Camera BG Layout"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::VERTEX | wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });
        let camera_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Camera BG"),
            layout: &camera_bg_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: camera_buffer.as_entire_binding(),
            }],
        });

        // Shader
        // We'll simulate a single "sun" far off in one direction: (1.0, 0.7, 0.2)
        // with warm color. Also small ambient. This effectively "bathes" everything in warm light.
        let shader_src = r#"
@group(0) @binding(0)
var<uniform> viewProj: mat4x4<f32>;

struct Instance {
    @location(5) pos: vec3<f32>,
    @location(6) scale: vec3<f32>,
    @location(7) color: vec3<f32>,
    @location(8) alpha: f32,
};

struct VSOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) normal: vec3<f32>,
    @location(1) color: vec3<f32>,
    @location(2) alpha: f32,
};

@vertex
fn vs_main(
    @location(0) inPos: vec3<f32>,
    @location(1) inNormal: vec3<f32>,
    @location(2) _uv: vec2<f32>,
    inst: Instance
) -> VSOutput {
    var out: VSOutput;
    // Apply non-uniform scale
    let scaledPos = vec3<f32>(inPos.x * inst.scale.x,
                              inPos.y * inst.scale.y,
                              inPos.z * inst.scale.z);
    let worldPos = scaledPos + inst.pos;
    out.position = viewProj * vec4<f32>(worldPos, 1.0);

    // The normal, also scaled. We'll just do approximate approach:
    // for small flattening, it should be fine to do normal = inNormal / scale
    let scaleFix = vec3<f32>(1.0/inst.scale.x, 1.0/inst.scale.y, 1.0/inst.scale.z);
    out.normal = normalize(inNormal * scaleFix);
    out.color = inst.color;
    out.alpha = inst.alpha;
    return out;
}

@fragment
fn fs_main(input: VSOutput) -> @location(0) vec4<f32> {
    let n = normalize(input.normal);

    // "Sun" direction & color
    let sunDir = normalize(vec3<f32>(1.0, 0.7, 0.2));
    let sunColor = vec3<f32>(1.0, 0.9, 0.7) * 1.5;
    // Lambert
    let lambert = max(dot(n, sunDir), 0.0);
    // small ambient
    let ambient = 0.2;
    let finalColor = input.color * (ambient + lambert * sunColor);
    return vec4<f32>(finalColor, input.alpha);
}
"#;

        let shader_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Sun Shader"),
            source: wgpu::ShaderSource::Wgsl(std::borrow::Cow::Borrowed(shader_src)),
        });

        // Pipeline
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Render Pipeline Layout"),
            bind_group_layouts: &[&camera_bg_layout],
            push_constant_ranges: &[],
        });
        let render_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("Pipeline"),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader_module,
                entry_point: "vs_main",
                buffers: &[
                    // geometry
                    wgpu::VertexBufferLayout {
                        array_stride: mem::size_of::<f32>() as wgpu::BufferAddress * 8,
                        step_mode: wgpu::VertexStepMode::Vertex,
                        attributes: &[
                            // position
                            wgpu::VertexAttribute {
                                offset: 0,
                                shader_location: 0,
                                format: wgpu::VertexFormat::Float32x3,
                            },
                            // normal
                            wgpu::VertexAttribute {
                                offset: (4 * 3) as wgpu::BufferAddress,
                                shader_location: 1,
                                format: wgpu::VertexFormat::Float32x3,
                            },
                            // uv
                            wgpu::VertexAttribute {
                                offset: (4 * 6) as wgpu::BufferAddress,
                                shader_location: 2,
                                format: wgpu::VertexFormat::Float32x2,
                            },
                        ],
                    },
                    // instance data
                    InstanceData::desc(),
                ],
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader_module,
                entry_point: "fs_main",
                targets: &[Some(wgpu::ColorTargetState {
                    format,
                    blend: Some(wgpu::BlendState {
                        color: wgpu::BlendComponent::OVER,
                        alpha: wgpu::BlendComponent::OVER,
                    }),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                front_face: wgpu::FrontFace::Ccw,
                cull_mode: Some(wgpu::Face::Back),
                polygon_mode: wgpu::PolygonMode::Fill,
                strip_index_format: None,
                unclipped_depth: false,
                conservative: false,
            },
            depth_stencil: Some(wgpu::DepthStencilState {
                format: wgpu::TextureFormat::Depth24Plus,
                depth_write_enabled: true,
                depth_compare: wgpu::CompareFunction::Less,
                stencil: wgpu::StencilState::default(),
                bias: wgpu::DepthBiasState::default(),
            }),
            multisample: wgpu::MultisampleState {
                count: 1,
                mask: !0,
                alpha_to_coverage_enabled: false,
            },
            multiview: None,
        });

        // Construct State
        Self {
            surface,
            device,
            queue,
            config,
            size,
            render_pipeline,
            depth_texture_view,

            vertex_buffer,
            index_buffer,
            num_indices,

            planet_voxels,
            atmosphere_voxels,
            cloud_voxels,

            dust_orbits,
            dust_instances,

            instance_buffer,
            num_instances_total: num_instances_total,

            // Two Perlin generators for "superposition" in color (optional usage)
            perlin1: Perlin::new(1),
            perlin2: Perlin::new(2),

            camera_uniform,
            camera_buffer,
            camera_bind_group,

            time: 0.0,

            camera_yaw: 0.0,
            camera_pitch: 0.3,
            camera_dist: 60.0,

            last_mouse_pos: None,
            mouse_pressed: false,
        }
    }

    // Resize
    fn resize(&mut self, new_size: PhysicalSize<u32>) {
        if new_size.width > 0 && new_size.height > 0 {
            self.size = new_size;
            self.config.width = new_size.width;
            self.config.height = new_size.height;
            self.surface.configure(&self.device, &self.config);

            // Recreate depth
            let depth_desc = wgpu::TextureDescriptor {
                label: Some("Depth Tex"),
                size: wgpu::Extent3d {
                    width: new_size.width,
                    height: new_size.height,
                    depth_or_array_layers: 1,
                },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: wgpu::TextureFormat::Depth24Plus,
                usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
                view_formats: &[],
            };
            let depth_texture = self.device.create_texture(&depth_desc);
            self.depth_texture_view = depth_texture.create_view(&wgpu::TextureViewDescriptor::default());
        }
    }

    // update
    fn update(&mut self) {
        self.time += 0.01;

        // Update planet color with superposition of perlin1 & perlin2
        // Also update atmosphere & clouds for "dynamic" effect
        let freq = 0.06;
        let freq_cloud = 0.03;
        let freq_atmo = 0.04;

        // Planet
        for v in &mut self.planet_voxels {
            let x = v.position[0] as f64 * freq;
            let y = v.position[1] as f64 * freq;
            let z = v.position[2] as f64 * freq;
            let t = self.time as f64 * 0.2;

            let val1 = self.perlin1.get([x, y, z + t]);
            let val2 = 0.5 * self.perlin2.get([z, x, y + t * 1.5]);
            let sum = val1 + val2;  // superposition
            let nval = 0.5*(sum + 1.0); // map [-1..1] to [0..1]
            let (r, g, b) = perlin_color_map(nval as f32);
            v.color = [r, g, b];
        }

        // Atmosphere
        for a in &mut self.atmosphere_voxels {
            let x = a.position[0] as f64 * freq_atmo;
            let y = a.position[1] as f64 * freq_atmo;
            let z = a.position[2] as f64 * freq_atmo;
            let t = self.time as f64 * 0.3;
            let val = self.perlin1.get([x, y, z + t]);
            let nval = 0.5*(val + 1.0);
            // from light blue to pinkish
            let (r, g, b) = atmosphere_color_map(nval as f32);
            a.color = [r, g, b];
        }

        // Clouds
        for c in &mut self.cloud_voxels {
            let x = c.position[0] as f64 * freq_cloud;
            let y = c.position[1] as f64 * freq_cloud;
            let z = c.position[2] as f64 * freq_cloud;
            let t = self.time as f64 * 0.4;
            let val = self.perlin2.get([x + t, y, z]);
            let nval = 0.5*(val + 1.0);
            // near white, but maybe tinted
            let (r, g, b) = cloud_color_map(nval as f32);
            c.color = [r, g, b];
        }

        // Update dust orbits
        for (i, orbit) in self.dust_orbits.iter_mut().enumerate() {
            orbit.angle += orbit.speed;
            let x = orbit.radius * orbit.angle.cos();
            let z = orbit.radius * orbit.angle.sin();
            let y = orbit.height;
            self.dust_instances[i].position = [x, y, z];
        }

        // Re-combine into single buffer
        let combined = [
            self.planet_voxels.as_slice(),
            self.atmosphere_voxels.as_slice(),
            self.cloud_voxels.as_slice(),
            self.dust_instances.as_slice(),
        ]
        .concat();
        self.queue.write_buffer(&self.instance_buffer, 0, bytemuck::cast_slice(&combined));

        // Recompute camera
        let aspect = self.config.width as f32 / self.config.height as f32;
        let fovy = 45f32.to_radians();
        let near = 0.1;
        let far = 2000.0;

        let eye_x = self.camera_dist * self.camera_yaw.cos() * self.camera_pitch.cos();
        let eye_y = self.camera_dist * self.camera_pitch.sin();
        let eye_z = self.camera_dist * self.camera_yaw.sin() * self.camera_pitch.cos();
        let eye = Vec3::new(eye_x, eye_y, eye_z);
        let center = Vec3::ZERO;
        let up = Vec3::Y;

        let view = Mat4::look_at_rh(eye, center, up);
        let proj = Mat4::perspective_rh(fovy, aspect, near, far);
        let vp = proj * view;

        self.camera_uniform.view_proj = vp.to_cols_array();
        self.queue
            .write_buffer(&self.camera_buffer, 0, bytemuck::cast_slice(&self.camera_uniform.view_proj));
    }

    // render
    fn render(&mut self) -> Result<(), wgpu::SurfaceError> {
        let frame = self.surface.get_current_texture()?;
        let view = frame.texture.create_view(&wgpu::TextureViewDescriptor::default());

        let mut encoder =
            self.device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("Render Encoder"),
                });

        {
            let mut rp = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("Main Render Pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color {
                            r: 0.0,
                            g: 0.0,
                            b: 0.01,
                            a: 1.0,
                        }),
                        store: true,
                    },
                })],
                depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                    view: &self.depth_texture_view,
                    depth_ops: Some(wgpu::Operations {
                        load: wgpu::LoadOp::Clear(1.0),
                        store: true,
                    }),
                    stencil_ops: None,
                }),
            });
            rp.set_pipeline(&self.render_pipeline);
            rp.set_bind_group(0, &self.camera_bind_group, &[]);
            rp.set_vertex_buffer(0, self.vertex_buffer.slice(..));
            rp.set_vertex_buffer(1, self.instance_buffer.slice(..));
            rp.set_index_buffer(self.index_buffer.slice(..), wgpu::IndexFormat::Uint16);
            rp.draw_indexed(0..self.num_indices, 0, 0..self.num_instances_total);
        }

        self.queue.submit(iter::once(encoder.finish()));
        frame.present();
        Ok(())
    }

    // Handle user input (mouse drag => orbit, scroll => zoom)
    fn input(&mut self, event: &WindowEvent) -> bool {
        match event {
            WindowEvent::MouseInput { state, button, .. } => {
                if *button == MouseButton::Left {
                    self.mouse_pressed = *state == ElementState::Pressed;
                }
                true
            }
            WindowEvent::CursorMoved { position, .. } => {
                if self.mouse_pressed {
                    let (x, y) = (position.x, position.y);
                    if let Some((lx, ly)) = self.last_mouse_pos {
                        let dx = (x - lx) as f32 * 0.005;
                        let dy = (y - ly) as f32 * 0.005;
                        self.camera_yaw += dx;
                        self.camera_pitch -= dy;
                        self.camera_pitch = self.camera_pitch.clamp(-1.4, 1.4);
                    }
                    self.last_mouse_pos = Some((x, y));
                } else {
                    self.last_mouse_pos = Some((position.x, position.y));
                }
                true
            }
            WindowEvent::MouseWheel { delta, .. } => {
                let scroll = match delta {
                    MouseScrollDelta::LineDelta(_, s) => *s,
                    MouseScrollDelta::PixelDelta(px) => px.y as f32 / 60.0,
                };
                self.camera_dist -= scroll * 2.0;
                if self.camera_dist < 5.0 { 
                    self.camera_dist = 5.0;
                }
                if self.camera_dist > 500.0 {
                    self.camera_dist = 500.0;
                }
                true
            }
            _ => false,
        }
    }
}

//
// ================= MAIN + EVENT LOOP =================
//

fn main() {
    env_logger::init();
    pollster::block_on(run());
}

async fn run() {
    let event_loop = EventLoop::new();
    let window = WindowBuilder::new()
        .with_title("Gigantic Sun + Planet + Disk Formation (wgpu)")
        .build(&event_loop)
        .unwrap();

    let mut state = State::new(&window).await;

    event_loop.run(move |event, _, control_flow| {
        match &event {
            Event::WindowEvent { event, window_id } if *window_id == window.id() => {
                match event {
                    WindowEvent::CloseRequested
                    | WindowEvent::KeyboardInput {
                        input:
                            KeyboardInput {
                                state: ElementState::Pressed,
                                virtual_keycode: Some(VirtualKeyCode::Escape),
                                ..
                            },
                        ..
                    } => *control_flow = ControlFlow::Exit,

                    WindowEvent::Resized(physical_size) => {
                        state.resize(*physical_size);
                    }
                    WindowEvent::ScaleFactorChanged { new_inner_size, .. } => {
                        state.resize(**new_inner_size);
                    }
                    _ => {
                        // Mouse / Scroll
                        state.input(event);
                    }
                }
            }
            Event::RedrawRequested(_) => {
                state.update();
                match state.render() {
                    Ok(_) => {}
                    Err(wgpu::SurfaceError::Lost) => {
                        state.resize(state.size);
                    }
                    Err(wgpu::SurfaceError::OutOfMemory) => {
                        *control_flow = ControlFlow::Exit;
                    }
                    Err(e) => eprintln!("Render error: {:?}", e),
                }
            }
            Event::MainEventsCleared => {
                window.request_redraw();
            }
            _ => {}
        }
    });
}

//
// ================= COLOR MAPS FOR PLANET, ATMOSPHERE, CLOUDS =================
//

/// Basic planet color mapping: maps [0..1] to some interesting color gradient
fn perlin_color_map(t: f32) -> (f32, f32, f32) {
    // We'll do something from greenish -> bluish -> pinkish
    // This is just an example.
    if t < 0.33 {
        let u = t / 0.33; 
        let r = 0.1 + 0.2*u;
        let g = 0.7 - 0.3*u;
        let b = 0.3 + 0.4*u;
        (r, g, b)
    } else if t < 0.66 {
        let u = (t - 0.33)/0.33; 
        let r = 0.3 + 0.5*u;
        let g = 0.4 - 0.2*u;
        let b = 0.7 + 0.2*u;
        (r, g, b)
    } else {
        let u = (t - 0.66)/0.34;
        let r = 0.8 + 0.2*u; 
        let g = 0.2 + 0.3*u;
        let b = 0.9 - 0.2*u;
        (r, g, b)
    }
}

/// Atmosphere color: usually bluish -> pinkish
fn atmosphere_color_map(t: f32) -> (f32, f32, f32) {
    // We'll do a mild gradient
    // 0 => (0.3, 0.5, 0.8), 1 => (1.0, 0.6, 0.7)
    let r = 0.3 + 0.7*t;
    let g = 0.5 + 0.1*t;
    let b = 0.8 - 0.1*t;
    (r, g, b)
}

/// Cloud color: near white, sometimes a bit grey/blue
fn cloud_color_map(t: f32) -> (f32, f32, f32) {
    // We'll clamp near white
    let grey = 0.8 + 0.2*t; 
    let tint = 0.9 + 0.1*t; 
    // just shift it slightly 
    (tint, grey, tint)
}
