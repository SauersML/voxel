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
// We'll reuse the same cube geometry for all voxel-based objects:
// the planet, its atmosphere, clouds, the orbiting disk particles,
// AND the big sun in the distance (represented as a large voxel-sphere).
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
   -0.5,  0.5,  0.5,    1.0,  0.0,  1.0,    1.0, 1.0,

    // -Z
   -0.5,  0.5, -0.5,    0.0,  0.0, -1.0,    1.0, 0.0,
   -0.5, -0.5, -0.5,    0.0,  0.0, -1.0,    0.0, 0.0,
    0.5, -0.5, -0.5,    0.0,  0.0, -1.0,    0.0, 1.0,
    0.5,  0.5, -0.5,    0.0,  0.0, -1.0,    1.0, 1.0,
];

#[rustfmt::skip]
const INDICES: &[u16] = &[
    0, 1, 2,   0, 2, 3,   
    4, 5, 6,   4, 6, 7,   
    8, 9, 10,  8, 10, 11,  
    12,13,14,  12,14,15, 
    16,17,18,  16,18,19, 
    20,21,22,  20,22,23, 
];

//
// ================== INSTANCE DATA: position, scale, color, alpha ==================
//
// We'll store each voxel's position, 3D scale (for disk flattening or large spheres),
// color, and alpha (for partial transparency).
//

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct InstanceData {
    position: [f32; 3],
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
// =========== CAMERA UNIFORM ===========
//

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct CameraUniform {
    view_proj: [f32; 16],
}

//
// =========== ORBIT INFO for Disk Particles ===========
//

#[derive(Clone, Copy)]
struct DustOrbit {
    radius: f32,
    angle: f32,
    speed: f32,
    height: f32,
}

//
// =========== PROGRAM STATE ===========
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

    // Planet
    planet_voxels: Vec<InstanceData>,
    atmosphere_voxels: Vec<InstanceData>,
    cloud_voxels: Vec<InstanceData>,

    // Disk
    dust_orbits: Vec<DustOrbit>,
    dust_instances: Vec<InstanceData>,

    // Sun
    sun_voxels: Vec<InstanceData>,

    // Combined buffer
    instance_buffer: wgpu::Buffer,
    num_instances_total: u32,

    // Noise
    perlin1: Perlin,
    perlin2: Perlin,

    // Camera
    camera_uniform: CameraUniform,
    camera_buffer: wgpu::Buffer,
    camera_bind_group: wgpu::BindGroup,

    // Animation
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
        // Create instance + surface + adapter
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
            .expect("Failed to find a GPU adapter.");

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

        // Depth texture
        let depth_desc = wgpu::TextureDescriptor {
            label: Some("Depth Texture"),
            size: wgpu::Extent3d {
                width: size.width,
                height: size.height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Depth24Plus,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            view_formats: &[],
        };
        let depth_texture = device.create_texture(&depth_desc);
        let depth_texture_view = depth_texture.create_view(&wgpu::TextureViewDescriptor::default());

        // Create cube vertex/index buffers
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

        //
        // Build Planet
        //
        let radius = 12.0;
        let mut planet_voxels = Vec::new();
        let mut rng = rand::thread_rng();
        for x in -16..16 {
            for y in -16..16 {
                for z in -16..16 {
                    let fx = x as f32 + 0.5;
                    let fy = y as f32 + 0.5;
                    let fz = z as f32 + 0.5;
                    let dist = (fx*fx + fy*fy + fz*fz).sqrt();
                    let noise_bump = rng.gen_range(0.0..2.0);
                    if dist < radius + noise_bump {
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

        //
        // White Polar Caps (post-process)
        //
        // If the voxel's |y| is near the top/bottom, color = white
        // We'll define a threshold fraction for polar region
        let polar_threshold = 0.75 * radius;
        for v in &mut planet_voxels {
            let fy = v.position[1];
            if fy.abs() > polar_threshold {
                // Mark as white
                v.color = [1.0, 1.0, 1.0];
            }
        }

        //
        // Atmosphere (shell ~1.05 bigger radius)
        //
        let mut atmosphere_voxels = Vec::new();
        let atmosphere_factor = 1.05;
        for v in &planet_voxels {
            let dist = (v.position[0]*v.position[0] +
                        v.position[1]*v.position[1] +
                        v.position[2]*v.position[2]).sqrt();
            if dist > radius - 1.0 {
                let outward = atmosphere_factor;
                let scale = [1.0, 1.0, 1.0];
                let pos = [
                    v.position[0] * outward / dist.max(0.0001),
                    v.position[1] * outward / dist.max(0.0001),
                    v.position[2] * outward / dist.max(0.0001),
                ];
                atmosphere_voxels.push(InstanceData {
                    position: pos,
                    scale,
                    color: [0.3, 0.5, 0.8],
                    alpha: 0.2,
                });
            }
        }

        //
        // Clouds (~1.1 bigger, partial alpha)
        //
        let mut cloud_voxels = Vec::new();
        let cloud_factor = 1.1;
        for v in &planet_voxels {
            let dist = (v.position[0]*v.position[0] +
                        v.position[1]*v.position[1] +
                        v.position[2]*v.position[2]).sqrt();
            if dist > radius - 1.0 && rng.gen_bool(0.3) {
                let outward = cloud_factor;
                let scale = [1.0, 1.0, 1.0];
                let pos = [
                    v.position[0] * outward / dist.max(0.0001),
                    v.position[1] * outward / dist.max(0.0001),
                    v.position[2] * outward / dist.max(0.0001),
                ];
                cloud_voxels.push(InstanceData {
                    position: pos,
                    scale,
                    color: [1.0, 1.0, 1.0],
                    alpha: 0.15,
                });
            }
        }

        //
        // Disk-like particles
        //
        let mut dust_orbits = Vec::new();
        let mut dust_instances = Vec::new();
        for _ in 0..300 {
            let orbit_radius = rng.gen_range(15.0..40.0);
            let angle = rng.gen_range(0.0..std::f32::consts::TAU);
            let speed = rng.gen_range(0.005..0.02);
            let height = rng.gen_range(-6.0..6.0);

            dust_orbits.push(DustOrbit {
                radius: orbit_radius,
                angle,
                speed,
                height,
            });
            // Flatten in Y => disk shape
            dust_instances.push(InstanceData {
                position: [0.0, 0.0, 0.0],
                scale: [0.2, 0.02, 0.2],
                color: [1.0, 1.0, 1.0],
                alpha: 0.2,
            });
        }

        //
        // Sun in the distance
        // We'll create a big sphere of bright blocks at some offset, e.g. (120, 10, -100).
        // Let radius=20 for the sun, to see it if you rotate the camera.
        //
        let mut sun_voxels = Vec::new();
        let sun_radius = 20.0;
        let sun_center = [120.0, 10.0, -100.0];
        for x in -(sun_radius as i32)..(sun_radius as i32) {
            for y in -(sun_radius as i32)..(sun_radius as i32) {
                for z in -(sun_radius as i32)..(sun_radius as i32) {
                    let fx = x as f32 + 0.5;
                    let fy = y as f32 + 0.5;
                    let fz = z as f32 + 0.5;
                    let dist = (fx*fx + fy*fy + fz*fz).sqrt();
                    if dist < sun_radius {
                        let pos = [
                            sun_center[0] + fx,
                            sun_center[1] + fy,
                            sun_center[2] + fz,
                        ];
                        // super bright color
                        sun_voxels.push(InstanceData {
                            position: pos,
                            scale: [1.0, 1.0, 1.0],
                            color: [4.0, 3.5, 0.2], // bright
                            alpha: 1.0,
                        });
                    }
                }
            }
        }

        //
        // Combine all instances
        //
        let total_count = planet_voxels.len() 
            + atmosphere_voxels.len()
            + cloud_voxels.len()
            + dust_instances.len()
            + sun_voxels.len();
        let num_instances_total = total_count as u32;

        let combined_data = [
            planet_voxels.as_slice(),
            atmosphere_voxels.as_slice(),
            cloud_voxels.as_slice(),
            dust_instances.as_slice(),
            sun_voxels.as_slice(),
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
        let camera_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Camera BGL"),
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
            layout: &camera_bgl,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: camera_buffer.as_entire_binding(),
            }],
        });

        //
        // WGSL SHADER
        // We'll simulate a single "sun" direction (somewhere up-right), plus small ambient.
        // We won't physically cast shadow from the sun sphere. It's just visual geometry + light direction.
        //
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
    // Non-uniform scale
    let scaledPos = vec3<f32>(
        inPos.x * inst.scale.x,
        inPos.y * inst.scale.y,
        inPos.z * inst.scale.z
    );
    let worldPos = scaledPos + inst.pos;
    out.position = viewProj * vec4<f32>(worldPos, 1.0);

    // approximate normal
    let scaleFix = vec3<f32>(
        1.0/inst.scale.x,
        1.0/inst.scale.y,
        1.0/inst.scale.z
    );
    out.normal = normalize(inNormal * scaleFix);

    out.color = inst.color;
    out.alpha = inst.alpha;
    return out;
}

@fragment
fn fs_main(in: VSOutput) -> @location(0) vec4<f32> {
    // single directional sun
    let sunDir = normalize(vec3<f32>(1.0, 0.7, 0.2));  // direction
    let sunColor = vec3<f32>(1.0, 0.9, 0.7) * 1.5;     // warm color
    let n = normalize(in.normal);

    let lambert = max(dot(n, sunDir), 0.0);
    let ambient = 0.2;
    let finalColor = in.color * (ambient + lambert * sunColor);
    return vec4<f32>(finalColor, in.alpha);
}
"#;

        let shader_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Sun Shader"),
            source: wgpu::ShaderSource::Wgsl(std::borrow::Cow::Borrowed(shader_src)),
        });

        // Pipeline
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Render Pipeline Layout"),
            bind_group_layouts: &[&camera_bgl],
            push_constant_ranges: &[],
        });
        let render_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("Render Pipeline"),
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
                            wgpu::VertexAttribute {
                                offset: 0,
                                shader_location: 0,
                                format: wgpu::VertexFormat::Float32x3,
                            },
                            wgpu::VertexAttribute {
                                offset: (4*3) as wgpu::BufferAddress,
                                shader_location: 1,
                                format: wgpu::VertexFormat::Float32x3,
                            },
                            wgpu::VertexAttribute {
                                offset: (4*6) as wgpu::BufferAddress,
                                shader_location: 2,
                                format: wgpu::VertexFormat::Float32x2,
                            },
                        ],
                    },
                    // instance
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

        // Build final State
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
            sun_voxels,

            instance_buffer,
            num_instances_total,

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

    /// Resize
    fn resize(&mut self, new_size: PhysicalSize<u32>) {
        if new_size.width > 0 && new_size.height > 0 {
            self.size = new_size;
            self.config.width = new_size.width;
            self.config.height = new_size.height;
            self.surface.configure(&self.device, &self.config);

            // Recreate depth
            let depth_desc = wgpu::TextureDescriptor {
                label: Some("Depth Texture"),
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
            self.depth_texture_view =
                depth_texture.create_view(&wgpu::TextureViewDescriptor::default());
        }
    }

    /// Update each frame: animate colors, orbits, camera
    fn update(&mut self) {
        self.time += 0.01;

        // We'll do some Perlin-based color changes
        let freq_planet = 0.06;
        let freq_atmo = 0.04;
        let freq_cloud = 0.03;

        // Planet color shift
        for v in &mut self.planet_voxels {
            let x = v.position[0] as f64 * freq_planet;
            let y = v.position[1] as f64 * freq_planet;
            let z = v.position[2] as f64 * freq_planet;
            let t = self.time as f64 * 0.2;
            // Combine two Perlin noises
            let val1 = self.perlin1.get([x, y, z + t]);
            let val2 = 0.5 * self.perlin2.get([z, x, y + 1.5 * t]);
            let sum = val1 + val2;
            let mapped = 0.5*(sum + 1.0); // [-1..1]->[0..1]

            // If near poles, we keep it white
            let fy = v.position[1];
            let dist = (v.position[0]*v.position[0]
                      + v.position[2]*v.position[2]).sqrt();
            let radius = 12.0;
            if fy.abs() > 0.75*radius {
                // keep white
                v.color = [1.0, 1.0, 1.0];
            } else {
                let (r, g, b) = planet_color_map(mapped as f32);
                v.color = [r, g, b];
            }
        }

        // Atmosphere
        for a in &mut self.atmosphere_voxels {
            let x = a.position[0] as f64 * freq_atmo;
            let y = a.position[1] as f64 * freq_atmo;
            let z = a.position[2] as f64 * freq_atmo;
            let t = self.time as f64 * 0.3;
            let val = self.perlin1.get([x, y, z + t]);
            let mapped = 0.5*(val + 1.0);
            let (r, g, b) = atmosphere_color_map(mapped as f32);
            a.color = [r, g, b];
        }

        // Clouds
        for c in &mut self.cloud_voxels {
            let x = c.position[0] as f64 * freq_cloud;
            let y = c.position[1] as f64 * freq_cloud;
            let z = c.position[2] as f64 * freq_cloud;
            let t = self.time as f64 * 0.4;
            let val = self.perlin2.get([x + t, y, z]);
            let mapped = 0.5*(val + 1.0);
            let (r, g, b) = cloud_color_map(mapped as f32);
            c.color = [r, g, b];
        }

        // Disk orbits
        for (i, orbit) in self.dust_orbits.iter_mut().enumerate() {
            orbit.angle += orbit.speed;
            let x = orbit.radius * orbit.angle.cos();
            let z = orbit.radius * orbit.angle.sin();
            let y = orbit.height;
            self.dust_instances[i].position = [x, y, z];
        }

        // Sun is static in position & color for now.

        // Re-combine into single buffer
        let combined_data = [
            self.planet_voxels.as_slice(),
            self.atmosphere_voxels.as_slice(),
            self.cloud_voxels.as_slice(),
            self.dust_instances.as_slice(),
            self.sun_voxels.as_slice(),
        ]
        .concat();
        self.queue
            .write_buffer(&self.instance_buffer, 0, bytemuck::cast_slice(&combined_data));

        // Recompute camera matrix
        let aspect = self.config.width as f32 / self.config.height as f32;
        let fovy = 45.0f32.to_radians();
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
        self.queue.write_buffer(
            &self.camera_buffer,
            0,
            bytemuck::cast_slice(&self.camera_uniform.view_proj),
        );
    }

    /// Render
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

    /// Handle input (mouse drag => orbit, mouse wheel => zoom in/out).
    /// If you want "pinch to zoom" on a trackpad, that may appear as
    /// different `WindowEvent`s (depends on OS).
    fn input(&mut self, event: &WindowEvent) -> bool {
        match event {
            // Mouse press
            WindowEvent::MouseInput { state, button, .. } => {
                if *button == MouseButton::Left {
                    self.mouse_pressed = *state == ElementState::Pressed;
                }
                true
            }
            // Mouse move
            WindowEvent::CursorMoved { position, .. } => {
                if self.mouse_pressed {
                    let (x, y) = (position.x, position.y);
                    if let Some((lx, ly)) = self.last_mouse_pos {
                        // rotate
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
            // Scroll => zoom, reversed so up => zoom in
            WindowEvent::MouseWheel { delta, .. } => {
                let scroll_amt = match delta {
                    MouseScrollDelta::LineDelta(_, s) => *s,
                    MouseScrollDelta::PixelDelta(px) => px.y as f32 / 60.0,
                };
                // Reverse the sign so up => negative => smaller dist => zoom in
                let factor = -2.0;
                self.camera_dist += scroll_amt * factor;
                if self.camera_dist < 5.0 {
                    self.camera_dist = 5.0;
                }
                if self.camera_dist > 1000.0 {
                    self.camera_dist = 1000.0;
                }
                true
            }
            _ => false,
        }
    }
}

//
// =========== MAIN + EVENT LOOP ===========
//

fn main() {
    env_logger::init();
    pollster::block_on(run());
}

async fn run() {
    let event_loop = EventLoop::new();
    let window = WindowBuilder::new()
        .with_title("Gigantic Sun, Planet + Disk + Clouds + Polar Caps (wgpu)")
        .build(&event_loop)
        .unwrap();

    let mut state = State::new(&window).await;

    event_loop.run(move |event, _, control_flow| match &event {
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
                    // Mouse or scroll input
                    state.input(event);
                }
            }
        }
        Event::RedrawRequested(_) => {
            state.update();
            match state.render() {
                Ok(_) => {}
                Err(wgpu::SurfaceError::Lost) => state.resize(state.size),
                Err(wgpu::SurfaceError::OutOfMemory) => *control_flow = ControlFlow::Exit,
                Err(e) => eprintln!("Render error: {:?}", e),
            }
        }
        Event::MainEventsCleared => {
            // keep redrawing
            window.request_redraw();
        }
        _ => {}
    });
}

//
// =========== COLOR MAPS ===========
//

/// Basic planet color mapping: we do a small gradient from green->blue->pink
/// but ignoring poles if they are labeled white.
fn planet_color_map(t: f32) -> (f32, f32, f32) {
    // quick gradient
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

/// Atmosphere color: from light blue to pinkish
fn atmosphere_color_map(t: f32) -> (f32, f32, f32) {
    let r = 0.3 + 0.7*t;
    let g = 0.5 + 0.1*t;
    let b = 0.8 - 0.1*t;
    (r, g, b)
}

/// Cloud color: near white, slightly tinted
fn cloud_color_map(t: f32) -> (f32, f32, f32) {
    let grey = 0.8 + 0.2*t;
    let tint = 0.9 + 0.1*t;
    (tint, grey, tint)
}
