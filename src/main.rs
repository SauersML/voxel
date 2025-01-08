use std::iter;
use std::mem;

use wgpu::util::DeviceExt;
use winit::{
    dpi::PhysicalSize,
    event::*,
    event_loop::{ControlFlow, EventLoop},
    window::WindowBuilder,
};

use glam::{Mat4, Vec3, Vec4};
use rand::Rng;
use noise::{NoiseFn, Perlin};

//
// ========================== GEOMETRY DATA ==========================
//
// We'll reuse a single cube geometry for all voxel-like objects.
// The sun, planet, atmosphere, clouds, dust, etc. are drawn as
// scaled cubes in 3D, via GPU instancing.
//

#[rustfmt::skip]
const VERTICES: &[f32] = &[
    // position (x,y,z), normal (nx,ny,nz), uv (unused)
    // +X
    0.5,  0.5, -0.5,   1.0,  0.0,  0.0,  0.0, 0.0,
    0.5, -0.5, -0.5,   1.0,  0.0,  0.0,  0.0, 0.0,
    0.5, -0.5,  0.5,   1.0,  0.0,  0.0,  0.0, 0.0,
    0.5,  0.5,   0.5,   1.0,  0.0,  0.0,  0.0, 0.0,

    // -X
   -0.5,  0.5,  0.5,  -1.0,  0.0,  0.0,  0.0, 0.0,
   -0.5, -0.5,  0.5,  -1.0,  0.0,  0.0,  0.0, 0.0,
   -0.5, -0.5, -0.5,  -1.0,  0.0,  0.0,  0.0, 0.0,
   -0.5,  0.5, -0.5,  -1.0,  0.0,  0.0,  0.0, 0.0,

    // +Y
   -0.5,  0.5,  0.5,   0.0,  1.0,  0.0,  0.0, 0.0,
   -0.5,  0.5, -0.5,   0.0,  1.0,  0.0,  0.0, 0.0,
    0.5,  0.5, -0.5,   0.0,  1.0,  0.0,  0.0, 0.0,
    0.5,  0.5,  0.5,   0.0,  1.0,  0.0,  0.0, 0.0,

    // -Y
    0.5, -0.5,  0.5,   0.0, -1.0,  0.0,  0.0, 0.0,
    0.5, -0.5, -0.5,   0.0, -1.0,  0.0,  0.0, 0.0,
   -0.5, -0.5, -0.5,   0.0, -1.0,  0.0,  0.0, 0.0,
   -0.5, -0.5,  0.5,   0.0, -1.0,  0.0,  0.0, 0.0,

    // +Z
    0.5,  0.5,  0.5,   0.0,  0.0,  1.0,  0.0, 0.0,
    0.5, -0.5,  0.5,   0.0,  0.0,  1.0,  0.0, 0.0,
   -0.5, -0.5,  0.5,   0.0,  0.0,  1.0,  0.0, 0.0,
   -0.5,  0.5,  0.5,   0.0,  0.0,  1.0,  0.0, 0.0,

    // -Z
   -0.5,  0.5, -0.5,   0.0,  0.0, -1.0,  0.0, 0.0,
   -0.5, -0.5, -0.5,   0.0,  0.0, -1.0,  0.0, 0.0,
    0.5, -0.5, -0.5,   0.0,  0.0, -1.0,  0.0, 0.0,
    0.5,  0.5, -0.5,   0.0,  0.0, -1.0,  0.0, 0.0,
];

#[rustfmt::skip]
const INDICES: &[u16] = &[
    0,1,2,  0,2,3,
    4,5,6,  4,6,7,
    8,9,10, 8,10,11,
    12,13,14, 12,14,15,
    16,17,18, 16,18,19,
    20,21,22, 20,22,23,
];

//
// =================== INSTANCE DATA (position, scale, color, alpha) ===================
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
// =================== CAMERA UNIFORM ===================
//

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct CameraUniform {
    view_proj: [f32; 16],
}

//
// =================== Additional Uniform: The Sun's Position ===================
//
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct SunUniform {
    // We'll store sunPos in world space (x,y,z,1)
    sun_pos: [f32; 4],
}

//
// ========== DUST ORBITS ==========
//
#[derive(Clone, Copy)]
struct DustOrbit {
    radius: f32,
    angle: f32,
    speed: f32,
    height: f32,
}

//
// =================== PROGRAM STATE ===================
//
struct State {
    surface: wgpu::Surface,
    device: wgpu::Device,
    queue: wgpu::Queue,
    config: wgpu::SurfaceConfiguration,
    size: PhysicalSize<u32>,

    render_pipeline: wgpu::RenderPipeline,
    depth_texture_view: wgpu::TextureView,

    // geometry
    vertex_buffer: wgpu::Buffer,
    index_buffer: wgpu::Buffer,
    num_indices: u32,

    // planet
    planet_voxels: Vec<InstanceData>,
    atmosphere_voxels: Vec<InstanceData>,
    cloud_voxels: Vec<InstanceData>,

    // dust
    dust_orbits: Vec<DustOrbit>,
    dust_instances: Vec<InstanceData>,

    // sun
    sun_voxels: Vec<InstanceData>,

    // combined
    instance_buffer: wgpu::Buffer,
    num_instances_total: u32,

    // camera
    camera_uniform: CameraUniform,
    camera_buffer: wgpu::Buffer,

    // sun pos uniform
    sun_uniform: SunUniform,
    sun_buffer: wgpu::Buffer,

    // combined bind group
    bind_group: wgpu::BindGroup,

    // noise
    perlin1: Perlin,
    perlin2: Perlin,

    time: f32,
    camera_yaw: f32,
    camera_pitch: f32,
    camera_dist: f32,

    last_mouse_pos: Option<(f64, f64)>,
    mouse_pressed: bool,
}

impl State {
    async fn new(window: &winit::window::Window) -> Self {
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

        // surface config
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

        // depth
        let depth_tex_desc = wgpu::TextureDescriptor {
            label: Some("DepthTex"),
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
        let depth_texture = device.create_texture(&depth_tex_desc);
        let depth_texture_view = depth_texture.create_view(&wgpu::TextureViewDescriptor::default());

        // geometry
        let vertex_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("CubeVB"),
            contents: bytemuck::cast_slice(VERTICES),
            usage: wgpu::BufferUsages::VERTEX,
        });
        let index_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("CubeIB"),
            contents: bytemuck::cast_slice(INDICES),
            usage: wgpu::BufferUsages::INDEX,
        });
        let num_indices = INDICES.len() as u32;

        //
        // Build planet, atmosphere, clouds
        //
        let planet_radius = 12.0;
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
                    if dist < planet_radius + noise_bump {
                        // default color placeholder
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

        // smaller ice caps => if |fy| > 0.9 * planet_radius
        let polar_threshold = 0.9 * planet_radius;
        for v in &mut planet_voxels {
            let fy = v.position[1];
            if fy.abs() > polar_threshold {
                v.color = [1.0, 1.0, 1.0];
            }
        }

        // atmosphere (shell ~1.05 bigger)
        let mut atmosphere_voxels = Vec::new();
        for v in &planet_voxels {
            let dist = (v.position[0]*v.position[0]
                       + v.position[1]*v.position[1]
                       + v.position[2]*v.position[2]).sqrt();
            if dist > planet_radius - 1.0 {
                let outward = 1.05;
                let scale = [1.0, 1.0, 1.0];
                let pos = [
                    v.position[0]*outward/dist.max(0.0001),
                    v.position[1]*outward/dist.max(0.0001),
                    v.position[2]*outward/dist.max(0.0001),
                ];
                atmosphere_voxels.push(InstanceData {
                    position: pos,
                    scale,
                    color: [0.3, 0.5, 0.8],
                    alpha: 0.2, // bigger alpha => we can see
                });
            }
        }

        // clouds (~1.1 bigger, partial alpha, random subset)
        let mut cloud_voxels = Vec::new();
        for v in &planet_voxels {
            let dist = (v.position[0]*v.position[0]
                       + v.position[1]*v.position[1]
                       + v.position[2]*v.position[2]).sqrt();
            if dist > planet_radius - 1.0 && rng.gen_bool(0.3) {
                let outward = 1.1;
                let scale = [1.0, 1.0, 1.0];
                let pos = [
                    v.position[0]*outward/dist.max(0.0001),
                    v.position[1]*outward/dist.max(0.0001),
                    v.position[2]*outward/dist.max(0.0001),
                ];
                cloud_voxels.push(InstanceData {
                    position: pos,
                    scale,
                    color: [1.0, 1.0, 1.0],
                    alpha: 0.4, // more alpha => more visible
                });
            }
        }

        //
        // Disk-like dust
        //
        let mut dust_orbits = Vec::new();
        let mut dust_instances = Vec::new();
        for _ in 0..300 {
            let orbit_r = rng.gen_range(15.0..40.0);
            let angle = rng.gen_range(0.0..std::f32::consts::TAU);
            let speed = rng.gen_range(0.005..0.02);
            let height = rng.gen_range(-6.0..6.0);

            dust_orbits.push(DustOrbit {
                radius: orbit_r,
                angle,
                speed,
                height,
            });
            dust_instances.push(InstanceData {
                position: [0.0, 0.0, 0.0],
                scale: [0.2, 0.02, 0.2],
                color: [1.0, 1.0, 1.0],
                alpha: 0.2,
            });
        }

        //
        // Gigantic sun
        // radius = 1200 => 100x bigger than planet's 12
        // further away => let's place center around (4000, 0, -2000)
        // each block is scaled up to e.g. [2,2,2]
        //
        let sun_radius = 1200.0;
        let sun_center = [4000.0, 0.0, -2000.0];
        let mut sun_voxels = Vec::new();

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
                        // We apply a scale so each sun-voxel is bigger:
                        // We'll do [2,2,2] => 2 meter cubes
                        sun_voxels.push(InstanceData {
                            position: pos,
                            scale: [2.0, 2.0, 2.0],
                            color: [4.0, 3.5, 0.2], // placeholder
                            alpha: 1.0,
                        });
                    }
                }
            }
        }

        // combine all
        let total_count = planet_voxels.len()
            + atmosphere_voxels.len()
            + cloud_voxels.len()
            + dust_instances.len()
            + sun_voxels.len();
        let num_instances_total = total_count as u32;

        let combined_instances = [
            planet_voxels.as_slice(),
            atmosphere_voxels.as_slice(),
            cloud_voxels.as_slice(),
            dust_instances.as_slice(),
            sun_voxels.as_slice(),
        ]
        .concat();
        let instance_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("InstanceBuffer"),
            contents: bytemuck::cast_slice(&combined_instances),
            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
        });

        //
        // Camera
        //
        let camera_uniform = CameraUniform {
            view_proj: Mat4::IDENTITY.to_cols_array(),
        };
        let camera_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("CameraBuffer"),
            contents: bytemuck::cast_slice(&camera_uniform.view_proj),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        //
        // Sun pos uniform
        //
        let sun_uniform = SunUniform {
            sun_pos: [sun_center[0], sun_center[1], sun_center[2], 1.0],
        };
        let sun_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("SunPosBuffer"),
            contents: bytemuck::bytes_of(&sun_uniform),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        // bind group layout
        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Globals BGL"),
            entries: &[
                // camera
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::VERTEX_FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // sun pos
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::VERTEX_FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });

        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Globals BG"),
            layout: &bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: camera_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: sun_buffer.as_entire_binding(),
                },
            ],
        });

        //
        // WGSL SHADER: point light from sun, no ambient
        // We'll pass worldPos from vs->fs, then compute direction = sunPos - worldPos
        // dot with normal => lambert.
        // If it's the sun itself, we do a dynamic color from a perlin we embed in "color"...
        // Actually we'll just do the same lambert approach, but the sun is so large that
        // effectively it won't be shadowed. We'll do "self-luminous" approach => skip lambert => color=in.color
        // We'll do a small hack: if (in.color.r>3.0) => treat as luminous. That way the sun glows.
        //

        let shader_src = r#"
struct Globals {
    viewProj: mat4x4<f32>,
};
struct Sun {
    sunPos: vec4<f32>,
};

@group(0) @binding(0) var<uniform> globals: Globals;
@group(0) @binding(1) var<uniform> sunData: Sun;

struct Instance {
    @location(5) pos: vec3<f32>,
    @location(6) scale: vec3<f32>,
    @location(7) color: vec3<f32>,
    @location(8) alpha: f32,
};

struct VSOut {
    @builtin(position) clipPos: vec4<f32>,
    @location(0) normal: vec3<f32>,
    @location(1) color: vec3<f32>,
    @location(2) alpha: f32,
    @location(3) worldPos: vec3<f32>,
};

@vertex
fn vs_main(
    @location(0) inPos: vec3<f32>,
    @location(1) inNormal: vec3<f32>,
    @location(2) _uv: vec2<f32>,
    inst: Instance
) -> VSOut {
    var out: VSOut;
    let scaled = vec3<f32>(
        inPos.x * inst.scale.x,
        inPos.y * inst.scale.y,
        inPos.z * inst.scale.z
    );
    let worldPos = scaled + inst.pos;
    out.worldPos = worldPos;

    // approximate normal:
    let fix = vec3<f32>(
        1.0/inst.scale.x,
        1.0/inst.scale.y,
        1.0/inst.scale.z
    );
    out.normal = normalize(inNormal * fix);

    out.clipPos = globals.viewProj * vec4<f32>(worldPos, 1.0);
    out.color = inst.color;
    out.alpha = inst.alpha;
    return out;
}

@fragment
fn fs_main(in: VSOut) -> @location(0) vec4<f32> {
    // if color.r>3 => this is a sun voxel => self-luminous, no shading
    if (in.color.r > 3.0) {
        return vec4<f32>(in.color, 1.0);
    }

    // otherwise, lambert from point light
    let sunPos = sunData.sunPos.xyz;
    let dir = normalize(sunPos - in.worldPos);
    let lambert = max(dot(in.normal, dir), 0.0);

    // no ambient
    let finalColor = in.color * lambert;
    return vec4<f32>(finalColor, in.alpha);
}
"#;

        let shader_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("SunPointShader"),
            source: wgpu::ShaderSource::Wgsl(std::borrow::Cow::Borrowed(shader_src)),
        });

        // pipeline layout
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("PipelineLayout"),
            bind_group_layouts: &[&bind_group_layout],
            push_constant_ranges: &[],
        });

        let render_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("RenderPipeline"),
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
                cull_mode: Some(wgpu::Face::Back),
                front_face: wgpu::FrontFace::Ccw,
                polygon_mode: wgpu::PolygonMode::Fill,
                ..Default::default()
            },
            depth_stencil: Some(wgpu::DepthStencilState {
                format: wgpu::TextureFormat::Depth24Plus,
                depth_write_enabled: true,
                depth_compare: wgpu::CompareFunction::Less,
                stencil: wgpu::StencilState::default(),
                bias: wgpu::DepthBiasState::default(),
            }),
            multisample: wgpu::MultisampleState::default(),
            multiview: None,
        });

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

            camera_uniform,
            camera_buffer,
            sun_uniform,
            sun_buffer,
            bind_group,

            perlin1: Perlin::new(1),
            perlin2: Perlin::new(2),

            time: 0.0,
            camera_yaw: 0.0,
            camera_pitch: 0.3,
            camera_dist: 80.0, // start a bit further
            last_mouse_pos: None,
            mouse_pressed: false,
        }
    }

    fn resize(&mut self, new_size: PhysicalSize<u32>) {
        if new_size.width>0 && new_size.height>0 {
            self.size = new_size;
            self.config.width = new_size.width;
            self.config.height = new_size.height;
            self.surface.configure(&self.device, &self.config);

            let depth_tex_desc = wgpu::TextureDescriptor {
                label: Some("DepthTex"),
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
            let depth_texture = self.device.create_texture(&depth_tex_desc);
            self.depth_texture_view = depth_texture.create_view(&wgpu::TextureViewDescriptor::default());
        }
    }

    fn update(&mut self) {
        self.time += 0.01;

        //
        // Planet color changes, atmosphere, clouds
        //
        let freq_planet = 0.06;
        let freq_atmo   = 0.04;
        let freq_cloud  = 0.03;

        // planet
        for v in &mut self.planet_voxels {
            let (x, y, z) = (v.position[0], v.position[1], v.position[2]);
            let dist_xy = (x*x + z*z).sqrt();
            let planet_r = 12.0;

            // If it's in polar region => keep white
            let polar_threshold = 0.9*planet_r;
            if y.abs() > polar_threshold {
                // skip
                continue;
            }

            // otherwise, do perlin-based color shift
            let px = x as f64 * freq_planet;
            let py = y as f64 * freq_planet;
            let pz = z as f64 * freq_planet;
            let t = self.time as f64 * 0.2;

            let val1 = self.perlin1.get([px, py, pz + t]);
            let val2 = 0.5*self.perlin2.get([pz, px, py + 1.5*t]);
            let sum = val1 + val2;
            let mapped = 0.5*(sum + 1.0);
            let (r,g,b) = planet_color_map(mapped as f32);
            v.color = [r,g,b];
        }

        // atmosphere
        for a in &mut self.atmosphere_voxels {
            let px = a.position[0] as f64 * freq_atmo;
            let py = a.position[1] as f64 * freq_atmo;
            let pz = a.position[2] as f64 * freq_atmo;
            let t  = self.time as f64 * 0.3;
            let val = self.perlin1.get([px, py, pz + t]);
            let mapped = 0.5*(val + 1.0);
            let (r,g,b) = atmosphere_color_map(mapped as f32);
            a.color = [r,g,b];
        }

        // clouds
        for c in &mut self.cloud_voxels {
            let px = c.position[0] as f64 * freq_cloud;
            let py = c.position[1] as f64 * freq_cloud;
            let pz = c.position[2] as f64 * freq_cloud;
            let t  = self.time as f64 * 0.4;
            let val = self.perlin2.get([px + t, py, pz]);
            let mapped = 0.5*(val + 1.0);
            let (r,g,b) = cloud_color_map(mapped as f32);
            c.color = [r,g,b];
        }

        //
        // Disk orbits
        //
        for (i, orbit) in self.dust_orbits.iter_mut().enumerate() {
            orbit.angle += orbit.speed;
            let x = orbit.radius * orbit.angle.cos();
            let z = orbit.radius * orbit.angle.sin();
            let y = orbit.height;
            self.dust_instances[i].position = [x,y,z];
        }

        //
        // Sun: dynamic noisy surface (yellow/orange)
        //
        let freq_sun = 0.02; // bigger features
        for s in &mut self.sun_voxels {
            let sx = s.position[0];
            let sy = s.position[1];
            let sz = s.position[2];
            // convert to local coords
            // sun center = (4000, 0, -2000)
            let lx = sx - 4000.0;
            let ly = sy - 0.0;
            let lz = sz + 2000.0;

            let px = lx as f64 * freq_sun;
            let py = ly as f64 * freq_sun;
            let pz = lz as f64 * freq_sun;
            let t  = self.time as f64 * 0.3;

            let val1 = self.perlin1.get([px, py + t, pz]);
            let val2 = 0.5*self.perlin2.get([py, pz + t, px]);
            let sum  = val1 + val2;
            let mapped = 0.5*(sum + 1.0);

            // map [0..1] => some range of yellow/orange
            // e.g. (3..4, 2.5..3.5, 0.0..0.5)
            let r = 3.0 + 1.0*mapped;   // [3..4]
            let g = 2.5 + 1.0*mapped;  // [2.5..3.5]
            let b = 0.0 + 0.5*mapped;  // [0..0.5]
            s.color = [r as f32, g as f32, b as f32];
        }

        // re-combine
        let combined = [
            self.planet_voxels.as_slice(),
            self.atmosphere_voxels.as_slice(),
            self.cloud_voxels.as_slice(),
            self.dust_instances.as_slice(),
            self.sun_voxels.as_slice(),
        ]
        .concat();
        self.queue.write_buffer(
            &self.instance_buffer,
            0,
            bytemuck::cast_slice(&combined),
        );

        // camera
        let aspect = self.config.width as f32 / self.config.height as f32;
        let fovy = 45f32.to_radians();
        let near = 0.1;
        let far  = 999999.0;

        let eye_x = self.camera_dist * self.camera_yaw.cos() * self.camera_pitch.cos();
        let eye_y = self.camera_dist * self.camera_pitch.sin();
        let eye_z = self.camera_dist * self.camera_yaw.sin() * self.camera_pitch.cos();
        let eye   = Vec3::new(eye_x, eye_y, eye_z);
        let center= Vec3::ZERO;
        let up    = Vec3::Y;

        let view = Mat4::look_at_rh(eye, center, up);
        let proj = Mat4::perspective_rh(fovy, aspect, near, far);
        let vp   = proj * view;
        self.camera_uniform.view_proj = vp.to_cols_array();
        self.queue.write_buffer(
            &self.camera_buffer,
            0,
            bytemuck::cast_slice(&self.camera_uniform.view_proj)
        );
    }

    fn render(&mut self) -> Result<(), wgpu::SurfaceError> {
        let frame = self.surface.get_current_texture()?;
        let view  = frame.texture.create_view(&wgpu::TextureViewDescriptor::default());

        let mut encoder = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("RenderEncoder"),
        });

        {
            let mut rp = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("MainRenderPass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color {
                            r: 0.0,
                            g: 0.0,
                            b: 0.01,
                            a: 1.0
                        }),
                        store: true,
                    }
                })],
                depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                    view: &self.depth_texture_view,
                    depth_ops: Some(wgpu::Operations {
                        load: wgpu::LoadOp::Clear(1.0),
                        store: true
                    }),
                    stencil_ops: None,
                }),
            });

            rp.set_pipeline(&self.render_pipeline);
            rp.set_bind_group(0, &self.bind_group, &[]);
            rp.set_vertex_buffer(0, self.vertex_buffer.slice(..));
            rp.set_vertex_buffer(1, self.instance_buffer.slice(..));
            rp.set_index_buffer(self.index_buffer.slice(..), wgpu::IndexFormat::Uint16);

            rp.draw_indexed(0..self.num_indices, 0, 0..self.num_instances_total);
        }

        self.queue.submit(iter::once(encoder.finish()));
        frame.present();
        Ok(())
    }

    fn input(&mut self, event: &WindowEvent) -> bool {
        match event {
            // mouse press
            WindowEvent::MouseInput { state, button, .. } => {
                if *button == MouseButton::Left {
                    self.mouse_pressed = *state == ElementState::Pressed;
                }
                true
            }
            // mouse move
            WindowEvent::CursorMoved{ position, .. } => {
                if self.mouse_pressed {
                    let (x, y) = (position.x, position.y);
                    if let Some((lx, ly)) = self.last_mouse_pos {
                        let dx = (x - lx) as f32 * 0.005;
                        let dy = (y - ly) as f32 * 0.005;
                        self.camera_yaw   += dx;
                        self.camera_pitch -= dy;
                        self.camera_pitch = self.camera_pitch.clamp(-1.4, 1.4);
                    }
                    self.last_mouse_pos = Some((x,y));
                } else {
                    self.last_mouse_pos = Some((position.x, position.y));
                }
                true
            }
            // scroll => zoom in/out
            WindowEvent::MouseWheel { delta, .. } => {
                let amt = match delta {
                    MouseScrollDelta::LineDelta(_, s) => *s,
                    MouseScrollDelta::PixelDelta(px) => px.y as f32 / 60.0,
                };
                // up => zoom in => reduce distance
                self.camera_dist += amt * -2.0;
                if self.camera_dist < 5.0 {
                    self.camera_dist = 5.0;
                }
                if self.camera_dist > 30000.0 {
                    self.camera_dist = 30000.0;
                }
                true
            }
            _ => false
        }
    }
}

//
// =========== MAIN + EVENT LOOP ===========
//

fn main(){
    env_logger::init();
    pollster::block_on(run());
}

async fn run() {
    let event_loop = EventLoop::new();
    let window = WindowBuilder::new()
        .with_title("Gigantic Sun + Planet + Clouds + Perlin (wgpu)")
        .build(&event_loop)
        .unwrap();

    let mut state = State::new(&window).await;

    event_loop.run(move |event, _, control_flow| {
        match &event {
            Event::WindowEvent { event, window_id} if *window_id == window.id() => {
                match event {
                    WindowEvent::CloseRequested
                    | WindowEvent::KeyboardInput {
                        input:
                            KeyboardInput{
                                state: ElementState::Pressed,
                                virtual_keycode: Some(VirtualKeyCode::Escape),
                                ..
                            },
                        ..
                    } => *control_flow = ControlFlow::Exit,

                    WindowEvent::Resized(physical_size) => {
                        state.resize(*physical_size);
                    }
                    WindowEvent::ScaleFactorChanged{ new_inner_size, ..} => {
                        state.resize(**new_inner_size);
                    }
                    _ => {
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
                        *control_flow = ControlFlow::Exit
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
// ===================== COLOR MAPS =====================
//

fn planet_color_map(t: f32) -> (f32, f32, f32) {
    // same gradient as before
    if t < 0.33 {
        let u = t / 0.33;
        let r = 0.1 + 0.2*u;
        let g = 0.7 - 0.3*u;
        let b = 0.3 + 0.4*u;
        (r,g,b)
    } else if t < 0.66 {
        let u = (t - 0.33)/0.33;
        let r = 0.3 + 0.5*u;
        let g = 0.4 - 0.2*u;
        let b = 0.7 + 0.2*u;
        (r,g,b)
    } else {
        let u = (t - 0.66)/0.34;
        let r = 0.8 + 0.2*u;
        let g = 0.2 + 0.3*u;
        let b = 0.9 - 0.2*u;
        (r,g,b)
    }
}

fn atmosphere_color_map(t: f32) -> (f32, f32, f32) {
    // from light blue to pinkish
    let r = 0.3 + 0.7*t;
    let g = 0.5 + 0.1*t;
    let b = 0.8 - 0.1*t;
    (r,g,b)
}

fn cloud_color_map(t: f32) -> (f32, f32, f32) {
    // near white but slightly tinted
    let grey = 0.8 + 0.2*t;
    let tint = 0.9 + 0.1*t;
    (tint, grey, tint)
}
