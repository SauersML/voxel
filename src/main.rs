use std::iter;

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
// ========================== GEOMETRY DATA ==========================
//
// We'll reuse a single cube geometry for all voxel-like objects:
// planet, atmosphere, clouds, dust, and the giant sun.
//

#[rustfmt::skip]
const VERTICES: &[f32] = &[
    // position (x,y,z), normal (nx,ny,nz), uv (unused)
    // +X
    0.5,  0.5, -0.5,    1.0,  0.0,  0.0,   0.0, 0.0,
    0.5, -0.5, -0.5,    1.0,  0.0,  0.0,   0.0, 0.0,
    0.5, -0.5,  0.5,    1.0,  0.0,  0.0,   0.0, 0.0,
    0.5,  0.5,  0.5,    1.0,  0.0,  0.0,   0.0, 0.0,

    // -X
   -0.5,  0.5,  0.5,   -1.0,  0.0,  0.0,   0.0, 0.0,
   -0.5, -0.5,  0.5,   -1.0,  0.0,  0.0,   0.0, 0.0,
   -0.5, -0.5, -0.5,   -1.0,  0.0,  0.0,   0.0, 0.0,
   -0.5,  0.5, -0.5,   -1.0,  0.0,  0.0,   0.0, 0.0,

    // +Y
   -0.5,  0.5,  0.5,    0.0,  1.0,  0.0,   0.0, 0.0,
   -0.5,  0.5, -0.5,    0.0,  1.0,  0.0,   0.0, 0.0,
    0.5,  0.5, -0.5,    0.0,  1.0,  0.0,   0.0, 0.0,
    0.5,  0.5,  0.5,    0.0,  1.0,  0.0,   0.0, 0.0,

    // -Y
    0.5, -0.5,  0.5,    0.0, -1.0,  0.0,   0.0, 0.0,
    0.5, -0.5, -0.5,    0.0, -1.0,  0.0,   0.0, 0.0,
   -0.5, -0.5, -0.5,    0.0, -1.0,  0.0,   0.0, 0.0,
   -0.5, -0.5,  0.5,    0.0, -1.0,  0.0,   0.0, 0.0,

    // +Z
    0.5,  0.5,  0.5,    0.0,  0.0,  1.0,   0.0, 0.0,
    0.5, -0.5,  0.5,    0.0,  0.0,  1.0,   0.0, 0.0,
   -0.5, -0.5,  0.5,    0.0,  0.0,  1.0,   0.0, 0.0,
   -0.5,  0.5,  0.5,    0.0,  0.0,  1.0,   0.0, 0.0,

    // -Z
   -0.5,  0.5, -0.5,    0.0,  0.0, -1.0,   0.0, 0.0,
   -0.5, -0.5, -0.5,    0.0,  0.0, -1.0,   0.0, 0.0,
    0.5, -0.5, -0.5,    0.0,  0.0, -1.0,   0.0, 0.0,
    0.5,  0.5, -0.5,    0.0,  0.0, -1.0,   0.0, 0.0,
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
// =================== INSTANCE DATA ===================
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
        wgpu::VertexBufferLayout {
            array_stride: std::mem::size_of::<InstanceData>() as wgpu::BufferAddress,
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
                    offset: 4 * 3,
                    shader_location: 6,
                    format: wgpu::VertexFormat::Float32x3,
                },
                // color
                wgpu::VertexAttribute {
                    offset: 4 * 6,
                    shader_location: 7,
                    format: wgpu::VertexFormat::Float32x3,
                },
                // alpha
                wgpu::VertexAttribute {
                    offset: 4 * 9,
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
// =============== Additional Uniform: The Sun's Position ===============
//
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct SunUniform {
    sun_pos: [f32; 4], // stored in world space
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
// ============================ PERLIN COLOR MAPS ============================
//
fn planet_color_map(t: f32) -> (f32, f32, f32) {
    if t < 0.33 {
        let u = t / 0.33;
        (0.1 + 0.2 * u, 0.7 - 0.3 * u, 0.3 + 0.4 * u)
    } else if t < 0.66 {
        let u = (t - 0.33) / 0.33;
        (0.3 + 0.5 * u, 0.4 - 0.2 * u, 0.7 + 0.2 * u)
    } else {
        let u = (t - 0.66) / 0.34;
        (0.8 + 0.2 * u, 0.2 + 0.3 * u, 0.9 - 0.2 * u)
    }
}

fn atmosphere_color_map(t: f32) -> (f32, f32, f32) {
    // from light blue to pinkish
    (
        0.3 + 0.7 * t,
        0.5 + 0.1 * t,
        0.8 - 0.1 * t,
    )
}

fn cloud_color_map(t: f32) -> (f32, f32, f32) {
    // near white but slightly tinted
    let grey = 0.8 + 0.2 * t;
    let tint = 0.9 + 0.1 * t;
    (tint, grey, tint)
}

//
// ============================ SUN CHUNK BUFFERS ============================
//
// Instead of holding a huge CPU Vec for the entire radius=1200 sun, we'll
// generate it in many small chunk buffers, drawn incrementally. We'll still
// update them each frame for Perlin color changes, matching original logic.
//
// No feature is removed: the sun is still fully volumetric, with dynamic color.
//

struct SunChunk {
    buffer: wgpu::Buffer,
    count: u32, // number of instances in this chunk
    chunk_data_cpu: Vec<InstanceData>, // CPU copy for dynamic updates
}

fn inside_sun(x: i32, y: i32, z: i32, rad: f32) -> bool {
    let fx = x as f32 + 0.5;
    let fy = y as f32 + 0.5;
    let fz = z as f32 + 0.5;
    (fx * fx + fy * fy + fz * fz).sqrt() < rad
}

//
// This struct manages chunk-based generation & updating of the giant sun.
//
struct SunManager {
    radius: i32,      // e.g. 1200
    chunk_size: i32,  // e.g. 32
    center: [f32; 3], // e.g. (4000,0,-2000)

    // we'll keep them in a Vec
    chunks: Vec<SunChunk>,
    // to generate them incrementally:
    bx: i32, // current chunk index in x dimension
    by: i32,
    bz: i32,
    xblocks: i32,
    yblocks: i32,
    zblocks: i32,

    // perlin references
    perlin1: Perlin,
    perlin2: Perlin,
}

impl SunManager {
    fn new(radius: f32, chunk_size: i32, p1: Perlin, p2: Perlin) -> Self {
        // total blocks for the [(-r)..r]^3 region
        let r_i = radius as i32;
        let side = 2 * r_i;
        let blocks = (side as f32 / chunk_size as f32).ceil() as i32;
        SunManager {
            radius: r_i,
            chunk_size,
            center: [4000.0, 0.0, -2000.0],
            chunks: Vec::new(),
            bx: 0,
            by: 0,
            bz: 0,
            xblocks: blocks,
            yblocks: blocks,
            zblocks: blocks,

            perlin1: p1,
            perlin2: p2,
        }
    }

    // generate one chunk, store it in GPU buffer
    fn generate_one_chunk(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
    ) -> bool {
        if self.bx >= self.xblocks {
            return false; // done generating
        }

        let x0 = -self.radius + self.bx * self.chunk_size;
        let y0 = -self.radius + self.by * self.chunk_size;
        let z0 = -self.radius + self.bz * self.chunk_size;

        // build CPU data
        let mut block_data = Vec::with_capacity((self.chunk_size * self.chunk_size * self.chunk_size) as usize);
        for xx in 0..self.chunk_size {
            for yy in 0..self.chunk_size {
                for zz in 0..self.chunk_size {
                    let gx = x0 + xx;
                    let gy = y0 + yy;
                    let gz = z0 + zz;
                    if inside_sun(gx, gy, gz, self.radius as f32) {
                        let fx = gx as f32 + 0.5;
                        let fy = gy as f32 + 0.5;
                        let fz = gz as f32 + 0.5;
                        let pos = [
                            self.center[0] + fx,
                            self.center[1] + fy,
                            self.center[2] + fz,
                        ];
                        // scale => [2,2,2] for giant sun blocks
                        block_data.push(InstanceData {
                            position: pos,
                            scale: [2.0, 2.0, 2.0],
                            color: [4.0, 3.5, 0.2], // placeholder, updated each frame
                            alpha: 1.0,
                        });
                    }
                }
            }
        }
        if block_data.is_empty() {
            // skip building buffer
        } else {
            let size_bytes = (block_data.len() * std::mem::size_of::<InstanceData>()) as wgpu::BufferAddress;
            let buf = device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("SunChunkBuffer"),
                size: size_bytes,
                usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
            // write to GPU
            queue.write_buffer(&buf, 0, bytemuck::cast_slice(&block_data));

            self.chunks.push(SunChunk {
                buffer: buf,
                count: block_data.len() as u32,
                chunk_data_cpu: block_data,
            });
        }

        // proceed to next chunk
        self.bz += 1;
        if self.bz >= self.zblocks {
            self.bz = 0;
            self.by += 1;
            if self.by >= self.yblocks {
                self.by = 0;
                self.bx += 1;
            }
        }
        true
    }

    // call this each frame => generate e.g. 5 new chunks
    fn generate_some_chunks(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
    ) {
        // generate up to N new chunks
        for _ in 0..5 {
            if !self.generate_one_chunk(device, queue) {
                // done generating all chunks
                break;
            }
        }
    }

    // per-frame update => do perlin color
    fn update_perlin_colors(&mut self, queue: &wgpu::Queue, time: f32) {
        // freq = 0.02, as original
        let freq_sun = 0.02;
        for chunk in &mut self.chunks {
            // update chunk_data_cpu, then rewrite
            for s in &mut chunk.chunk_data_cpu {
                let sx = s.position[0];
                let sy = s.position[1];
                let sz = s.position[2];
                // local coords => subtract center
                let lx = sx - self.center[0];
                let ly = sy - self.center[1];
                let lz = sz - (-self.center[2]); // center[2] = -2000 => minus negative => plus
                let px = lx as f64 * freq_sun;
                let py = ly as f64 * freq_sun;
                let pz = lz as f64 * freq_sun;
                let t = time as f64 * 0.3;
                let val1 = self.perlin1.get([px, py + t, pz]);
                let val2 = 0.5 * self.perlin2.get([py, pz + t, px]);
                let sum = val1 + val2;
                let mapped = 0.5 * (sum + 1.0);

                let r = 3.0 + 1.0 * mapped;   // [3..4]
                let g = 2.5 + 1.0 * mapped;  // [2.5..3.5]
                let b = 0.0 + 0.5 * mapped;  // [0..0.5]
                s.color = [r as f32, g as f32, b as f32];
            }
            // rewrite entire chunk to GPU
            let bytes = bytemuck::cast_slice(&chunk.chunk_data_cpu);
            queue.write_buffer(&chunk.buffer, 0, bytes);
        }
    }

    // draw all chunks
    fn draw_all<'a>(&'a self, rp: &mut wgpu::RenderPass<'a>, index_count: u32) {
        for chunk in &self.chunks {
            rp.set_vertex_buffer(1, chunk.buffer.slice(..));
            rp.draw_indexed(0..index_count, 0, 0..chunk.count);
        }
    }
}

//
// ========================= PROGRAM STATE =========================
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

    //
    // planet
    //
    planet_voxels: Vec<InstanceData>,
    atmosphere_voxels: Vec<InstanceData>,
    cloud_voxels: Vec<InstanceData>,

    //
    // dust
    //
    dust_orbits: Vec<DustOrbit>,
    dust_instances: Vec<InstanceData>,

    //
    // "small" combined buffer => planet + atmosphere + clouds + dust
    //
    base_instance_buffer: wgpu::Buffer,
    base_instance_count: u32,

    //
    // sun manager => chunk-based
    //
    sun_manager: SunManager, // no single huge Vec => chunked

    // camera
    camera_uniform: CameraUniform,
    camera_buffer: wgpu::Buffer,

    // sun pos
    sun_uniform: SunUniform,
    sun_buffer: wgpu::Buffer,

    // bind group
    bind_group: wgpu::BindGroup,

    // noise
    perlin1: Perlin,
    perlin2: Perlin,

    //
    // dynamic
    //
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
        let adapter = instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            compatible_surface: Some(&surface),
            force_fallback_adapter: false,
        })
        .await
        .expect("No GPU adapter found!");

        let (device, queue) = adapter.request_device(
            &wgpu::DeviceDescriptor {
                label: Some("Device"),
                features: wgpu::Features::empty(),
                limits: wgpu::Limits::default(),
            },
            None,
        )
        .await
        .unwrap();

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
        let depth_desc = wgpu::TextureDescriptor {
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
        let depth_tex = device.create_texture(&depth_desc);
        let depth_texture_view = depth_tex.create_view(&wgpu::TextureViewDescriptor::default());

        // geometry buffers
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
        // Planet, atmosphere, clouds, dust
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
                    let dist = (fx * fx + fy * fy + fz * fz).sqrt();
                    let noise_bump = rng.gen_range(0.0..2.0);
                    if dist < planet_radius + noise_bump {
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
        // color poles white
        let pole_thresh = 0.9 * planet_radius;
        for v in &mut planet_voxels {
            if v.position[1].abs() > pole_thresh {
                v.color = [1.0, 1.0, 1.0];
            }
        }

        // atmosphere
        let mut atmosphere_voxels = Vec::new();
        for v in &planet_voxels {
            let dist = (v.position[0].powi(2) + v.position[1].powi(2) + v.position[2].powi(2)).sqrt();
            if dist > planet_radius - 1.0 {
                let outward = 1.05;
                let pos = [
                    v.position[0] * outward / dist.max(0.0001),
                    v.position[1] * outward / dist.max(0.0001),
                    v.position[2] * outward / dist.max(0.0001),
                ];
                atmosphere_voxels.push(InstanceData {
                    position: pos,
                    scale: [1.0, 1.0, 1.0],
                    color: [0.3, 0.5, 0.8],
                    alpha: 0.2,
                });
            }
        }

        // clouds
        let mut cloud_voxels = Vec::new();
        for v in &planet_voxels {
            let dist = (v.position[0].powi(2) + v.position[1].powi(2) + v.position[2].powi(2)).sqrt();
            if dist > planet_radius - 1.0 && rng.gen_bool(0.3) {
                let outward = 1.1;
                let pos = [
                    v.position[0] * outward / dist.max(0.0001),
                    v.position[1] * outward / dist.max(0.0001),
                    v.position[2] * outward / dist.max(0.0001),
                ];
                cloud_voxels.push(InstanceData {
                    position: pos,
                    scale: [1.0, 1.0, 1.0],
                    color: [1.0, 1.0, 1.0],
                    alpha: 0.4,
                });
            }
        }

        // dust
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

        // combine planet+atmo+cloud+dust in one buffer
        let base_combined = [
            planet_voxels.as_slice(),
            atmosphere_voxels.as_slice(),
            cloud_voxels.as_slice(),
            dust_instances.as_slice(),
        ].concat();
        let base_instance_count = base_combined.len() as u32;
        let base_instance_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("BaseInstanceBuffer"),
            contents: bytemuck::cast_slice(&base_combined),
            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
        });

        //
        // Sun => chunk-based approach
        //
        let sun_manager = SunManager::new(1200.0, 32, Perlin::new(1), Perlin::new(2));

        // camera
        let camera_uniform = CameraUniform {
            view_proj: Mat4::IDENTITY.to_cols_array(),
        };
        let camera_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("CameraBuffer"),
            contents: bytemuck::cast_slice(&camera_uniform.view_proj),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        // sun pos
        let sun_uniform = SunUniform {
            sun_pos: [4000.0, 0.0, -2000.0, 1.0],
        };
        let sun_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("SunPosBuffer"),
            contents: bytemuck::bytes_of(&sun_uniform),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        // global bind group
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
        // WGSL Shader (point light from sun, same logic, self-luminous if color.r>3)
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
    // if color.r>3 => this is sun => self-luminous
    if (in.color.r > 3.0) {
        return vec4<f32>(in.color, 1.0);
    }
    let sunPos = sunData.sunPos.xyz;
    let dir = normalize(sunPos - in.worldPos);
    let lambert = max(dot(in.normal, dir), 0.0);

    let finalColor = in.color * lambert;
    return vec4<f32>(finalColor, in.alpha);
}
"#;

        // important: use device.create_shader_module(wgpu::ShaderModuleDescriptor), no ampersand
        let shader_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("SunPointShader"),
            source: wgpu::ShaderSource::Wgsl(shader_src.into()),
        });

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
                        array_stride: std::mem::size_of::<f32>() as wgpu::BufferAddress * 8,
                        step_mode: wgpu::VertexStepMode::Vertex,
                        attributes: &[
                            wgpu::VertexAttribute {
                                offset: 0,
                                shader_location: 0,
                                format: wgpu::VertexFormat::Float32x3,
                            },
                            wgpu::VertexAttribute {
                                offset: (4 * 3) as wgpu::BufferAddress,
                                shader_location: 1,
                                format: wgpu::VertexFormat::Float32x3,
                            },
                            wgpu::VertexAttribute {
                                offset: (4 * 6) as wgpu::BufferAddress,
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

            base_instance_buffer,
            base_instance_count,

            sun_manager,

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
            camera_dist: 80.0,
            last_mouse_pos: None,
            mouse_pressed: false,
        }
    }

    fn resize(&mut self, new_size: PhysicalSize<u32>) {
        if new_size.width > 0 && new_size.height > 0 {
            self.size = new_size;
            self.config.width = new_size.width;
            self.config.height = new_size.height;
            self.surface.configure(&self.device, &self.config);

            let depth_desc = wgpu::TextureDescriptor {
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
            let depth_tex = self.device.create_texture(&depth_desc);
            self.depth_texture_view = depth_tex.create_view(&wgpu::TextureViewDescriptor::default());
        }
    }

    fn update(&mut self) {
        self.time += 0.01;

        //
        // ============ Planet color changes, atmosphere, clouds ============
        //
        let freq_planet = 0.06;
        let freq_atmo   = 0.04;
        let freq_cloud  = 0.03;

        // planet
        for v in &mut self.planet_voxels {
            let (x, y, z) = (v.position[0], v.position[1], v.position[2]);
            let planet_r = 12.0;
            let polar_threshold = 0.9 * planet_r;
            if y.abs() > polar_threshold {
                continue; // keep white
            }
            let px = x as f64 * freq_planet;
            let py = y as f64 * freq_planet;
            let pz = z as f64 * freq_planet;
            let t = self.time as f64 * 0.2;

            let val1 = self.perlin1.get([px, py, pz + t]);
            let val2 = 0.5 * self.perlin2.get([pz, px, py + 1.5 * t]);
            let sum = val1 + val2;
            let mapped = 0.5 * (sum + 1.0);
            let (r, g, b) = planet_color_map(mapped as f32);
            v.color = [r, g, b];
        }

        // atmosphere
        for a in &mut self.atmosphere_voxels {
            let px = a.position[0] as f64 * freq_atmo;
            let py = a.position[1] as f64 * freq_atmo;
            let pz = a.position[2] as f64 * freq_atmo;
            let t = self.time as f64 * 0.3;
            let val = self.perlin1.get([px, py, pz + t]);
            let mapped = 0.5 * (val + 1.0);
            let (r, g, b) = atmosphere_color_map(mapped as f32);
            a.color = [r, g, b];
        }

        // clouds
        for c in &mut self.cloud_voxels {
            let px = c.position[0] as f64 * freq_cloud;
            let py = c.position[1] as f64 * freq_cloud;
            let pz = c.position[2] as f64 * freq_cloud;
            let t = self.time as f64 * 0.4;
            let val = self.perlin2.get([px + t, py, pz]);
            let mapped = 0.5 * (val + 1.0);
            let (r, g, b) = cloud_color_map(mapped as f32);
            c.color = [r, g, b];
        }

        //
        // ============ Disk orbits ============
        //
        for (i, orbit) in self.dust_orbits.iter_mut().enumerate() {
            orbit.angle += orbit.speed;
            let x = orbit.radius * orbit.angle.cos();
            let z = orbit.radius * orbit.angle.sin();
            let y = orbit.height;
            self.dust_instances[i].position = [x, y, z];
        }

        //
        // ============ Re-write planet+atmo+cloud+dust to GPU buffer ============
        //
        let re_combined = [
            self.planet_voxels.as_slice(),
            self.atmosphere_voxels.as_slice(),
            self.cloud_voxels.as_slice(),
            self.dust_instances.as_slice(),
        ].concat();
        self.queue.write_buffer(
            &self.base_instance_buffer,
            0,
            bytemuck::cast_slice(&re_combined),
        );

        //
        // ============ Sun: chunk-based generation & dynamic color updates ============
        //
        // generate some new chunks (incrementally) => e.g. 5 per frame
        self.sun_manager.generate_some_chunks(&self.device, &self.queue);

        // update perlin color in each chunk
        self.sun_manager.update_perlin_colors(&self.queue, self.time);

        //
        // ============ Camera ============
        //
        let aspect = self.config.width as f32 / self.config.height as f32;
        let fovy = 45f32.to_radians();
        let near = 0.1;
        let far = 999999.0;

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

    fn render(&mut self) -> Result<(), wgpu::SurfaceError> {
        let frame = self.surface.get_current_texture()?;
        let view = frame.texture.create_view(&wgpu::TextureViewDescriptor::default());

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
            rp.set_bind_group(0, &self.bind_group, &[]);

            // draw planet+atmosphere+clouds+dust
            rp.set_vertex_buffer(0, self.vertex_buffer.slice(..));
            rp.set_index_buffer(self.index_buffer.slice(..), wgpu::IndexFormat::Uint16);
            rp.set_vertex_buffer(1, self.base_instance_buffer.slice(..));
            rp.draw_indexed(0..self.num_indices, 0, 0..self.base_instance_count);

            // draw each sun chunk
            self.sun_manager.draw_all(&mut rp, self.num_indices);
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
            // scroll => zoom
            WindowEvent::MouseWheel { delta, .. } => {
                let amt = match delta {
                    MouseScrollDelta::LineDelta(_, s) => *s,
                    MouseScrollDelta::PixelDelta(px) => px.y as f32 / 60.0,
                };
                self.camera_dist += amt * -2.0;
                if self.camera_dist < 5.0 {
                    self.camera_dist = 5.0;
                }
                if self.camera_dist > 30000.0 {
                    self.camera_dist = 30000.0;
                }
                true
            }
            _ => false,
        }
    }
}

//
// ===================== MAIN + EVENT LOOP =====================
//

fn main() {
    env_logger::init();
    pollster::block_on(run());
}

async fn run() {
    let event_loop = EventLoop::new();
    let window = WindowBuilder::new()
        .with_title("Gigantic Sun + Planet + Clouds + Dust + Perlin (chunked) - EXACT same features")
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
