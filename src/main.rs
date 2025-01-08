use std::iter;
use std::mem;

use wgpu::util::DeviceExt;
use winit::{
    dpi::PhysicalSize,
    event::*,
    event_loop::{ControlFlow, EventLoop},
    window::WindowBuilder,
};

use glam::{Mat4, Vec2, Vec3};
use rand::Rng;
use noise::{NoiseFn, Perlin};

//
// ========================= CUBE GEOMETRY =========================
//

#[rustfmt::skip]
const VERTICES: &[f32] = &[
    // position         normal           uv
    // +X
    0.5,  0.5, -0.5,   1.0,  0.0,  0.0,  1.0, 0.0,
    0.5, -0.5, -0.5,   1.0,  0.0,  0.0,  0.0, 0.0,
    0.5, -0.5,  0.5,   1.0,  0.0,  0.0,  0.0, 1.0,
    0.5,  0.5,  0.5,   1.0,  0.0,  0.0,  1.0, 1.0,

    // -X
   -0.5,  0.5,  0.5,  -1.0,  0.0,  0.0,  1.0, 0.0,
   -0.5, -0.5,  0.5,  -1.0,  0.0,  0.0,  0.0, 0.0,
   -0.5, -0.5, -0.5,  -1.0,  0.0,  0.0,  0.0, 1.0,
   -0.5,  0.5, -0.5,  -1.0,  0.0,  0.0,  1.0, 1.0,

    // +Y
   -0.5,  0.5,  0.5,   0.0,  1.0,  0.0,  1.0, 0.0,
   -0.5,  0.5, -0.5,   0.0,  1.0,  0.0,  0.0, 0.0,
    0.5,  0.5, -0.5,   0.0,  1.0,  0.0,  0.0, 1.0,
    0.5,  0.5,  0.5,   0.0,  1.0,  0.0,  1.0, 1.0,

    // -Y
    0.5, -0.5,  0.5,   0.0, -1.0,  0.0,  1.0, 0.0,
    0.5, -0.5, -0.5,   0.0, -1.0,  0.0,  0.0, 0.0,
   -0.5, -0.5, -0.5,   0.0, -1.0,  0.0,  0.0, 1.0,
   -0.5, -0.5,  0.5,   0.0, -1.0,  0.0,  1.0, 1.0,

    // +Z
    0.5,  0.5,  0.5,   0.0,  0.0,  1.0,  1.0, 0.0,
    0.5, -0.5,  0.5,   0.0,  0.0,  1.0,  0.0, 0.0,
   -0.5, -0.5,  0.5,   0.0,  0.0,  1.0,  0.0, 1.0,
   -0.5,  0.5,  0.5,   0.0,  0.0,  1.0,  1.0, 1.0,

    // -Z
   -0.5,  0.5, -0.5,   0.0,  0.0, -1.0,  1.0, 0.0,
   -0.5, -0.5, -0.5,   0.0,  0.0, -1.0,  0.0, 0.0,
    0.5, -0.5, -0.5,   0.0,  0.0, -1.0,  0.0, 1.0,
    0.5,  0.5, -0.5,   0.0,  0.0, -1.0,  1.0, 1.0,
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
// ========================= INSTANCE DATA =========================
//

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct InstanceData {
    position: [f32; 3],
    scale: f32,
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
                    format: wgpu::VertexFormat::Float32,
                },
                // color
                wgpu::VertexAttribute {
                    offset: (4 * 4) as wgpu::BufferAddress,
                    shader_location: 7,
                    format: wgpu::VertexFormat::Float32x3,
                },
                // alpha
                wgpu::VertexAttribute {
                    offset: (4 * 7) as wgpu::BufferAddress,
                    shader_location: 8,
                    format: wgpu::VertexFormat::Float32,
                },
            ],
        }
    }
}

//
// ========================= CAMERA UNIFORM =========================
//

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct CameraUniform {
    view_proj: [f32; 16],
}

//
// ========================= CUSTOM ORBITING DUST DATA =========================
//

/// For each dust particle, we store its orbit radius, speed, and current angle.
#[derive(Clone, Copy)]
struct DustOrbit {
    radius: f32,
    angle: f32,
    speed: f32,
    height: f32, // how high above the origin it orbits
}

//
// ========================= APP STATE =========================
//

struct State {
    // WGPU + Window
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
    sphere_instances: Vec<InstanceData>,
    dust_orbits: Vec<DustOrbit>, // separate CPU data for dust orbiting
    dust_instances: Vec<InstanceData>,
    instance_buffer: wgpu::Buffer,
    num_instances_total: u32,

    // Perlin noise for color
    perlin: Perlin,

    // Camera
    camera_uniform: CameraUniform,
    camera_buffer: wgpu::Buffer,
    camera_bind_group: wgpu::BindGroup,

    // Animation
    time: f32,

    // Camera controls
    camera_yaw: f32,   // horizontal rotation
    camera_pitch: f32, // vertical rotation
    camera_dist: f32,  // distance from center

    last_mouse_pos: Option<(f64, f64)>,
    mouse_pressed: bool,
}

impl State {
    async fn new(window: &winit::window::Window) -> Self {
        // Setup instance + surface + adapter
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
            .expect("Failed to find GPU adapter!");

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
        let surface_format = surface_caps.formats[0];
        let config = wgpu::SurfaceConfiguration {
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            format: surface_format,
            width: size.width,
            height: size.height,
            present_mode: surface_caps.present_modes[0],
            alpha_mode: surface_caps.alpha_modes[0],
            view_formats: vec![],
        };
        surface.configure(&device, &config);

        // Depth texture
        let depth_texture_desc = &wgpu::TextureDescriptor {
            label: Some("DepthTexture"),
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

        // Buffers for geometry
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
        // Build main voxel sphere
        //
        let mut sphere_instances = Vec::new();
        let mut rng = rand::thread_rng();
        let radius = 12.0;
        // We’ll do a 32^3 region, check distance from center
        // We'll sample Perlin to pick color in patches
        for x in -16..16 {
            for y in -16..16 {
                for z in -16..16 {
                    let fx = x as f32 + 0.5;
                    let fy = y as f32 + 0.5;
                    let fz = z as f32 + 0.5;
                    let dist = (fx*fx + fy*fy + fz*fz).sqrt();
                    let noise = rng.gen_range(0.0..2.0);
                    if dist < radius + noise {
                        // We'll fill real color from Perlin later in update() to make it shift
                        let placeholder_color = [0.5, 0.5, 0.5];
                        sphere_instances.push(InstanceData {
                            position: [fx, fy, fz],
                            scale: 1.0,
                            color: placeholder_color,
                            alpha: 1.0,
                        });
                    }
                }
            }
        }

        //
        // Build orbiting dust
        //
        // Instead of random position, we store orbit info (radius, speed, angle).
        // We'll update them each frame and fill into "dust_instances".
        let mut dust_orbits = Vec::new();
        let mut dust_instances = Vec::new();
        for _ in 0..300 {
            let orbit_radius = rng.gen_range(15.0..35.0);
            let speed = rng.gen_range(0.005..0.02); // vary speed
            let angle = rng.gen_range(0.0..std::f32::consts::TAU);
            let height = rng.gen_range(-5.0..15.0);

            dust_orbits.push(DustOrbit {
                radius: orbit_radius,
                angle,
                speed,
                height,
            });
            // placeholder instance
            dust_instances.push(InstanceData {
                position: [0.0, 0.0, 0.0],
                scale: 0.2,
                color: [1.0, 1.0, 1.0],
                alpha: 0.15,
            });
        }

        // Combined instance buffer
        let num_instances_total = (sphere_instances.len() + dust_instances.len()) as u32;
        let combined = [sphere_instances.as_slice(), dust_instances.as_slice()].concat();
        let instance_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Instance Buffer"),
            contents: bytemuck::cast_slice(&combined),
            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
        });

        //
        // Camera
        //
        let camera_uniform = CameraUniform {
            view_proj: Mat4::IDENTITY.to_cols_array(),
        };
        let camera_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Camera Buffer"),
            contents: bytemuck::cast_slice(&camera_uniform.view_proj),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });
        let camera_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
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
            layout: &camera_bind_group_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: camera_buffer.as_entire_binding(),
            }],
        });

        //
        // Shader
        //
        // We’ll do simpler Lambert shading with two directional lights + small ambient.
        // No rim light. Reduced brightness for more subtle shading.
        //
        let shader_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Voxel Shader"),
            source: wgpu::ShaderSource::Wgsl(std::borrow::Cow::Borrowed(
                r#"
@group(0) @binding(0) 
var<uniform> viewProj: mat4x4<f32>;

struct Instance {
    @location(5) pos: vec3<f32>,
    @location(6) scale: f32,
    @location(7) color: vec3<f32>,
    @location(8) alpha: f32,
};

struct VSOut {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) normal: vec3<f32>,
    @location(1) color: vec3<f32>,
    @location(2) alpha: f32,
};

@vertex
fn vs_main(
    @location(0) inPos: vec3<f32>,
    @location(1) inNormal: vec3<f32>,
    @location(2) inUV: vec2<f32>,
    inst: Instance
) -> VSOut {
    var out: VSOut;
    let worldPos = inPos * inst.scale + inst.pos;
    out.clip_position = viewProj * vec4<f32>(worldPos, 1.0);
    out.normal = normalize(inNormal); // scale is uniform, so no real distortion
    out.color = inst.color;
    out.alpha = inst.alpha;
    return out;
}

@fragment
fn fs_main(input: VSOut) -> @location(0) vec4<f32> {
    let n = normalize(input.normal);

    // Two directional lights
    let lightDir1 = normalize(vec3<f32>(0.2, 1.0, 0.3));
    let lightDir2 = normalize(vec3<f32>(-0.4, 0.7, -0.2));
    let color1    = vec3<f32>(1.0, 0.95, 0.9);
    let color2    = vec3<f32>(0.6, 0.7, 1.0);

    // Lambert
    let d1 = max(dot(n, lightDir1), 0.0);
    let d2 = max(dot(n, lightDir2), 0.0);

    // minimal ambient
    let ambient = 0.15;

    let finalColor = input.color * (ambient + 0.5*d1*color1 + 0.5*d2*color2);

    return vec4<f32>(finalColor, input.alpha);
}
"#
            )),
        });

        //
        // Pipeline
        //
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Pipeline Layout"),
            bind_group_layouts: &[&camera_bind_group_layout],
            push_constant_ranges: &[],
        });

        let render_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("Render Pipeline"),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader_module,
                entry_point: "vs_main",
                buffers: &[
                    // Cube geometry
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
                    // Instance data
                    InstanceData::desc(),
                ],
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader_module,
                entry_point: "fs_main",
                targets: &[Some(wgpu::ColorTargetState {
                    format: config.format,
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

        // Final
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

            sphere_instances,
            dust_orbits,
            dust_instances,
            instance_buffer,
            num_instances_total,

            // Generate a random u32 seed using the existing RNG
            let seed: u32 = rng.gen();
            
            // Initialize Perlin with the generated seed
            perlin: Perlin::new(seed),
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

            let depth_texture_desc = &wgpu::TextureDescriptor {
                label: Some("DepthTexture"),
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
            let depth_texture = self.device.create_texture(depth_texture_desc);
            self.depth_texture_view = depth_texture.create_view(&wgpu::TextureViewDescriptor::default());
        }
    }

    /// Update every frame
    fn update(&mut self) {
        self.time += 0.01;

        // 1) Update sphere color with Perlin noise that shifts over time
        // We sample noise in [x, y, z, time], then map to color
        let freq = 0.06; // scale of noise
        for inst in &mut self.sphere_instances {
            let x = inst.position[0] as f64 * freq;
            let y = inst.position[1] as f64 * freq;
            let z = inst.position[2] as f64 * freq;
            let t = self.time as f64 * 0.3; // shift over time
            // sample in 4D: (x, y, z+ t)
            let val = self.perlin.get([x, y, z + t]);
            // val in [-1..1], map to [0..1]
            let normalized = 0.5 * (val + 1.0);
            // We'll pick a color in e.g. green-blue-pinkish range
            let (r, g, b) = perlin_color_map(normalized as f32);
            inst.color = [r, g, b];
        }

        // 2) Update dust orbit
        for (i, orbit) in self.dust_orbits.iter_mut().enumerate() {
            orbit.angle += orbit.speed;
            let x = orbit.radius * orbit.angle.cos();
            let z = orbit.radius * orbit.angle.sin();
            let y = orbit.height;
            self.dust_instances[i].position = [x, y, z];
        }

        // 3) Combine into single instance buffer
        let combined = [
            self.sphere_instances.as_slice(),
            self.dust_instances.as_slice(),
        ]
        .concat();
        self.queue.write_buffer(
            &self.instance_buffer,
            0,
            bytemuck::cast_slice(&combined),
        );

        // 4) Recompute camera
        let aspect = self.config.width as f32 / self.config.height as f32;
        let fovy = 45f32.to_radians();
        let near = 0.1;
        let far = 2000.0;

        // Convert yaw/pitch/dist to eye pos
        let eye_x = self.camera_dist * self.camera_yaw.cos() * self.camera_pitch.cos();
        let eye_y = self.camera_dist * self.camera_pitch.sin();
        let eye_z = self.camera_dist * self.camera_yaw.sin() * self.camera_pitch.cos();
        let eye = Vec3::new(eye_x, eye_y, eye_z);

        let center = Vec3::ZERO;
        let up = Vec3::Y;

        let view = Mat4::look_at_rh(eye, center, up);
        let proj = Mat4::perspective_rh(fovy, aspect, near, far);
        let view_proj = proj * view;

        self.camera_uniform.view_proj = view_proj.to_cols_array();
        self.queue.write_buffer(
            &self.camera_buffer,
            0,
            bytemuck::cast_slice(&self.camera_uniform.view_proj),
        );
    }

    /// Render a frame
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
                        // Slightly dark background
                        load: wgpu::LoadOp::Clear(wgpu::Color {
                            r: 0.01,
                            g: 0.01,
                            b: 0.02,
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

    /// Handle input events
    fn input(&mut self, event: &WindowEvent) -> bool {
        match event {
            // Mouse press
            WindowEvent::MouseInput { state, button, .. } => {
                if *button == MouseButton::Left {
                    self.mouse_pressed = *state == ElementState::Pressed;
                }
                false
            }
            // Mouse move
            WindowEvent::CursorMoved { position, .. } => {
                if self.mouse_pressed {
                    // Drag => rotate
                    let (x, y) = (position.x, position.y);
                    if let Some((last_x, last_y)) = self.last_mouse_pos {
                        let dx = (x - last_x) as f32 * 0.005;
                        let dy = (y - last_y) as f32 * 0.005;
                        self.camera_yaw += dx;
                        self.camera_pitch -= dy;
                        // clamp pitch
                        self.camera_pitch = self.camera_pitch.clamp(-1.5, 1.5);
                    }
                    self.last_mouse_pos = Some((x, y));
                } else {
                    self.last_mouse_pos = Some((position.x, position.y));
                }
                false
            }
            // Scroll => zoom
            WindowEvent::MouseWheel { delta, .. } => {
                let scroll = match delta {
                    MouseScrollDelta::LineDelta(_, scroll) => *scroll,
                    MouseScrollDelta::PixelDelta(px) => px.y as f32 / 60.0,
                };
                self.camera_dist -= scroll * 2.0;
                if self.camera_dist < 10.0 {
                    self.camera_dist = 10.0;
                }
                if self.camera_dist > 300.0 {
                    self.camera_dist = 300.0;
                }
                false
            }
            _ => false,
        }
    }
}

//
// ========================= MAIN ENTRY =========================
//

fn main() {
    env_logger::init();
    pollster::block_on(run());
}

async fn run() {
    let event_loop = EventLoop::new();
    let window = WindowBuilder::new()
        .with_title("Noisy Sphere + Perlin Color + Orbiting Dust (wgpu)")
        .build(&event_loop)
        .unwrap();

    let mut state = State::new(&window).await;

    event_loop.run(move |event, _, control_flow| {
        match &event {
            Event::WindowEvent {
                ref event,
                window_id,
            } if *window_id == window.id() => {
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

                    // Handle mouse/pan input
                    _ => {
                        if state.input(event) {
                            // handled
                        }
                    }
                }
            }
            Event::RedrawRequested(_) => {
                state.update();
                match state.render() {
                    Ok(_) => {}
                    Err(wgpu::SurfaceError::Lost) => state.resize(state.size),
                    Err(wgpu::SurfaceError::OutOfMemory) => *control_flow = ControlFlow::Exit,
                    Err(e) => eprintln!("render error: {:?}", e),
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
// ========================= COLOR MAPPING =========================
//

/// Map a single float in [0..1] to some "cool" color range for the sphere.
fn perlin_color_map(t: f32) -> (f32, f32, f32) {
    // We'll do a simple gradient from greenish to pinkish to bluish, etc.
    // t in [0..1].
    // Let's do a few "key" points, then interpolate.
    // For a quick approach, we can just do a rainbow-like pattern in HSV or do a few ifs.
    // We'll do a quick manual gradient:

    // e.g. 0.0 => (0.2, 0.7, 0.3)  green
    // 0.5 => (0.8, 0.2, 0.7) pinkish
    // 1.0 => (0.2, 0.4, 0.9) bluish
    if t < 0.5 {
        let u = t * 2.0; // [0..1]
        let r = 0.2 + 0.6 * u; // 0.2 -> 0.8
        let g = 0.7 - 0.5 * u; // 0.7 -> 0.2
        let b = 0.3 + 0.4 * u; // 0.3 -> 0.7
        (r, g, b)
    } else {
        let u = (t - 0.5) * 2.0; // [0..1]
        let r = 0.8 - 0.6 * u; // 0.8 -> 0.2
        let g = 0.2 + 0.2 * u; // 0.2 -> 0.4
        let b = 0.7 + 0.2 * u; // 0.7 -> 0.9
        (r, g, b)
    }
}
