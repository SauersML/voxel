use std::iter;
use std::mem;
use wgpu::util::DeviceExt;
use winit::{
    dpi::PhysicalSize,
    event::*,
    event_loop::{ControlFlow, EventLoop},
    window::WindowBuilder,
};

// We'll use glam for vector/matrix math, and rand for the noisy color/sphere generation
use glam::{Mat4, Vec3};
use rand::Rng;

/// Vertex data for a single cube. We'll use one cube and then draw it many times via instancing.
#[rustfmt::skip]
const VERTICES: &[f32] = &[
    // Positions          // Normals         // UVs (unused but placeholder)
    // Each "face" has 4 vertices, total 6 faces * 4 = 24
    // Cube center at (0,0,0) with side length=1; extends from -0.5 to +0.5
    // Face +X
    0.5,  0.5, -0.5,     1.0,  0.0,  0.0,     1.0, 0.0,
    0.5, -0.5, -0.5,     1.0,  0.0,  0.0,     0.0, 0.0,
    0.5, -0.5,  0.5,     1.0,  0.0,  0.0,     0.0, 1.0,
    0.5,  0.5,  0.5,     1.0,  0.0,  0.0,     1.0, 1.0,
    // Face -X
   -0.5,  0.5,  0.5,    -1.0,  0.0,  0.0,     1.0, 0.0,
   -0.5, -0.5,  0.5,    -1.0,  0.0,  0.0,     0.0, 0.0,
   -0.5, -0.5, -0.5,    -1.0,  0.0,  0.0,     0.0, 1.0,
   -0.5,  0.5, -0.5,    -1.0,  0.0,  0.0,     1.0, 1.0,
    // Face +Y
   -0.5,  0.5,  0.5,     0.0,  1.0,  0.0,     1.0, 0.0,
   -0.5,  0.5, -0.5,     0.0,  1.0,  0.0,     0.0, 0.0,
    0.5,  0.5, -0.5,     0.0,  1.0,  0.0,     0.0, 1.0,
    0.5,  0.5,  0.5,     0.0,  1.0,  0.0,     1.0, 1.0,
    // Face -Y
    0.5, -0.5,  0.5,     0.0, -1.0,  0.0,     1.0, 0.0,
    0.5, -0.5, -0.5,     0.0, -1.0,  0.0,     0.0, 0.0,
   -0.5, -0.5, -0.5,     0.0, -1.0,  0.0,     0.0, 1.0,
   -0.5, -0.5,  0.5,     0.0, -1.0,  0.0,     1.0, 1.0,
    // Face +Z
    0.5,  0.5,  0.5,     0.0,  0.0,  1.0,     1.0, 0.0,
    0.5, -0.5,  0.5,     0.0,  0.0,  1.0,     0.0, 0.0,
   -0.5, -0.5,  0.5,     0.0,  0.0,  1.0,     0.0, 1.0,
   -0.5,  0.5,  0.5,     0.0,  0.0,  1.0,     1.0, 1.0,
    // Face -Z
   -0.5,  0.5, -0.5,     0.0,  0.0, -1.0,     1.0, 0.0,
   -0.5, -0.5, -0.5,     0.0,  0.0, -1.0,     0.0, 0.0,
    0.5, -0.5, -0.5,     0.0,  0.0, -1.0,     0.0, 1.0,
    0.5,  0.5, -0.5,     0.0,  0.0, -1.0,     1.0, 1.0,
];

/// Indices for the 12 triangles (2 per face, 6 faces) = 36 indices total.
#[rustfmt::skip]
const INDICES: &[u16] = &[
    0, 1, 2,  0, 2, 3,      // +X
    4, 5, 6,  4, 6, 7,      // -X
    8, 9, 10, 8, 10,11,     // +Y
    12,13,14, 12,14,15,     // -Y
    16,17,18, 16,18,19,     // +Z
    20,21,22, 20,22,23,     // -Z
];

/// The data we need per voxel instance: position and color. We'll build a per-instance transform in the shader.
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct InstanceData {
    position: [f32; 3],
    color: [f32; 3],
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
                // color
                wgpu::VertexAttribute {
                    offset: 4 * 3,
                    shader_location: 6,
                    format: wgpu::VertexFormat::Float32x3,
                },
            ],
        }
    }
}

/// Uniforms for our camera (MVP matrix).
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct CameraUniform {
    view_proj: [f32; 16],
}

/// The main rendering state.
struct State {
    surface: wgpu::Surface,
    device: wgpu::Device,
    queue: wgpu::Queue,
    config: wgpu::SurfaceConfiguration,
    size: PhysicalSize<u32>,

    render_pipeline: wgpu::RenderPipeline,
    depth_texture_view: wgpu::TextureView,

    vertex_buffer: wgpu::Buffer,
    index_buffer: wgpu::Buffer,
    num_indices: u32,

    instance_buffer: wgpu::Buffer,
    num_instances: u32,

    camera_uniform: CameraUniform,
    camera_buffer: wgpu::Buffer,
    camera_bind_group: wgpu::BindGroup,

    frame: f32, // We'll use this to rotate the camera slowly.
}

impl State {
    /// Create the State and all GPU resources.
    async fn new(window: &winit::window::Window) -> Self {
        // Instance, surface, adapter, device, queue
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
            .expect("Failed to find a suitable GPU adapter");
        let (device, queue) = adapter
            .request_device(
                &wgpu::DeviceDescriptor {
                    label: None,
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

        // Depth texture for 3D rendering
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

        // Create buffers for our cube
        let vertex_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Vertex Buffer"),
            contents: bytemuck::cast_slice(VERTICES),
            usage: wgpu::BufferUsages::VERTEX,
        });

        let index_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Index Buffer"),
            contents: bytemuck::cast_slice(INDICES),
            usage: wgpu::BufferUsages::INDEX,
        });
        let num_indices = INDICES.len() as u32;

        // Generate a noisy sphere set of instance data
        // We'll do a 32^3 region centered at the origin. 
        // radius ~ 12 plus some random "noise" (0..2).
        let mut instances = Vec::new();
        let mut rng = rand::thread_rng();
        let half = 16.0;
        for x in -16..16 {
            for y in -16..16 {
                for z in -16..16 {
                    let vx = x as f32 + 0.5;
                    let vy = y as f32 + 0.5;
                    let vz = z as f32 + 0.5;
                    let dist = (vx*vx + vy*vy + vz*vz).sqrt();
                    let noise = rng.gen_range(0.0..2.0);
                    if dist < 12.0 + noise {
                        // We keep this voxel
                        let r = rng.gen_range(0.0..1.0);
                        let g = rng.gen_range(0.0..1.0);
                        let b = rng.gen_range(0.0..1.0);
                        instances.push(InstanceData {
                            position: [vx, vy, vz],
                            color: [r, g, b],
                        });
                    }
                }
            }
        }

        let instance_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Instance Buffer"),
            contents: bytemuck::cast_slice(&instances),
            usage: wgpu::BufferUsages::VERTEX,
        });
        let num_instances = instances.len() as u32;

        // Camera uniform
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
                label: Some("Camera Bind Group Layout"),
                entries: &[
                    wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::VERTEX,
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
            label: Some("Camera Bind Group"),
            layout: &camera_bind_group_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: camera_buffer.as_entire_binding(),
            }],
        });

        // Create a basic shader
        let shader_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Voxel Shader"),
            source: wgpu::ShaderSource::Wgsl(std::borrow::Cow::Borrowed(
                r#"
@group(0) @binding(0)
var<uniform> viewProj: mat4x4<f32>;

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) normal: vec3<f32>,
    @location(1) fragColor: vec3<f32>,
};

struct Instance {
    @location(5) instancePos: vec3<f32>,
    @location(6) instanceColor: vec3<f32>,
};

@vertex
fn vs_main(
    @location(0) position: vec3<f32>,
    @location(1) normal: vec3<f32>,
    inst: Instance
) -> VertexOutput {
    var out: VertexOutput;
    let worldPos = position + inst.instancePos;
    out.clip_position = viewProj * vec4<f32>(worldPos, 1.0);
    out.normal = normal;
    out.fragColor = inst.instanceColor;
    return out;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    // simple directional shading for a "3D" feel
    let lightDir = normalize(vec3<f32>(0.5, 1.0, 0.3));
    let diffuse = max(dot(in.normal, lightDir), 0.0) * 0.8 + 0.2;
    return vec4<f32>(in.fragColor * diffuse, 1.0);
}
"#,
            )),
        });

        // Create pipeline layout
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Render Pipeline Layout"),
            bind_group_layouts: &[&camera_bind_group_layout],
            push_constant_ranges: &[],
        });

        // Create render pipeline
        let render_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("Render Pipeline"),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader_module,
                entry_point: "vs_main",
                buffers: &[
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
                            // uv (unused)
                            wgpu::VertexAttribute {
                                offset: (4 * 6) as wgpu::BufferAddress,
                                shader_location: 2,
                                format: wgpu::VertexFormat::Float32x2,
                            },
                        ],
                    },
                    InstanceData::desc(),
                ],
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader_module,
                entry_point: "fs_main",
                targets: &[Some(wgpu::ColorTargetState {
                    format: config.format,
                    blend: Some(wgpu::BlendState::ALPHA_BLENDING),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                strip_index_format: None,
                front_face: wgpu::FrontFace::Ccw,
                cull_mode: Some(wgpu::Face::Back),
                polygon_mode: wgpu::PolygonMode::Fill,
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
            instance_buffer,
            num_instances,
            camera_uniform,
            camera_buffer,
            camera_bind_group,
            frame: 0.0,
        }
    }

    /// Called when the window is resized.
    fn resize(&mut self, new_size: PhysicalSize<u32>) {
        if new_size.width > 0 && new_size.height > 0 {
            self.size = new_size;
            self.config.width = new_size.width;
            self.config.height = new_size.height;
            self.surface.configure(&self.device, &self.config);

            // Update depth texture
            let depth_texture_desc = &wgpu::TextureDescriptor {
                label: Some("Depth Texture"),
                size: wgpu::Extent3d {
                    width: self.config.width,
                    height: self.config.height,
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
            self.depth_texture_view =
                depth_texture.create_view(&wgpu::TextureViewDescriptor::default());
        }
    }

    /// Update (called every frame). We'll rotate the camera around the origin.
    fn update(&mut self) {
        // We'll spin around the Y axis
        self.frame += 0.01;
        let aspect = self.config.width as f32 / self.config.height as f32;
        let fovy = 45.0f32.to_radians();
        let near = 0.1;
        let far = 1000.0;

        // Build a simple orbit camera around the origin:
        let radius = 40.0;
        let eye = Vec3::new(
            self.frame.cos() * radius,
            10.0,
            self.frame.sin() * radius,
        );
        let center = Vec3::ZERO;
        let up = Vec3::Y;
        let view = Mat4::look_at_rh(eye, center, up);
        let proj = Mat4::perspective_rh(fovy, aspect, near, far);
        let vp = proj * view;

        self.camera_uniform.view_proj = vp.to_cols_array();
        // Copy to buffer
        self.queue.write_buffer(
            &self.camera_buffer,
            0,
            bytemuck::cast_slice(&self.camera_uniform.view_proj),
        );
    }

    /// Render a frame.
    fn render(&mut self) -> Result<(), wgpu::SurfaceError> {
        let output = self.surface.get_current_texture()?;
        let view = output
            .texture
            .create_view(&wgpu::TextureViewDescriptor::default());

        // Encoder
        let mut encoder =
            self.device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("Render Encoder"),
                });

        {
            let mut render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("Render Pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color {
                            r: 0.02,
                            g: 0.02,
                            b: 0.06,
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
            render_pass.set_pipeline(&self.render_pipeline);
            render_pass.set_bind_group(0, &self.camera_bind_group, &[]);
            render_pass.set_vertex_buffer(0, self.vertex_buffer.slice(..));
            render_pass.set_vertex_buffer(1, self.instance_buffer.slice(..));
            render_pass.set_index_buffer(self.index_buffer.slice(..), wgpu::IndexFormat::Uint16);
            render_pass.draw_indexed(0..self.num_indices, 0, 0..self.num_instances);
        }

        // Submit
        self.queue.submit(iter::once(encoder.finish()));
        output.present();

        Ok(())
    }
}

/// Entry point
fn main() {
    env_logger::init();
    pollster::block_on(run());
}

/// Set up winit, create the State, run the event loop.
async fn run() {
    let event_loop = EventLoop::new();
    let window = WindowBuilder::new()
        .with_title("Noisy Sphere Voxel Visualizer (wgpu)")
        .build(&event_loop)
        .unwrap();

    let mut state = State::new(&window).await;

    event_loop.run(move |event, _, control_flow| match event {
        Event::WindowEvent {
            ref event,
            window_id,
        } if window_id == window.id() => match event {
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
            _ => {}
        },
        Event::RedrawRequested(_) => {
            state.update();
            match state.render() {
                Ok(_) => {}
                // Reconfigure the surface if lost
                Err(wgpu::SurfaceError::Lost) => state.resize(state.size),
                // The system is out of memory, we should quit.
                Err(wgpu::SurfaceError::OutOfMemory) => *control_flow = ControlFlow::Exit,
                // All other errors (Outdated, Timeout) should be resolved by the next frame
                Err(e) => eprintln!("{:?}", e),
            }
        }
        Event::MainEventsCleared => {
            window.request_redraw();
        }
        _ => {}
    });
}
