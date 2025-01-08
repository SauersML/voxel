use std::iter;
use std::mem;
use wgpu::util::DeviceExt;
use winit::{
    dpi::PhysicalSize,
    event::*,
    event_loop::{ControlFlow, EventLoop},
    window::WindowBuilder,
};

// We'll use glam for vector/matrix math, and rand for color/position generation
use glam::{Mat4, Vec3};
use rand::Rng;

//
// GEOMETRY DATA (one cube, used by all voxels & dust)
//

// Each face has 4 vertices, and a cube has 6 faces => 24 vertices total
// Here, we store position, normal, and UV (although the UV isn't used)
#[rustfmt::skip]
const VERTICES: &[f32] = &[
    // Positions           // Normals          // UVs
    // Face +X
    0.5,  0.5, -0.5,      1.0,  0.0,  0.0,     1.0, 0.0,
    0.5, -0.5, -0.5,      1.0,  0.0,  0.0,     0.0, 0.0,
    0.5, -0.5,  0.5,      1.0,  0.0,  0.0,     0.0, 1.0,
    0.5,  0.5,  0.5,      1.0,  0.0,  0.0,     1.0, 1.0,

    // Face -X
   -0.5,  0.5,  0.5,     -1.0,  0.0,  0.0,     1.0, 0.0,
   -0.5, -0.5,  0.5,     -1.0,  0.0,  0.0,     0.0, 0.0,
   -0.5, -0.5, -0.5,     -1.0,  0.0,  0.0,     0.0, 1.0,
   -0.5,  0.5, -0.5,     -1.0,  0.0,  0.0,     1.0, 1.0,

    // Face +Y
   -0.5,  0.5,  0.5,      0.0,  1.0,  0.0,     1.0, 0.0,
   -0.5,  0.5, -0.5,      0.0,  1.0,  0.0,     0.0, 0.0,
    0.5,  0.5, -0.5,      0.0,  1.0,  0.0,     0.0, 1.0,
    0.5,  0.5,  0.5,      0.0,  1.0,  0.0,     1.0, 1.0,

    // Face -Y
    0.5, -0.5,  0.5,      0.0, -1.0,  0.0,     1.0, 0.0,
    0.5, -0.5, -0.5,      0.0, -1.0,  0.0,     0.0, 0.0,
   -0.5, -0.5, -0.5,      0.0, -1.0,  0.0,     0.0, 1.0,
   -0.5, -0.5,  0.5,      0.0, -1.0,  0.0,     1.0, 1.0,

    // Face +Z
    0.5,  0.5,  0.5,      0.0,  0.0,  1.0,     1.0, 0.0,
    0.5, -0.5,  0.5,      0.0,  0.0,  1.0,     0.0, 0.0,
   -0.5, -0.5,  0.5,      0.0,  0.0,  1.0,     0.0, 1.0,
   -0.5,  0.5,  0.5,      0.0,  0.0,  1.0,     1.0, 1.0,

    // Face -Z
   -0.5,  0.5, -0.5,      0.0,  0.0, -1.0,     1.0, 0.0,
   -0.5, -0.5, -0.5,      0.0,  0.0, -1.0,     0.0, 0.0,
    0.5, -0.5, -0.5,      0.0,  0.0, -1.0,     0.0, 1.0,
    0.5,  0.5, -0.5,      0.0,  0.0, -1.0,     1.0, 1.0,
];

// Each face = 2 triangles = 6 faces * 2 = 12 triangles => 36 indices
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
// PER-INSTANCE DATA
//
// We’ll store position, scale, color, and alpha for each instance.
// That way, we can draw both "voxels" and "dust" in the same draw call, 
// just with different scales/colors/alphas.
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
// CAMERA UNIFORM
//

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct CameraUniform {
    view_proj: [f32; 16],
}

//
// RENDERING STATE
//

struct State {
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

    // Instances (voxels + dust)
    instance_buffer: wgpu::Buffer,
    num_instances: u32,

    // Camera
    camera_uniform: CameraUniform,
    camera_buffer: wgpu::Buffer,
    camera_bind_group: wgpu::BindGroup,

    // Animation frame
    frame: f32,
}

impl State {
    /// Create the State and all GPU resources
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
            .expect("Failed to find a suitable GPU adapter");

        // Create device + queue
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

        // Surface configuration
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

        // Create vertex & index buffers (cube)
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
        // Generate noisy sphere + dust instance data
        //

        let mut instances: Vec<InstanceData> = Vec::new();
        let mut rng = rand::thread_rng();

        // 1) Voxel "noisy sphere"
        //    We'll sample in a 32^3 region around the origin
        //    radius ~ 12 plus up to +2 noise
        for x in -16..16 {
            for y in -16..16 {
                for z in -16..16 {
                    let vx = x as f32 + 0.5;
                    let vy = y as f32 + 0.5;
                    let vz = z as f32 + 0.5;
                    let dist = (vx * vx + vy * vy + vz * vz).sqrt();
                    let noise = rng.gen_range(0.0..2.0);
                    if dist < 12.0 + noise {
                        // Colors focusing on purples, pinks, greens, blues
                        // Example: pick a random hue in [180..360], random saturation/value
                        let hue = rng.gen_range(180.0..360.0);
                        let saturation = rng.gen_range(0.6..1.0);
                        let value = rng.gen_range(0.6..1.0);
                        let (r, g, b) = hsv_to_rgb(hue, saturation, value);

                        let voxel = InstanceData {
                            position: [vx, vy, vz],
                            scale: 1.0,     // normal voxel size
                            color: [r, g, b],
                            alpha: 1.0,     // fully opaque
                        };
                        instances.push(voxel);
                    }
                }
            }
        }

        // 2) Subtle dust particles
        //    We'll scatter, say, 300 dust cubes in a bigger region
        //    They are small and partially transparent.
        for _ in 0..300 {
            let dx = rng.gen_range(-30.0..30.0);
            let dy = rng.gen_range(-10.0..30.0);
            let dz = rng.gen_range(-30.0..30.0);

            let hue = rng.gen_range(180.0..360.0);
            let saturation = rng.gen_range(0.2..0.4);
            let value = rng.gen_range(0.7..1.0);
            let (r, g, b) = hsv_to_rgb(hue, saturation, value);

            let dust = InstanceData {
                position: [dx, dy, dz],
                scale: 0.2,         // smaller than a normal voxel
                color: [r, g, b],
                alpha: 0.15,        // mostly transparent
            };
            instances.push(dust);
        }

        let instance_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Instance Buffer"),
            contents: bytemuck::cast_slice(&instances),
            usage: wgpu::BufferUsages::VERTEX,
        });
        let num_instances = instances.len() as u32;

        //
        // Camera + Uniform
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
                label: Some("Camera Bind Group Layout"),
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
            label: Some("Camera Bind Group"),
            layout: &camera_bind_group_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: camera_buffer.as_entire_binding(),
            }],
        });

        //
        // WGSL SHADER
        //  - Accepts instance data for position + scale + color + alpha
        //  - Uses two directional lights + a rim light + a small glow factor
        //  - Also supports alpha blending for subtle dust
        //
        let shader_module = device.create_shader_module(&wgpu::ShaderModuleDescriptor {
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

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) normal: vec3<f32>,
    @location(1) fragColor: vec3<f32>,
    @location(2) fragAlpha: f32,
    @location(3) worldPos: vec3<f32>,
};

@vertex
fn vs_main(
    @location(0) inPos: vec3<f32>,
    @location(1) inNormal: vec3<f32>,
    @location(2) _inUV: vec2<f32>,
    inst: Instance
) -> VertexOutput {
    var out: VertexOutput;
    // Scale the cube by inst.scale, then translate
    let worldPos = (inPos * inst.scale) + inst.pos;
    out.clip_position = viewProj * vec4<f32>(worldPos, 1.0);

    // Normal is scaled by inst.scale too, though if scale is uniform, direction remains same
    out.normal = normalize(inNormal);
    out.fragColor = inst.color;
    out.fragAlpha = inst.alpha;
    out.worldPos = worldPos;
    return out;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    // Two directional lights for a more interesting look
    let lightDir1 = normalize(vec3<f32>(0.5, 1.0, 0.3));
    let lightDir2 = normalize(vec3<f32>(-0.8, 0.4, 0.2));
    let lightColor1 = vec3<f32>(1.0, 0.95, 0.9);
    let lightColor2 = vec3<f32>(0.6, 0.8, 1.0);
    let n = normalize(in.normal);

    let diff1 = max(dot(n, lightDir1), 0.0);
    let diff2 = max(dot(n, lightDir2), 0.0);
    let diffuse = diff1 * lightColor1 + diff2 * lightColor2;

    // Rim light (fake “Fresnel”)
    // We'll approximate the "view direction" as from above or from camera at +Z in local coords
    let viewDir = normalize(vec3<f32>(0.0, 0.0, 1.0));
    let rim = 1.0 - max(dot(n, viewDir), 0.0);
    let rim_color = vec3<f32>(1.0, 0.2, 0.9) * pow(rim, 3.0);

    // Slight "glow" factor
    let glow = 0.05;

    let base = in.fragColor;
    let color = base + diffuse + rim_color + glow;

    return vec4<f32>(color, in.fragAlpha);
}
"#
            )),
        });

        //
        // PIPELINE LAYOUT + RENDER PIPELINE
        //
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Render Pipeline Layout"),
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
                    wgpu::VertexBufferLayout {
                        array_stride: mem::size_of::<f32>() as wgpu::BufferAddress * 8,
                        step_mode: wgpu::VertexStepMode::Vertex,
                        attributes: &[
                            // position (x, y, z)
                            wgpu::VertexAttribute {
                                offset: 0,
                                shader_location: 0,
                                format: wgpu::VertexFormat::Float32x3,
                            },
                            // normal (x, y, z)
                            wgpu::VertexAttribute {
                                offset: (4 * 3) as wgpu::BufferAddress,
                                shader_location: 1,
                                format: wgpu::VertexFormat::Float32x3,
                            },
                            // uv (x, y) [unused]
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
                    // Enable alpha blending so dust can be partially transparent
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
                count: 1,              // no MSAA
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

    /// Resize callback
    fn resize(&mut self, new_size: PhysicalSize<u32>) {
        if new_size.width > 0 && new_size.height > 0 {
            self.size = new_size;
            self.config.width = new_size.width;
            self.config.height = new_size.height;
            self.surface.configure(&self.device, &self.config);

            // Recreate depth texture
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

    /// Called every frame, we spin around the Y-axis
    fn update(&mut self) {
        self.frame += 0.01;

        let aspect = self.config.width as f32 / self.config.height as f32;
        let fovy = 45.0f32.to_radians();
        let near = 0.1;
        let far = 1000.0;

        // Orbit around the origin
        let radius = 40.0;
        let eye = Vec3::new(
            self.frame.cos() * radius,
            20.0,
            self.frame.sin() * radius,
        );
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
        let output = self.surface.get_current_texture()?;
        let view = output.texture.create_view(&wgpu::TextureViewDescriptor::default());

        let mut encoder =
            self.device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("Render Encoder"),
                });

        {
            let mut render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("Main Render Pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        // Dark background with a slight bluish tint
                        load: wgpu::LoadOp::Clear(wgpu::Color {
                            r: 0.02,
                            g: 0.02,
                            b: 0.07,
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

            // Draw all instances at once
            render_pass.draw_indexed(0..self.num_indices, 0, 0..self.num_instances);
        }

        self.queue.submit(iter::once(encoder.finish()));
        output.present();

        Ok(())
    }
}

//
// ENTRY POINT
//

fn main() {
    env_logger::init();
    pollster::block_on(run());
}

async fn run() {
    let event_loop = EventLoop::new();
    let window = WindowBuilder::new()
        .with_title("Colorful Noisy Sphere + Dust (wgpu)")
        .build(&event_loop)
        .unwrap();

    let mut state = State::new(&window).await;

    event_loop.run(move |event, _, control_flow| match event {
        Event::WindowEvent {
            event,
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
                // If lost, reconfigure surface
                Err(wgpu::SurfaceError::Lost) => state.resize(state.size),
                // If out of memory, exit
                Err(wgpu::SurfaceError::OutOfMemory) => *control_flow = ControlFlow::Exit,
                // Other errors (Outdated, Timeout) are transient
                Err(e) => eprintln!("{:?}", e),
            }
        }
        Event::MainEventsCleared => {
            // Request a redraw
            window.request_redraw();
        }
        _ => {}
    });
}

//
// HELPER: Convert HSV -> RGB for pretty random colors
//
fn hsv_to_rgb(h: f32, s: f32, v: f32) -> (f32, f32, f32) {
    let c = v * s;
    let hh = h / 60.0;
    let x = c * (1.0 - ((hh % 2.0) - 1.0).abs());

    let (r1, g1, b1) = if hh < 1.0 {
        (c, x, 0.0)
    } else if hh < 2.0 {
        (x, c, 0.0)
    } else if hh < 3.0 {
        (0.0, c, x)
    } else if hh < 4.0 {
        (0.0, x, c)
    } else if hh < 5.0 {
        (x, 0.0, c)
    } else {
        (c, 0.0, x)
    };

    let m = v - c;
    (r1 + m, g1 + m, b1 + m)
}
