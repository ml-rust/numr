//! WGSL compute pipeline infrastructure
//!
//! Provides pipeline caching and dispatch utilities for WGSL compute shaders.
//! Follows the same pattern as CUDA kernel launchers for DRY consistency.

use parking_lot::Mutex;
use std::collections::HashMap;
use std::sync::Arc;
use wgpu::{
    BindGroup, BindGroupDescriptor, BindGroupEntry, BindGroupLayout, BindGroupLayoutDescriptor,
    BindGroupLayoutEntry, BindingType, Buffer, BufferBindingType, ComputePipeline,
    ComputePipelineDescriptor, Device, PipelineLayoutDescriptor, Queue, ShaderModule,
    ShaderModuleDescriptor, ShaderSource, ShaderStages,
};

use crate::dtype::DType;
use crate::error::{Error, Result};

/// Workgroup size for compute shaders (matches CUDA BLOCK_SIZE)
pub const WORKGROUP_SIZE: u32 = 256;

// ============================================================================
// Pipeline Cache
// ============================================================================

/// Cache for compute pipelines keyed by (shader_name, dtype)
pub struct PipelineCache {
    device: Arc<Device>,
    #[allow(dead_code)]
    queue: Arc<Queue>,
    /// Cached shader modules by name
    modules: Mutex<HashMap<&'static str, Arc<ShaderModule>>>,
    /// Cached pipelines by (shader_name, entry_point)
    pipelines: Mutex<HashMap<(&'static str, &'static str), Arc<ComputePipeline>>>,
    /// Cached bind group layouts by layout key
    layouts: Mutex<HashMap<LayoutKey, Arc<BindGroupLayout>>>,
}

/// Key for bind group layout cache
#[derive(Clone, Copy, PartialEq, Eq, Hash)]
pub struct LayoutKey {
    /// Number of storage buffers in the layout
    pub num_storage_buffers: u32,
    /// Number of uniform buffers in the layout
    pub num_uniform_buffers: u32,
}

impl PipelineCache {
    /// Create a new pipeline cache
    pub fn new(device: Arc<Device>, queue: Arc<Queue>) -> Self {
        Self {
            device,
            queue,
            modules: Mutex::new(HashMap::new()),
            pipelines: Mutex::new(HashMap::new()),
            layouts: Mutex::new(HashMap::new()),
        }
    }

    /// Get or create a shader module
    pub fn get_or_create_module(&self, name: &'static str, source: &str) -> Arc<ShaderModule> {
        let mut modules = self.modules.lock();
        if let Some(module) = modules.get(name) {
            return module.clone();
        }

        let module = self.device.create_shader_module(ShaderModuleDescriptor {
            label: Some(name),
            source: ShaderSource::Wgsl(source.into()),
        });

        let module = Arc::new(module);
        modules.insert(name, module.clone());
        module
    }

    /// Get or create a compute pipeline
    pub fn get_or_create_pipeline(
        &self,
        shader_name: &'static str,
        entry_point: &'static str,
        module: &ShaderModule,
        layout: &BindGroupLayout,
    ) -> Arc<ComputePipeline> {
        let key = (shader_name, entry_point);
        let mut pipelines = self.pipelines.lock();

        if let Some(pipeline) = pipelines.get(&key) {
            return pipeline.clone();
        }

        let pipeline_layout = self
            .device
            .create_pipeline_layout(&PipelineLayoutDescriptor {
                label: Some(&format!("{}_layout", shader_name)),
                bind_group_layouts: &[layout],
                immediate_size: 0, // Not using push constants
            });

        let pipeline = self
            .device
            .create_compute_pipeline(&ComputePipelineDescriptor {
                label: Some(&format!("{}_{}", shader_name, entry_point)),
                layout: Some(&pipeline_layout),
                module,
                entry_point: Some(entry_point),
                compilation_options: Default::default(),
                cache: None,
            });

        let pipeline = Arc::new(pipeline);
        pipelines.insert(key, pipeline.clone());
        pipeline
    }

    /// Get or create a bind group layout for storage buffers
    pub fn get_or_create_layout(&self, key: LayoutKey) -> Arc<BindGroupLayout> {
        let mut layouts = self.layouts.lock();

        if let Some(layout) = layouts.get(&key) {
            return layout.clone();
        }

        let mut entries = Vec::new();

        // Storage buffers (read-write)
        for i in 0..key.num_storage_buffers {
            entries.push(BindGroupLayoutEntry {
                binding: i,
                visibility: ShaderStages::COMPUTE,
                ty: BindingType::Buffer {
                    ty: BufferBindingType::Storage { read_only: false },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            });
        }

        // Uniform buffers (read-only params)
        for i in 0..key.num_uniform_buffers {
            entries.push(BindGroupLayoutEntry {
                binding: key.num_storage_buffers + i,
                visibility: ShaderStages::COMPUTE,
                ty: BindingType::Buffer {
                    ty: BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            });
        }

        let layout = self
            .device
            .create_bind_group_layout(&BindGroupLayoutDescriptor {
                label: Some("linalg_layout"),
                entries: &entries,
            });

        let layout = Arc::new(layout);
        layouts.insert(key, layout.clone());
        layout
    }

    /// Create a bind group from buffers
    pub fn create_bind_group(&self, layout: &BindGroupLayout, buffers: &[&Buffer]) -> BindGroup {
        let entries: Vec<BindGroupEntry> = buffers
            .iter()
            .enumerate()
            .map(|(i, buffer)| BindGroupEntry {
                binding: i as u32,
                resource: buffer.as_entire_binding(),
            })
            .collect();

        self.device.create_bind_group(&BindGroupDescriptor {
            label: Some("compute_bind_group"),
            layout,
            entries: &entries,
        })
    }

    /// Get device reference
    pub fn device(&self) -> &Device {
        &self.device
    }
}

// ============================================================================
// Dispatch Helpers
// ============================================================================

/// Compute number of workgroups for n elements
#[inline]
pub fn workgroup_count(n: usize) -> u32 {
    ((n as u32) + WORKGROUP_SIZE - 1) / WORKGROUP_SIZE
}

/// Map DType to WGSL entry point suffix
#[allow(dead_code)]
pub fn dtype_suffix(dtype: DType) -> Result<&'static str> {
    match dtype {
        DType::F32 => Ok("f32"),
        DType::F64 => Err(Error::UnsupportedDType {
            dtype,
            op: "WGSL (f64 not supported in WebGPU)",
        }),
        _ => Err(Error::UnsupportedDType {
            dtype,
            op: "linalg",
        }),
    }
}

/// Get entry point name for operation and dtype
#[allow(dead_code)]
pub fn entry_point(op: &str, dtype: DType) -> Result<String> {
    let suffix = dtype_suffix(dtype)?;
    Ok(format!("{}_{}", op, suffix))
}
