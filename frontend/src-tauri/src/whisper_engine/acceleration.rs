use crate::audio::{GpuType, PerformanceTier};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WhisperCompiledBackend {
    Metal,
    Cuda,
    Vulkan,
    HipBlas,
    Cpu,
}

impl WhisperCompiledBackend {
    pub fn current() -> Self {
        if cfg!(feature = "cuda") {
            Self::Cuda
        } else if cfg!(feature = "vulkan") {
            Self::Vulkan
        } else if cfg!(feature = "hipblas") {
            Self::HipBlas
        } else if cfg!(target_os = "macos") || cfg!(feature = "metal") {
            Self::Metal
        } else {
            Self::Cpu
        }
    }

    pub fn as_str(self) -> &'static str {
        match self {
            Self::Metal => "Metal",
            Self::Cuda => "Cuda",
            Self::Vulkan => "Vulkan",
            Self::HipBlas => "HipBlas",
            Self::Cpu => "Cpu",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct WhisperContextAcceleration {
    pub compiled_backend: WhisperCompiledBackend,
    pub runtime_detected_gpu: GpuType,
    pub use_gpu: bool,
    pub flash_attn: bool,
    pub gpu_device: i32,
}

impl WhisperContextAcceleration {
    pub fn status_label(self) -> &'static str {
        match (self.compiled_backend, self.flash_attn) {
            (WhisperCompiledBackend::Metal, true) => "Metal GPU with Flash Attention (Ultra-Fast)",
            (WhisperCompiledBackend::Metal, false) => "Metal GPU acceleration",
            (WhisperCompiledBackend::Cuda, true) => "CUDA GPU with Flash Attention (Ultra-Fast)",
            (WhisperCompiledBackend::Cuda, false) => "CUDA GPU acceleration",
            (WhisperCompiledBackend::Vulkan, _) => "Vulkan GPU acceleration",
            (WhisperCompiledBackend::HipBlas, _) => "HIP BLAS GPU acceleration",
            (WhisperCompiledBackend::Cpu, _) => "CPU processing only",
        }
    }
}

pub fn whisper_context_acceleration_for(
    compiled_backend: WhisperCompiledBackend,
    runtime_detected_gpu: GpuType,
    performance_tier: PerformanceTier,
) -> WhisperContextAcceleration {
    let use_gpu = !matches!(compiled_backend, WhisperCompiledBackend::Cpu);
    let fast_tier = matches!(performance_tier, PerformanceTier::High | PerformanceTier::Ultra);
    let flash_attn = match compiled_backend {
        WhisperCompiledBackend::Metal | WhisperCompiledBackend::Cuda => fast_tier,
        WhisperCompiledBackend::Vulkan | WhisperCompiledBackend::HipBlas | WhisperCompiledBackend::Cpu => false,
    };

    WhisperContextAcceleration {
        compiled_backend,
        runtime_detected_gpu,
        use_gpu,
        flash_attn: use_gpu && flash_attn,
        gpu_device: 0,
    }
}

/// What the Whisper engine actually runs on, for display in the settings UI.
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize)]
pub struct WhisperRuntimeAcceleration {
    /// Compiled backend: "Cuda", "Metal", "Vulkan", "HipBlas" or "Cpu".
    pub backend: String,
    /// True when transcription is expected to run on a GPU.
    pub uses_gpu: bool,
    /// GPU name when the backend can report it (CUDA only).
    pub device: Option<String>,
    /// Short human-readable label, e.g. "NVIDIA GPU (CUDA): NVIDIA GeForce RTX 2080 Ti".
    pub label: String,
}

#[cfg(feature = "cuda")]
mod cuda_probe {
    use std::os::raw::{c_char, c_int};

    // Provided by the ggml CUDA backend that whisper-rs-sys links statically.
    extern "C" {
        fn ggml_backend_cuda_get_device_count() -> c_int;
        fn ggml_backend_cuda_get_device_description(device: c_int, description: *mut c_char, size: usize);
    }

    /// Names of the CUDA devices ggml can use (empty when none is usable).
    pub fn devices() -> Vec<String> {
        let count = unsafe { ggml_backend_cuda_get_device_count() };
        (0..count.max(0))
            .map(|device| {
                let mut buf = [0 as c_char; 256];
                unsafe { ggml_backend_cuda_get_device_description(device, buf.as_mut_ptr(), buf.len()) };
                unsafe { std::ffi::CStr::from_ptr(buf.as_ptr()) }
                    .to_string_lossy()
                    .trim()
                    .to_string()
            })
            .collect()
    }
}

/// Describe the hardware Whisper uses. For CUDA builds this asks the CUDA
/// runtime (once, then cached), so a missing or unusable GPU is reported as CPU.
pub fn whisper_runtime_acceleration() -> WhisperRuntimeAcceleration {
    static CACHE: std::sync::OnceLock<WhisperRuntimeAcceleration> = std::sync::OnceLock::new();
    CACHE.get_or_init(probe_runtime_acceleration).clone()
}

fn probe_runtime_acceleration() -> WhisperRuntimeAcceleration {
    let backend = WhisperCompiledBackend::current();
    let (uses_gpu, device, label) = match backend {
        #[cfg(feature = "cuda")]
        WhisperCompiledBackend::Cuda => match cuda_probe::devices().into_iter().next() {
            Some(name) => (true, Some(name.clone()), format!("NVIDIA GPU (CUDA): {}", name)),
            None => (false, None, "CPU (CUDA build, but no usable NVIDIA GPU was found)".to_string()),
        },
        #[cfg(not(feature = "cuda"))]
        WhisperCompiledBackend::Cuda => (true, None, "NVIDIA GPU (CUDA)".to_string()),
        WhisperCompiledBackend::Metal => (true, None, "Apple GPU (Metal)".to_string()),
        WhisperCompiledBackend::Vulkan => (true, None, "GPU (Vulkan)".to_string()),
        WhisperCompiledBackend::HipBlas => (true, None, "AMD GPU (ROCm)".to_string()),
        WhisperCompiledBackend::Cpu => (false, None, "CPU (this build has no GPU support)".to_string()),
    };
    WhisperRuntimeAcceleration { backend: backend.as_str().to_string(), uses_gpu, device, label }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn acceleration_vulkan_backend_ignores_runtime_cuda_flash_attention() {
        let params = whisper_context_acceleration_for(
            WhisperCompiledBackend::Vulkan,
            GpuType::Cuda,
            PerformanceTier::High,
        );

        assert_eq!(params.compiled_backend, WhisperCompiledBackend::Vulkan);
        assert_eq!(params.runtime_detected_gpu, GpuType::Cuda);
        assert!(params.use_gpu);
        assert!(!params.flash_attn);
    }

    #[test]
    fn acceleration_vulkan_backend_keeps_gpu_without_runtime_gpu_detection() {
        let params = whisper_context_acceleration_for(
            WhisperCompiledBackend::Vulkan,
            GpuType::None,
            PerformanceTier::Low,
        );

        assert!(params.use_gpu);
        assert!(!params.flash_attn);
    }

    #[test]
    fn acceleration_cuda_backend_enables_flash_attention_for_fast_tiers() {
        let high = whisper_context_acceleration_for(
            WhisperCompiledBackend::Cuda,
            GpuType::Cuda,
            PerformanceTier::High,
        );
        let ultra = whisper_context_acceleration_for(
            WhisperCompiledBackend::Cuda,
            GpuType::Cuda,
            PerformanceTier::Ultra,
        );

        assert!(high.use_gpu);
        assert!(high.flash_attn);
        assert!(ultra.use_gpu);
        assert!(ultra.flash_attn);
    }

    #[test]
    fn acceleration_cpu_backend_disables_gpu_and_flash_attention() {
        for runtime_gpu in [GpuType::None, GpuType::Cuda, GpuType::Vulkan] {
            let params = whisper_context_acceleration_for(
                WhisperCompiledBackend::Cpu,
                runtime_gpu,
                PerformanceTier::Ultra,
            );

            assert!(!params.use_gpu);
            assert!(!params.flash_attn);
        }
    }

    #[cfg(not(any(feature = "cuda", feature = "vulkan", feature = "hipblas", feature = "metal", target_os = "macos")))]
    #[test]
    fn runtime_acceleration_reports_cpu_for_cpu_builds() {
        let info = whisper_runtime_acceleration();
        assert_eq!(info.backend, "Cpu");
        assert!(!info.uses_gpu);
        assert!(info.label.starts_with("CPU"));
    }
}
