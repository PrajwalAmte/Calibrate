pub mod llamacpp;
#[cfg(target_os = "macos")]
pub mod metal;
pub mod onnxruntime;
pub mod subprocess;
pub mod tensorrt;
pub mod torchscript;
pub mod transformers;

use crate::bench::runtime::RuntimeDescriptor;

/// All registered runtimes, in display order. Availability and format
/// compatibility are checked by the harness before anything is created.
pub fn registry() -> Vec<RuntimeDescriptor> {
    #[allow(unused_mut)]
    let mut runtimes = vec![
        transformers::descriptor(),
        torchscript::descriptor(),
        onnxruntime::descriptor(),
        tensorrt::descriptor(),
        llamacpp::LlamaCppRuntime::descriptor(),
    ];
    #[cfg(target_os = "macos")]
    runtimes.push(metal::MetalRuntime::descriptor());
    runtimes
}

/// Names of every runtime, for error messages.
pub fn names() -> Vec<&'static str> {
    registry().iter().map(|d| d.name).collect()
}
