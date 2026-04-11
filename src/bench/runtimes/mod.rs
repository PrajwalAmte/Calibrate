pub mod candle;
pub mod llamacpp;
pub mod onnxruntime;
pub mod tensorrt;
pub mod torchscript;

use crate::bench::runtime::RuntimeDescriptor;

/// Return the full set of registered runtime descriptors.
///
/// The harness calls `is_available()` and `is_compatible()` on each descriptor
/// at startup to build the list of runtimes that will actually run. The order
/// here determines the display order in the output table.
pub fn registry() -> Vec<RuntimeDescriptor> {
    vec![
        candle::CandleRuntime::descriptor(),
        onnxruntime::OnnxRuntime::descriptor(),
        llamacpp::LlamaCppRuntime::descriptor(),
        torchscript::TorchScriptRuntime::descriptor(),
        tensorrt::TensorRtRuntime::descriptor(),
    ]
}
