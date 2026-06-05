# Learnings from Building calibrate

Discoveries, surprises, and things that broke during development of `watch`, `bench`, and `plan`. Ordered roughly by the sequence they were encountered during development.

---

### Rust module system

The compiler does not discover files automatically. Every directory needs a `mod.rs` (or a file named after the directory, e.g. `bench.rs` at the parent level), and every submodule must be declared explicitly with `pub mod foo;` in its parent. This felt like ceremony at first.

Understanding why it works this way changed the perspective. Because the module tree is explicit and complete, the compiler knows the entire dependency graph before evaluating a single expression. That is why borrow errors can be so precise — the compiler is not guessing at scope; it has a complete, unambiguous picture of every path through the codebase.

The practical consequence: there is no "implicit re-export". If `plan/providers/runpod.rs` defines a struct, nothing outside `plan/providers/` can see it unless `mod.rs` re-exports it with `pub use`. This forced a conscious decision about every public API, which turned out to be useful rather than annoying. The module boundary is also the unit of visibility — `pub(crate)` means visible everywhere in the binary, `pub(super)` means visible only to the parent module, and unmarked items are private to their own file.

---

### `#[derive(Debug, Clone, Serialize, Deserialize)]`

The instinct was to add derives only when needed, treating them as a precision tool. That was wrong. The cost is four words per struct. The benefit is avoiding a constant tax of going back to add them later.

`Debug` enables `{:?}` and `{:#?}` formatting. Without it, `println!` and `dbg!` do not compile. The first time a large struct was missing `Debug` and the compiler error was not immediately obvious, the derive got added to every struct going forward.

`Clone` removes a significant class of borrow-checker friction. When data needs to cross a module boundary, be captured by a closure, or be stored in two places simultaneously, the borrow checker will reject it if there is only one owner. `.clone()` on a non-`Clone` type does not compile. Having it available by default means the question becomes "should this be cloned here?" rather than "why doesn't this compile?"

`Serialize`/`Deserialize` from `serde` unlock JSON serialization and deserialization for free. Every report struct in calibrate — `BenchReport`, `PlanReport` — is just `serde_json::to_string_pretty(&report)?`. The test suite uses `serde_json::from_str` to round-trip structs and verify field names. Two field name mismatches were caught this way that would have been silent bugs for any JSON consumer.

---

### `Result<T, E>` and the `?` operator

Rust has no exceptions. Every function that can fail returns `Result<T, E>`, where `T` is the success value and `E` is the error type. The caller must handle both cases — the compiler will not let a `Result` be silently ignored.

The `?` operator is syntactic sugar for: if this is `Ok(v)`, unwrap and continue with `v`; if this is `Err(e)`, return `Err(e.into())` immediately from the current function. The `.into()` is important — it means `?` can convert between compatible error types automatically.

Combined with `anyhow::Result<T>` (which uses `Box<dyn Error>` as the error type), `?` becomes a universal propagator. Any error type that implements `std::error::Error` can be promoted to `anyhow::Error` with `?`. Adding `.context("message")` or `.with_context(|| format!("..."))`  wraps the error with call-site information that appears in the chain when the error is printed.

The practical result: most command functions in calibrate read as a straight-line happy path with `?` on every fallible call, and errors print with full context chains without any explicit error-handling code. The only `match` blocks on errors are in tests and in one place where the code genuinely needs to distinguish variants.

---

### `clap` derive macros

`clap` has two APIs: a builder API where argument parsers are constructed programmatically, and a derive API where the CLI is specified as annotated Rust structs and enums.

The derive API was chosen and there was no reason to look back. The pattern:

```rust
#[derive(Parser)]
struct PlanArgs {
    #[arg(long)]
    model: String,

    #[arg(long, default_value_t = 1)]
    epochs: u32,

    #[arg(long, value_enum, default_value_t = FinetuneMethod::Lora)]
    method: FinetuneMethod,
}

#[derive(Clone, clap::ValueEnum)]
enum FinetuneMethod { Full, Lora, Qlora }
```

`#[derive(Parser)]` generates the parser. `#[arg(long)]` makes the field a `--long-flag`. `default_value_t` sets a typed default. `value_enum` tells clap to accept the enum's variant names as strings (kebab-cased by default). The generated `--help` output is assembled from field names, types, defaults, and any `#[arg(help = "...")]` annotations — all automatically.

The three calibrate subcommands (`watch`, `bench`, `plan`) are a nested `Commands` enum. Adding a new subcommand is one enum variant and one `match` arm in `main.rs`. The entire CLI evolved from a single subcommand to three with ~20 total flags without a single line of manual argument parsing.

---

### String formatting with named arguments

`format!("{base} Note: {extra}")` is Rust's equivalent of Python f-strings. Named variables from the local scope are captured directly without positional indexing. This was a pleasant surprise — it was expected to require something like `format!("{0} Note: {1}", base, extra)`.

The more substantive discovery was Rust's format alignment specifiers. The general form is `{value:fill_char align width.precision}`:
- `{:<9}` — left-align in a field of width 9
- `{:>7.2}` — right-align a float with 2 decimal places in a field of width 7
- `{:-<25}` — left-align and pad with `-` characters instead of spaces

The entire bench and plan terminal tables are built with these format strings. No external table library was needed. The column widths are hardcoded constants chosen to fit the expected data ranges. This is less flexible than a proper table renderer but the code is ten lines instead of a library dependency.

One gotcha: format string width specifiers take a constant or a variable via the `width$` syntax (`{:<width$}` where `width` is a `usize` in scope). Dynamic column widths in the bench table use this to right-size the "Runtime" column based on the longest runtime name in the result set.

---

### Ownership and the borrow checker in practice

The borrow checker enforces one rule: there can be any number of shared (immutable) references to a value, *or* exactly one exclusive (mutable) reference, but never both simultaneously. This prevents data races at compile time with no runtime cost.

The friction in practice is almost never the rule itself — it is data that is owned in the wrong place. The typical failure pattern: a struct holds a value, multiple parts of the codebase want to read from it and one wants to write to it, and the borrow checker rejects the combination. The temptation is to sprinkle `.clone()` until it compiles. The right response is to ask whether that value should even be shared state, or whether it should be passed explicitly.

In calibrate, the session state went through two rewrites because the initial design had `SessionSnapshot` passing through too many layers by reference. The fix was making `SessionSnapshot` cheap to clone (`#[derive(Clone)]` on all its fields) and passing owned copies across boundaries instead of shared references. After that, the borrow checker stopped being an obstacle.

`Arc<T>` (atomically reference-counted shared ownership) and `parking_lot::RwLock<T>` (a read-write lock) appear in the NVML collector for data that genuinely needs shared access across threads. `Arc<RwLock<MetricsSnapshot>>` is the idiom: multiple readers can hold a read lock simultaneously, one writer can hold the write lock exclusively. Outside of that pattern, ownership is passed, not shared.

---

### Traits as interfaces

A trait is a set of method signatures that a type promises to implement. Any type that implements a trait can be used wherever the trait is expected — this is Rust's equivalent of an interface.

`bench/runtime.rs` defines:

```rust
pub trait Runtime: Send {
    fn generate_input(&self, batch: usize, seq_len: usize) -> BenchInput;
    fn load(&mut self, model_path: &Path) -> Result<()>;
    fn infer(&mut self, input: &BenchInput) -> Result<Duration>;
    fn pre_collected_samples(&mut self) -> Option<(Vec<u64>, f64, u64)> { None }
    fn teardown(&mut self) {}
}
```

Five backends implement this trait: Candle, ONNX Runtime, TorchScript, llama.cpp, TensorRT. The harness holds `Vec<Box<dyn Runtime>>` — a list of trait objects. At runtime, method calls dispatch through a vtable to the correct implementation. The harness has no `if backend == "candle"` logic anywhere.

`Box<dyn Trait>` is heap allocation + dynamic dispatch (vtable). `impl Trait` in a function argument is static dispatch (monomorphized at compile time, zero overhead). The bench harness uses `Box<dyn Runtime>` because the list of backends is runtime-determined. The output renderers use `impl Renderer` where possible to keep the stack allocation and avoid unnecessary heap allocation.

The revealing moment: writing a test-only `MockRuntime` that implements `Runtime` with trivially controlled return values. The harness tests run against this mock without touching a real model file. Same harness code, zero binary changes — that is what the interface boundary buys.

---

### `async/await` and the `tokio` runtime

Async Rust is cooperative multitasking. An `async fn` returns a `Future` — a value that can be polled for completion. `.await` polls the future and, if it is not ready, suspends the current task and yields back to the executor. The executor (tokio) picks another task that is ready to make progress. When the awaited future eventually becomes ready, the executor resumes the suspended task from where it left off.

The key distinction from OS threads: no stack is allocated per suspended task. A suspended future is just a state machine stored on the heap. This is why tokio can drive tens of thousands of concurrent futures on a small fixed-size thread pool.

The confusing part: blocking code inside an async context. If a task calls `std::thread::sleep` or any blocking syscall, it does not suspend cooperatively — it literally occupies a tokio thread for the duration, starving other tasks. The rule: inside `async` code, never call anything that blocks. Use `tokio::time::sleep` instead of `std::thread::sleep`. Use `tokio::fs` instead of `std::fs`. For unavoidably blocking work (C FFI, synchronous libraries), use `tokio::task::spawn_blocking` to run the work on a separate thread pool that does not starve the async executor.

The NVML case was more involved than `spawn_blocking` because NVML needs to be polled continuously on its own thread, not dispatched occasionally. That is why it gets a dedicated `std::thread`. See the NVML threading section below.

---

### `tokio::join!` for parallel async work

Sequential awaiting executes futures one after another:

```rust
let runpod = fetch_runpod().await?;   // waits ~800ms
let lambda = fetch_lambda().await?;   // then waits ~600ms
let vastai  = fetch_vastai().await?;  // then waits ~400ms
// total: ~1800ms
```

`tokio::join!` runs all three concurrently and returns when all complete:

```rust
let (runpod, lambda, vastai) = tokio::join!(
    fetch_runpod(),
    fetch_lambda(),
    fetch_vastai(),
);
// total: ~800ms (the slowest one)
```

The crucial difference from Python's `asyncio.gather`: `join!` is a *macro*, not a function. It expands at compile time into a state machine that polls each future in turn, so there is no heap allocation and no boxing of futures. The performance is equivalent to hand-written state machine code.

The subtlety that was not obvious upfront: `join!` does not short-circuit on errors — all three arms run to completion regardless of individual failures. Each returns a `Result`, and the error handling happens after the join. This is exactly the right behavior for provider fetching — a Vast.ai timeout should not abort the RunPod and Lambda requests that are already in flight. The design came from choosing the right primitive rather than from deliberate error-handling design.

---

### `serde` and round-trip serialization

`serde` is a framework for serializing and deserializing Rust data structures. The `Serialize` and `Deserialize` derives generate the code to convert a struct to and from any format that has a serde implementation — JSON, TOML, MessagePack, CBOR, and many others. The `serde_json` crate provides the JSON format.

The generated code handles nested structs, enums, `Option<T>` (serialized as `null` or the value), `Vec<T>`, `HashMap<K,V>`, and primitive types automatically. Enum variants serialize as their string names by default, which is why `AvailabilityStatus::Available` serializes as `"Available"` in the JSON output.

Customization attributes worth knowing:
- `#[serde(rename = "param_count")]` — use a different key name in the serialized form
- `#[serde(skip_serializing_if = "Option::is_none")]` — omit the field entirely when it is `None`
- `#[serde(rename_all = "camelCase")]` on the struct — convert all field names to camelCase
- `#[serde(default)]` on a field — use `Default::default()` when the field is missing during deserialization

The round-trip test pattern (`serialize → deserialize → assert field equality`) caught two real bugs: one where a field was renamed in the struct but the JSON consumer expected the old name, and one where an `Option` field was serializing as `null` but the consumer expected the key to be absent entirely. Both were invisible until a round-trip test existed.

---

### Error handling: `thiserror` vs `anyhow`

`thiserror` derives `std::error::Error` on a custom enum, allowing structured error types with typed variants:

```rust
#[derive(thiserror::Error, Debug)]
pub enum CalibrateError {
    #[error("process {pid} not found")]
    ProcessNotFound { pid: u32 },
    #[error("NVML unavailable: {0}")]
    NvmlUnavailable(String),
}
```

Callers can `match` on `CalibrateError::ProcessNotFound` and handle each case differently. This is the right tool when the error type is part of a public API or when call sites genuinely need to distinguish variants.

`anyhow` takes the opposite approach: `anyhow::Error` is an opaque, heap-allocated error that can wrap any `std::error::Error`. It is designed for propagation, not matching. The strength is `.context()`:

```rust
resolve_model(&args.model)
    .with_context(|| format!("could not resolve model '{}'", args.model))?;
```

This adds a human-readable message at every level of the call stack. When the error prints, the full chain appears: "could not resolve model 'llama3-8b': HTTP request failed: connection refused".

The calibrate codebase uses both: `thiserror` for `error.rs` (the domain boundary where test code matches on variants) and `anyhow::Result` in all command code (where errors are propagated to the user, not matched on). The decision about which to use at any given site: is the caller going to `match` on error variants? `thiserror`. Is the error just going to be propagated upward and eventually displayed? `anyhow`.

---

### HDR histograms for latency measurement

A naïve latency accumulator collects a `Vec<u64>` of nanosecond samples and sorts them to compute percentiles. This works but has two problems: memory grows linearly with sample count, and the percentile calculation requires a full sort of all samples.

`hdrhistogram` uses a different representation: a fixed-size array of buckets where each bucket covers a range of values. The bucket structure provides configurable relative precision (e.g., 1% error) over a dynamic range (e.g., 1 nanosecond to 60 seconds). Memory usage is constant regardless of sample count. Percentile queries are O(1) — scan the bucket array until the cumulative count exceeds the requested percentile.

Construction requires specifying the maximum trackable value:

```rust
let mut hist = Histogram::<u64>::new_with_bounds(1, 60_000_000_000, 3).unwrap();
//                                                   ^                ^
//                                            max 60 seconds     3 significant figures
```

Setting the maximum too low causes silently saturated samples — outliers above the max are recorded at the max value rather than rejected or flagged. This is correct behavior but surprising if not anticipated.

The `bench/stats.rs` `StatAccumulator` wraps HDR histogram and exposes `p50()`, `p95()`, `p99()`, `min()`, `max()`, and `mean()`. The mean comes from a separate running sum because HDR histogram's mean is less precise (it is derived from bucket midpoints rather than exact values).

---

### Process metrics via `/proc`

Linux exposes every running process's state through the `/proc` virtual filesystem. No system calls beyond `open` and `read` are needed — the kernel populates the files on demand.

Key files:
- `/proc/<pid>/stat` — 52 space-separated fields including process state, CPU time (in clock ticks), virtual memory size, and start time
- `/proc/<pid>/status` — human-readable key-value pairs including VmRSS (resident set size), VmPeak, and thread count
- `/proc/<pid>/cmdline` — the full command line, null-byte separated

The parsing trap in `/proc/<pid>/stat`: field 2 is the process name in parentheses and can contain spaces and even parentheses. Splitting naively on whitespace gives the wrong field indices for everything after field 2. The correct approach: find the index of the last `)` character, split that suffix on whitespace, and index from there. This is documented in `man 5 proc` but not prominent enough to avoid the mistake on first read.

CPU utilization requires two readings, not one: record `(utime + stime)` at time T1, sleep, record again at T2, then divide the delta by the elapsed time scaled by the clock tick rate (`sysconf(_SC_CLK_TCK)`, typically 100 on Linux). This is what `top` and `ps` do. They are not magical — they are reading the same files.

---

### Hexagonal (ports and adapters) architecture

The core domain — `session/`, `metrics/`, `analysis/` — has no imports from `collectors/` or `output/`. It defines data types and transformations. `collectors/` translates hardware APIs (NVML, `/proc`) *into* those types. `output/` translates those types *into* terminal renders or JSON. The domain is the center; everything else is an adapter at the edge.

The immediate practical benefit: tests for `session/lifecycle.rs` pass in a `Vec<MetricsSnapshot>` directly without touching NVML. The session state machine, the MFU calculation, the sliding window, the bottleneck classifier — all of this is tested against controlled fake data and the tests run in milliseconds on any machine.

The benefit that was not anticipated: when a second output format (JSON) and a third output format (markdown for bench) needed adding, there was no session logic to touch. Adding an output format is adding a file in `output/` and a match arm. Adding a new collector (the CPU-only fallback) required no changes to session logic. The domain boundary enforces a discipline that makes extension cheap.

The cost that was felt: the number of types increases. There is a `GpuSample` (raw NVML data), a `MetricsSnapshot` (domain computation result), and a `SessionSnapshot` (everything a renderer needs). Three types where one "clever" implementation might have used one. Each boundary has a conversion function. For a project of this size, the verbosity is manageable and each type has a clear responsibility.

---

### Typestate pattern for session lifecycle

The naive session implementation used a state enum:

```rust
enum SessionState { NotStarted, Running, Ended }

struct Session {
    state: SessionState,
    // ...
}

impl Session {
    fn snapshot(&self) -> Result<SessionSnapshot> {
        if self.state != SessionState::Running {
            return Err(anyhow!("session is not running"));
        }
        // ...
    }
}
```

This compiles and works, but it means a programming error — calling `snapshot()` after the session ended — becomes a runtime error. The compiler accepts the call.

The typestate pattern encodes state in the type parameter:

```rust
struct Session<S> { state: S, data: SessionData }
struct NotStarted;
struct Running { collector: Box<dyn Collector> }
struct Ended;

impl Session<Running> {
    fn snapshot(&self) -> SessionSnapshot { ... }
}
// snapshot() doesn't exist on Session<NotStarted> or Session<Ended>
```

Now calling `snapshot()` on an ended session does not compile. The state machine transitions are functions that consume the current state and return the next:

```rust
impl Session<NotStarted> {
    fn start(self, collector: Box<dyn Collector>) -> Session<Running> { ... }
}
impl Session<Running> {
    fn end(self) -> Session<Ended> { ... }
}
```

`self` is consumed (moved) by each transition, so it is impossible to call `start()` twice — the first call moves `self` and the second call has no value to operate on.

The trade-off: the type signatures become complex (`fn takes_session(s: &Session<Running>)`), and functions that work on sessions in any state need either generics or separate implementations. For the calibrate session lifecycle, where invalid transitions have real consequences (wasted GPU time, incorrect metrics), the compile-time enforcement is worth the verbosity.

---

### NVML threading model

NVML (NVIDIA Management Library) is a C library. The Rust wrapper `nvml-wrapper` exposes safe bindings but cannot change the underlying threading constraints: NVML must be initialized once per process, and per-device handles should be used from the thread that created them, or with explicit per-thread context setup.

The first attempt called NVML from inside `tokio::spawn`. This caused non-deterministic panics with errors like "nvml not initialized on this thread". The issue: `tokio::spawn` tasks can migrate between threads in the thread pool. A handle created on thread A might be polled later on thread B.

The fix: a dedicated `std::thread` that owns all NVML state for its entire lifetime. The thread runs a polling loop, constructs `MetricsSnapshot` values from NVML readings, and sends them over a `flume::Sender<MetricsSnapshot>` channel. The async runtime receives from a `flume::Receiver` on the async side.

```
std::thread (owns NVML)
  └─ poll every N ms
  └─ flume::Sender ──────────────────────►  tokio task
                                             flume::Receiver
                                             └─ select! in watch loop
```

`flume` was chosen over `std::sync::mpsc` because it provides an async-compatible receiver (`receiver.recv_async().await`) that integrates cleanly with the tokio select loop without blocking a thread. `std::sync::mpsc::Receiver::recv()` is blocking and would starve the tokio executor.

The general pattern here — dedicated `std::thread` for blocking/unsafe work, channel to cross into async — is the correct solution for any FFI library with threading constraints.

---

### Dual-path measurement for subprocess runtimes

The bench harness was written first for Candle, which loads a SafeTensors file in-process. The measurement loop is straightforward:

```
for each warmup iteration: call infer(), discard
for each measurement iteration:
    t0 = Instant::now()
    infer()
    samples.push(t0.elapsed().as_nanos())
```

Then the ONNX backend was added. ONNX Runtime does not have Rust bindings — it is invoked via a Python subprocess that loads the model, runs inference, prints timing results, and exits. The first run of the harness against this backend produced data like: `[892ms, 0ms, 0ms, 0ms, ...]`. One real sample, 99 zeros.

The reason: `infer()` for the ONNX backend spawns a subprocess and waits for it to exit. The subprocess runs all 100 iterations internally, prints a summary, and exits. From the harness perspective, the first call to `infer()` blocks for the entire run and returns the subprocess's total time. Subsequent calls spawn a new subprocess that finds no model path remaining and returns immediately.

The fix introduced a new method on the `Runtime` trait:

```rust
fn pre_collected_samples(&mut self) -> Option<(Vec<u64>, f64, u64)> {
    None  // default: in-process runtime, use the standard loop
}
```

Subprocess runtimes override this: on the first `infer()` call, they run the subprocess, parse its output into a `Vec<u64>` of nanosecond samples, a peak VRAM reading in GiB, and a total-tokens count, then cache everything. When `pre_collected_samples()` is called, they return the cached data. The harness checks the return value after the first `infer()`:

```rust
if let Some(samples) = runtime.pre_collected_samples() {
    // delegated path: use subprocess's own measurements
} else {
    // in-process path: run the measurement loop
}
```

In hindsight, the distinction should have been in the trait design from the start. The mistake made the correct abstraction obvious in a way that upfront design reasoning did not.

---

### OOM detection without panicking

The initial expectation: a model that does not fit in VRAM would cause `load()` or `infer()` to return `Err(...)` with a descriptive message. In practice, CUDA OOM errors surface in at least four different ways depending on the backend and CUDA version:

1. **In-process panic** — Candle's allocator panics rather than returning an error in some code paths
2. **`Err` with a message string** — other in-process paths return `Err("CUDA out of memory")` or `Err("OOM")`
3. **Subprocess non-zero exit** — the Python subprocess exits with code 1 and prints the OOM message to stderr
4. **Subprocess exit code 137** — the OS kills the process for exceeding memory limits (SIGKILL, exit 128+9)

The harness handles all four:

```rust
// Catch in-process panics
let result = std::panic::catch_unwind(AssertUnwindSafe(|| runtime.infer(&input)));
match result {
    Err(_)           => { /* panic = OOM or crash */ }
    Ok(Err(e)) if is_oom_error(&e) => { /* Err with OOM message */ }
    Ok(Err(e))       => { /* other error */ }
    Ok(Ok(duration)) => { /* success */ }
}

fn is_oom_error(e: &anyhow::Error) -> bool {
    let msg = e.to_string().to_lowercase();
    msg.contains("out of memory") || msg.contains("oom") || msg.contains("cuda error")
}
```

The important design decision: OOM is recorded as data, not failure. `BenchResult { oom: true, runtime: "onnxruntime", ... }` appears in the report table with an `!` marker. The bench continues with the next runtime. This required treating OOM as a known outcome of the measurement process rather than an exceptional error.

---

### FLOP-based training duration estimation

The floating-point operation count for one forward pass through a transformer is approximately:

$$\text{FLOPs}_{\text{forward}} = 2 \times P \times L_{\text{seq}}$$

where $P$ is parameter count and $L_\text{seq}$ is sequence length. The factor of 2 comes from the multiply-add (each MAC = 2 FLOPs) nature of matrix multiplication. The backward pass requires approximately $2\times$ the FLOPs of the forward pass (computing gradients for both activations and weights). The total per-step cost is therefore approximately $6 \times P \times L_\text{seq}$.

Total training FLOPs:

$$\text{FLOPs}_{\text{total}} = 6 \times P \times L_\text{seq} \times \frac{D}{B} \times E$$

where $D$ is dataset rows, $B$ is batch size, and $E$ is epochs. Wall-clock time follows from:

$$t = \frac{\text{FLOPs}_{\text{total}}}{\text{GPU TFLOPS} \times \text{MFU}}$$

The formula comes from Kaplan et al. (2020) and Hoffmann et al. (2022). It has known limitations: it ignores attention FLOP complexity (which is $O(L^2)$ and dominates at long sequences), data loading overhead, and optimizer step cost. For typical fine-tuning workloads with sequences under 2048 tokens, the approximation is within 20–30% of observed wall-clock time.

The MFU sensitivity was underestimated initially. The default assumption is 30% MFU (conservative for a fine-tuning run on a single GPU with suboptimal batch size). At 45% MFU — achievable with Unsloth's kernel optimizations — the estimated duration shrinks by 33%. This is why the integration between `calibrate watch` (which measures actual MFU) and `calibrate plan --mfu` (which uses that measurement) is more consequential than it first appeared.

Presenting the result as a `[0.8×, 1.5×]` range was a late decision, prompted by noticing that a point estimate of "1.87 hours" communicates false precision. The range acknowledges that the model is an approximation.

---

### VRAM component modelling for fine-tuning

The initial VRAM estimate was `params × bytes_per_param`. This is wrong by 3–6× in realistic fine-tuning scenarios because it accounts for model weights only. The full VRAM budget during training has five components:

**Weights**: `params × bytes_per_param`. For fp16/bf16, 2 bytes per parameter. For 4-bit quantization, 0.5 bytes. A 7B parameter model in fp16 occupies 14 GiB.

**Gradients**: In full fine-tuning, gradient tensors have the same shape as weight tensors — another 2 bytes per parameter, another 14 GiB for a 7B model. With LoRA, only the adapter parameters (approximately 1% of total parameters) have gradients. For a 7B model, that is 70M parameters × 2 bytes = 140 MiB instead of 14 GiB.

**Optimizer states**: Adam maintains two momentum tensors per trainable parameter — the first moment (mean of gradients) and second moment (uncentered variance). Both are stored in fp32 regardless of model precision. That is 8 bytes per trainable parameter, or approximately 56 GiB for a 7B full fine-tune. With LoRA and 8-bit Adam (bitsandbytes), this drops to the adapter fraction at half precision — under 1 GiB for a 7B model.

**Activations**: The forward pass produces intermediate tensors at every layer that the backward pass needs to compute gradients. Without gradient checkpointing, these are kept in memory simultaneously: `batch × seq_len × hidden_size × num_layers × dtype_bytes / GiB`. With gradient checkpointing, activations are discarded after each layer's forward pass and recomputed during the backward pass, reducing activation memory by approximately 4–8× at the cost of ~30% extra compute.

**KV cache**: Transformer attention requires storing key and value projections for each token in the sequence: `2 × num_layers × num_kv_heads × head_dim × seq_len × batch × dtype_bytes`.

The discovery that changed the understanding of LoRA: it does not just reduce gradients — it eliminates nearly the entire optimizer state cost. Full Adam on 7B parameters requires ~56 GiB of optimizer state. LoRA adapter Adam requires ~200 MiB. This is the real reason LoRA makes large model fine-tuning tractable on consumer hardware, not the weight reduction.

The Unsloth savings factor of 45% applied to the subtotal is a deliberate approximation of a complex set of kernel-level optimizations (Flash Attention 2, fused rotary embeddings, quantized backward passes). The actual savings vary between 40% and 60% depending on model architecture and hardware, so 45% is a reasonable conservative midpoint.

---

### Aggregating unreliable external APIs gracefully

The early `providers/` implementation wrapped the entire provider fetch in a single `Result` and propagated failures upward:

```rust
pub async fn fetch_all() -> Result<Vec<GpuListing>> {
    let mut listings = vec![];
    listings.extend(runpod::fetch().await?);   // ? here aborts everything on failure
    listings.extend(lambda::fetch().await?);
    listings.extend(vastai::fetch().await?);
    Ok(listings)
}
```

Testing against live APIs immediately revealed the problem: Vast.ai returns HTTP 429 under normal calling patterns, Lambda's availability endpoint occasionally returns 503, and RunPod's GraphQL endpoint has a ~2% error rate. A tool that fails completely when any one of three external services has a bad moment is not trustworthy.

The correct design: each provider failure is isolated and reported, not propagated:

```rust
pub async fn fetch_all(filter: Option<&str>) -> (Vec<GpuListing>, Vec<SkippedProvider>) {
    let (r, l, v) = tokio::join!(
        runpod::fetch(),
        lambda::fetch(),
        vastai::fetch(),
    );
    let mut listings = vec![];
    let mut skipped  = vec![];
    for (name, result) in [("RunPod", r), ("Lambda", l), ("Vast.ai", v)] {
        match result {
            Ok(ls)  => listings.extend(ls),
            Err(e)  => skipped.push(SkippedProvider { name: name.into(), reason: e.to_string() }),
        }
    }
    (listings, skipped)
}
```

`fetch_all` now returns `(data, failures)` instead of `Result<data>`. The command layer shows which providers were skipped and produces a recommendation from whatever data was available. The test cases for skipped providers — verifying that the command still produces output when one or two providers fail — became some of the most valuable tests in the suite. They encode behavior that would otherwise only surface in production under real network conditions.

---

### HDR histogram percentile correctness

A test in `bench/stats.rs` was constructed with 99 samples at 10ms and 1 sample at 100ms, then asserted `p99 == 100ms`. The test failed: p99 returned 10ms.

The reason took careful reading to understand. Percentile semantics: p99 means "the value below which 99% of observations fall". With 100 samples, 99% of 100 = 99 samples. The 99th-smallest value is 10ms (the last of the 99 fast samples). The 100th value (100ms) is at the 100th percentile — p100, not p99. p99 never reaches the outlier.

The fix: use 98 samples at 10ms and 2 samples at 100ms. Now 98% of observations are fast and 2% are slow. p99 falls in the slow region (the 99th-smallest value is now one of the 100ms samples), and the assertion passes.

The general lesson: percentile boundary tests require thinking in terms of the percentile definition, not intuition about "the 99th value". With N samples, the value at percentile P is the `ceil(N × P/100)`-th smallest value. At exactly 100 samples:
- p99 = the 99th-smallest value = the 99th out of 100
- p99 ≠ the largest value (which is p100)
- To make p99 capture an outlier, more than 1% of samples must be that outlier

---

### Graceful degradation without a GPU

The first version of collector selection was:

```rust
let collector = NvmlCollector::new(pid)?;  // fails hard if NVML unavailable
```

On a development laptop or in CI without an NVIDIA driver, this terminates immediately with "NVML initialization failed". The tool was useless for development on the machines it was developed on.

The fix was an explicit fallback chain:

```rust
let collector: Box<dyn Collector> = match NvmlCollector::new(pid) {
    Ok(c)  => Box::new(c),
    Err(e) => {
        eprintln!("NVML unavailable ({e}), falling back to CPU-only metrics");
        Box::new(CpuOnlyCollector::new(pid)?)
    }
};
```

`CpuOnlyCollector` implements the same `Collector` trait, reporting CPU utilization, memory, and process state from `/proc` but returning `None` for GPU fields. The session logic and metrics pipeline do not know or care which collector is in use — they operate on `MetricsSnapshot` values and render `None` GPU fields as "n/a".

This is the hexagonal architecture paying off again: the fallback required adding one new file (`cpu_only.rs`) and two lines in the startup code. No session logic, no metrics code, no output code needed to change.

The broader lesson: a CLI tool's failure modes are part of its interface. A tool that works reliably on 60% of machines gets a reputation as a "sometimes works" tool and is reached for less. The question to ask at every failure mode: is there a degraded result that is more useful than no result?

---

### Platform-conditional compilation with `#[cfg]`

Rust's conditional compilation is resolved entirely at compile time. The `#[cfg(target_os = "linux")]` attribute causes the annotated item to be compiled in only when the target operating system is Linux. On any other target, the item does not exist — it generates no code, no symbol, and no binary size. This is different from a runtime `if` branch, where both sides are compiled and only one is executed.

The three forms used throughout the macOS extension:

```rust
// Attribute form — applies to the next item (function, struct, impl, mod, use)
#[cfg(target_os = "macos")]
pub mod apple_gpu;

// cfg! macro form — evaluates to a bool at compile time; used inside expressions
let on_mac = cfg!(target_os = "macos");

// cfg_if! (external crate) — if/else chains for longer platform switches
```

Guarding entire `mod` declarations is the cleanest way to handle platform-specific modules. Declaring `#[cfg(target_os = "linux")] pub mod nvml;` means the `nvml` module simply does not exist on macOS. Attempting to reference `crate::collectors::nvml::NvmlCollector` in macOS-only code produces a compile error immediately, rather than a confusing linker failure.

The alternative — putting `#[cfg]` attributes inside functions — leads to scattered guards that are hard to audit. Guarding the module boundary once is cleaner.

The compound form `#[cfg(any(target_os = "linux", target_os = "macos"))]` accepts either platform. The inverse `#[cfg(not(any(...)))]` guards the bail-out for unsupported platforms. This pattern — support A, support B, fail clearly on everything else — became the standard for every platform branch in the watch and probe commands.

One trap: `#[cfg]` on a `use` import does not suppress "unused import" warnings on other platforms; the import simply doesn't exist, which means the code that was using it also needs its own `#[cfg]` guard. Guarding imports (`use`) independently from the code that uses them is unnecessary ceremony. The cleanest solution is to keep the `use` imports inside the `#[cfg]` blocks where the code lives.

---

### Raw FFI to Apple system frameworks

`IOKit` and `CoreFoundation` are C frameworks shipped with every macOS installation. Rust can call into them directly through `extern "C"` blocks with `#[link(name = "IOKit", kind = "framework")]`. No binding-generation tool (`bindgen`) is needed for a small, stable set of functions — the declarations can be written by hand.

The function signatures must exactly match the C headers. The critical types:

```rust
type IOObject = u32;   // mach_port_t / io_object_t on 64-bit macOS
type KernReturn = i32; // kern_return_t
```

`IOObject` being `u32` is the important one. The underlying C type is `mach_port_t`, which is `unsigned int` (32-bit) on both 32-bit and 64-bit macOS — not a pointer — so `u32` is correct.

`CFDictionary`, `CFString`, and `CFNumber` are all `*mut c_void` / `*const c_void` on the Rust side. CoreFoundation is a reference-counted C framework: every `Create` function returns a +1 retain count that the caller must balance with `CFRelease`. Missing a `CFRelease` leaks memory; calling it twice causes a use-after-free. The discipline: pair every `Create` call with a `CFRelease` in the same scope, and never `Release` a pointer you do not own (e.g. values returned from `CFDictionaryGetValue` are owned by the dictionary, not the caller).

`IOServiceGetMatchingServices` has one unusual ownership rule: it *consumes* the `CFDictionary` returned by `IOServiceMatching`. After `IOServiceGetMatchingServices` returns, the matching dictionary has been released by IOKit regardless of whether the call succeeded. Calling `CFRelease` on it afterward is a double-free. This is documented in the IOKit headers but not in any Rust binding; it must be handled by simply not storing the pointer after the call.

The `unsafe` scope for IOKit calls must be as narrow as possible. The pattern used throughout `apple_gpu.rs`: unsafe FFI calls are isolated to small private functions (`query_gpu_stats`, `extract_snapshot`, `cf_dict_i64`) that present safe interfaces to the rest of the module. The public `run()` method on `AppleGpuCollector` contains no `unsafe` — all unsafe is inside those private helpers.

---

### IOKit GPU performance statistics on Apple Silicon

Apple Silicon GPU utilization is exposed through the IOKit registry under the `IOAccelerator` class. On Apple Silicon, the concrete subclass is `AGXAccelerator` (Apple GPU Accelerator), but it conforms to the `IOAccelerator` protocol, so `IOServiceMatching("IOAccelerator")` discovers it.

Each matched service has a property dictionary accessible via `IORegistryEntryCreateCFProperties`. Within that dictionary, the `PerformanceStatistics` sub-dictionary contains the metrics that correspond to what Activity Monitor calls "GPU":

- `Device Utilization %` — overall GPU utilization (0–100), Apple Silicon
- `GPU Core Utilization` — equivalent key on some AMD/Intel Mac GPUs
- `In use system memory` — bytes of GPU/unified memory currently in use

On Apple Silicon, `vram_total` does not exist as a distinct hardware quantity — the GPU's "VRAM" is a portion of unified DRAM. The total memory pool is queried via `sysctl("hw.memsize")`, which returns total physical RAM. This is the correct denominator: on M-series chips, the GPU can in principle use all available RAM (subject to OS pressure), so total system RAM is the right VRAM budget to display.

The limitation that must be understood and communicated: these counters are system-wide, not per-process. macOS does not expose per-process GPU memory allocation to userspace without Metal Performance Shaders instrumentation. The `vram_used_mib` field in `RawSample` therefore reflects total GPU memory pressure across all processes, not just the monitored training job. This matches what Activity Monitor shows and is the best available approximation at the OS level.

---

### Process liveness detection across platforms

The Linux approach — checking for the existence of `/proc/{pid}` — works only on Linux. The POSIX-portable equivalent is `kill(pid, 0)`. From `man 2 kill`:

> If sig is 0, then no signal is sent, but existence and permission checks are still performed; this can be used to check for the existence of a process ID or process group ID.

`kill(pid, 0)` returns 0 if the process exists and the calling process has permission to send it a signal, or -1 with errno `ESRCH` (no such process) or `EPERM` (process exists but permission denied). For liveness purposes, both 0 and `EPERM` mean the process is alive — only `ESRCH` means it has exited.

The implementation:
```rust
fn process_is_alive(pid: u32) -> bool {
    unsafe { libc::kill(pid as libc::pid_t, 0) == 0 }
}
```

This is `unsafe` because it's a raw syscall, but it contains no undefined behavior — `kill` with signal 0 is specified by POSIX and safe to call. The same code works on Linux, macOS, and any other POSIX system.

The asymmetry with the Linux `/proc` check: `/proc/{pid}` can also detect permission failures by checking file metadata with `stat`. The `kill(pid, 0)` approach conflates "process exists but I have no permission" with "process exists and I have permission" — both return liveness as `true`. For the `watch` command's purpose (deciding when to stop monitoring), this conflation is acceptable: if the process exists but we have no permission, we will hit errors reading metrics and the collector will exit on its own.

---

### `sysctl` for hardware information on macOS

`sysctl` is the POSIX interface for reading kernel parameters. On macOS it is the primary API for hardware information that has no `/proc` equivalent. The pattern is consistent across all queries:

```rust
let mut value: T = unsafe { std::mem::zeroed() };
let mut len = std::mem::size_of::<T>();
let ret = unsafe {
    libc::sysctlbyname(
        b"key.name\0".as_ptr() as *const libc::c_char,
        &mut value as *mut T as *mut libc::c_void,
        &mut len,
        std::ptr::null_mut(),  // newp: null = read-only
        0,                     // newlen: 0 = read-only
    )
};
```

The NUL-terminated byte literal `b"hw.memsize\0"` is important — `sysctlbyname` expects a C string. Missing the NUL terminator is a memory safety bug: the C runtime will read past the end of the Rust slice looking for the NUL byte. Using a byte literal with an explicit `\0` makes this correct and visible.

Two keys used in calibrate:

- `hw.memsize` → `u64` — total physical RAM in bytes. Never changes after boot.
- `machdep.cpu.brand_string` → `[u8; 256]` — CPU/SoC brand string, e.g. `"Apple M3 Max"`.

`machdep.cpu.brand_string` requires a buffer, not a scalar. The returned `len` includes the NUL terminator; the displayed string should use `&buf[..len - 1]`. On non-Apple-Silicon Macs (Intel), this returns an Intel CPU string. The `apple_gpu_name()` function appends `" GPU"` to distinguish the GPU name from the chip name when matching against spec-DB keys, since IOKit does not provide a separate GPU product name.

---

### Designing for per-process vs system-wide metrics

NVML is per-process aware: `device.running_compute_processes()` returns PIDs, so the `NvmlCollector` can identify which GPU is being used by the specific PID being monitored. IOKit on macOS is not — it provides system-wide GPU counters without process attribution.

This distinction shapes the semantics of `RawSample` on each platform:

| Field | Linux (NVML) | macOS (IOKit) |
|---|---|---|
| `sm_utilization` | GPU utilization while target PID is active | Total GPU utilization, all processes |
| `vram_used_mib` | VRAM used by all compute processes on the device | Total GPU memory in use, all processes |
| `vram_total_mib` | Device VRAM capacity | Total system RAM |

The analytics pipeline (MFU, bottleneck detection, recommendations) consumes `RawSample` values identically regardless of platform. This means MFU on macOS is approximate in two ways: the TFLOPS denominator comes from spec-DB benchmarks (not live measurement), and the utilization numerator reflects all GPU activity on the machine, not just the training job.

The correct design decision was not to add a `is_system_wide: bool` flag to `RawSample` and scatter platform checks throughout the analytics code. Instead, the limitation is documented in the advisory printed at watch startup and in the summary report. The analytics pipeline running on approximate data and producing approximate results is more useful than no analytics at all. The user is informed once; the code is not cluttered.

---

### Metal compute pipeline setup

Apple's Metal framework exposes GPU compute through a pipeline of objects that must be created in a specific order:

1. **`Device`** — represents the physical GPU. `Device::system_default()` returns the primary GPU on any Mac.
2. **`CommandQueue`** — a queue of command buffers submitted to the device. Create once; reuse across many dispatches.
3. **Library** — compiled Metal Shading Language (MSL) code. `device.new_library_with_source(msl_string, &CompileOptions::new())` compiles MSL at runtime and returns a library object.
4. **Function** — a named entry point from the library. `library.get_function("kernel_name", None)`.
5. **`ComputePipelineState`** — the compiled pipeline that binds the function to the device. Create once per function; expensive to create but cheap to reuse.
6. **`CommandBuffer`** — a recording of GPU commands. Create per-dispatch from the queue.
7. **`ComputeCommandEncoder`** — records individual compute commands into a buffer. `command_buffer.new_compute_command_encoder()`.

For a `bench` workload, steps 1–5 happen in `load()` and are cached on the struct. Steps 6–7 happen in every `infer()` call. This separation is critical for accurate latency measurement: if pipeline creation happened inside `infer()`, the benchmark would measure compilation time, not inference time.

`wait_until_completed()` on the command buffer is the GPU synchronization barrier. Metal dispatch is asynchronous by default — `cmd_buf.commit()` returns immediately and the GPU executes the commands in parallel with the CPU. Without `wait_until_completed()`, `infer()` would return instantly (measuring only encoding time, ~microseconds) rather than the actual GPU execution time. Forgetting this synchronization call was the first bug encountered in `MetalRuntime`.

---

### Metal Shading Language and the GEMV kernel

MSL is a superset of C++14 with extensions for GPU parallelism. A compute kernel is a function annotated `kernel` that runs once per thread in a grid. The grid dimensions are specified at dispatch time from the CPU side.

The bench kernel is a GEMV (general matrix–vector multiply): each thread computes one output element by dot-producting its assigned row of the weight matrix with the input vector.

```metal
kernel void bench_matmul(
    device const float* weights [[buffer(0)]],
    device const float* input   [[buffer(1)]],
    device float*       output  [[buffer(2)]],
    constant uint2&     dims    [[buffer(3)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid >= dims.x) return;
    float sum = 0.0f;
    uint offset = gid * dims.y;
    for (uint k = 0; k < dims.y; ++k)
        sum += weights[offset + k] * input[k];
    output[gid] = sum;
}
```

`[[buffer(N)]]` is the binding index — it must match the index passed to `encoder.set_buffer(N, ...)` on the CPU side. `[[thread_position_in_grid]]` is the absolute thread index; the guard `if (gid >= dims.x) return;` handles the case where the grid is padded to a multiple of the threadgroup size.

The dispatch sizing uses `dispatch_threads` (non-uniform dispatch), which correctly handles non-power-of-two output sizes without wasting threads. The threadgroup size is capped at `min(max_total_threads_per_threadgroup, 256, M)` — using 256 is a practical heuristic for simple kernels; the M cap prevents launching threadgroups larger than the output dimension on small matrices.

The `constant` address space (`constant uint2& dims`) is for read-only data that is broadcast identically to all threads. It is stored in a faster memory path than `device` on Apple Silicon. The dimensions fit in 8 bytes and are passed directly via `set_bytes` rather than allocating a buffer, which avoids a small allocation per dispatch.

---

### `MTLResourceOptions::StorageModeShared` on unified memory

Metal has three storage modes for buffers:

- `StorageModeShared` — buffer is accessible from both CPU and GPU. On Apple Silicon with unified memory, this is a single physical allocation visible to both processors. No explicit synchronization needed.
- `StorageModeManaged` (macOS only with discrete GPU) — separate CPU and GPU copies that must be synchronized explicitly via `didModifyRange` / `synchronizeResource`.
- `StorageModePrivate` — GPU-only. CPU cannot read or write directly. Requires a blit (copy) command to populate.

On Apple Silicon, `StorageModeShared` is optimal. There is no discrete GPU VRAM; the GPU reads directly from the same DRAM as the CPU. Choosing `StorageModePrivate` and uploading via blit commands would add unnecessary copy overhead on this hardware.

The bench input buffer reuse logic exploits this: after the first `infer()` call allocates a `SharedMode` buffer, subsequent calls write directly to `buffer.contents() as *mut f32` — a raw pointer into the shared allocation — and the GPU sees the updated values on the next dispatch without any explicit cache flush. On discrete GPUs, this would require a `didModifyRange` call; on Apple Silicon unified memory, the CPU write is immediately coherent with the GPU's view.

---

### CoreML and MPS execution providers for subprocess runtimes

Both ONNX Runtime and PyTorch have macOS-native acceleration paths:

**ONNX Runtime + CoreML**: CoreMLExecutionProvider delegates ONNX graph nodes to Apple's Neural Engine and GPU via CoreML. It requires `onnxruntime >= 1.17` with the CoreML package installed (`pip install onnxruntime-silicon` or the standard package on Apple Silicon). The provider is selected by name:

```python
providers = ["CoreMLExecutionProvider", "CPUExecutionProvider"]
session = ort.InferenceSession(model_path, providers=providers)
```

Not all ONNX ops are supported by CoreML — unsupported nodes fall back to CPU automatically. The benchmark result therefore represents a mix of Neural Engine/GPU (for supported ops) and CPU (for the remainder). This is exactly what a production ONNX deployment on macOS would use, so the benchmark is representative.

**PyTorch + MPS**: Metal Performance Shaders backend for PyTorch, available since PyTorch 1.12 on Apple Silicon. Device selection:

```python
if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
    model = model.to("mps")
    device = "mps"
```

`torch.mps.synchronize()` is required for accurate latency measurement — same reason as Metal's `wait_until_completed()`. Without synchronization, `time.perf_counter()` captures only the time to enqueue GPU commands, not the time to execute them. Forgetting synchronization produces latency readings of 0.1–1ms for operations that actually take 20–100ms.

The detection order in the torchscript driver is: MPS first, then CUDA, then CPU. This ensures that the same script produces the correct behavior across macOS (MPS), Linux (CUDA), and CPU-only machines without any platform-specific argument.

---

### GPU spec matching with a token scoring algorithm

The spec database uses short canonical names (`"M3 Max"`, `"A100 SXM"`) while the strings that arrive at runtime are verbose and inconsistent (`"Apple M3 Max GPU"` from `sysctl`, `"NVIDIA A100-SXM4-80GB"` from NVML). The matching algorithm must handle both gracefully.

The approach is two-stage:

**Stage 1 — Normalization** (`normalize_for_match`): strips vendor prefixes (`"nvidia "`, `"geforce "`, `"tesla "`, `"apple "`), replaces hyphens with spaces, and removes tokens that are pure noise for matching:
- Memory-size tokens: any token ending in `"gb"` whose prefix is a parseable integer (e.g. `"80gb"`, `"40gb"`)
- Generic suffix tokens: `"gpu"` — present in macOS IOKit names but absent from spec keys

**Stage 2 — Token prefix scoring** (`match_score`): every token in the spec key must prefix-match at least one token in the normalized query. The score is the sum of matched character lengths. Longer (more specific) spec keys score higher for the same query.

The prefix-match direction matters: spec tokens prefix-match query tokens, not the reverse. This allows `"SXM"` in the spec to match `"SXM4"` in the query (the query is more specific than the spec key). If the direction were reversed, `"SXM4"` in the query would need to start with `"SXM4"` in the spec, which would require a separate entry for every SXM variant.

The scoring system correctly handles the four-way disambiguation that was validated with tests:
- `"M3"` query matches `"M3"` but not `"M3 Max"` (M3 Max requires all its tokens to match; "Max" has no counterpart in the query)
- `"M3 Max"` query matches both `"M3"` and `"M3 Max"`, but `"M3 Max"` scores higher (two tokens matched vs one)
- `"A10G"` query matches `"A10G"` (score 4) and `"A10"` (score 3); `"A10G"` wins
- NVIDIA query never matches Apple spec keys: `"rtx 4090"` has no tokens that prefix-match `"m"` in `"M3 Max"`

The test `apple_spec_not_matched_by_nvidia_query` and `nvidia_spec_not_matched_by_apple_query` encode this cross-contamination guarantee explicitly. Without them, a future change to the normalization or scoring could silently break isolation between vendor families.

---

### Apple Silicon GPU TFLOPS figures and their sources

Apple does not publish official peak TFLOPS figures. The values in `fallback_specs.json` are derived from:

1. **MLPerf Inference benchmarks** — the MLPerf committee publishes throughput results for Apple Silicon across standardized models. Back-calculating from throughput and model FLOPs gives a practical TFLOPS estimate.
2. **Empirical measurements** — Metal benchmarks (`metal-bench`, MLX performance tests) run on known models and compare against theoretical GPU core counts × frequency × MAC throughput.
3. **Apple silicon chip specifications** — Apple publishes GPU core counts and approximate clock speeds. Peak theoretical TFLOPS = `cores × MACs_per_core_per_cycle × frequency × 2`. For BF16 on M2+, the multiply-accumulate units operate at full BF16 precision, giving `fp32_tflops × 2` as the BF16 figure.

M1 is the exception: the M1 Neural Engine supports BF16 but the GPU shader cores do not — GPU BF16 operations execute at the same throughput as FP32 on M1. This is why `"M1"` has `bf16_tflops == fp32_tflops` in the spec table. M2 and later introduced native BF16 in the GPU shader cores, giving the 2× ratio.

The `boost_clock_mhz` field is populated with the nominal GPU clock from Apple's chip documentation, used for the clock-speed ratio in MFU calculations. Apple does not expose live GPU clock frequency via any public macOS API (unlike NVML's `device.clock_info(Clock::SM)`), so `sm_clock_mhz` in `RawSample` remains `0` on macOS and the MFU calculator falls back to using the spec's `boost_clock_mhz` as the reference.

The correct way to communicate this in the tool's output is to mark Apple Silicon MFU values as "estimated" — the denominator is accurate to within approximately 20% based on empirical evidence, but is not the certified figure that NVIDIA publishes for its datacenter GPUs.

---

### `vram_gib` semantics on unified memory architectures

The `GpuSpec` struct has a `vram_gib: u32` field that represents GPU memory capacity. For discrete GPUs, this is unambiguous — there is a fixed amount of GDDR/HBM on the PCIe card. For Apple Silicon, the field is populated with the base unified memory configuration for each chip variant (e.g., 36 for M3 Max, which ships with 36 GiB or 128 GiB depending on configuration).

This is a deliberate simplification. The `vram_total_mib` field in `RawSample` reflects the actual system RAM at runtime (from `sysctl hw.memsize`), which is authoritative and correct. The spec-DB `vram_gib` is used only for the `plan` command's VRAM sufficiency check — whether a given GPU can fit a model with a given VRAM requirement. Using the base memory configuration is conservative: it will recommend the minimum tier that fits and let the user upgrade if they have a higher-memory SKU.

A potential improvement would be to query system RAM at `plan` time and use it as the VRAM budget directly, but that would only be accurate on the machine running the `plan` command, not when planning for a remote cloud instance. The spec-DB base figure is the right choice for planning purposes.

---

### Eliminating dead-code warnings with `#[cfg]` on `mod` declarations

When platform-specific modules are guarded with `#[cfg]` attributes on the `pub mod` declaration itself, the compiler excludes the entire module from the compilation unit for unsupported platforms. This has a cascading benefit: any code that uses types from that module only inside its own `#[cfg]` block also becomes dead from the compiler's perspective, and no dead-code warnings are emitted.

Contrast with guarding only the call sites:

```rust
// This produces dead_code warnings on macOS for NvmlCollector, ProcCollector, etc.
pub mod nvml;  // module compiled on all platforms
pub mod proc;

// In watch.rs:
#[cfg(target_os = "linux")]
{
    let collector = NvmlCollector::new(...);  // only used here
}
```

The `nvml` module is compiled on macOS (its types exist), but the types are never used (the call sites are cfg-guarded). The compiler reports `NvmlCollector` as never constructed, `discover_gpu_indices` as never used, etc.

The fix — `#[cfg(target_os = "linux")] pub mod nvml;` — makes the module not exist on macOS. The types never enter the type system. No dead-code warning is possible because there is nothing to warn about.

The transition from 27 dead-code warnings after Phase 1 to zero warnings after Phase 2 came entirely from moving the `#[cfg]` guard from call sites to module declarations. The general principle: guard at the highest level of the module hierarchy that is practical, not at individual use sites.

---

### Extending the `Runtime` trait without breaking existing implementations

Adding a new runtime (`MetalRuntime`) to a trait-based system requires implementing every method in the trait. The `Runtime` trait has four required methods (`name`, `load`, `infer`, `unload`) and one optional method with a default implementation (`pre_collected_samples`).

Because `MetalRuntime` is an in-process runtime (not a subprocess), it uses the standard measurement loop — `pre_collected_samples` returns `None` by default and no override is needed. The harness's dual-path logic (subprocess vs in-process) required no changes.

The `RuntimeDescriptor` struct uses function pointers rather than trait objects:

```rust
pub struct RuntimeDescriptor {
    pub name: &'static str,
    pub is_available: fn() -> bool,
    pub is_compatible: fn(ModelFormat) -> bool,
    pub create: fn() -> Box<dyn Runtime>,
}
```

This is intentional: descriptors are static and cheap to copy (three function pointers and a string reference), while `Box<dyn Runtime>` objects are expensive to create and hold heap-allocated state. The registry returns `Vec<RuntimeDescriptor>` — a list of descriptions — and the harness creates `Box<dyn Runtime>` instances only for the runtimes that pass both `is_available` and `is_compatible` checks.

Inserting the Metal descriptor at position 1 in the registry (`runtimes.insert(1, ...)`) under a `#[cfg(target_os = "macos")]` guard required no changes to any existing descriptor or to the harness. The platform guard ensures `metal::MetalRuntime::descriptor()` is not even compiled on Linux, which means the `metal` crate dependency is not linked on Linux either.

---

### Platform-conditional dependencies in Cargo.toml

Cargo supports target-specific dependencies via `[target.'cfg(...)'.dependencies]` sections:

```toml
[target.'cfg(target_os = "macos")'.dependencies]
metal = "0.29"
```

This is the cargo manifest equivalent of `#[cfg]` on Rust code. The `metal` crate is not downloaded, not compiled, and not linked on Linux or Windows. The dependency section uses the same cfg predicate syntax as Rust's `#[cfg]` attributes — `target_os`, `target_arch`, `feature`, and compound expressions using `all(...)`, `any(...)`, `not(...)`.

The distinction from `[dependencies]` with a feature flag is important: a feature flag still compiles and links the crate on all platforms (the feature just controls which code paths are active). A `[target.'cfg(...)'.dependencies]` entry truly excludes the crate from non-matching platforms.

This matters for the `metal` crate specifically because it links against Metal and ObjectiveC runtime libraries (`-framework Metal`, `libobj.a`) that are only present on macOS. Listing it under `[dependencies]` unconditionally would cause a linker failure on Linux. The target-specific section is not a convenience — it is a correctness requirement.

One subtlety with the string quoting: the single quotes around `cfg(target_os = "macos")` in TOML are required because the string contains double quotes. If the inner string used single quotes, the TOML outer quotes would need to be double quotes. Cargo's TOML parser accepts both forms.

---

### Testing cross-platform spec matching without running on both platforms

The spec matching code (`normalize_for_match`, `match_score`, `find_best_match`) is pure Rust with no platform-specific code or FFI. This means all test scenarios — including Apple Silicon GPU name resolution — can be tested on Linux in CI and on macOS locally, running identical test code on both platforms.

The tests for Apple Silicon resolution:

```rust
#[test]
fn fallback_resolves_apple_m3_max() {
    let spec = FallbackRepository.get_by_name("Apple M3 Max GPU").unwrap();
    assert_eq!(spec.name, "M3 Max");
    assert!((spec.bf16_tflops - 28.4).abs() < 0.1);
}
```

This test passes on Linux even though no Apple GPU is present, because `get_by_name` is a pure in-memory lookup. The `"Apple M3 Max GPU"` string is exactly what `attach::apple_gpu_name()` would return at runtime on an M3 Max machine — the test is validating the entire name→spec resolution path end-to-end without requiring the hardware.

The `nvidia_spec_not_matched_by_apple_query` and `apple_spec_not_matched_by_nvidia_query` tests guard against a class of regression that would only manifest at runtime on the respective hardware but can be caught in any CI environment. This is the property of pure functions: they can be tested exhaustively on any platform.

The 141-test suite runs in 0.05 seconds. The fast feedback loop is a consequence of keeping all analytics, spec matching, and planning logic free of I/O and FFI — the only real-time tests that exist are in `process::attach` where liveness is checked, and even those use `kill(pid, 0)` which is a negligible syscall.
