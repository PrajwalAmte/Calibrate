/// Tera floating-point operations per second.
#[derive(Debug, Clone, Copy, PartialEq, PartialOrd, serde::Serialize, serde::Deserialize)]
pub struct Tflops(pub f64);

/// Temperature in degrees Celsius.
#[derive(Debug, Clone, Copy, PartialEq, PartialOrd, serde::Serialize, serde::Deserialize)]
pub struct Celsius(pub f32);

/// Power in Watts.
#[derive(Debug, Clone, Copy, PartialEq, PartialOrd, serde::Serialize, serde::Deserialize)]
pub struct Watts(pub f32);

/// Memory size in Mebibytes.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, serde::Serialize, serde::Deserialize)]
pub struct Mib(pub u64);

/// Clock frequency in MHz.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, serde::Serialize, serde::Deserialize)]
pub struct Mhz(pub u32);

/// A percentage value in the range 0.0 – 100.0.
#[derive(Debug, Clone, Copy, PartialEq, PartialOrd, serde::Serialize, serde::Deserialize)]
pub struct Percent(pub f32);

impl Percent {
    /// Clamp the inner value to [0, 100].
    pub fn clamped(v: f32) -> Self {
        Self(v.clamp(0.0, 100.0))
    }

    pub fn as_f32(self) -> f32 {
        self.0
    }
}

impl std::fmt::Display for Tflops {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{:.1} TFLOPS", self.0)
    }
}

impl std::fmt::Display for Celsius {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{:.0}°C", self.0)
    }
}

impl std::fmt::Display for Watts {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{:.0}W", self.0)
    }
}

impl std::fmt::Display for Mib {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        if self.0 >= 1024 {
            write!(f, "{:.1} GiB", self.0 as f64 / 1024.0)
        } else {
            write!(f, "{} MiB", self.0)
        }
    }
}

impl std::fmt::Display for Mhz {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{} MHz", self.0)
    }
}

impl std::fmt::Display for Percent {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{:.1}%", self.0)
    }
}
