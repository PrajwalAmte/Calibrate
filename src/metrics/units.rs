//! Formatting helpers for optional hardware readings.

/// "8.0 GiB", "512 MiB".
pub fn fmt_mib(mib: u64) -> String {
    if mib >= 1024 {
        format!("{:.1} GiB", mib as f64 / 1024.0)
    } else {
        format!("{mib} MiB")
    }
}

pub fn fmt_opt<T>(v: Option<T>, f: impl FnOnce(T) -> String) -> String {
    v.map(f).unwrap_or_else(|| "N/A".to_string())
}

pub fn fmt_pct(v: Option<f32>) -> String {
    fmt_opt(v, |p| format!("{p:.1}%"))
}

pub fn fmt_temp(v: Option<f32>) -> String {
    fmt_opt(v, |t| format!("{t:.0}°C"))
}

pub fn fmt_watts(v: Option<f32>) -> String {
    fmt_opt(v, |w| format!("{w:.0}W"))
}

/// "00:04:22"
pub fn fmt_hms(secs: f64) -> String {
    let s = secs.max(0.0) as u64;
    format!("{:02}:{:02}:{:02}", s / 3600, (s % 3600) / 60, s % 60)
}

/// Human duration for estimates: "45s", "12m", "3.4h", "2.1d".
pub fn fmt_duration(secs: f64) -> String {
    if !secs.is_finite() || secs < 0.0 {
        return "n/a".into();
    }
    if secs < 90.0 {
        format!("{secs:.0}s")
    } else if secs < 90.0 * 60.0 {
        format!("{:.0}m", secs / 60.0)
    } else if secs < 48.0 * 3600.0 {
        format!("{:.1}h", secs / 3600.0)
    } else {
        format!("{:.1}d", secs / 86400.0)
    }
}

/// "$0.42", "$1,234", "<$0.01".
pub fn fmt_usd(v: f64) -> String {
    if !v.is_finite() {
        return "n/a".into();
    }
    if v > 0.0 && v < 0.01 {
        return "<$0.01".into();
    }
    if v >= 1000.0 {
        let whole = v.round() as u64;
        let s = whole.to_string();
        let mut out = String::new();
        for (i, c) in s.chars().enumerate() {
            if i > 0 && (s.len() - i).is_multiple_of(3) {
                out.push(',');
            }
            out.push(c);
        }
        return format!("${out}");
    }
    format!("${v:.2}")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn formats() {
        assert_eq!(fmt_mib(8192), "8.0 GiB");
        assert_eq!(fmt_mib(300), "300 MiB");
        assert_eq!(fmt_temp(None), "N/A");
        assert_eq!(fmt_watts(Some(210.4)), "210W");
        assert_eq!(fmt_hms(262.0), "00:04:22");
        assert_eq!(fmt_duration(30.0), "30s");
        assert_eq!(fmt_duration(1800.0), "30m");
        assert_eq!(fmt_duration(3.0 * 3600.0), "3.0h");
        assert_eq!(fmt_duration(3.0 * 86400.0), "3.0d");
        assert_eq!(fmt_usd(0.004), "<$0.01");
        assert_eq!(fmt_usd(0.0), "$0.00");
        assert_eq!(fmt_usd(12.345), "$12.35");
        assert_eq!(fmt_usd(1234567.0), "$1,234,567");
    }
}
