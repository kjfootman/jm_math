use thiserror::Error;

#[derive(Debug, Error)]
pub enum Error {
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),

    #[error("Maximum iteration exceeded - iteration: {0}, residual: {1:.2E}")]
    Convergence(usize, f64),

    #[error("Dimension mismatched: {0}")]
    DimensionMismatch(String),

    #[error("Missing diagonal element at row: {0}")]
    MissingDiagonal(usize),

    #[error("Value error: {0}")]
    ValueError(String),

    #[error("Parse error: {0}")]
    ParseInt(#[from] std::num::ParseIntError),

    #[error("Parse error: {0}")]
    ParseFloat(#[from] std::num::ParseFloatError),

    #[error("Logger setting error: {0}")]
    LoggerSetting(#[from] log::SetLoggerError),
}
