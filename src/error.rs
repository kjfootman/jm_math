use thiserror::Error;

#[derive(Debug, Error)]
pub enum Error {
    #[error("I/O error\n{0}")]
    Io(#[from] std::io::Error),

    #[error("Parse error\n{0}")]
    ParseInt(#[from] std::num::ParseIntError),

    #[error("Parse error\n{0}")]
    ParseFloat(#[from] std::num::ParseFloatError),

    #[error("Dimension mismatched\n{0}")]
    DimensionMismatch(String),

    #[error("Value error\n{0}")]
    ValueError(String),
}
