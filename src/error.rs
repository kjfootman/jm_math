use thiserror::Error;

#[derive(Debug, Error)]
pub enum Error {
    #[error("I/O error\n{0}")]
    Io(#[from] std::io::Error),

    #[error("Dimension mismatched\n{0}")]
    DimensionMismatch(String),

    #[error("Value error\n{0}")]
    ValueError(String),

    #[error("Parse error\n{0}")]
    Parse(String),
}
