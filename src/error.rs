use thiserror::Error;

#[derive(Debug, Error)]
pub enum Error {
    #[error("dimension mismatched.\n{0}")]
    DimensionMismatch(String),

    #[error("value error.\n{0}")]
    ValueError(String),
}
