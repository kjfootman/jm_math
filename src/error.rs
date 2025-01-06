//! error module
use std::error::Error;
use std::fmt::Display;

// #[derive(Debug)]
// pub struct MyError<'a>(pub &'a str);

// impl Display for MyError<'_> {
//     fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
//         write!(f, "{}", self.0)
//     }
// }

// impl Error for MyError<'_> {}

#[derive(Debug)]
pub enum JmError {
    // Dimension error.
    DimError(String),
    // Convergence error.
    ConvErr(String),
    // Value error.
    ValueErr(String),
    // Not available error
    NotAvailable(String),
}

impl JmError {
    fn get_message(&self) -> &str {
        match self {
            JmError::DimError(msg) => msg,
            JmError::ConvErr(msg) => msg,
            JmError::ValueErr(msg) => msg,
            JmError::NotAvailable(msg) => msg,
        }
    }
}

impl Display for JmError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.get_message())
    }
}

impl Error for JmError {}
