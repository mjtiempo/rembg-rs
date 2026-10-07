//! Remove image backgrounds — a Rust port of python-rembg.

mod bg;
mod matting;
pub mod session;

pub use bg::{remove, remove_bytes, RemoveOptions};
pub use session::{download_models, model_names, Session};
