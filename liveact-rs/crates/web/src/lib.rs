//! LiveAct-RS web frontend (Rust -> WASM, no JS framework).
//!
//! Entry point: `start()`, called by the bootstrap snippet in index.html.

mod app;
mod audio;
mod avatar;
mod dom;
mod expression;
mod model_lab;
mod net;
mod viseme;

use wasm_bindgen::prelude::*;

#[wasm_bindgen]
pub fn start() -> Result<(), JsValue> {
    // Video Model Lab on /model.html (or #model), live studio otherwise
    let hash_model = dom::window()
        .location()
        .hash()
        .map(|h| h.contains("model"))
        .unwrap_or(false);
    let path_model = dom::window()
        .location()
        .pathname()
        .map(|p| p.contains("model"))
        .unwrap_or(false);
    if hash_model || path_model {
        model_lab::boot().map_err(|e| JsValue::from_str(&e))?;
        return Ok(());
    }
    let app = app::boot().map_err(|e| JsValue::from_str(&e))?;
    app::wire_ui(&app);
    app::start_render_loop(app);
    Ok(())
}
