//! HTTP routes: embedded static frontend + JSON APIs.

use std::sync::Arc;

use axum::extract::State;
use axum::http::{header, HeaderMap, HeaderValue, StatusCode};
use axum::response::{IntoResponse, Response};
use axum::routing::get;
use axum::Router;
use include_dir::{include_dir, Dir};

use crate::state::App;

static DIST: Dir<'_> = include_dir!("$CARGO_MANIFEST_DIR/../../assets/dist");

fn mime_for(path: &str) -> &'static str {
    match path.rsplit('.').next().unwrap_or("") {
        "html" | "htm" => "text/html; charset=utf-8",
        "css" => "text/css; charset=utf-8",
        "js" | "mjs" => "text/javascript; charset=utf-8",
        "wasm" => "application/wasm",
        "json" => "application/json",
        "svg" => "image/svg+xml",
        "png" => "image/png",
        "ico" => "image/x-icon",
        "woff2" => "font/woff2",
        _ => "application/octet-stream",
    }
}

fn serve_dist(path: &str) -> Response {
    match DIST.get_file(path) {
        Some(f) => {
            let mut headers = HeaderMap::new();
            headers.insert(
                header::CONTENT_TYPE,
                HeaderValue::from_static(mime_for(path)),
            );
            headers.insert(
                header::CACHE_CONTROL,
                HeaderValue::from_static("no-cache"),
            );
            (StatusCode::OK, headers, f.contents()).into_response()
        }
        None => (StatusCode::NOT_FOUND, "not found").into_response(),
    }
}

async fn index() -> Response {
    serve_dist("index.html")
}

async fn model_page() -> Response {
    serve_dist("model.html")
}

async fn asset_path(axum::extract::Path(path): axum::extract::Path<String>) -> Response {
    serve_dist(&path)
}

async fn pkg_path(axum::extract::Path(path): axum::extract::Path<String>) -> Response {
    serve_dist(&format!("pkg/{path}"))
}

async fn health(State(app): State<Arc<App>>) -> Response {
    let body = serde_json::json!({
        "status": if app.engine.is_some() { "ok" } else { "degraded" },
        "model": app.engine.as_ref().map(|e| e.model_name()).unwrap_or("none"),
        "voices": app.available_voices().len(),
        "uptime_s": app.started_at.elapsed().as_secs(),
    });
    axum::Json(body).into_response()
}

async fn voices(State(app): State<Arc<App>>) -> Response {
    axum::Json(serde_json::json!({ "voices": app.available_voices() })).into_response()
}

pub fn router(app: Arc<App>) -> Router {
    Router::new()
        .route("/", get(index))
        .route("/model", get(model_page))
        .route("/model.html", get(model_page))
        .route("/api/health", get(health))
        .route("/api/voices", get(voices))
        .route("/ws", get(crate::ws::ws_handler))
        .route("/api/model/status", get(crate::videomodel::status))
        .route("/api/model/generate", axum::routing::post(crate::videomodel::generate))
        .route("/pkg/{*path}", get(pkg_path))
        .route("/assets/{*path}", get(asset_path))
        .with_state(app)
}
