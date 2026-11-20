//! Small DOM helpers over web-sys.

use wasm_bindgen::{closure::Closure, JsCast, UnwrapThrowExt};
use web_sys::{
    Document, HtmlButtonElement, HtmlCanvasElement, HtmlElement, HtmlInputElement,
    HtmlSelectElement, HtmlTextAreaElement,
};

pub fn window() -> web_sys::Window {
    web_sys::window().unwrap_throw()
}

pub fn document() -> Document {
    window().document().unwrap_throw()
}

pub fn body() -> HtmlElement {
    document().body().unwrap_throw()
}

pub fn get(id: &str) -> HtmlElement {
    document()
        .get_element_by_id(id)
        .unwrap_throw()
        .dyn_into::<HtmlElement>()
        .unwrap_throw()
}

pub fn input(id: &str) -> HtmlInputElement {
    get(id).dyn_into::<HtmlInputElement>().unwrap_throw()
}

pub fn textarea(id: &str) -> HtmlTextAreaElement {
    get(id).dyn_into::<HtmlTextAreaElement>().unwrap_throw()
}

pub fn select(id: &str) -> HtmlSelectElement {
    get(id).dyn_into::<HtmlSelectElement>().unwrap_throw()
}

pub fn button(id: &str) -> HtmlButtonElement {
    get(id).dyn_into::<HtmlButtonElement>().unwrap_throw()
}

pub fn canvas(id: &str) -> HtmlCanvasElement {
    get(id).dyn_into::<HtmlCanvasElement>().unwrap_throw()
}

pub fn set_text(id: &str, text: &str) {
    get(id).set_text_content(Some(text));
}

pub fn on_click(id: &str, mut f: impl FnMut() + 'static) {
    let el = get(id);
    let c = Closure::<dyn FnMut()>::new(move || f());
    el.add_event_listener_with_callback("click", c.as_ref().unchecked_ref())
        .unwrap_throw();
    c.forget();
}

pub fn on_input(id: &str, mut f: impl FnMut() + 'static) {
    let el = get(id);
    let c = Closure::<dyn FnMut(web_sys::Event)>::new(move |_| f());
    el.add_event_listener_with_callback("input", c.as_ref().unchecked_ref())
        .unwrap_throw();
    c.forget();
}

pub fn on_keydown_enter(id: &str, mut f: impl FnMut() + 'static) {
    let el = get(id);
    let c = Closure::<dyn FnMut(web_sys::KeyboardEvent)>::new(move |e: web_sys::KeyboardEvent| {
        if e.key() == "Enter" {
            e.prevent_default();
            f();
        }
    });
    el.add_event_listener_with_callback("keydown", c.as_ref().unchecked_ref())
        .unwrap_throw();
    c.forget();
}
