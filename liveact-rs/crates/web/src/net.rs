//! WebSocket session (JSON control frames + binary audio frames).

use std::cell::RefCell;
use std::rc::Rc;

use liveact_proto::{ClientMsg, ServerMsg};
use wasm_bindgen::{closure::Closure, JsCast, UnwrapThrowExt};
use web_sys::{BinaryType, MessageEvent, WebSocket};

type MsgFn = Rc<RefCell<Box<dyn FnMut(ServerMsg)>>>;
type BinFn = Rc<RefCell<Box<dyn FnMut(Vec<u8>)>>>;
type StateFn = Rc<RefCell<Box<dyn FnMut(&'static str)>>>;

pub struct Handlers {
    pub on_msg: MsgFn,
    pub on_binary: BinFn,
    pub on_state: StateFn,
}

pub struct WsSession {
    ws: WebSocket,
}

impl WsSession {
    pub fn connect(url: String, handlers: Handlers) -> Result<Self, String> {
        let ws = WebSocket::new(&url).map_err(|e| format!("ws open failed: {e:?}"))?;
        ws.set_binary_type(BinaryType::Arraybuffer);

        // onopen
        {
            let on_state = handlers.on_state.clone();
            let c = Closure::<dyn FnMut()>::new(move || {
                (on_state.borrow_mut())("connected");
            });
            ws.set_onopen(Some(c.as_ref().unchecked_ref()));
            c.forget();
        }
        // onclose (page reload reconnects; sessions are stateless)
        {
            let on_state = handlers.on_state.clone();
            let c = Closure::<dyn FnMut()>::new(move || {
                (on_state.borrow_mut())("disconnected");
            });
            ws.set_onclose(Some(c.as_ref().unchecked_ref()));
            c.forget();
        }
        // onerror
        {
            let on_state = handlers.on_state.clone();
            let c = Closure::<dyn FnMut(web_sys::ErrorEvent)>::new(
                move |e: web_sys::ErrorEvent| {
                    (on_state.borrow_mut())("error");
                    web_sys::console::log_1(&format!("ws error: {}", e.message()).into());
                },
            );
            ws.set_onerror(Some(c.as_ref().unchecked_ref()));
            c.forget();
        }
        // onmessage
        {
            let on_msg = handlers.on_msg.clone();
            let on_binary = handlers.on_binary.clone();
            let c = Closure::<dyn FnMut(MessageEvent)>::new(move |ev: MessageEvent| {
                let data = ev.data();
                if data.is_string() {
                    let text = data.as_string().unwrap_or_default();
                    match serde_json::from_str::<ServerMsg>(&text) {
                        Ok(m) => (on_msg.borrow_mut())(m),
                        Err(e) => web_sys::console::log_1(
                            &format!("bad server message: {e} :: {text}").into(),
                        ),
                    }
                } else if !data.is_null() && !data.is_undefined() {
                    let bytes = js_sys::Uint8Array::new(&data).to_vec();
                    (on_binary.borrow_mut())(bytes);
                }
            });
            ws.set_onmessage(Some(c.as_ref().unchecked_ref()));
            c.forget();
        }

        Ok(WsSession { ws })
    }

    pub fn send(&self, msg: &ClientMsg) -> bool {
        match serde_json::to_string(msg) {
            Ok(s) => self.ws.send_with_str(&s).is_ok(),
            Err(_) => false,
        }
    }
}
