//! Minimal zero-dependency oneshot channel for async WebGPU buffer mapping.

use std::future::Future;
use std::pin::Pin;
use std::sync::{Arc, Mutex};
use std::task::{Context, Poll, Waker};

struct State<T> {
    value: Option<T>,
    closed: bool,
    waker: Option<Waker>,
}

/// Sending half of a single-use async oneshot channel.
pub(crate) struct Sender<T> {
    state: Arc<Mutex<State<T>>>,
}

/// Receiving half of a single-use async oneshot channel.
pub(crate) struct Receiver<T> {
    state: Arc<Mutex<State<T>>>,
}

pub(crate) fn channel<T>() -> (Sender<T>, Receiver<T>) {
    let state = Arc::new(Mutex::new(State {
        value: None,
        closed: false,
        waker: None,
    }));
    (
        Sender {
            state: Arc::clone(&state),
        },
        Receiver { state },
    )
}

impl<T> Sender<T> {
    pub(crate) fn send(self, value: T) {
        let waker = {
            let mut guard = self.state.lock().unwrap_or_else(|e| e.into_inner());
            guard.value = Some(value);
            guard.closed = true;
            guard.waker.take()
        };
        if let Some(w) = waker {
            w.wake();
        }
    }
}

impl<T> Drop for Sender<T> {
    fn drop(&mut self) {
        let waker = {
            let mut guard = self.state.lock().unwrap_or_else(|e| e.into_inner());
            if guard.closed {
                return;
            }
            guard.closed = true;
            guard.waker.take()
        };
        if let Some(w) = waker {
            w.wake();
        }
    }
}

impl<T> Future for Receiver<T> {
    type Output = Option<T>;

    fn poll(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Self::Output> {
        let mut guard = self.state.lock().unwrap_or_else(|e| e.into_inner());
        if let Some(val) = guard.value.take() {
            return Poll::Ready(Some(val));
        }
        if guard.closed {
            return Poll::Ready(None);
        }
        guard.waker = Some(cx.waker().clone());
        Poll::Pending
    }
}
