## Install

```sh
rustup target install wasm32-unknown-unknown
cargo install wasm-bindgen-cli
```

## Build

```sh
cargo build --target wasm32-unknown-unknown --release
wasm-bindgen --out-dir pkg --target no-modules target/wasm32-unknown-unknown/release/My_Game.wasm
```

## Serve

```sh
python3 serve.py
```

Then open `http://localhost:8080` in your browser.
