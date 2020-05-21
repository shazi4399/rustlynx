cargo build --release

Start-Process Powershell.exe { 
    $Env:RUST_BACKTRACE=1
    cargo run cfg=cfg/Party0.toml
    pause
}
Start-Process Powershell.exe { 
    $Env:RUST_BACKTRACE=1
    cargo run cfg=cfg/Party1.toml
    pause
}
