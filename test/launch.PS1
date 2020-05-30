Start-Process Powershell.exe -WindowStyle Maximized { 
    $Env:RUST_BACKTRACE=1
    cargo test _ --  --nocapture --test-threads=1 0
    pause
}
Start-Process Powershell.exe -WindowStyle hidden { 
    cargo test _ -- --test-threads=1 1
}
