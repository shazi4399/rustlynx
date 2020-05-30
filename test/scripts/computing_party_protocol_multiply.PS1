Start-Process Powershell.exe -WindowStyle Maximized { 
    $Env:RUST_BACKTRACE=1
    cargo test _computing_party_protocol_multiply --  --nocapture 0
    pause
}
Start-Process Powershell.exe -WindowStyle hidden { 
    cargo test _computing_party_protocol_multiply -- 1
}
