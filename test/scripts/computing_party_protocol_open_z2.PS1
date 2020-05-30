Start-Process Powershell.exe -WindowStyle Maximized { 
    $Env:RUST_BACKTRACE=1
    cargo test _computing_party_protocol_open_z2 --  --nocapture 0
    pause
}
Start-Process Powershell.exe -WindowStyle hidden { 
    cargo test _computing_party_protocol_open_z2 -- 1
}
