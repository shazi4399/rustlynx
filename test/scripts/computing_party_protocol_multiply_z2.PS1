Start-Process Powershell.exe -WindowStyle Maximized { 
    $EnvRUST_BACKTRACE=1
    cargo test _computing_party_protocol_multiply_z2 --  --nocapture 0
    pause
}
Start-Process Powershell.exe -WindowStyle hidden { 
    cargo test _computing_party_protocol_multiply_z2 -- 1
}
