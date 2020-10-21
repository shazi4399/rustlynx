Start-Process Powershell.exe -WindowStyle Maximized { 
    $EnvRUST_BACKTRACE=1
    cargo test _computing_party_ml_naive_bayes_inference --  --nocapture 0
    pause
}
Start-Process Powershell.exe -WindowStyle hidden { 
    cargo test _computing_party_ml_naive_bayes_inference -- 1
}
