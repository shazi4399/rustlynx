function test($path, $id) {
	$fullpath=$path + "Party" + $id + ".toml"
	cargo test _computing_party_protocol_multiply --  --nocapture $fullpath 
}



