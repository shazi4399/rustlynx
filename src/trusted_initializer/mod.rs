use std::error::Error;

pub fn run(cfg_file: String) -> Result<(), Box<dyn Error>> {

	println!("rustlynx::trusted_initializer::run: {}", &cfg_file);

	Ok(())
	
}