use itertools::Itertools;
use std::error::Error;
use std::path::Path;

/* Initial command line arguments */
#[derive(Debug)]
pub struct Args {
	
	pub cfg_file: String,
	pub ti: bool,
}

pub fn argparse(args: Vec<String>) -> Result<Args, Box<dyn Error>> {

	let mut cfg_file = String::new();
	let mut ti       = false;

	for arg in args[1..].iter() {

		let (key, val) = match arg.split('=').next_tuple() {
			Some(tup) => tup,
			None => return Err( format!("missing key=val fmt for arg: '{}'", &arg).into() ), 
		};

		match key {
			"ti" => match val.parse() {
						Ok(boolean) => ti = boolean,
						Err(_) => return Err( format!("cannot parse '{}' into boolean", &val).into() ),
					},

			"cfg_file" => if Path::new(val).exists() { 
							cfg_file = String::from(val); 
						} else { 
							return Err( format!("file path '{}' not found", &val).into() ); 
						},

			_ => return Err( format!("unrecognized cmd line arg: {}", &arg).into() ),
		}
	}

	if cfg_file.is_empty() {
		return Err( format!("cfg filepath must be specified").into());
	}

	Ok(Args {
		cfg_file : cfg_file, 
		ti       : ti,
	})
}
