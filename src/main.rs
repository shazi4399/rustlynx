extern crate rustlynx;

use std::error::Error;
use std::env;

fn main() -> Result<(), Box<dyn Error>> {

	let args = rustlynx::init::argparse( env::args().collect() )?;

	rustlynx::run( args )?;

    Ok(())
}
