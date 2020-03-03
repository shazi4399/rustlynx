extern crate itertools;

use std::error::Error;

pub mod computing_party;
pub mod trusted_initializer;
pub mod init;


pub fn run( args: init::Args ) -> Result<(),Box<dyn Error>> {

	if args.ti {
		trusted_initializer::run( args.cfg_file )?;
	} else {
		computing_party::run( args.cfg_file )?;
	}

	Ok(())
}

#[cfg(test)]
mod tests {

	use super::init;

    #[test]
    fn init_argparse() {
    	/* ti=bool not specified, cfg_file specified + exists */
    	assert!( init::argparse(vec![ String::from(""),String::from("cfg_file=src/main.rs")]).is_ok() );
    	/* ti=false, cfg_file specified + exists */
    	assert!( init::argparse(vec![ String::from(""),String::from("ti=false"),String::from("cfg_file=src/main.rs")]).is_ok() );
    	/* ti=true, cfg_file specified + exists */
    	assert!( init::argparse(vec![ String::from(""),String::from("ti=true"),String::from("cfg_file=src/main.rs")]).is_ok() );
    	/* cfg_file not specified */
    	assert!( init::argparse(vec![ String::from("")]).is_err() );
    	/* cfg_file specified but doesn't exist */
    	assert!( init::argparse(vec![ String::from(""),String::from(""),String::from("cfg_file=not/a/real/file.txt")]).is_err() );
    	/* ti=bool specified incorrectly */
    	assert!( init::argparse(vec![ String::from(""),String::from("ti=NotTrueOrFalse"),String::from("cfg_file=src/main.rs")]).is_err() );
    	/* missing key=val format */
    	assert!( init::argparse(vec![ String::from(""),String::from(""),String::from("cfg_file:src/main.rs")]).is_err() );
    	/* unrecognized argument */
    	assert!( init::argparse(vec![ String::from(""),String::from("cfg_file=src/main.rs"), String::from("this=IsNotAValidArgument")]).is_err() );
    }
}
