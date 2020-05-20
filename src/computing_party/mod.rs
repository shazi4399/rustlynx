use std::error::Error;

pub mod protocol;
pub mod init;
pub mod constants;
mod util;

pub fn run( cfg_file : String ) -> Result<(), Box<dyn Error>> {

	let mut ctx = init::runtime_context( &cfg_file )?;
	init::connection( &mut ctx )?;

		

	println!("rustlynx::computing_party::run: runtime context initialized\n\n{}\n", &ctx);

	// match ctx.ml.model {
	// 	"logisticregression": super::ml::logistic_regression();
	// }

	Ok(())

}


