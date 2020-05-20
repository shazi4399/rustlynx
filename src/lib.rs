extern crate itertools;
use std::error::Error;

pub mod computing_party;
pub mod trusted_initializer;
pub mod init;
pub mod io;

pub fn run( args: init::Args ) -> Result<(), Box<dyn Error>> {

	if args.ti {
		trusted_initializer::run( args.cfg_file )?;
	} else {
		computing_party::run( args.cfg_file )?;
	}

	Ok(())
}


#[cfg(test)]
mod tests {

    #[test]
    fn _init_argparse() {

        use super::init;

    	/* ti=bool not specified, cfg specified + exists */
    	assert!( init::argparse(vec![ String::from(""),String::from("cfg=src/main.rs")]).is_ok() );
    	/* ti=false, cfg specified + exists */
    	assert!( init::argparse(vec![ String::from(""),String::from("ti=false"),String::from("cfg=src/main.rs")]).is_ok() );
    	/* ti=true, cfg specified + exists */
    	assert!( init::argparse(vec![ String::from(""),String::from("ti=true"),String::from("cfg=src/main.rs")]).is_ok() );
    	/* cfg not specified */
    	assert!( init::argparse(vec![ String::from("")]).is_err() );
    	/* cfg specified but doesn't exist */
    	assert!( init::argparse(vec![ String::from(""),String::from(""),String::from("cfg=not/a/real/file.txt")]).is_err() );
    	/* ti=bool specified incorrectly */
    	assert!( init::argparse(vec![ String::from(""),String::from("ti=NotTrueOrFalse"),String::from("cfg=src/main.rs")]).is_err() );
    	/* missing key=val format */
    	assert!( init::argparse(vec![ String::from(""),String::from(""),String::from("cfg:src/main.rs")]).is_err() );
    	/* unrecognized argument */
    	assert!( init::argparse(vec![ String::from(""),String::from("cfg=src/main.rs"), String::from("this=IsNotAValidArgument")]).is_err() );
    }

    #[test]
    fn _computing_party_protocol_open() {

        use std::env;
        use std::num::Wrapping;
        use std::time::SystemTime;
        use super::computing_party::protocol;
        use super::computing_party::init;
        use super::computing_party::constants;
        use super::io;
        let test_path = "test/files/computing_party_protocol_open";

        let args: Vec<String> = env::args().collect();
        let id = &args[args.len()-1];
        let test_cfg = format!("{}/Party{}.toml", test_path, id); 
        let mut ctx = init::runtime_context( &test_cfg ).unwrap();
        
        assert!( init::connection( &mut ctx ).is_ok() );
        
        /* random tests */
        let test_file = format!("{}/p{}.csv", test_path, ctx.num.asymm);
        let itc_file = format!("{}/itc.csv", test_path);

        let input = io::single_col_csv_to_wrapping_vec(&test_file).unwrap();
        let check = io::single_col_csv_to_wrapping_vec(&itc_file).unwrap();

        let result = protocol::open(&input, &mut ctx).unwrap();

        assert_eq!(&result, &check);

        /* empty vec test */
        let input: Vec<Wrapping<u64>> = Vec::new();
        assert_eq!(input, protocol::open(&input, &mut ctx).unwrap());

        /* runtime tests */
        for input_size in vec![100000, 1000000, 10000000].iter() {
        
            let input = (0..*input_size)
                .map(|x| Wrapping(if ctx.num.asymm == 0 {0} else {x})).collect(); 
            let check = (0..*input_size).map(|x| Wrapping(x)).collect::<Vec<Wrapping<u64>>>();
           
            for n_threads in vec![1, 2, 4, 8, 16, 32].iter() {    
                
                ctx.sys.threads.online = *n_threads;
            
                let mut avg_elapsed = 0.0;
                let num_tests = 10;
                for test_no in 0..num_tests {
                    let now = SystemTime::now();
                    let result = protocol::open(&input, &mut ctx).unwrap();
                    let elapsed = now.elapsed().unwrap().as_millis();
                    avg_elapsed += (elapsed as f64) / (num_tests as f64);
                    assert_eq!(result, check); 
                }

                println!("size={:10}, n_threads={:2}, work time {:5.0} ms", 
                        input_size, ctx.sys.threads.online, avg_elapsed);               
            } 
        }
    }

    #[test]
    fn _computing_party_protocol_multiply() {

        use std::env;
        use std::num::Wrapping;
        use std::time::SystemTime;
        use super::computing_party::protocol;
        use super::computing_party::init;
        use super::computing_party::constants;
        use super::io;
        let test_path = "test/files/computing_party_protocol_multiply";

        let args: Vec<String> = env::args().collect();
        let id = &args[args.len()-1];
        let test_cfg = format!("{}/Party{}.toml", test_path, id); 
        let mut ctx = init::runtime_context( &test_cfg ).unwrap();
        
        assert!( init::connection( &mut ctx ).is_ok() );
        
        /* empty vec test */
        let input = Vec::<Wrapping<u64>>::new();
        assert_eq!( &input, &protocol::multiply(&input, &input, &mut ctx).unwrap() );

        /* random tests */
        let x_file = format!("{}/p{}_x.csv", test_path, ctx.num.asymm);
        let y_file = format!("{}/p{}_y.csv", test_path, ctx.num.asymm);
        let itc_file = format!("{}/itc.csv", test_path);

        let x = io::single_col_csv_to_wrapping_vec(&x_file).unwrap();
        let y = io::single_col_csv_to_wrapping_vec(&y_file).unwrap();
        let check = io::single_col_csv_to_wrapping_vec(&itc_file).unwrap();

        let result = protocol::multiply(&x, &y, &mut ctx).unwrap();
        let result = protocol::open(&result, &mut ctx).unwrap();
        assert_eq!(&result, &check);
   
        /* runtime tests */
        for input_size in vec![100000, 1000000, 10000000].iter() {
        
            let input = (0..*input_size)
                .map(|x| Wrapping(if ctx.num.asymm == 0 {0} else {x})).collect(); 
            let check = (0..*input_size).map(|x| Wrapping(x*x)).collect::<Vec<Wrapping<u64>>>();
           
            for n_threads in vec![1, 2, 4, 8, 16, 32].iter() {    
                
                ctx.sys.threads.online = *n_threads;
            
                let mut avg_elapsed = 0.0;
                let num_tests = 1;
                for test_no in 0..num_tests {
                    let now = SystemTime::now();
                    let result = protocol::multiply(&input, &input, &mut ctx).unwrap();
                    let elapsed = now.elapsed().unwrap().as_millis();
                    avg_elapsed += (elapsed as f64) / (num_tests as f64);
                    assert_eq!(protocol::open(&result, &mut ctx).unwrap(), check); 
                }

                println!("size={:10}, n_threads={:2}, work time {:5.0} ms", 
                        input_size, ctx.sys.threads.online, avg_elapsed);               
            } 
        }
    }
}
