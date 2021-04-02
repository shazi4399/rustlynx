extern crate itertools;
extern crate rand;
use std::error::Error;

pub mod computing_party;
pub mod trusted_initializer;
pub mod init;
pub mod io;
pub mod util;
pub mod constants;

pub fn run( args: init::Args ) -> Result<(), Box<dyn Error>> {

	if args.ti {
		trusted_initializer::run( args.cfg_file )
	} else {
		computing_party::run( args.cfg_file )
    }  
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
    fn _io_single_col_csv_to_u128_vec() {

        use super::io;

        let test_file = "test/files/io/single_col_u128.csv";
        let vec_u128 = io::single_col_csv_to_u128_vec(&test_file).unwrap();

        assert_eq!(&vec_u128, &vec![0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 
            0xffffffffffffffffffffffffffffffff, 0xFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF]);

    }

    // #[test]
    // fn _util_compress_bit_vector() {

    //     use super::util;
    //     use rand::{thread_rng, Rng};
        
    //     /* empty vec -> empty vec */
    //     let output = util::compress_bit_vector(&vec![], 0, 1, false, 0).unwrap();
    //     assert_eq!(&output, &vec![]);

    //     // /* KNOWN ERRORS: bitlen less than 1 */
    //     assert!(util::compress_bit_vector(&vec![0x0 ; 1], 1, 0, false, 0).is_err());
    //     /* bitlen greater than 128 w/o padding */
    //     assert!(util::compress_bit_vector(&vec![0x0 ; 1], 1, 129, false, 0).is_err());
    //     /* bitlen greater than 128 w/ padding */
    //     assert!(util::compress_bit_vector(&vec![0x0 ; 1], 1, 128, true, 0).is_err());

    //     /* EDGE CASES: len: 128 of 0x1's -> len: 1 of 0xfff..f  */
    //     let output = util::compress_bit_vector(&vec![1u128 ; 128], 128, 1, false, 0).unwrap();
    //     assert_eq!(&output, &vec![0xffffffffffffffffffffffffffffffff ; 1]);
    //     /* 1's w/ padding 0's */
    //     let output = util::compress_bit_vector(&vec![1u128 ; 128], 128, 1, true, 0).unwrap();
    //     assert_eq!(&output, &vec![0xaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa ; 2]);
    //     /* 1's w/ padding 1's */
    //     let output = util::compress_bit_vector(&vec![1u128 ; 128], 128, 1, true, 1).unwrap();
    //     assert_eq!(&output, &vec![0xffffffffffffffffffffffffffffffff ; 2]);
    //     /* 0's w/ padding 0's */
    //     let output = util::compress_bit_vector(&vec![0u128 ; 128], 128, 1, true, 0).unwrap();
    //     assert_eq!(&output, &vec![0x00000000000000000000000000000000 ; 2]);
    //     /* 0's w/ padding 1's */
    //     let output = util::compress_bit_vector(&vec![0u128 ; 128], 128, 1, true, 1).unwrap();
    //     assert_eq!(&output, &vec![0x55555555555555555555555555555555 ; 2]);
    //     /* len: 128 of (2^127-1)'s -> len: 127 of 0xfff..f */
    //     let output = util::compress_bit_vector(&vec![0x7fffffffffffffffffffffffffffffff ; 128], 128, 127, false, 0).unwrap();
    //     assert_eq!(&output, &vec![0xffffffffffffffffffffffffffffffff ; 127]);
    //     /* 1's w/ padding 0's */
    //     let output = util::compress_bit_vector(&vec![0x7fffffffffffffffffffffffffffffff ; 128], 128, 127, true, 0).unwrap();
    //     assert_eq!(&output, &vec![0xfffffffffffffffffffffffffffffffe ; 128]);
    //     /* 1's w/ padding 1's */
    //     let output = util::compress_bit_vector(&vec![0x7fffffffffffffffffffffffffffffff ; 128], 128, 127, true, 1).unwrap();
    //     assert_eq!(&output, &vec![0xffffffffffffffffffffffffffffffff ; 128]);
    //     /* 0's w/ padding 0's */
    //     let output = util::compress_bit_vector(&vec![0x00000000000000000000000000000000 ; 128], 128, 127, true, 0).unwrap();
    //     assert_eq!(&output, &vec![0x00000000000000000000000000000000 ; 128]);
    //     /* 0's w/ padding 0's */
    //     let output = util::compress_bit_vector(&vec![0x00000000000000000000000000000000 ; 128], 128, 127, true, 1).unwrap();
    //     assert_eq!(&output, &vec![0x00000000000000000000000000000001 ; 128]);
    //     /* len: 1 of 0xfff..f -> len: 1 of 0xfff..f */
    //     let output = util::compress_bit_vector(&vec![0xffffffffffffffffffffffffffffffff ; 1], 1, 128, false, 0).unwrap();
    //     assert_eq!(&output, &vec![0xffffffffffffffffffffffffffffffff ; 1]);

    //     /* RANDOM TESTS */
    //     let mut rng = rand::thread_rng();
    //     let n_tests = 10000;
    //     let max_len = 10000;
        
    //     for i in 0..n_tests {

    //         let len = 1 + rng.gen::<usize>() % (max_len - 1); 
    //         let bitlen = 1 + rng.gen::<usize>() % 127;
    //         let pad = rng.gen::<bool>();
    //         let asymm = rng.gen::<u64>() & 1;
    //         let bitmask = (1 << bitlen) - 1;

    //         let input = (0..len).map(|_i| ((rng.gen::<u64>() as u128) | ((rng.gen::<u64>() as u128) << 64)) & bitmask).collect::<Vec<u128>>();
    //         let compressed = util::compress_bit_vector(&input, len, bitlen, pad, asymm).unwrap();
    //         let output = util::decompress_bit_vector(&compressed, len, bitlen, pad, asymm).unwrap();

    //         assert_eq!(&input, &output);
    //     }
    // }

    // #[test]
    // fn _util_decompress_bit_vector() {

    //     use super::util;

    //     /* empty vec -> empty vec */
    //     let output = util::decompress_bit_vector(&vec![], 0, 1, false, 0).unwrap();
    //     assert_eq!(&output, &vec![]);

    //     /* KNOWN ERRORS: bitlen less than 1 */
    //     assert!(util::decompress_bit_vector(&vec![0x0 ; 1], 1, 0, false, 0).is_err());
    //     /* bitlen greater than 128 w/o padding */
    //     assert!(util::decompress_bit_vector(&vec![0x0 ; 1], 1, 129, false, 0).is_err());
    //     /* bitlen greater than 128 w/ padding */
    //     assert!(util::decompress_bit_vector(&vec![0x0 ; 1], 1, 128, true, 0).is_err());

    //     /* EDGE CASES: len: 128 of 0x1's -> len: 1 of 0xfff..f  */
    //     let output = util::decompress_bit_vector(&vec![0xffffffffffffffffffffffffffffffff ; 1], 128, 1, false, 0).unwrap();
    //     assert_eq!(&output, &vec![0x1 ; 128]);
    //     /* 1's w/ padding 0's */
    //     let output = util::decompress_bit_vector(&vec![0xaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa ; 2], 128, 1, true, 0).unwrap();
    //     assert_eq!(&output, &vec![0x1 ; 128]);
    //     /* 1's w/ padding 1's */
    //     let output = util::decompress_bit_vector(&vec![0xffffffffffffffffffffffffffffffff ; 2], 128, 1, true, 1).unwrap();
    //     assert_eq!(&output, &vec![0x1 ; 128]);
    //     /* 0's w/ padding 0's */
    //     let output = util::decompress_bit_vector(&vec![0x00000000000000000000000000000000 ; 2], 128, 1, true, 0).unwrap();
    //     assert_eq!(&output, &vec![0u128 ; 128]);
    //     /* 0's w/ padding 1's */
    //     let output = util::decompress_bit_vector(&vec![0x55555555555555555555555555555555 ; 2], 128, 1, true, 1).unwrap();
    //     assert_eq!(&output,&vec![0u128 ; 128]);
    //     /* len: 128 of (2^127-1)'s -> len: 127 of 0xfff..f */
    //     let output = util::decompress_bit_vector(&vec![0xffffffffffffffffffffffffffffffff ; 127], 128, 127, false, 0).unwrap();
    //     assert_eq!(&output, &vec![0x7fffffffffffffffffffffffffffffff ; 128]);
    //     /* 1's w/ padding 0's */
    //     let output = util::decompress_bit_vector(&vec![0xfffffffffffffffffffffffffffffffe ; 128], 128, 127, true, 0).unwrap();
    //     assert_eq!(&output, &vec![0x7fffffffffffffffffffffffffffffff ; 128]);
    //     /* 1's w/ padding 1's */
    //     let output = util::decompress_bit_vector(&vec![0xffffffffffffffffffffffffffffffff ; 128], 128, 127, true, 1).unwrap();
    //     assert_eq!(&output, &vec![0x7fffffffffffffffffffffffffffffff ; 128]);
    //     /* 0's w/ padding 0's */
    //     let output = util::decompress_bit_vector(&vec![0x00000000000000000000000000000000 ; 128], 128, 127, true, 0).unwrap();
    //     assert_eq!(&output, &vec![0x00000000000000000000000000000000 ; 128]);
    //     /* 0's w/ padding 0's */
    //     let output = util::decompress_bit_vector(&vec![0x00000000000000000000000000000001 ; 128], 128, 127, true, 1).unwrap();
    //     assert_eq!(&output, &vec![0x00000000000000000000000000000000 ; 128]);
    //     /* len: 1 of 0xfff..f -> len: 1 of 0xfff..f */
    //     let output = util::decompress_bit_vector(&vec![0xffffffffffffffffffffffffffffffff ; 1], 1, 128, false, 0).unwrap();
    //     assert_eq!(&output, &vec![0xffffffffffffffffffffffffffffffff ; 1]);

    //     /* random tests in _util_compress_bit_vector */

    // }
 
    // #[test]
    // fn _util_compress_from_tesselated_bit_vector() {

    //     use super::util;
    //     use rand::{thread_rng, Rng};

    //     /* empty vec -> empty vec */
    //     let output = util::compress_from_tesselated_bit_vector(&vec![], 0, 1, false, 0).unwrap();
    //     assert_eq!(&output, &vec![]);

    //     let output = util::compress_from_tesselated_bit_vector(&vec![0xaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa], 64, 1, true, 0).unwrap();
    //     assert_eq!(&output, &vec![0xaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa]);

    //     let output = util::compress_from_tesselated_bit_vector(&vec![0xaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa], 64, 1, true, 1).unwrap();
    //     assert_eq!(&output, &vec![0xffffffffffffffffffffffffffffffff]);
        
    //     /* RANDOM TESTS */
    //     let mut rng = rand::thread_rng();
    //     let n_tests = 10000;
    //     let max_len = 1000;
        
    //     for i in 0..n_tests {

    //         let len = 10000 + rng.gen::<usize>() % (max_len - 1); 
    //         let bitlen = 1 + rng.gen::<usize>() % 64;
    //         let pad = true;
    //         let asymm = rng.gen::<u64>() & 1;
    //         let bitmask = (1 << bitlen) - 1;

    //         let input = (0..len).map(|_i| ((rng.gen::<u64>() as u128) | ((rng.gen::<u64>() as u128) << 64)) & bitmask).collect::<Vec<u128>>();
    //         let compressed = util::compress_bit_vector(&input, len, bitlen, pad, asymm).unwrap();
    //         let tesselated = compressed.iter().map(|x| x & 0xaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa).collect::<Vec<u128>>(); 
    //         let tesselated_bitlen = (bitlen + 1) >> 1;
    //         let compressed = util::compress_from_tesselated_bit_vector(&tesselated, len, tesselated_bitlen, pad, asymm).unwrap();
    //         let output = util::decompress_bit_vector(&compressed, len, tesselated_bitlen, pad, asymm).unwrap();

    //         let mut expected = vec![0u128 ; len];
    //         let shift = if pad && (bitlen & 1 == 1) {0} else {1};
    //         for i in 0..len {
    //             for j in 0..tesselated_bitlen {
    //                 expected[i] |= ((input[i] >> 2*j + shift ) & 1) << j;
    //             }
    //         }

    //         //assert_eq!(&expected, &output);
    //         assert_eq!(expected.len(), output.len());
    //         // print!("I:");
    //         // for el in &input {
    //         //     print!(" {:b}", el);
    //         // }
    //         // println!();

    //         // print!("O:");
    //         // for el in &output {
    //         //     print!(" {:b}", el);
    //         // }
    //         // println!();

    //         // print!("E:");
    //         // for el in &expected {
    //         //     print!(" {:b}", el);
    //         // }
    //         // println!();
    //     }



    // }

    // #[test]
    // fn _computing_party_protocol_open() {

    //     use std::env;
    //     use std::num::Wrapping;
    //     use std::time::SystemTime;
    //     use super::computing_party::protocol;
    //     use super::computing_party::init;
    //     use super::io;
    //     let test_path = "test/files/computing_party_protocol_open";

    //     let args: Vec<String> = env::args().collect();
    //     let id = &args[args.len()-1];
    //     let test_cfg = format!("{}/Party{}.toml", test_path, id); 
    //     let mut ctx = init::runtime_context( &test_cfg ).unwrap();
        
    //     /* connection */
    //     assert!(init::connection( &mut ctx ).is_ok() );
        
    //     /* empty vec test */
    //     let input: Vec<Wrapping<u64>> = Vec::new();
    //     assert_eq!(input, protocol::open(&input, &mut ctx).unwrap());

    //     /* random tests */
    //     let test_file = format!("{}/p{}.csv", test_path, ctx.num.asymm);
    //     let itc_file = format!("{}/itc.csv", test_path);

    //     let input = io::single_col_csv_to_wrapping_vec(&test_file).unwrap();
    //     let check = io::single_col_csv_to_wrapping_vec(&itc_file).unwrap();

    //     let result = protocol::open(&input, &mut ctx).unwrap();

    //     assert_eq!(&result, &check);

    //     /* runtime tests */
    //     for input_size in vec![100000, 1000000, 10000000].iter() {
        
    //         let input = (0..*input_size)
    //             .map(|x| Wrapping(if ctx.num.asymm == 0 {0} else {x})).collect(); 
    //         let check = (0..*input_size).map(|x| Wrapping(x)).collect::<Vec<Wrapping<u64>>>();
           
    //         for n_threads in vec![4, 8, 16, 32].iter() {    
                
    //             ctx.sys.threads.online = *n_threads;
            
    //             let mut avg_elapsed = 0.0;
    //             let num_tests = 1;
    //             for _test_no in 0..num_tests {
    //                 let now = SystemTime::now();
    //                 let result = protocol::open(&input, &mut ctx).unwrap();
    //                 let elapsed = now.elapsed().unwrap().as_millis();
    //                 avg_elapsed += (elapsed as f64) / (num_tests as f64);
    //                 assert_eq!(result, check); 
    //             }

    //             println!("size={:10}, n_threads={:2}, work time {:5.0} ms", 
    //                     input_size, ctx.sys.threads.online, avg_elapsed);               
    //         } 
    //     }
    // }

    // #[test]
    // fn _computing_party_protocol_open_z2() {

    //     use std::env;
    //     use std::num::Wrapping;
    //     use std::time::SystemTime;
    //     use super::computing_party::protocol;
    //     use super::computing_party::init;
    //     use super::io;
    //     let test_path = "test/files/computing_party_protocol_open_z2";

    //     let args: Vec<String> = env::args().collect();
    //     let id = &args[args.len()-1];
    //     let test_cfg = format!("{}/Party{}.toml", test_path, id); 
    //     let mut ctx = init::runtime_context( &test_cfg ).unwrap();
        
    //     /* connection */
    //     assert!( init::connection( &mut ctx ).is_ok() );
        
    //     /* empty vec test */
    //     let input: Vec<u128> = Vec::new();
    //     assert_eq!(input, protocol::open_z2(&input, &mut ctx).unwrap());

    //     /* random tests */
    //     let test_file = format!("{}/p{}.csv", test_path, ctx.num.asymm);
    //     let itc_file = format!("{}/itc.csv", test_path);

    //     let input = io::single_col_csv_to_u128_vec(&test_file).unwrap();
    //     let check = io::single_col_csv_to_u128_vec(&itc_file).unwrap();

    //     let result = protocol::open_z2(&input, &mut ctx).unwrap();

    //     assert_eq!(&result, &check);

    //     /* runtime tests */
    //     for input_size in vec![100000, 1000000, 10000000].iter() {
        
    //         let input = (0..*input_size)
    //             .map(|x| if ctx.num.asymm == 0 {0} else {x}).collect::<Vec<u128>>(); 
    //         let check = (0..*input_size).map(|x| x).collect::<Vec<u128>>();
           
    //         for n_threads in vec![4, 8, 16, 32].iter() {    
                
    //             ctx.sys.threads.online = *n_threads;
            
    //             let mut avg_elapsed = 0.0;
    //             let num_tests = 1;
    //             for _test_no in 0..num_tests {
    //                 let now = SystemTime::now();
    //                 let result = protocol::open_z2(&input, &mut ctx).unwrap();
    //                 let elapsed = now.elapsed().unwrap().as_millis();
    //                 avg_elapsed += (elapsed as f64) / (num_tests as f64);
    //                 assert_eq!(result, check); 
    //             }

    //             println!("size={:10}, n_threads={:2}, work time {:5.0} ms", 
    //                     input_size, ctx.sys.threads.online, avg_elapsed);               
    //         } 
    //     }
    // }

    // #[test]
    // fn _computing_party_protocol_multiply() {

    //     use std::env;
    //     use std::num::Wrapping;
    //     use std::time::SystemTime;
    //     use super::computing_party::protocol;
    //     use super::computing_party::init;
    //     use super::io;
    //     let test_path = "test/files/computing_party_protocol_multiply";

    //     let args: Vec<String> = env::args().collect();
    //     let id = &args[args.len()-1];
    //     let test_cfg = format!("{}/Party{}.toml", test_path, id); 
    //     let mut ctx = init::runtime_context( &test_cfg ).unwrap();
        
    //     /* connect */
    //     assert!( init::connection( &mut ctx ).is_ok() );
        
    //     /* empty vec test */
    //     let input = Vec::<Wrapping<u64>>::new();
    //     assert_eq!( &input, &protocol::multiply(&input, &input, &mut ctx).unwrap() );

    //     /* random tests */
    //     let x_file = format!("{}/p{}_x.csv", test_path, ctx.num.asymm);
    //     let y_file = format!("{}/p{}_y.csv", test_path, ctx.num.asymm);
    //     let itc_file = format!("{}/itc.csv", test_path);

    //     let x = io::single_col_csv_to_wrapping_vec(&x_file).unwrap();
    //     let y = io::single_col_csv_to_wrapping_vec(&y_file).unwrap();
    //     let check = io::single_col_csv_to_wrapping_vec(&itc_file).unwrap();

    //     let result = protocol::multiply(&x, &y, &mut ctx).unwrap();
    //     let result = protocol::open(&result, &mut ctx).unwrap();
    //     assert_eq!(&result, &check);
   
    //     /* runtime tests */
    //     for input_size in vec![100000, 1000000, 10000000].iter() {
        
    //         let input = (0..*input_size)
    //             .map(|x| Wrapping(if ctx.num.asymm == 0 {0} else {x})).collect(); 
    //         let check = (0..*input_size).map(|x| Wrapping(x*x)).collect::<Vec<Wrapping<u64>>>();
           
    //         for n_threads in vec![4, 8, 16, 32].iter() {    
                
    //             ctx.sys.threads.online = *n_threads;
            
    //             let mut avg_elapsed = 0.0;
    //             let num_tests = 1;
    //             for _test_no in 0..num_tests {
    //                 let now = SystemTime::now();
    //                 let result = protocol::multiply(&input, &input, &mut ctx).unwrap();
    //                 let elapsed = now.elapsed().unwrap().as_millis();
    //                 avg_elapsed += (elapsed as f64) / (num_tests as f64);
    //                 assert_eq!(protocol::open(&result, &mut ctx).unwrap(), check); 
    //             }

    //             println!("size={:10}, n_threads={:2}, work time {:5.0} ms", 
    //                     input_size, ctx.sys.threads.online, avg_elapsed);               
    //         } 
    //     }
    // }

    // #[test]
    // fn _computing_party_protocol_multiply_z2() {

    //     use std::env;
    //     use std::num::Wrapping;
    //     use std::time::SystemTime;
    //     use super::computing_party::protocol;
    //     use super::computing_party::init;
    //     use super::io;
    //     let test_path = "test/files/computing_party_protocol_multiply_z2";

    //     let args: Vec<String> = env::args().collect();
    //     let id = &args[args.len()-1];
    //     let test_cfg = format!("{}/Party{}.toml", test_path, id); 
    //     let mut ctx = init::runtime_context( &test_cfg ).unwrap();
        
    //     /* connect */
    //     assert!( init::connection( &mut ctx ).is_ok() );
        
    //     /* empty vec test */
    //     let input = Vec::<u128>::new();
    //     assert_eq!( &input, &protocol::multiply_z2(&input, &input, &mut ctx).unwrap() );

    //     /* random tests */
    //     let x_file = format!("{}/p{}_x.csv", test_path, ctx.num.asymm);
    //     let y_file = format!("{}/p{}_y.csv", test_path, ctx.num.asymm);
    //     let itc_file = format!("{}/itc.csv", test_path);

    //     let x = io::single_col_csv_to_u128_vec(&x_file).unwrap();
    //     let y = io::single_col_csv_to_u128_vec(&y_file).unwrap();
    //     let check = io::single_col_csv_to_u128_vec(&itc_file).unwrap();

    //     let result = protocol::multiply_z2(&x, &y, &mut ctx).unwrap();
    //     let result = protocol::open_z2(&result, &mut ctx).unwrap();
    //     assert_eq!(&result, &check);
   
    //     /* runtime tests */
    //     for input_size in vec![100000, 1000000, 10000000].iter() {
        
    //         let input = (0..*input_size)
    //             .map(|x| if ctx.num.asymm == 0 {0} else {x}).collect::<Vec<u128>>(); 
    //         let check = (0..*input_size).map(|x| x).collect::<Vec<u128>>();
           
    //         for n_threads in vec![4, 8, 16, 32].iter() {    
                
    //             ctx.sys.threads.online = *n_threads;
            
    //             let mut avg_elapsed = 0.0;
    //             let num_tests = 1;
    //             for _test_no in 0..num_tests {
    //                 let now = SystemTime::now();
    //                 let result = protocol::multiply_z2(&input, &input, &mut ctx).unwrap();
    //                 let elapsed = now.elapsed().unwrap().as_millis();
    //                 avg_elapsed += (elapsed as f64) / (num_tests as f64);
    //                 assert_eq!(protocol::open_z2(&result, &mut ctx).unwrap(), check); 
    //             }

    //             println!("size={:10}, n_threads={:2}, work time {:5.0} ms", 
    //                     input_size, ctx.sys.threads.online, avg_elapsed);               
    //         } 
    //     }
    // }

    // #[test]
    // fn _computing_party_protocol_pairwise_mult_z2() {

    //     use std::env;
    //     use std::num::Wrapping;
    //     use std::time::SystemTime;
    //     use super::util;
    //     use super::computing_party::protocol;
    //     use super::computing_party::init;
    //     use super::io;
    //     use rand::{thread_rng, Rng};
    //     let test_path = "test/files/computing_party_protocol_pairwise_mult_z2";

    //     let args: Vec<String> = env::args().collect();
    //     let id = &args[args.len()-1];
    //     let test_cfg = format!("{}/Party{}.toml", test_path, id); 
    //     let mut ctx = init::runtime_context( &test_cfg ).unwrap();
        
    //     /* connect */
    //     assert!( init::connection( &mut ctx ).is_ok() );
        
    //     /* empty vec test */
    //     let empty_vec = Vec::<u128>::new();
    //     assert_eq!(empty_vec, protocol::pairwise_mult_z2(&vec![], 0, 0, &mut ctx).unwrap());
        
    //     /* known results */
    //     let input = vec![ if ctx.num.asymm == 0 {0xffffffffffffffffffffffffffffffffu128} else {0u128}];
    //     let expected = vec![0xffffffffffffffffu128]; 
    //     let output = protocol::pairwise_mult_z2(&input, 1, 128, &mut ctx).unwrap();
    //     let output = util::compress_from_tesselated_bit_vector(&output, 1, 64, true, ctx.num.asymm).unwrap();
    //     let output = util::decompress_bit_vector(&output, 1, 64, true, ctx.num.asymm).unwrap();
    //     let output = protocol::open_z2(&output, &mut ctx).unwrap();

    //     println!("I: {:x?}", &input);
    //     println!("O: {:x?}", &output);
    //     println!("E: {:x?}", &expected);

    //     assert_eq!(&output, &expected);

    //     /* RANDOM TESTS */
    //     let mut rng = rand::thread_rng();
  
    //     for &len in vec![1, 2, 3, 5, 7, 11, 11177].iter() {
    //         for bitlen in 1..65 {

            
    //             let pad = true;
    //             let asymm = ctx.num.asymm;
    //             let bitmask: u128 = (1u128 << bitlen) - 1u128;

    //             let input = (0..len).map(|_i| ((rng.gen::<u64>() as u128) | ((rng.gen::<u64>() as u128) << 64)) & bitmask).collect::<Vec<u128>>();

    //             let opened_input = protocol::open_z2(&input, &mut ctx).unwrap();
    //             let compressed = util::compress_bit_vector(&input, len, bitlen, pad, asymm).unwrap();                
    //             let tesselated = protocol::pairwise_mult_z2(&compressed, len, bitlen, &mut ctx).unwrap(); 
  
    //             let tesselated_bitlen = (bitlen + 1) >> 1;
    //             let compressed = util::compress_from_tesselated_bit_vector(&tesselated, len, tesselated_bitlen, pad, asymm).unwrap();
    //             let output = util::decompress_bit_vector(&compressed, len, tesselated_bitlen, pad, asymm).unwrap();
    //             let output = protocol::open_z2(&output, &mut ctx).unwrap(); 

    //             let mut expected = vec![0u128 ; len];
    //             if pad && (bitlen & 1 == 1) {
    //                 for i in 0..len {
    //                     expected[i] |= opened_input[i] & 1;
    //                     for j in 1..tesselated_bitlen {
    //                         let left_bit = (opened_input[i] >> 2*j - 1);
    //                         let right_bit = (opened_input[i] >> 2*j);
    //                         expected[i] |= (left_bit & right_bit & 1) << j;
    //                     }
    //                 }
    //             } else {
    //                 for i in 0..len {
    //                     for j in 0..tesselated_bitlen {
    //                         let left_bit = (opened_input[i] >> 2*j);
    //                         let right_bit = (opened_input[i] >> 2*j + 1);
    //                         expected[i] |= (left_bit & right_bit & 1) << j;
    //                     }
    //                 }
    //             }

    //             // println!("[len={}][bitlen={}]", len, bitlen);
    //             // println!("[len={}][bitlen={}] I: {:x?}, O: {:x?}, E: {:x?}", len, bitlen, &opened_input, &output, &expected);
    //             // println!("\t expected: {:x?}", &expected);

    //             assert_eq!(&expected, &output);
    //         }
    //     }
    // }

    // #[test]
    // fn _computing_party_protocol_parallel_mult_z2() {

    //     use std::env;
    //     use std::num::Wrapping;
    //     use std::time::SystemTime;
    //     use super::util;
    //     use super::computing_party::protocol;
    //     use super::computing_party::init;
    //     use super::io;
    //     use rand::{thread_rng, Rng};
    //     let test_path = "test/files/computing_party_protocol_parallel_mult_z2";

    //     let args: Vec<String> = env::args().collect();
    //     let id = &args[args.len()-1];
    //     let test_cfg = format!("{}/Party{}.toml", test_path, id); 
    //     let mut ctx = init::runtime_context( &test_cfg ).unwrap();
        
    //     /* connect */
    //     assert!( init::connection( &mut ctx ).is_ok() );
        
    //     /* empty vec test */
    //     let empty_vec = Vec::<u128>::new();
    //     assert_eq!(empty_vec, protocol::parallel_mult_z2(&vec![], 0, 0, &mut ctx).unwrap());
        
    //     /* known results */
    //     let input = vec![if ctx.num.asymm == 0 {0xffffffffffffffffu128} else {0u128} ; 2];
    //     let expected = vec![0x1u128 ; 2]; 
    //     let output = protocol::parallel_mult_z2(&input, 2, 64, &mut ctx).unwrap();
    //     let output = protocol::open_z2(&output, &mut ctx).unwrap();

    //     println!("I: {:x?}", &input);
    //     println!("O: {:x?}", &output);
    //     println!("E: {:x?}", &expected);

    //     assert_eq!(&output, &expected);

    //     /* RANDOM TESTS */
    //     let mut rng = rand::thread_rng();

    //     for &len in vec![1, 2, 3, 5, 7, 11, 11177].iter() {
    //         for bitlen in 1..65 {
   
    //             let pad = true;
    //             let asymm = ctx.num.asymm;
    //             let bitmask: u128 = (1u128 << bitlen) - 1u128;

    //             let input = vec![if ctx.num.asymm == 0 {bitmask} else {0} ; len];

    //             let opened_input = protocol::open_z2(&input, &mut ctx).unwrap();
    //             let output = protocol::parallel_mult_z2(&input, len, bitlen, &mut ctx).unwrap(); 
    //             let output = protocol::open_z2(&output, &mut ctx).unwrap(); 

    //             assert_eq!(&vec![1u128 ; len], &output);
    //         }
    //     }

    //     for &len in vec![1, 2, 3, 5, 7, 11, 11177].iter() {
    //         for &bitlen in vec![2, 8, 32, 64].iter() {
   
    //             for i in 0..bitlen {

    //                 let pad = true;
    //                 let asymm = ctx.num.asymm;
    //                 let bitmask: u128 = (1u128 << bitlen) - 1u128;
    
    //                 let input = vec![if ctx.num.asymm == 0 {bitmask} else {1 << i} ; len];
    
    //                 let opened_input = protocol::open_z2(&input, &mut ctx).unwrap();
    //                 let output = protocol::parallel_mult_z2(&input, len, bitlen, &mut ctx).unwrap(); 
    //                 let output = protocol::open_z2(&output, &mut ctx).unwrap(); 
    
    //                 assert_eq!(&vec![0u128 ; len], &output);

    //             }

    //         }
    //     }

    //     for &len in vec![1, 2, 3, 5, 7, 11, 11177].iter() {
    //         for bitlen in 1..65 {

            
    //             let pad = true;
    //             let asymm = ctx.num.asymm;
    //             let bitmask: u128 = (1u128 << bitlen) - 1u128;

    //             let input = (0..len).map(|_i| ((rng.gen::<u64>() as u128) | ((rng.gen::<u64>() as u128) << 64)) & bitmask).collect::<Vec<u128>>();

    //             let opened_input = protocol::open_z2(&input, &mut ctx).unwrap();
    //             let output = protocol::parallel_mult_z2(&input, len, bitlen, &mut ctx).unwrap(); 
    //             let output = protocol::open_z2(&output, &mut ctx).unwrap(); 

    //             let mut expected = vec![0u128 ; len];
    //             for i in 0..len {
    //                 expected[i] = opened_input[i] & 1;
    //                 for j in 1..bitlen {
    //                     expected[i] &= opened_input[i] >> j;
    //                 }
    //             }
            
    //             // println!("[len={}][bitlen={}]", len, bitlen);
    //             // println!("[len={}][bitlen={}] I: {:x?}, O: {:x?}, E: {:x?}", len, bitlen, &opened_input, &output, &expected);
    //             // println!("\t expected: {:x?}", &expected);

    //             assert_eq!(&expected, &output);
    //         }
    //     }
    // }

    // #[test]
    // fn _computing_party_protocol_equality_from_z2() {

    //     use std::env;
    //     use std::num::Wrapping;
    //     use std::time::SystemTime;
    //     use super::util;
    //     use super::computing_party::protocol;
    //     use super::computing_party::init;
    //     use super::io;
    //     use rand::{thread_rng, Rng};
    //     let test_path = "test/files/computing_party_protocol_equality_from_z2";

    //     let args: Vec<String> = env::args().collect();
    //     let id = &args[args.len()-1];
    //     let test_cfg = format!("{}/Party{}.toml", test_path, id); 
    //     let mut ctx = init::runtime_context( &test_cfg ).unwrap();
        
    //     /* connect */
    //     assert!( init::connection( &mut ctx ).is_ok() );
        
    //     /* empty vec test */
    //     let empty_vec = Vec::<u128>::new();
    //     assert_eq!(empty_vec, protocol::equality_from_z2(&vec![], &vec![], 0, 0, &mut ctx).unwrap());
        
    //     /* known results */
    //     let all_1s = if ctx.num.asymm == 0 {0xffffffffffffffffu128 } else { 0u128 };
    //     let x = vec![ all_1s, 0u128, all_1s, 0u128 ];
    //     let y = vec![ all_1s, 0u128, 0u128, all_1s ];
        
    //     let expected = vec![1u128, 1u128, 0u128, 0u128]; 
    //     let output = protocol::equality_from_z2(&x, &y, 4, 64, &mut ctx).unwrap();
    //     let output = protocol::open_z2(&output, &mut ctx).unwrap();

    //     println!("I: x={:x?}, y={:x?}", &x, &y);
    //     println!("O: {:x?}", &output);
    //     println!("E: {:x?}", &expected);

    //     assert_eq!(&output, &expected);

    //     // /* RANDOM TESTS */
    //     let mut rng = rand::thread_rng();

    //     for &bitlen in vec![5].iter() {
    //         let max_val = (1 << bitlen) - 1;
    //         let bitmask: u128 = (1u128 << bitlen) - 1u128;

    //         let x = (0..max_val).map(|idx| if ctx.num.asymm == 0 {idx as u128} else {0u128}).collect::<Vec<u128>>();

    //         for j in 0..max_val {

    //             let y = vec![ if ctx.num.asymm == 1 {j as u128} else {0u128} ; max_val ];

    //             let output = protocol::equality_from_z2(&x, &y, max_val, bitlen, &mut ctx).unwrap(); 
    //             let output = protocol::open_z2(&output, &mut ctx).unwrap(); 

    //             println!("O: {:x?}", &output);

    //             let expected = (0..max_val).map(|i| if i == j {1u128} else {0u128}).collect::<Vec<u128>>();
    //             assert_eq!(&expected, &output); 
    //         }
    //     }
    
    // }

    // #[test]
    // fn _computing_party_protocol_bit_extract() {

    //     use std::env;
    //     use std::num::Wrapping;
    //     use std::time::SystemTime;
    //     use super::util;
    //     use super::computing_party::protocol;
    //     use super::computing_party::init;
    //     use super::io;
    //     use rand::{thread_rng, Rng};
    //     let test_path = "test/files/computing_party_protocol_bit_extract";

    //     let args: Vec<String> = env::args().collect();
    //     let id = &args[args.len()-1];
    //     let test_cfg = format!("{}/Party{}.toml", test_path, id); 
    //     let mut ctx = init::runtime_context( &test_cfg ).unwrap();
        
    //     /* connect */
    //     assert!( init::connection( &mut ctx ).is_ok() );
     
    //     let mut rng = rand::thread_rng();
    //     let n_tests = 100;
           
    //     for test in 0..n_tests {

    //         let input = rng.gen::<u64>();
    //         let input_opened = protocol::open(&vec![Wrapping(input)], &mut ctx).unwrap()[0].0 as u128;

    //         for bit_pos in 0..64 {

    //             let output = protocol::bit_extract(&Wrapping(input), bit_pos, &mut ctx).unwrap();
    //             let output = protocol::open_z2(&vec![output], &mut ctx).unwrap()[0];

    //             // println!("[bitpos={}] input opened : {}", bit_pos, input_opened);
    //             // println!("[bitpos={}] output opened: {}", bit_pos, output);    

    //             assert_eq!((input_opened >> bit_pos) & 1, output)

    //         } 
    //     }
    // }

    // #[test]
    // fn _computing_party_protocol_geq() {

    //     use std::env;
    //     use std::num::Wrapping;
    //     use std::time::SystemTime;
    //     use super::util;
    //     use super::computing_party::protocol;
    //     use super::computing_party::init;
    //     use super::io;
    //     use rand::{thread_rng, Rng};
    //     let test_path = "test/files/computing_party_protocol_bit_extract";

    //     let args: Vec<String> = env::args().collect();
    //     let id = &args[args.len()-1];
    //     let test_cfg = format!("{}/Party{}.toml", test_path, id); 
    //     let mut ctx = init::runtime_context( &test_cfg ).unwrap();
        
    //     /* connect */
    //     assert!( init::connection( &mut ctx ).is_ok() );
     
    //     let mut rng = rand::thread_rng();
    //     let n_tests = 100;
    //     let bitmask: u64 = (1 << (ctx.num.precision_int + ctx.num.precision_frac)) - 1;   
        
    //     for test in 0..n_tests {

    //         let mut x = 0u64;
    //         let mut y = 0u64;
    //         let x_neg = rng.gen::<bool>();
    //         let y_neg = rng.gen::<bool>();
            
    //         if ctx.num.asymm == 0 {

    //             x = rng.gen::<u64>() & bitmask;
    //             y = rng.gen::<u64>() & bitmask;
                
    //             if x_neg {
    //                 x = (- Wrapping(x)).0; 
    //             }

    //             if y_neg {
    //                 y = (- Wrapping(y)).0; 
    //             }
    //         }

    //         let xy_open = protocol::open(&vec![Wrapping(x), Wrapping(y)], &mut ctx).unwrap();
    //         let x_open = xy_open[0];
    //         let y_open = xy_open[1];

    //         let expected = if !(x_neg ^ y_neg) {
    //             if x_open >= y_open {
    //                 1u128
    //             } else {
    //                 0u128
    //             }
    //         } else if x_neg && !y_neg {
    //             0u128
    //         } else {
    //             1u128
    //         };

    //         // println!("x: {}, y: {}, output: {} expected: {}", x_open, y_open, output, expected);

    //         let output = protocol::geq(&Wrapping(x), &Wrapping(y), &mut ctx).unwrap();
    //         let output = protocol::open_z2(&vec![output], &mut ctx).unwrap()[0];

    //         if ctx.num.asymm == 0 {

    //             assert_eq!(expected, output);

    //         }

    //     }
    // }

    // #[test]
    // fn _test_create_selection_vectors() {
    //     use std::env;
    //     use std::num::Wrapping;
    //     use std::time::SystemTime;
    //     use super::util;
    //     use super::computing_party::protocol;
    //     use super::computing_party::init;
    //     use super::computing_party::ml::decision_trees::extra_trees::extra_trees;

    //     let test_path = "test/files/computing_party_protocol_bit_extract";
    //     let args: Vec<String> = env::args().collect();
    //     let id = &args[args.len()-1];
    //     let test_cfg = format!("{}/Party{}.toml", test_path, id); 
    //     let mut ctx = init::runtime_context( &test_cfg ).unwrap();
    //     let vecs = extra_trees::create_selection_vectors(1000, 100, &mut ctx).unwrap();
    //     let open_vecs = protocol::open(&vecs.into_iter().flatten().collect(), &mut ctx);
    //     let counts: Vec<usize> = open_vecs.iter().map(|x| x.iter().filter(|y| y.0 == 1u64).count()).collect();

    //     assert_eq!(vec![1; 100], counts);

    // }

    #[test]
    fn _computing_party_protocol_pairwise_mult_zq() {

        use std::env;
        use std::num::Wrapping;
        use std::time::SystemTime;
        use super::util;
        use super::computing_party::protocol;
        use super::computing_party::init;
        use super::io;
        use rand::{thread_rng, Rng};
        let test_path = "test/files/computing_party_protocol_pairwise_mult_zq";

        let args: Vec<String> = env::args().collect();

        println!("ARGS HERE {:?}", args);

        let id = &args[args.len()-1];
        let test_cfg = format!("{}/Party{}.toml", test_path, id); 
        let mut ctx = init::runtime_context( &test_cfg ).unwrap();
        
        /* connect */
        assert!( init::connection( &mut ctx ).is_ok() );
        
        /* empty vec test */
        let empty_vec = Vec::<Wrapping<u64>>::new();
        assert_eq!(empty_vec, protocol::pairwise_mult_zq(&vec![], &mut ctx).unwrap());

        /* known results */
        let input = if ctx.num.asymm == 0 
        { vec![vec![Wrapping(1), Wrapping(2)], vec![Wrapping(3), Wrapping(4)], vec![Wrapping(5), Wrapping(6)]] } else 
        { vec![vec![Wrapping(0), Wrapping(0)], vec![Wrapping(0), Wrapping(0)], vec![Wrapping(0), Wrapping(0)]] };
        let expected = [Wrapping(2), Wrapping(12), Wrapping(30)]; 
        let output = protocol::pairwise_mult_zq(&input, &mut ctx).unwrap();
        let output = protocol::open(&output, &mut ctx).unwrap();

        println!("I: {:x?}", &input);
        println!("O: {:x?}", &output);
        println!("E: {:x?}", &expected);

        assert_eq!(&output, &expected);

        /* known results */
        let input = if ctx.num.asymm == 0 
        { vec![vec![Wrapping(1), Wrapping(2), Wrapping(1)], vec![Wrapping(3), Wrapping(4), Wrapping(1)], vec![Wrapping(5), Wrapping(6), Wrapping(1)]] } else 
        { vec![vec![Wrapping(0), Wrapping(0), Wrapping(0)], vec![Wrapping(0), Wrapping(0), Wrapping(0)], vec![Wrapping(0), Wrapping(0), Wrapping(0)]] };
        let expected = [Wrapping(2), Wrapping(12), Wrapping(30)]; 
        let output = protocol::pairwise_mult_zq(&input, &mut ctx).unwrap();
        let output = protocol::open(&output, &mut ctx).unwrap();

        println!("I: {:x?}", &input);
        println!("O: {:x?}", &output);
        println!("E: {:x?}", &expected);

        assert_eq!(&output, &expected);


        /* known results */
        let input = if ctx.num.asymm == 0 
        { vec![vec![Wrapping(1), Wrapping(2), Wrapping(1), Wrapping(2), Wrapping(1)], vec![Wrapping(3), Wrapping(4), Wrapping(1), Wrapping(4), Wrapping(1)], vec![Wrapping(5), Wrapping(6), Wrapping(1), Wrapping(6), Wrapping(1)]] } else 
        { vec![vec![Wrapping(0), Wrapping(0), Wrapping(0), Wrapping(0), Wrapping(0)], vec![Wrapping(0), Wrapping(0), Wrapping(0), Wrapping(0), Wrapping(0)], vec![Wrapping(0), Wrapping(0), Wrapping(0), Wrapping(0), Wrapping(0)]] };
        let expected = [Wrapping(4), Wrapping(12 * 4), Wrapping(30 * 6)]; 
        let output = protocol::pairwise_mult_zq(&input, &mut ctx).unwrap();
        let output = protocol::open(&output, &mut ctx).unwrap();

        println!("I: {:x?}", &input);
        println!("O: {:x?}", &output);
        println!("E: {:x?}", &expected);

        assert_eq!(&output, &expected);
        
        // /* RANDOM TESTS */
        let mut rng = rand::thread_rng();
  
        // seems to break on my end when I populate the vector with multiple values. If you want to test, just put a different number in vec![]
        for &len in vec![11].iter() { 

            let mut input = vec![vec![Wrapping(0); len]; len];
            
            if ctx.num.asymm == 0 {
                for i in 0.. len {
                    for j in 0.. len {
                        // keep random number small to reduce chance of overflow
                        input[i][j] = Wrapping((rng.gen::<u64>() % 8 + 1) as u64); 
                    }
                }
            }

            let mut expected: Vec<Wrapping<u64>> = vec![];
            for vec in &input {

                let mut product = Wrapping(1 as u64);

                for val in vec {
                    product = val * product;
                }

                expected.push(product);
            }

            let output = protocol::pairwise_mult_zq(&input, &mut ctx).unwrap();
            let output = protocol::open(&output, &mut ctx).unwrap();
    
            println!("I: {:x?}", &input);
            println!("O: {:x?}", &output);
            println!("E: {:x?}", &expected);
    
            assert_eq!(&output, &expected);

        }

    }

}
