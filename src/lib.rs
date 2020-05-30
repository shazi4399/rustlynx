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

    #[test]
    fn _util_bitset_compress() {

        use super::util::Bitset;
        use rand::{thread_rng, Rng};
        
        let mut bits = Bitset::new( vec![0xdeadbeef, 0xffffffff], 65, 1);
        println!("{}\n", &bits);
        bits.compress(true);
        println!("{}\n", &bits);
        bits.tesselate();
        println!("{}\n", &bits);
        bits.compress(true);
        println!("{}\n", &bits);
        bits.decompress();
        println!("{}\n", &bits);

        

        /* RANDOM TESTS */
        let mut rng = rand::thread_rng();
        let n_tests = 10;
        let max_len = 1000;
        
        for i in 0..n_tests {

            for bitlen in 1..65 {

                let len = 1 + rng.gen::<usize>() % (max_len - 1); 
                let asymm = 1;
                let bitmask = (1 << bitlen) - 1;

                // println!("i={:2}, bitlen={:3}, len={:2}", i, bitlen, len);

                let input = (0..len).map(|_i| ((rng.gen::<u64>() as u128) | ((rng.gen::<u64>() as u128) << 64)) & bitmask).collect::<Vec<u128>>();
                let mut bitset = Bitset::new( input, bitlen, asymm );
                
                let check = bitset.clone();
                bitset.compress(true);
                bitset.tesselate();
                bitset.compress(true);
                bitset.decompress();
            
                assert_eq!(&bitset, &check);
            }
        }

    }

    #[test]
    fn _util_compress_bit_vector() {

        use super::util;
        use rand::{thread_rng, Rng};
        
        /* empty vec -> empty vec */
        let output = util::compress_bit_vector(&vec![], 0, 1, false, 0).unwrap();
        assert_eq!(&output, &vec![]);

        /* KNOWN ERRORS: bitlen less than 1 */
        assert!(util::compress_bit_vector(&vec![0x0 ; 1], 1, 0, false, 0).is_err());
        /* bitlen greater than 128 w/o padding */
        assert!(util::compress_bit_vector(&vec![0x0 ; 1], 1, 129, false, 0).is_err());
        /* bitlen greater than 128 w/ padding */
        assert!(util::compress_bit_vector(&vec![0x0 ; 1], 1, 128, true, 0).is_err());

        /* EDGE CASES: len: 128 of 0x1's -> len: 1 of 0xfff..f  */
        let output = util::compress_bit_vector(&vec![1u128 ; 128], 128, 1, false, 0).unwrap();
        assert_eq!(&output, &vec![0xffffffffffffffffffffffffffffffff ; 1]);
        /* 1's w/ padding 0's */
        let output = util::compress_bit_vector(&vec![1u128 ; 128], 128, 1, true, 0).unwrap();
        assert_eq!(&output, &vec![0xaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa ; 2]);
        /* 1's w/ padding 1's */
        let output = util::compress_bit_vector(&vec![1u128 ; 128], 128, 1, true, 1).unwrap();
        assert_eq!(&output, &vec![0xffffffffffffffffffffffffffffffff ; 2]);
        /* 0's w/ padding 0's */
        let output = util::compress_bit_vector(&vec![0u128 ; 128], 128, 1, true, 0).unwrap();
        assert_eq!(&output, &vec![0x00000000000000000000000000000000 ; 2]);
        /* 0's w/ padding 1's */
        let output = util::compress_bit_vector(&vec![0u128 ; 128], 128, 1, true, 1).unwrap();
        assert_eq!(&output, &vec![0x55555555555555555555555555555555 ; 2]);
        /* len: 128 of (2^127-1)'s -> len: 127 of 0xfff..f */
        let output = util::compress_bit_vector(&vec![0x7fffffffffffffffffffffffffffffff ; 128], 128, 127, false, 0).unwrap();
        assert_eq!(&output, &vec![0xffffffffffffffffffffffffffffffff ; 127]);
        /* 1's w/ padding 0's */
        let output = util::compress_bit_vector(&vec![0x7fffffffffffffffffffffffffffffff ; 128], 128, 127, true, 0).unwrap();
        assert_eq!(&output, &vec![0xfffffffffffffffffffffffffffffffe ; 128]);
        /* 1's w/ padding 1's */
        let output = util::compress_bit_vector(&vec![0x7fffffffffffffffffffffffffffffff ; 128], 128, 127, true, 1).unwrap();
        assert_eq!(&output, &vec![0xffffffffffffffffffffffffffffffff ; 128]);
        /* 0's w/ padding 0's */
        let output = util::compress_bit_vector(&vec![0x00000000000000000000000000000000 ; 128], 128, 127, true, 0).unwrap();
        assert_eq!(&output, &vec![0x00000000000000000000000000000000 ; 128]);
        /* 0's w/ padding 0's */
        let output = util::compress_bit_vector(&vec![0x00000000000000000000000000000000 ; 128], 128, 127, true, 1).unwrap();
        assert_eq!(&output, &vec![0x00000000000000000000000000000001 ; 128]);
        /* len: 1 of 0xfff..f -> len: 1 of 0xfff..f */
        let output = util::compress_bit_vector(&vec![0xffffffffffffffffffffffffffffffff ; 1], 1, 128, false, 0).unwrap();
        assert_eq!(&output, &vec![0xffffffffffffffffffffffffffffffff ; 1]);

        /* RANDOM TESTS */
        let mut rng = rand::thread_rng();
        let n_tests = 10;
        let max_len = 10000;
        
        for i in 0..n_tests {

            let len = 1 + rng.gen::<usize>() % (max_len - 1); 
            let bitlen = 1 + rng.gen::<usize>() % 127;
            let pad = rng.gen::<bool>();
            let asymm = rng.gen::<u64>() & 1;
            let bitmask = (1 << bitlen) - 1;

            let input = (0..len).map(|_i| ((rng.gen::<u64>() as u128) | ((rng.gen::<u64>() as u128) << 64)) & bitmask).collect::<Vec<u128>>();
            // println!("len: {}, bitlen: {}, pad: {}, asymm: {}, bitmask: 0x{:x}, input: [{:x}, ... ]",
            //     len, bitlen, pad, asymm, bitmask, input[0]);

            let compressed = util::compress_bit_vector(&input, len, bitlen, pad, asymm).unwrap();
            let output = util::decompress_bit_vector(&compressed, len, bitlen, pad, asymm).unwrap();

            assert_eq!(&input, &output);
        }
    }

    #[test]
    fn _util_decompress_bit_vector() {

        use super::util;

        /* empty vec -> empty vec */
        let output = util::decompress_bit_vector(&vec![], 0, 1, false, 0).unwrap();
        assert_eq!(&output, &vec![]);

        /* KNOWN ERRORS: bitlen less than 1 */
        assert!(util::decompress_bit_vector(&vec![0x0 ; 1], 1, 0, false, 0).is_err());
        /* bitlen greater than 128 w/o padding */
        assert!(util::decompress_bit_vector(&vec![0x0 ; 1], 1, 129, false, 0).is_err());
        /* bitlen greater than 128 w/ padding */
        assert!(util::decompress_bit_vector(&vec![0x0 ; 1], 1, 128, true, 0).is_err());

        /* EDGE CASES: len: 128 of 0x1's -> len: 1 of 0xfff..f  */
        let output = util::decompress_bit_vector(&vec![0xffffffffffffffffffffffffffffffff ; 1], 128, 1, false, 0).unwrap();
        assert_eq!(&output, &vec![0x1 ; 128]);
        /* 1's w/ padding 0's */
        let output = util::decompress_bit_vector(&vec![0xaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa ; 2], 128, 1, true, 0).unwrap();
        assert_eq!(&output, &vec![0x1 ; 128]);
        /* 1's w/ padding 1's */
        let output = util::decompress_bit_vector(&vec![0xffffffffffffffffffffffffffffffff ; 2], 128, 1, true, 1).unwrap();
        assert_eq!(&output, &vec![0x1 ; 128]);
        /* 0's w/ padding 0's */
        let output = util::decompress_bit_vector(&vec![0x00000000000000000000000000000000 ; 2], 128, 1, true, 0).unwrap();
        assert_eq!(&output, &vec![0u128 ; 128]);
        /* 0's w/ padding 1's */
        let output = util::decompress_bit_vector(&vec![0x55555555555555555555555555555555 ; 2], 128, 1, true, 1).unwrap();
        assert_eq!(&output,&vec![0u128 ; 128]);
        /* len: 128 of (2^127-1)'s -> len: 127 of 0xfff..f */
        let output = util::decompress_bit_vector(&vec![0xffffffffffffffffffffffffffffffff ; 127], 128, 127, false, 0).unwrap();
        assert_eq!(&output, &vec![0x7fffffffffffffffffffffffffffffff ; 128]);
        /* 1's w/ padding 0's */
        let output = util::decompress_bit_vector(&vec![0xfffffffffffffffffffffffffffffffe ; 128], 128, 127, true, 0).unwrap();
        assert_eq!(&output, &vec![0x7fffffffffffffffffffffffffffffff ; 128]);
        /* 1's w/ padding 1's */
        let output = util::decompress_bit_vector(&vec![0xffffffffffffffffffffffffffffffff ; 128], 128, 127, true, 1).unwrap();
        assert_eq!(&output, &vec![0x7fffffffffffffffffffffffffffffff ; 128]);
        /* 0's w/ padding 0's */
        let output = util::decompress_bit_vector(&vec![0x00000000000000000000000000000000 ; 128], 128, 127, true, 0).unwrap();
        assert_eq!(&output, &vec![0x00000000000000000000000000000000 ; 128]);
        /* 0's w/ padding 0's */
        let output = util::decompress_bit_vector(&vec![0x00000000000000000000000000000001 ; 128], 128, 127, true, 1).unwrap();
        assert_eq!(&output, &vec![0x00000000000000000000000000000000 ; 128]);
        /* len: 1 of 0xfff..f -> len: 1 of 0xfff..f */
        let output = util::decompress_bit_vector(&vec![0xffffffffffffffffffffffffffffffff ; 1], 1, 128, false, 0).unwrap();
        assert_eq!(&output, &vec![0xffffffffffffffffffffffffffffffff ; 1]);

        /* random tests in _util_compress_bit_vector */

    }

    #[test]
    fn _util_remove_tesselated_padding() {

        use super::util;

        /* empty vec -> empty vec */
        let output = util::remove_tesselated_padding(&vec![], 0).unwrap();
        assert_eq!(&output, &vec![]);

        let output = util::remove_tesselated_padding(&vec![0xffffffffffffffffffffffffffffffff ; 1], 1).unwrap();
        assert_eq!(&output, &vec![0xffffffffffffffff0000000000000000; 1]);
        
        let output = util::remove_tesselated_padding(&vec![0xffffffffffffffffffffffffffffffff ; 2], 2).unwrap();
        assert_eq!(&output, &vec![0xffffffffffffffffffffffffffffffff; 1]);
        
        let output = util::remove_tesselated_padding(&vec![0xaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa ; 2], 2).unwrap();
        assert_eq!(&output, &vec![0xffffffffffffffffffffffffffffffff; 1]);
        
        let output = util::remove_tesselated_padding(&vec![0x55555555555555555555555555555555 ; 2], 2).unwrap();
        assert_eq!(&output, &vec![0x00000000000000000000000000000000; 1]);
              
        let output = util::remove_tesselated_padding(&vec![0xffffffffffffffffffffffffffffffff ; 3], 3).unwrap();
        assert_eq!(&output, &vec![0xffffffffffffffffffffffffffffffff, 0xffffffffffffffff0000000000000000]);
        
        let output = util::remove_tesselated_padding(&vec![0xaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa ; 3], 3).unwrap();
        assert_eq!(&output, &vec![0xffffffffffffffffffffffffffffffff, 0xffffffffffffffff0000000000000000]);
        
        let output = util::remove_tesselated_padding(&vec![0x55555555555555555555555555555555 ; 3], 3).unwrap();
        assert_eq!(&output, &vec![0x00000000000000000000000000000000; 2]);

    }

    
    #[test]
    /* unfinished */
    fn _util_compress_from_tesselated() {

        use super::util;

        /* empty vec -> empty vec */
        let output = util::compress_from_tesselated(&vec![], 0, 1, false, 0).unwrap();
        assert_eq!(&output, &vec![]);

        let output = util::compress_from_tesselated(&vec![0xffffffffffffffffffffffffffffffff], 64, 1, true, 0).unwrap();
        assert_eq!(&output, &vec![0xaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa]);

        let output = util::compress_from_tesselated(&vec![0xffffffffffffffffffffffffffffffff], 64, 1, true, 1).unwrap();
        assert_eq!(&output, &vec![0xffffffffffffffffffffffffffffffff]);
        
        let output = util::compress_from_tesselated(&vec![0xffffffffffffffffffffffffffffffff ; 2], 63, 1, true, 0).unwrap();
        //assert_eq!(&output, &vec![0xaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa]);
        println!("{:x?}", &output);
    }

    #[test]
    fn _computing_party_protocol_open() {

        use std::env;
        use std::num::Wrapping;
        use std::time::SystemTime;
        use super::computing_party::protocol;
        use super::computing_party::init;
        use super::io;
        let test_path = "test/files/computing_party_protocol_open";

        let args: Vec<String> = env::args().collect();
        let id = &args[args.len()-1];
        let test_cfg = format!("{}/Party{}.toml", test_path, id); 
        let mut ctx = init::runtime_context( &test_cfg ).unwrap();
        
        /* connection */
        assert!( init::connection( &mut ctx ).is_ok() );
        
        /* empty vec test */
        let input: Vec<Wrapping<u64>> = Vec::new();
        assert_eq!(input, protocol::open(&input, &mut ctx).unwrap());

        /* random tests */
        let test_file = format!("{}/p{}.csv", test_path, ctx.num.asymm);
        let itc_file = format!("{}/itc.csv", test_path);

        let input = io::single_col_csv_to_wrapping_vec(&test_file).unwrap();
        let check = io::single_col_csv_to_wrapping_vec(&itc_file).unwrap();

        let result = protocol::open(&input, &mut ctx).unwrap();

        assert_eq!(&result, &check);

        /* runtime tests */
        for input_size in vec![100000, 1000000, 10000000].iter() {
        
            let input = (0..*input_size)
                .map(|x| Wrapping(if ctx.num.asymm == 0 {0} else {x})).collect(); 
            let check = (0..*input_size).map(|x| Wrapping(x)).collect::<Vec<Wrapping<u64>>>();
           
            for n_threads in vec![4, 8, 16, 32].iter() {    
                
                ctx.sys.threads.online = *n_threads;
            
                let mut avg_elapsed = 0.0;
                let num_tests = 1;
                for _test_no in 0..num_tests {
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
    fn _computing_party_protocol_open_z2() {

        use std::env;
        use std::num::Wrapping;
        use std::time::SystemTime;
        use super::computing_party::protocol;
        use super::computing_party::init;
        use super::io;
        let test_path = "test/files/computing_party_protocol_open_z2";

        let args: Vec<String> = env::args().collect();
        let id = &args[args.len()-1];
        let test_cfg = format!("{}/Party{}.toml", test_path, id); 
        let mut ctx = init::runtime_context( &test_cfg ).unwrap();
        
        /* connection */
        assert!( init::connection( &mut ctx ).is_ok() );
        
        /* empty vec test */
        let input: Vec<u128> = Vec::new();
        assert_eq!(input, protocol::open_z2(&input, &mut ctx).unwrap());

        /* random tests */
        let test_file = format!("{}/p{}.csv", test_path, ctx.num.asymm);
        let itc_file = format!("{}/itc.csv", test_path);

        let input = io::single_col_csv_to_u128_vec(&test_file).unwrap();
        let check = io::single_col_csv_to_u128_vec(&itc_file).unwrap();

        let result = protocol::open_z2(&input, &mut ctx).unwrap();

        assert_eq!(&result, &check);

        /* runtime tests */
        for input_size in vec![100000, 1000000, 10000000].iter() {
        
            let input = (0..*input_size)
                .map(|x| if ctx.num.asymm == 0 {0} else {x}).collect::<Vec<u128>>(); 
            let check = (0..*input_size).map(|x| x).collect::<Vec<u128>>();
           
            for n_threads in vec![4, 8, 16, 32].iter() {    
                
                ctx.sys.threads.online = *n_threads;
            
                let mut avg_elapsed = 0.0;
                let num_tests = 1;
                for _test_no in 0..num_tests {
                    let now = SystemTime::now();
                    let result = protocol::open_z2(&input, &mut ctx).unwrap();
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
        use super::io;
        let test_path = "test/files/computing_party_protocol_multiply";

        let args: Vec<String> = env::args().collect();
        let id = &args[args.len()-1];
        let test_cfg = format!("{}/Party{}.toml", test_path, id); 
        let mut ctx = init::runtime_context( &test_cfg ).unwrap();
        
        /* connect */
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
           
            for n_threads in vec![4, 8, 16, 32].iter() {    
                
                ctx.sys.threads.online = *n_threads;
            
                let mut avg_elapsed = 0.0;
                let num_tests = 1;
                for _test_no in 0..num_tests {
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

    #[test]
    fn _computing_party_protocol_multiply_z2() {

        use std::env;
        use std::num::Wrapping;
        use std::time::SystemTime;
        use super::computing_party::protocol;
        use super::computing_party::init;
        use super::io;
        let test_path = "test/files/computing_party_protocol_multiply_z2";

        let args: Vec<String> = env::args().collect();
        let id = &args[args.len()-1];
        let test_cfg = format!("{}/Party{}.toml", test_path, id); 
        let mut ctx = init::runtime_context( &test_cfg ).unwrap();
        
        /* connect */
        assert!( init::connection( &mut ctx ).is_ok() );
        
        /* empty vec test */
        let input = Vec::<u128>::new();
        assert_eq!( &input, &protocol::multiply_z2(&input, &input, &mut ctx).unwrap() );

        /* random tests */
        let x_file = format!("{}/p{}_x.csv", test_path, ctx.num.asymm);
        let y_file = format!("{}/p{}_y.csv", test_path, ctx.num.asymm);
        let itc_file = format!("{}/itc.csv", test_path);

        let x = io::single_col_csv_to_u128_vec(&x_file).unwrap();
        let y = io::single_col_csv_to_u128_vec(&y_file).unwrap();
        let check = io::single_col_csv_to_u128_vec(&itc_file).unwrap();

        let result = protocol::multiply_z2(&x, &y, &mut ctx).unwrap();
        let result = protocol::open_z2(&result, &mut ctx).unwrap();
        assert_eq!(&result, &check);
   
        /* runtime tests */
        for input_size in vec![100000, 1000000, 10000000].iter() {
        
            let input = (0..*input_size)
                .map(|x| if ctx.num.asymm == 0 {0} else {x}).collect::<Vec<u128>>(); 
            let check = (0..*input_size).map(|x| x).collect::<Vec<u128>>();
           
            for n_threads in vec![4, 8, 16, 32].iter() {    
                
                ctx.sys.threads.online = *n_threads;
            
                let mut avg_elapsed = 0.0;
                let num_tests = 1;
                for _test_no in 0..num_tests {
                    let now = SystemTime::now();
                    let result = protocol::multiply_z2(&input, &input, &mut ctx).unwrap();
                    let elapsed = now.elapsed().unwrap().as_millis();
                    avg_elapsed += (elapsed as f64) / (num_tests as f64);
                    assert_eq!(protocol::open_z2(&result, &mut ctx).unwrap(), check); 
                }

                println!("size={:10}, n_threads={:2}, work time {:5.0} ms", 
                        input_size, ctx.sys.threads.online, avg_elapsed);               
            } 
        }
    }

    #[test]
    fn _computing_party_protocol_pairwise_mult_z2() {

        use std::env;
        use std::num::Wrapping;
        use std::time::SystemTime;
        use super::computing_party::protocol;
        use super::computing_party::init;
        use super::io;
        use super::util::Bitset;
        use rand::{thread_rng, Rng};
        let test_path = "test/files/computing_party_protocol_pairwise_mult_z2";

        let args: Vec<String> = env::args().collect();
        let id = &args[args.len()-1];
        let test_cfg = format!("{}/Party{}.toml", test_path, id); 
        let mut ctx = init::runtime_context( &test_cfg ).unwrap();
        
        /* connect */
        assert!( init::connection( &mut ctx ).is_ok() );
        
        /* empty vec test */
        // assert_eq!( &Vec::<u128>::new(), &protocol::pairwise_mult_z2(&vec![], 0, 1, &mut ctx).unwrap() );

        // let mut rng = rand::thread_rng();

        // for &len in vec![1, 10, 100, 1000].iter() {
        //     for bitlen in (2..128).step_by(2) {

        //         println!("len: {}, bitlen: {}", len, bitlen);

        //         let bitmask = (-Wrapping(1u128)).0 >> (128 - bitlen);
        //         let input = if ctx.num.asymm == 0 {
        //             (0..len).map(|_i| (bitmask & (rng.gen::<u64>() as u128 | (rng.gen::<u64>() as u128) << 64))).collect::<Vec<u128>>()
        //         } else {
        //             vec![ 0x0 ; len]
        //         };
        //         let check = protocol::open_z2(&input, &mut ctx).unwrap();
        
        //         let mut check_eq = vec![0u128 ; len];
        //         for (j, &elem) in check.iter().enumerate() {
        //             for i in (1..128).step_by(2) {
        //                 let val = (elem >> i) & (elem >> (i-1)) & 1;
        //                 check_eq[j] |=  val << i;
        //             }
        //         }

        //         let mut bitset = Bitset::new(input, bitlen, ctx.num.asymm);
        //         println!("{}", &bitset);
        //         bitset.compress(true);
        //         println!("{}", &bitset);

        //         let mut output = protocol::pairwise_mult_z2(&bitset, &mut ctx).unwrap();
        //         output.compress(false);
        //         output.decompress();

        //         let output = protocol::open_z2(&output.bits, &mut ctx).unwrap();

        //         // println!("check  {:x?} -- output {:x?}", &check_eq, &output);
        //         assert_eq!(&check_eq, &output);
        //     }
        // }



    }

    #[test]
    fn _computing_party_protocol_parallel_mult_z2() {

        use std::env;
        use std::num::Wrapping;
        use std::time::SystemTime;
        use super::computing_party::protocol;
        use super::computing_party::init;
        use super::io;
        use super::util;
        use super::util::Bitset;
        use rand::{thread_rng, Rng};
        
        let test_path = "test/files/computing_party_protocol_parallel_mult_z2";
        
        let args: Vec<String> = env::args().collect();
        let id = &args[args.len()-1];
        let test_cfg = format!("{}/Party{}.toml", test_path, id); 
        let mut ctx = init::runtime_context( &test_cfg ).unwrap();
        
        /* connect */
        assert!( init::connection( &mut ctx ).is_ok() );

        let input = if ctx.num.asymm == 0 {vec![0xffffffff, 0xfffeffff, 0xffffffff]} else {vec![0u128; 3]};
        let mut bits = Bitset::new( input, 32, ctx.num.asymm );
        
        /* RANDOM TESTS */
        let mut rng = rand::thread_rng();
        let n_tests = 1 << 14;
        let len = 10;
          
        for bitlen in 14..15 {
            for i in 0..(1 << bitlen) {

                let asymm = ctx.num.asymm;
                let bitmask = (Wrapping(1u128 << bitlen) - Wrapping(1u128)).0;
                
                let input = if asymm == 0 {
                vec![i ; len]
                } else {
                    vec![0u128 ; len]
                };
                let check = protocol::open_z2(&input, &mut ctx).unwrap();

                let mut bitset = Bitset::new( input, bitlen, asymm );

                let now = SystemTime::now();
                let result = protocol::parallel_mult_z2(&bitset, &mut ctx).unwrap().bits;


                // let mut eq_check = vec![0u128 ; len];
                // for i in 0..len {
                //     if check[i] < (1 << bitlen) - 1 {
                //         eq_check[i] = 0;
                //     } else {
                //         eq_check[i] = 1;
                //     }
                // }
                
                println!("test={}, work time: {:5} ms", i, now.elapsed().unwrap().as_millis());

                // let opened = protocol::open_z2(&result, &mut ctx).unwrap();
                // if check[0] == (1 << bitlen) - 1 {
                //     println!("result: i={:2}, bitlen={:3}, len={:2}", i, bitlen, len);
                //     println!("check:  {:x?} = opened {:x?}", &eq_check, &opened);
                        
                // }


                // assert_eq!(&protocol::open_z2(&result, &mut ctx).unwrap(), &eq_check);
                    
            }
        }
        
    }

    #[test]
    fn _computing_party_protocol_equality_from_z2() {

        use std::env;
        use std::num::Wrapping;
        use std::time::SystemTime;
        use super::computing_party::protocol;
        use super::computing_party::init;
        use super::io;
        use super::util;
        use super::util::Bitset;
        use rand::{thread_rng, Rng};
        
        let test_path = "test/files/computing_party_protocol_parallel_mult_z2";
        
        let args: Vec<String> = env::args().collect();
        let id = &args[args.len()-1];
        let test_cfg = format!("{}/Party{}.toml", test_path, id); 
        let mut ctx = init::runtime_context( &test_cfg ).unwrap();
        
        /* connect */
        assert!( init::connection( &mut ctx ).is_ok() );

        /* RANDOM TESTS */
        let mut rng = rand::thread_rng();
        

        for bitlen in 14..15 {

            
            let x = (0..(1 << bitlen))
            .map(|i| if ctx.num.asymm == 0 {i as u128} else {0u128})
            .collect::<Vec<u128>>();

            let x = Bitset::new(x, bitlen, ctx.num.asymm);


            for i in 0..(1 << bitlen) {

                let y = vec![ if ctx.num.asymm == 1 {0u128} else {i as u128} ; 1 << bitlen ];
                let y = Bitset::new(y, bitlen, ctx.num.asymm);


                let eq_result = protocol::equality_from_z2(&x, &y, &mut ctx).unwrap();

                // println!("i={}, eq_result: {:?}", i, protocol::open_z2(&eq_result.bits, &mut ctx).unwrap());

                assert_eq!( 
                    protocol::open_z2(&eq_result.bits, &mut ctx).unwrap(),
                    (0..(1 << bitlen)).map(|j| if i == j {1u128} else {0u128} ).collect::<Vec<u128>>()
                );
            }      
            

        }

        
    }

    #[test]
    fn _computing_party_ml_naive_bayes_inference() {

        use std::env;
        use std::num::Wrapping;
        use std::time::SystemTime;
        use super::computing_party::protocol;
        use super::computing_party::init;
        use super::io;
        use super::util;
        use super::util::Bitset;
        use rand::{thread_rng, Rng};
        use super::computing_party::ml::naive_bayes;
        
        let test_path = "test/files/computing_party_ml_naive_bayes_inference";
        
        let args: Vec<String> = env::args().collect();
        let id = &args[args.len()-1];
        let test_cfg = format!("{}/Party{}.toml", test_path, id); 
        let mut ctx = init::runtime_context( &test_cfg ).unwrap();
      
        /* connect */
        assert!( init::connection( &mut ctx ).is_ok() );

        naive_bayes::inference::run(&mut ctx);

        // for &dict_size in vec![10, 100, 1000, 10000].iter() {
        //     for &ex_size in vec![5, 10, 25, 50, 100, 500].iter() {
        //         for &n_classes in vec![2, 3, 4].iter() {

        //             let now = SystemTime::now();
        //             naive_bayes::inference::run(dict_size, ex_size, n_classes, &mut ctx);
        //             println!("dict_size: {:5}, ex_size: {:5}, n_classes: {:5} -- work time {:5} ms",
        //                 dict_size, ex_size, n_classes, now.elapsed().unwrap().as_millis());
                    

        //         }
        //     }
        // }


    }

}
