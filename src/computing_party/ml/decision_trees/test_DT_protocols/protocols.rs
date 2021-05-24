use std::num::Wrapping;
use crate::computing_party::Context;
use crate::computing_party::ml::decision_trees::decision_tree::TrainingContext;
use std::error::Error;
use crate::computing_party::protocol::{discretize_into_ohe_batch, batch_matmul};
use crate::{util, io};
use crate::computing_party::ml::decision_trees::extra_trees::extra_trees::{load_arvs_from_file, two_dim_to_3_dim};
use rand::{self, Rng};
use rand::SeedableRng;
use std::io::{Write, Read};
use crate::computing_party::protocol;
use std::fs::OpenOptions;
use std::io::{BufWriter};
use std::time::{Instant};
use std::fs::File;

#[derive(Default)]
pub struct TestContext {
    pub test_size: usize,
}

pub fn test_protocol(ctx: &mut Context) -> Result<(), Box<dyn Error>> {

    let tctx = init(&ctx.ml.cfg)?;

    let initial_size = 100000;

    let test_sizes = vec![10, 100, 1000, 10000, 100000];

    let bucket_sizes = vec![2,3,5,8];

    let col_size = 1000;
    let row_sizes = vec![1000, 10000, 100000];

    for b in bucket_sizes {

        for r in row_sizes.clone() {

            let data = vec![vec![Wrapping(0); r]; col_size];

            let start = Instant::now();
            let processed_data_com = protocol::discretize_into_ohe_batch(&data, b, ctx);
            let time = format!("{:?}", start.elapsed());

            let result = format!("\n<><><><><><> cols: {}, rows {}, buckets: {} <><><><><><>\n {} seconds", 
            col_size, r, b, time);

            println!("{}", result);
        
            let path = "results_runtime.txt";
        
            let b = std::path::Path::new(path).exists();
        
            if !b {
                let f = File::create(path).expect("unable to create file");
                let mut f = BufWriter::new(f);
                write!(f, "{}\n", result).expect("unable to write");
            } else {
                let f = OpenOptions::new()
                .write(true)
                .append(true)
                .open(path)
                .expect("unable to open file");
                let mut f = BufWriter::new(f);

                write!(f, "{}\n", result).expect("unable to write");
            }

        }

    }

    

    // for i in test_sizes {

    //     let test_size = initial_size * i;

    //     let test_vec1 = vec![Wrapping(0); test_size];
    //     let test_vec2 = vec![Wrapping(0); test_size];
    
    //     println!("performing inequality for size of {}", test_size);
    //     //inequality
    //     let start = Instant::now();
    //     let res = protocol::batch_geq(&test_vec1, &test_vec2, ctx)?;
    //     let geq_time = format!("{:?}", start.elapsed());
        
    //     println!("performing conversion for size of {}", test_size);
    //     //2toq
    //     let start = Instant::now();
    //     protocol::z2_to_zq(&res, ctx);
    //     let z2_conversion_time = format!("{:?}", start.elapsed());
    
    //     println!("performing minmax for size of {}", test_size);
    //     //minmax
    //     let start = Instant::now();
    //     protocol::minmax_batch(&vec![test_vec1], ctx);
    //     let minmax_time = format!("{:?}", start.elapsed());
    
    //     let result = format!("\n<><><><><><> SIZE: {} <><><><><><>\ninequality: {} 2toq: {} seconds, minmax: {} seconds", test_size, geq_time, z2_conversion_time, minmax_time);
    
    //     println!("{}", result);
    
    //     let path = "results_runtime.txt";
    
    //     let b = std::path::Path::new(path).exists();
    
    //     if !b {
    //         let f = File::create(path).expect("unable to create file");
    //         let mut f = BufWriter::new(f);
    //         write!(f, "{}\n", result).expect("unable to write");
    //     } else {
    //         let f = OpenOptions::new()
    //         .write(true)
    //         .append(true)
    //         .open(path)
    //         .expect("unable to open file");
    //         let mut f = BufWriter::new(f);

    //         write!(f, "{}\n", result).expect("unable to write");
    //     }
    
    // }


    Ok(())
}

pub fn init(cfg_file: &String) -> Result<(TestContext), Box<dyn Error>>{
    let mut settings = config::Config::default();
    settings
        .merge(config::File::with_name(cfg_file.as_str())).unwrap()
        .merge(config::Environment::with_prefix("APP")).unwrap();


    let test_size: usize = settings.get_int("test_size")? as usize;

    let tctx = TestContext {
        test_size,
    };

    Ok((tctx))
}