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

    let test_size = tctx.test_size;
    // let processed_data_com = discretize_into_ohe_batch(&util::transpose(data)?,bucket_size,ctx);
    // let discretized_ohe_data = processed_data_com.0;
    // let full_splits = processed_data_com.1;
    // println!("discretized_ohe_data");
    // // discretized_ohe_data.iter().for_each(|x| println!("{:?}", protocol::open(&x, ctx).unwrap()));

    let test_vec1 = vec![Wrapping(0); test_size];
    let test_vec2 = vec![Wrapping(0); test_size];

    //inequality
    let start = Instant::now();
    let res = protocol::batch_geq(&test_vec1, &test_vec2, ctx)?;
    let geq_time = format!("{:?}", start.elapsed());
    
    //2toq
    let start = Instant::now();
    protocol::z2_to_zq(&res, ctx);
    let z2_conversion_time = format!("{:?}", start.elapsed());

    //minmax
    let start = Instant::now();
    protocol::minmax_batch(&vec![test_vec1], ctx);
    let minmax_time = format!("{:?}", start.elapsed());

    let result = format!("\n<><><><><><> SIZE: {} <><><><><><>\ninequality: {} 2toq: {} seconds, minmax: {} seconds", test_size, geq_time, z2_conversion_time, minmax_time);

    println!("{}", result);

    let path = "results_rf.txt";

    let b = std::path::Path::new(path).exists();

    if ctx.num.asymm == 0 {

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