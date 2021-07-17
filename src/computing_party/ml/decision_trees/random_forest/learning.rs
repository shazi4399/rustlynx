use std::error::Error;
use super::super::super::super::Context;
use super:: random_forest;
use super::inference;
use super::super::decision_tree;
use crate::computing_party::protocol;
use super::super::inference::classify_softvote;
use super::super::inference::classify_argmax;
use super::super::inference::count_dummy_nodes;
use std::fs::OpenOptions;
use std::io::{BufWriter, Write};
use std::time::{Instant};
use std::fs::File;

pub fn run(ctx: &mut Context) -> Result<(), Box<dyn Error>> {

    let mut rev_trees = vec![];

    let start = Instant::now();

    let (mut rfctx, data, classes ) = random_forest::init(&ctx.ml.cfg)?;
    let iterations = if rfctx.tc.single_tree_training {rfctx.tc.bulk_qty} else {1};
    for i in 0 .. iterations {
        let (processed_data, modified_classes, arvs, splits) = random_forest::rf_preprocess(&data, &classes, &mut rfctx, ctx)?;
        let trees = decision_tree::sid3t(&processed_data, &modified_classes, &arvs, &splits, ctx, &mut rfctx.tc)?;
        println!("Training complete, revealing trees for testing");
        trees.iter().for_each(|x| rev_trees.push(decision_tree::reveal_tree(x, ctx).unwrap()));
    }

    let duration = start.elapsed();

    let path = format!("cfg/ml/randomforest/inference{}.toml", ctx.num.asymm);
    let (_x, test_data, classes_single_col, ic) = inference::init(&path, false)?;
    let mut test_data_open = vec![];

    let test_lab_open = protocol::open(&classes_single_col, ctx)?;

    let test_lab_open_trunc: Vec<u64> = test_lab_open.iter().map(|x| x.0 >> ctx.num.precision_frac).collect();

    for row in test_data{
        test_data_open.push(protocol::open(&row, ctx)?);
    }
    let argmax_results = classify_argmax(&rev_trees, &test_data_open, &test_lab_open_trunc, &ic, ctx.num.precision_int, ctx.num.precision_frac)?;
    let softvote_results = classify_softvote(&rev_trees, &test_data_open, &test_lab_open_trunc, &ic, ctx.num.precision_int, ctx.num.precision_frac)?;
    let dummy_nodes = count_dummy_nodes(&rev_trees, &test_data_open, &test_lab_open_trunc, &ic, ctx.num.precision_int, ctx.num.precision_frac)?;

    let result = format!("argmax: {} %, softvote: {} %, {:?} seconds - real nodes: {}, dummy nodes {}, ratio: {}", 
    argmax_results * 100.0, softvote_results * 100.0, duration, dummy_nodes.0, dummy_nodes.1, dummy_nodes.0 as f64 / (dummy_nodes.0 as f64 + dummy_nodes.1 as f64 ) as f64);

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