use std::error::Error;
use std::num::Wrapping;
use super::super::super::super::Context;
use super::extra_trees::*;
use super::super::decision_tree::*;
use super::super::inference::classify_softvote;
use super::super::inference::classify_argmax;
use crate::io;
use crate::util;
use serde_json;
use crate::computing_party::protocol;
use super::inference;
use std::fs::OpenOptions;
use std::io::{BufWriter, Write};
use std::fs::File;

use std::time::{Duration, Instant};

pub fn run(ctx: &mut Context) -> Result<(), Box<dyn Error>> {

    let start = Instant::now();

    let (mut xt_ctx, data, classes) = init(&ctx.ml.cfg)?;
    // for i in 0 .. xt_ctx.tc.class_label_count {
    //     println!("Class {:?}:\n{:?}", i, protocol::open(&classes[0][i], ctx)?);
    // }
    let (processed_data, arvs, splits) = xt_preprocess(&data, &mut xt_ctx, ctx)?;
    let trees = sid3t(&processed_data, &classes, &arvs, &splits, ctx, &mut xt_ctx.tc)?;


    let duration = start.elapsed();


    io::write_to_file(&xt_ctx.tc.save_location, &serde_json::to_string(&trees)?)?;
    let mut rev_trees = vec![];
    trees.iter().for_each(|x| rev_trees.push(reveal_tree(x, ctx).unwrap()));
    if ctx.num.asymm == 1 {
        io::write_to_file("treedata/rev_trees.json", &serde_json::to_string_pretty(&rev_trees)?)?;
    }
    let path = format!("cfg/ml/extratrees/inference{}.toml", ctx.num.asymm);
    let (_, test_data, test_lab, infctx) = inference::init(&path, true)?; 

    // --------------
    // Done to streamline testing, in general, the inference should be seperate from the learning phase
    let mut test_data_open = vec![];

    let test_lab_open = protocol::open(&test_lab, ctx)?;

    let test_lab_open_trunc: Vec<u64> = test_lab_open.iter().map(|x| x.0 >> ctx.num.precision_frac).collect();

    for row in test_data{
        test_data_open.push(protocol::open(&row, ctx)?);
    }

    //let mut file = File::open("treedata/rev_trees.json")?;
    // let mut contents = String::new();
    // file.read_to_string(&mut contents)?;

    // let trees: Vec<Vec<TreeNode>> = serde_json::from_str(&contents)?;
    let argmax_results = classify_argmax(&rev_trees, &test_data_open, &test_lab_open_trunc, &infctx)?;

    let softvote_results = classify_softvote(&rev_trees, &test_data_open, &test_lab_open_trunc, &infctx)?;
    
    let result = format!("argmax: {} %, softvote: {} %, {:?} seconds", argmax_results * 100.0, softvote_results * 100.0, duration);

    println!("{}", result);

    let path = "results.txt";

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

pub fn init(cfg_file: &String) -> Result<(XTContext, Vec<Vec<Wrapping<u64>>>, Vec<Vec<Vec<Wrapping<u64>>>>), Box<dyn Error>> {
	let mut settings = config::Config::default();
    settings
        .merge(config::File::with_name(cfg_file.as_str())).unwrap()
        .merge(config::Environment::with_prefix("APP")).unwrap();

    let class_label_count: usize = settings.get_int("class_label_count")? as usize;
    let attribute_count: usize = settings.get_int("attribute_count")? as usize;
    let instance_count: usize = settings.get_int("instance_count")? as usize;
    let feature_count: usize = settings.get_int("feature_count")? as usize;
    let tree_count: usize = settings.get_int("tree_count")? as usize;
    let max_depth: usize = settings.get_int("max_depth")? as usize;
    let epsilon: f64 = settings.get_int("epsilon")? as f64;
    let save_location = settings.get_str("save_location")?;
    let original_attr_count = attribute_count;
    let bin_count = 2usize;

    let data = io::matrix_csv_to_wrapping_vec(&settings.get_str("data")?)?;
    let mut classes = io::matrix_csv_to_wrapping_vec(&settings.get_str("classes")?)?;

    let instance_count = data.len(); // ADDED BY DAVID, hopefully won't cause issues. This relives huge headaches though
    let attribute_count = data[0].len(); // ADDED BY DAVID, hopefully won't cause issues. This relives huge headaches though
    println!("{}", attribute_count);

    classes = util::transpose(&classes)?;
    let mut dup_classes = vec![];
    for i in 0 .. tree_count {
        dup_classes.push(classes.clone());
    }


    let tc = TrainingContext {
        instance_count,
        class_label_count,
        attribute_count,
        original_attr_count,
        bin_count,
        tree_count,
        max_depth,
        epsilon,
        save_location,
    };
    let xt = XTContext {
        tc,
        feature_count
    };

    Ok((xt, data, dup_classes))
}