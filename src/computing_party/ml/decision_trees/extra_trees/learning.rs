use std::error::Error;
use std::num::Wrapping;
use super::super::super::super::Context;
use super::extra_trees::*;
use super::super::decision_tree::*;
use crate::io;
use crate::util;
use serde_json;

pub fn run(ctx: &mut Context) -> Result<(), Box<dyn Error>> {
    let (mut xt_ctx, data, classes) = init(&ctx.ml.cfg)?;
    let (processed_data, arvs, splits) = xt_preprocess(&data, &mut xt_ctx, ctx)?;
    let trees = sid3t(&processed_data, &classes, &arvs, &splits, ctx, &mut xt_ctx.tc)?;
    io::write_to_file(&xt_ctx.tc.save_location, &serde_json::to_string(&trees)?)?;
    let mut rev_trees = vec![];
    trees.iter().for_each(|x| rev_trees.push(reveal_tree(x, ctx).unwrap()));
    if ctx.num.asymm == 1 {
        io::write_to_file("treedata/rev_trees.json", &serde_json::to_string(&rev_trees)?)?;
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