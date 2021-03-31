use std::error::Error;
use super::super::super::super::Context;
use std::num::Wrapping;
use super::super::decision_tree::TreeNode;
use super::super::inference::classify_in_the_clear;
use super::super::inference::{InferenceContext,};
use crate::io;
use crate::util;

pub fn run(ctx: &mut Context) -> Result<(), Box<dyn Error>> {
    let (trees, data, labels, inf_ctx) = init(&ctx.ml.cfg)?;
    //batch_classify(&data, &trees, inf_ctx, ctx);
    let acc = classify_in_the_clear(&data, &labels, inf_ctx);
    Ok(())
}

pub fn init(cfg_file: &String) -> Result<(Vec<Vec<TreeNode>>, Vec<Vec<Wrapping<u64>>>, Vec<Wrapping<u64>>, InferenceContext), Box<dyn Error>> {
    let mut settings = config::Config::default();
    settings
        .merge(config::File::with_name(cfg_file.as_str())).unwrap()
        .merge(config::Environment::with_prefix("APP")).unwrap();

    let class_label_count: usize = settings.get_int("class_label_count")? as usize;
    let attribute_count: usize = settings.get_int("attribute_count")? as usize;
    let instance_count: usize = settings.get_int("instance_count")? as usize;
    let max_depth: usize = settings.get_int("max_depth")? as usize;
    let trees: Vec<Vec<TreeNode>> = serde_json::from_str(&settings.get_str("tree_location")?)?;
    let bin_count = 2usize;

    let data = io::matrix_csv_to_wrapping_vec(&settings.get_str("data")?)?;
    let mut classes = io::matrix_csv_to_wrapping_vec(&settings.get_str("classes")?)?;

    classes = util::transpose(&classes)?;
    let classes_single_col = classes[0].clone();

    let ic = InferenceContext {
        instance_count,
        class_label_count,
        attribute_count,
        bin_count,
        max_depth,
    };

    Ok((trees, data, classes_single_col, ic))
}