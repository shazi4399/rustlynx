use std::error::Error;
use super::super::super::super::Context;
use std::num::Wrapping;
use super::super::decision_tree::TreeNode;
use crate::io;
use crate::util;
use super::super::inference::{InferenceContext,};

pub fn run(ctx: &mut Context) -> Result<(), Box<dyn Error>> {

    Ok(())
}
pub fn init(cfg_file: &String, no_tree_load:bool) -> Result<(Vec<Vec<TreeNode>>, Vec<Vec<Wrapping<u64>>>, Vec<Wrapping<u64>>, InferenceContext), Box<dyn Error>> {
    let mut settings = config::Config::default();
    settings
        .merge(config::File::with_name(cfg_file.as_str())).unwrap()
        .merge(config::Environment::with_prefix("APP")).unwrap();

    let class_label_count: usize = settings.get_int("class_label_count")? as usize;
    let bin_count: usize = settings.get_int("bin_count")? as usize;
    let max_depth: usize = settings.get_int("max_depth")? as usize;
    // let trees: Vec<Vec<TreeNode>> = if !no_tree_load {serde_json::from_str(&settings.get_str("tree_location")?)?} else {vec![]};
    let trees = vec![];

    let data = io::matrix_csv_to_wrapping_vec(&settings.get_str("data")?)?;
    let mut classes = io::matrix_csv_to_wrapping_vec(&settings.get_str("classes")?)?;

    classes = util::transpose(&classes)?;
    let classes_single_col = classes[1].clone();

    let ic = InferenceContext {
        instance_count: data.len(),
        class_label_count,
        attribute_count: data[0].len(),
        bin_count,
        max_depth,
    };

    Ok((trees, data, classes_single_col, ic))
}