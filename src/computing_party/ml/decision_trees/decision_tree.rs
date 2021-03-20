use std::error::Error;
use super::super::super::Context;
use std::num::Wrapping;

#[derive(Default)]
pub struct TreeNode {
    split_point: Wrapping<u64>,
    attribute_sel_vec: Vec<Wrapping<u64>>,
    classification: Wrapping<u64>,
}

#[derive(Default)]
pub struct TrainingContext {
    instance_count: usize,
    attribute_count: usize, //attribute count in training context
    tree_count: usize,
    max_depth: usize,
}

pub fn run(ctx: &mut Context) -> Result<(), Box<dyn Error>> {

    Ok(())
}

//Additive stares are Wrapping<u64>, binary are u128
pub fn sid3t(input: Vec<Vec<u128>>, ctx: Context, train_ctx: TrainingContext) -> Result<Vec<Vec<TreeNode>>, Box<dyn Error>>{

    let placeholder = vec![vec![]];
    Ok(placeholder)
}