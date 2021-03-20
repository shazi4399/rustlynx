use std::error::Error;
use super::super::super::Context;

#[derive(Default)]
struct TreeNode {
    split_point: Wrapping<u64>,
    attribute_sel_vec: Vec<Wrapping<u64>>,
    classification: Wrapping<u64>,
}

#[derive(Default)]
struct TrainingContext {
    instance_count: usize,
    attribute_count: usize, //attribute count in training context
    tree_count: usize,
    max_depth: usize,
}

pub fn run(ctx: &mut Context) -> Result<(), Box<dyn Error>> {

    Ok(())
}

//Additive stares are Wrapping<u64>, binary are u128
pub fn SID3T(input: Vec<Vec<u128>>, ctx: Context, train_ctx: TrainingContext) -> Vec<Vec<TreeNode>> {

}