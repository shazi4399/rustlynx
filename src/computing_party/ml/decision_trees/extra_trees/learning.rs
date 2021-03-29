use std::error::Error;
use super::super::super::super::Context;
use super::extra_trees::*;
use super::super::decision_tree::*;
use super::extra_trees;

pub fn run(ctx: &mut Context) -> Result<(), Box<dyn Error>> {
    let (mut xt_ctx, data, classes) = extra_trees::init(&ctx.ml.cfg)?;
    let (processed_data, arvs, splits) = extra_trees::xt_preprocess(&data, &mut xt_ctx, ctx)?;
    let trees = sid3t(&processed_data, &classes, &arvs, &splits, ctx, &mut xt_ctx.tc)?;
    trees.iter().for_each(|x| reveal_tree(x, ctx).unwrap());
    Ok(())
}

