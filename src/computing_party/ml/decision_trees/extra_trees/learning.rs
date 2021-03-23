use std::error::Error;
use super::super::super::super::Context;
use super::extra_trees::*;
use super::super::decision_tree::*;
use super::extra_trees;

pub fn run(ctx: &mut Context) -> Result<(), Box<dyn Error>> {
    let (xt_ctx, data, classes) = extra_trees::init(&ctx.ml.cfg)?;
    let processed_data = extra_trees::xt_preprocess(data, &xt_ctx, ctx);
    let trees = sid3t(processed_data, classes, ctx, &mut xt_ctx.tc);
    Ok(())
}

