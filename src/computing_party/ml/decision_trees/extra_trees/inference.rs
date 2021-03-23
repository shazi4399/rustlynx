use std::error::Error;
use super::super::super::super::Context;
use super::extra_trees;

pub fn run(ctx: &mut Context) -> Result<(), Box<dyn Error>> {
    let (xt_ctx, data, classes) = extra_trees::init(&ctx.ml.cfg)?;
    Ok(())
}