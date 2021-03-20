use std::error::Error;
use super::super::super::super::Context;
use super::super::decision_tree::TrainingContext;
use std::num::Wrapping;

#[derive(Default)]
pub struct XTContext {
    tc: TrainingContext,
    feature_count: usize,
}

//TODO LIST
//Write init, setting up context
//port the preprocessing phase

//Accepts columnwise input,
//Outputs binary discretized sets according to random split points
pub fn xt_preprocess(data: &Vec<Vec<Wrapping<u64>>>, xtctx: &XTContext, ctx: &mut Context) -> Result<Vec<Vec<Vec<u128>>>, Box<dyn Error>> {
    //assume data has already been loaded
    //only return the processed data. that is all that is required, I think.

    let placeholder = vec![vec![vec![]]];
    Ok(placeholder)
}