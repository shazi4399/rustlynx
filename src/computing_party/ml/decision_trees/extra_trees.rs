use super::super::super::Context;

#[derive(Default)]
struct XTContext {
    TrainingContext tc,
    feature_count: usize,
}

//Accepts columnwise input,
//Outputs binary discretized sets according to random split points
pub fn xt_preprocess(data: &Vec<Vec<Wrapping<u64>>>, xtctx: &XTTrainingContext, ctx: &mut Context) -> Vec<Vec<Vec<u128>> {
    //assume data has already been loaded
    //only return the processed data. that is all that is required, I think.
}