use std::error::Error;
use super::super::super::super::Context;
use super::super::decision_tree::TrainingContext;
use std::num::Wrapping;
use crate::io;

#[derive(Default)]
pub struct XTContext {
    pub tc: TrainingContext,
    pub feature_count: usize,
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

pub fn init(cfg_file: &String) -> Result<(XTContext, Vec<Vec<Wrapping<u64>>>), Box<dyn Error>> {
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

    let bin_count = 2usize;

    let data = io::matrix_csv_to_wrapping_vec(&settings.get_str("data")?)?;

    let tc = TrainingContext {
        instance_count,
        class_label_count,
        attribute_count,
        bin_count,
        tree_count,
        max_depth,
    };
    let xt = XTContext {
        tc,
        feature_count
    };

    Ok((xt, data))
}

pub fn create_selection_vectors(quant: usize, size: usize) {

}