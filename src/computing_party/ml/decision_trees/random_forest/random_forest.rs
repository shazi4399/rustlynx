//TODO LIST
//Write init, setting up context
//port the preprocessing phase

use std::num::Wrapping;
use crate::computing_party::Context;
use crate::computing_party::ml::decision_trees::decision_tree::TrainingContext;
use std::error::Error;
use crate::computing_party::protocol::{discretize_into_ohe_batch, batch_matmul};
use crate::{util, io};
use crate::computing_party::ml::decision_trees::extra_trees::extra_trees::{load_arvs_from_file, two_dim_to_3_dim};
use rand::{self, Rng};
use rand::SeedableRng;
use std::io::{Write, Read};
use crate::computing_party::protocol;

#[derive(Default)]
pub struct RFContext {
    pub tc: TrainingContext,
    pub feature_count: usize,
    pub attr_value_count:usize,
    pub instance_selected_count:usize,
    pub decision_tree: bool,
}

pub fn rf_preprocess(data: &Vec<Vec<Wrapping<u64>>>, classes: &Vec<Vec<Wrapping<u64>>>, rfctx: &mut RFContext, ctx: &mut Context)->Result<(Vec<Vec<Vec<Wrapping<u64>>>>, Vec<Vec<Vec<Wrapping<u64>>>>, Vec<Vec<Vec<Wrapping<u64>>>>, Vec<Vec<Vec<Wrapping<u64>>>>), Box<dyn Error>> {
    let bucket_size = rfctx.attr_value_count;
    let decision_tree = rfctx.decision_tree;
    let attribute_count = rfctx.tc.original_attr_count;
    let feature_count = if decision_tree {attribute_count} else {rfctx.feature_count};
    let instance_select_count = rfctx.instance_selected_count;
    let instance_count = rfctx.tc.original_instance_count;
    let tree_count = rfctx.tc.tree_count;
    let fsv_amount = tree_count * feature_count;
    
    // data.iter().for_each(|x| println!("{:?}", protocol::open(&x, ctx).unwrap()));

    let processed_data_com = discretize_into_ohe_batch(&util::transpose(data)?,bucket_size, ctx);
    let discretized_ohe_data = processed_data_com.0;
    let full_splits = processed_data_com.1;
    println!("discretized_ohe_data");
    // discretized_ohe_data.iter().for_each(|x| println!("{:?}", protocol::open(&x, ctx).unwrap()));

    let use_pregenerated_selections = false;
    let arv_path = "custom_randomness/arvs.csv";
    let seed = 0;
    // let column_major_arvs_unexp = if use_pregenerated_selections {load_arvs_from_file(arv_path, ctx.num.asymm as usize, feature_count, attribute_count, tree_count)?} else {create_selection_vectors(fsv_amount, attribute_count, seed, ctx)?};
    let (column_major_arvs_unexp, column_major_arvs) = create_selection_vectors_rf(fsv_amount, attribute_count, feature_count, bucket_size, seed, decision_tree, ctx)?;
    println!("column_major_arvs_unexp");
    // column_major_arvs_unexp.iter().for_each(|x| println!("{:?}", protocol::open(&x, ctx).unwrap()));
    println!("column_major_arvs");
    // column_major_arvs.iter().for_each(|x| println!("{:?}", protocol::open(&x, ctx).unwrap()));
    let column_major_arvs_unexp_row_maj: Vec<Vec<Vec<Wrapping<u64>>>> = two_dim_to_3_dim(&column_major_arvs_unexp, feature_count)?.iter().map(|x| util::transpose(&x).unwrap()).collect();
    let row_major_arvs: Vec<Vec<Vec<Wrapping<u64>>>> = two_dim_to_3_dim(&column_major_arvs, feature_count * bucket_size)?.iter().map(|x| util::transpose(&x).unwrap()).collect();
    // let row_major_arvs: Vec<Vec<Vec<Wrapping<u64>>>> = two_dim_to_3_dim(&column_major_arvs, feature_count)?;
    let mut column_major_arvs_grouped = two_dim_to_3_dim(&column_major_arvs, feature_count * bucket_size)?;
    let final_col_maj_arv_unexp = two_dim_to_3_dim(&column_major_arvs_unexp, feature_count)?;
    let final_row_major_arvs: Vec<Vec<Vec<Wrapping<u64>>>> =  column_major_arvs_grouped.iter().map(|x| util::transpose(&x).unwrap()).collect();
    println!("final_row_major_arvs");
    // final_row_major_arvs.iter().for_each(|x| x.iter().for_each(|y| println!("{:?}", protocol::open(&y, ctx).unwrap())));
    let mut attribute_reduced_sets = if decision_tree {vec![util::transpose(&discretized_ohe_data)?]} else {batch_matmul(&discretized_ohe_data,&final_row_major_arvs, ctx)?};
    println!("attribute_reduced_sets");
    // attribute_reduced_sets.iter().for_each(|x| x.iter().for_each(|y| println!("{:?}", protocol::open(&y, ctx).unwrap())));
    let instance_selection = instance_selection(instance_count, instance_select_count, tree_count, seed, ctx)?;
    let mut final_datasets = vec![];
    if decision_tree {
        final_datasets = attribute_reduced_sets;
    } else {
        for i in 0 .. tree_count {
            //println!("tree {} matmul", i);
            final_datasets.push(batch_matmul(&util::transpose(&attribute_reduced_sets[i])?,&vec![util::transpose(&instance_selection[i])?],ctx)?.into_iter().flatten().collect());
        }
    }
    let final_classes: Vec<Vec<Vec<Wrapping<u64>>>> = if decision_tree {vec![classes.clone()]} else {batch_matmul(&classes, &instance_selection.iter().map(|x| util::transpose(&x).unwrap()).collect(), ctx)?};
    println!("final_splits");
    let final_splits = if decision_tree {vec![full_splits]} else {batch_matmul(&util::transpose(&full_splits)?, &column_major_arvs_unexp_row_maj, ctx)?.iter().map(|x| util::transpose(&x).unwrap()).collect()};
    rfctx.tc.attribute_count = if decision_tree {attribute_count * bucket_size} else {bucket_size * feature_count};
    rfctx.tc.instance_count = if instance_select_count == 0 || instance_select_count >= instance_count {instance_count} else {instance_select_count};


    Ok((final_datasets, final_classes, final_col_maj_arv_unexp, final_splits))

}

fn instance_selection(total_instances: usize, instances_to_select: usize, tree_count: usize, seed: usize, ctx: &Context) -> Result<Vec<Vec<Vec<Wrapping<u64>>>>, Box<dyn Error>> {
    let mut ret = vec![vec![vec![Wrapping(0); total_instances]; instances_to_select]; tree_count];
    if ctx.num.asymm == 0 {
        return Ok(ret);
    }
    let mut rng = if seed > 0 {rand::StdRng::from_seed(&[seed])} else {rand::StdRng::new()?};
    for i in 0 .. tree_count {
        for j in 0 .. instances_to_select {
            let index = rng.gen_range(0, total_instances);
            ret[i][j][index] = Wrapping(1);
        }
    }
    Ok(ret)
}

pub fn create_selection_vectors_rf(quant: usize, size: usize, amount_per_tree: usize, bin_count: usize, seed: usize, decision_tree: bool, ctx: &mut Context) -> Result<(Vec<Vec<Wrapping<u64>>>, Vec<Vec<Wrapping<u64>>>), Box<dyn Error> >{
    if ctx.num.asymm == 0 {
        if decision_tree {
            return Ok((vec![vec![Wrapping(0); size]; size], vec![vec![Wrapping(0); size * bin_count]; size * bin_count]));
        } else {
            return Ok((vec![vec![Wrapping(0); size]; quant], vec![vec![Wrapping(0); size * bin_count]; quant * bin_count]));
        }
    }
    let mut rng = if seed > 0 {rand::StdRng::from_seed(&[seed])} else {rand::StdRng::new()?};
    let mut results: Vec<Vec<Wrapping<u64>>> = vec![];
    let mut results_exp = vec![];
    let mut indices: Vec<usize> = (0 .. size).collect();

    let fsvs_per_tree = if decision_tree {size} else {amount_per_tree};
    for i in 0 .. quant {
        if i % fsvs_per_tree == 0 && !decision_tree {
            rng.shuffle(indices.as_mut_slice());
        }
        let index = indices[i % fsvs_per_tree];
        let mut att_sel_vec = vec![Wrapping(0); size];
        att_sel_vec[index] = Wrapping(1);
        results.push(att_sel_vec);
        let mod_index = bin_count * index;
        for j in 0 .. bin_count {
            let mut sel_vec_exp = vec![Wrapping(0); size * bin_count];
            sel_vec_exp[mod_index + j]  = Wrapping(1);
            results_exp.push(sel_vec_exp);
        }
    }
    Ok((results, results_exp))
}

pub fn init(cfg_file: &String) -> Result<(RFContext, Vec<Vec<Wrapping<u64>>>, Vec<Vec<Wrapping<u64>>>), Box<dyn Error>>{
    let mut settings = config::Config::default();
    settings
        .merge(config::File::with_name(cfg_file.as_str())).unwrap()
        .merge(config::Environment::with_prefix("APP")).unwrap();

    let class_label_count: usize = settings.get_int("class_label_count")? as usize;
    let mut attribute_count: usize = settings.get_int("attribute_count")? as usize;
    let mut instance_count: usize = settings.get_int("instance_count")? as usize;
    let feature_count: usize = settings.get_int("feature_count")? as usize;
    let attr_value_count: usize = settings.get_int("attr_value_count")? as usize;
    let tree_count: usize = settings.get_int("tree_count")? as usize;
    let max_depth: usize = settings.get_int("max_depth")? as usize;
    let bulk_qty: usize = settings.get_int("bulk_qty")? as usize;
    let single_tree_training: bool = settings.get_bool("single_tree_training")? as bool;
    let decision_tree: bool = settings.get_bool("decision_tree")? as bool;
    let mut instance_selected_count: usize = settings.get_int("instance_selected_count")? as usize;
    let epsilon: f64 = settings.get_int("epsilon")? as f64;
    let save_location = settings.get_str("save_location")?;
    let original_attr_count = attribute_count;

    let data = io::matrix_csv_to_wrapping_vec(&settings.get_str("data")?)?;
    let mut classes = io::matrix_csv_to_wrapping_vec(&settings.get_str("classes")?)?;

    instance_count = data.len() as usize;
    if decision_tree {instance_selected_count = instance_count}
    attribute_count = data[0].len() as usize;

    classes = util::transpose(&classes)?;
    let tc = TrainingContext {
        instance_count,
        class_label_count,
        attribute_count,
        original_attr_count,
        original_instance_count: instance_count,
        bin_count: attr_value_count,
        tree_count,
        max_depth,
        epsilon,
        save_location,
        bulk_qty,
        single_tree_training,
    };
    let rf = RFContext {
        tc,
        feature_count,
        attr_value_count,
        instance_selected_count,
        decision_tree,
    };

    Ok((rf, data, classes))
}