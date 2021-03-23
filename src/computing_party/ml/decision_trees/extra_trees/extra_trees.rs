use std::error::Error;
use super::super::super::super::Context;
use super::super::decision_tree::TrainingContext;
use std::num::Wrapping;
use crate::io;
use super::super::super::super::protocol::*;
use crate::constants;
use rand::*;
use crate::util;

#[derive(Default)]
pub struct XTContext {
    pub tc: TrainingContext,
    pub feature_count: usize,
}

//TODO LIST
//Write init, setting up context
//port the preprocessing phase

//Accepts rowwise data, columnwise classes
//Outputs binary discretized sets according to random split points
//THIS WILL CHANGE TO RETURNING A U128 BINARY VECTOR
pub fn xt_preprocess(data: &Vec<Vec<Wrapping<u64>>>, xtctx: &XTContext, ctx: &mut Context) -> Result<Vec<Vec<Vec<Wrapping<u64>>>>, Box<dyn Error>> {
    //assume data has already been loaded
    //only return the processed data. that is all that is required, I think.

    let feature_count = xtctx.feature_count;
    let attribute_count = xtctx.tc.attribute_count;
    let decimal_precision = ctx.num.precision_frac;
    let asym = ctx.num.asymm;


    let minmax = minmax_batch(&util::transpose(data)?, ctx)?;

    let mins = minmax.0;
    let maxes = minmax.1;
    println!("maxes and mins found.");

    let fsv_amount = xtctx.tc.tree_count * xtctx.feature_count;
    let column_major_arvs = create_selection_vectors(fsv_amount, xtctx.tc.attribute_count, ctx)?;
    let mut column_major_arvs_flat_dup: Vec<Wrapping<u64>> = column_major_arvs.into_iter().flatten().collect();
    column_major_arvs_flat_dup.extend(&column_major_arvs_flat_dup);

    let mut mins_concat = vec![];
    let mut maxes_concat = vec![];
    for _i in 0 .. fsv_amount * feature_count {
        mins_concat.extend(&mins);
        maxes_concat.extend(&maxes);
    }
    mins_concat.append(&mut maxes_concat); //consolidate into mins, now mins represents both mins and maxes
    //let split_select = SystemTime::now();
    let mul_min_max = multiply(&column_major_arvs_flat_dup, &mins_concat, ctx)?;
    let selected_vals: Vec<Wrapping<u64>> = mul_min_max.chunks(attribute_count).map(|x| x.iter().fold(Wrapping(0), |acc, y| acc + y)).collect();

    let (selected_mins, selected_maxes)= selected_vals.split_at(selected_vals.len()/2);
    let selected_ranges: Vec<Wrapping<u64>> = selected_mins.iter().zip(selected_maxes.iter()).map(|(x, y)| y - x).collect();

    let random_ratios = create_random_ratios(selected_ranges.len(), ctx)?;
    
    let range_times_ratio = multiply(&selected_ranges, &random_ratios, ctx)?;
    let selected_splits: Vec<Wrapping<u64>> = range_times_ratio.iter().zip(selected_mins.iter()).map(|(x, y)| util::truncate(*x, decimal_precision, asym) + y).collect();
    //println!("split_select finished. Time taken: {:?}ms", split_select.elapsed().unwrap().as_millis());

    //println!("Matmul beginning.");
    //let matmultime = SystemTime::now();
    // let u = ctx.dt_shares.matmul_u.clone();
    // let mut e: Vec<Vec<Wrapping<u64>>> = vec![];
    // for i in 0..u.len() {
    //     e.push(reveal_wrapping(&dataset[i].iter().zip(u[i].iter()).map(|(&x, &u)| x - u ).collect(), ctx));
    // }
    // let v = ctx.dt_shares.matmul_vs.clone();
    // let w = ctx.dt_shares.matmul_ws.clone();
    
    //apply the CRVs to the dataset
    let column_reduced_datasets = matmul(
        &data,
        &column_major_arvs,
        ctx,
    );
    
    //println!("Matmul finished. Time taken: {:?}ms", matmultime.elapsed().unwrap().as_millis());
    //The splits have been found. The discretized datasets must now be made.

    //the sets must be changed to column major.
    let col_maj_time = SystemTime::now();
    let mut sets_col: Vec<Vec<Vec<Wrapping<u64>>>> = vec![];
    for i in 0 .. column_reduced_datasets.len() / ctx.dt_training.feature_count {
        let mut set: Vec<Vec<Wrapping<u64>>> = vec![];
        for j in 0 .. ctx.dt_training.feature_count {
            set.push(column_reduced_datasets[i * ctx.dt_training.feature_count + j].clone());
        }
        sets_col.push(set);
    }
    println!("col_maj finished. Time taken: {:?}ms", col_maj_time.elapsed().unwrap().as_millis());

    //The sets are now column oriented. Next is to compare the contents to the chosen split point.
    let total_sets = sets_col.len();
    println!("Binarizing sets.");
    let set_compil = SystemTime::now();
    let asym = Wrapping(ctx.asymmetric_bit as u64);
    let val_set = column_reduced_datasets.into_iter().flatten().collect();
    let mut split_set = vec![];
    for i in 0 .. total_sets {
        for j in 0 .. ctx.dt_training.feature_count {
            split_set.append(&mut vec![selected_splits[i * ctx.dt_training.feature_count + j]; ctx.dt_data.instance_count]);
        }
    }
    println!("set_compil finished. Time taken: {:?}ms", set_compil.elapsed().unwrap().as_millis());
    let comp_time = SystemTime::now();
    let cmp_res = xor_share_to_additive(&batch_compare(&val_set, &split_set, ctx), ctx, 1);
    println!("comp finished. Time taken: {:?}ms", comp_time.elapsed().unwrap().as_millis());
    let cmp_res_neg: Vec<Wrapping<u64>> = cmp_res.iter().map(|x| -x + asym).collect();
    
    let comp_extract_time = SystemTime::now();
    let comparison_results: Vec<Vec<Vec<Wrapping<u64>>>> = cmp_res.par_chunks(ctx.dt_data.instance_count * ctx.dt_training.feature_count).map(|x| x.chunks(ctx.dt_data.instance_count).map(|x| x.to_vec()).collect()).collect();
    let neg_comparison_results: Vec<Vec<Vec<Wrapping<u64>>>> = cmp_res_neg.par_chunks(ctx.dt_data.instance_count * ctx.dt_training.feature_count).map(|x| x.chunks(ctx.dt_data.instance_count).map(|x| x.to_vec()).collect()).collect();
    println!("comp_extract finished. Time taken: {:?}ms", comp_extract_time.elapsed().unwrap().as_millis());

    println!("Sets binarized.");
    let placeholder = vec![vec![vec![]]];
    Ok(placeholder)
}

pub fn init(cfg_file: &String) -> Result<(XTContext, Vec<Vec<Wrapping<u64>>>, Vec<Vec<Wrapping<u64>>>), Box<dyn Error>> {
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
    let mut classes = io::matrix_csv_to_wrapping_vec(&settings.get_str("classes")?)?;

    classes = util::transpose(&classes)?;

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

    Ok((xt, data, classes))
}

//Not in ring
pub fn create_selection_vectors(quant: usize, size: usize, ctx: &mut Context) -> Result<Vec<Vec<Wrapping<u64>>>, Box<dyn Error> >{
    let seed = [1234usize];
    let mut rng = rand::StdRng::from_seed(&seed);
    let (zero, one) = if ctx.num.asymm == 0 {(Wrapping(constants::TEST_CR_WRAPPING_SEL_VEC_U64_0.0), Wrapping(constants::TEST_CR_WRAPPING_SEL_VEC_U64_1.0))} 
    else {(Wrapping(constants::TEST_CR_WRAPPING_SEL_VEC_U64_0.1), Wrapping(constants::TEST_CR_WRAPPING_SEL_VEC_U64_1.1))};

    let mut results = vec![vec![]; quant];
    for i in 0 .. quant {
        let mut sel_vec = vec![zero; size];
        let index: usize = rng.gen_range(0, size);
        sel_vec[index] = one;
        results.push(sel_vec);
    }

    Ok(results)
}
//in ring
pub fn create_random_ratios(quant: usize, ctx: &mut Context) -> Result<Vec<Wrapping<u64>>, Box<dyn Error> >{
    let seed = [1234usize];
    let mut rng = rand::StdRng::from_seed(&seed);
    let upper_bound = u64::pow(2, ctx.num.precision_frac as u32);

    let mut results = vec![];
    for i in 0 .. quant {
        let val = if ctx.num.asymm == 1 {Wrapping(rng.gen_range(0, upper_bound))} else {Wrapping(0)};
        results.push(val);
    }

    Ok(results)
}