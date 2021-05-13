use std::{error::Error};
use super::super::super::super::Context;
use super::super::decision_tree::TrainingContext;
use std::num::Wrapping;
use itertools::izip;
use crate::io;
use super::super::super::super::protocol::*;
// use crate::constants;
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
pub fn xt_preprocess(data: &Vec<Vec<Wrapping<u64>>>, xtctx: &mut XTContext, ctx: &mut Context) -> 
Result<(Vec<Vec<Vec<Wrapping<u64>>>>, Vec<Vec<Vec<Wrapping<u64>>>>, Vec<Vec<Vec<Wrapping<u64>>>>), Box<dyn Error>> {
    //assume data has already been loaded

    // let vecs = create_selection_vectors(1000, 100, ctx).unwrap();
    // let open_vecs = open(&vecs.into_iter().flatten().collect(), ctx);
    // let counts: Vec<usize> = open_vecs.iter().map(|x| x.iter().filter(|y| y.0 == 1u64).count()).collect();
    // println!("counts:{:?}", counts);
    // assert_eq!(vec![1; 100], counts);

    // println!("test passed");

    let feature_count = xtctx.feature_count;
    let attribute_count = xtctx.tc.attribute_count;
    let tree_count = xtctx.tc.tree_count;
    let instance_count = xtctx.tc.instance_count;
    let decimal_precision = ctx.num.precision_frac;
    let asym = ctx.num.asymm;

    let use_pregenerated_splits_and_selections = false;
    let arv_path = "custom_randomness/arvs.csv";
    let splits_path = "custom_randomness/splits.csv";
    let seed = 0;

    let minmax = minmax_batch(&util::transpose(data)?, ctx)?;

    let mins = minmax.0;
    let maxes = minmax.1;

    // println!("MINS:{:?}", util::ring_to_float(&open(&mins, ctx)?, ctx.num.precision_int, ctx.num.precision_frac));
    // println!("MAXES:{:?}", util::ring_to_float(&open(&maxes, ctx)?, ctx.num.precision_int, ctx.num.precision_frac));

    println!("maxes and mins found.");

    let fsv_amount = xtctx.tc.tree_count * xtctx.feature_count;
    let column_major_arvs = if use_pregenerated_splits_and_selections {load_arvs_from_file(arv_path, ctx.num.asymm as usize, feature_count, attribute_count, tree_count)?} else {create_selection_vectors(fsv_amount, xtctx.tc.attribute_count, seed, ctx)?};
    let row_major_arvs: Vec<Vec<Vec<Wrapping<u64>>>> = two_dim_to_3_dim(&column_major_arvs, feature_count)?.iter().map(|x| util::transpose(&x).unwrap()).collect();
    // column_major_arvs.iter().for_each(|x| println!("{:?}", open(x, ctx).unwrap()));
    let final_column_major_arvs = two_dim_to_3_dim(&column_major_arvs, feature_count)?;
    let column_major_arvs_flat: Vec<Wrapping<u64>> = final_column_major_arvs.clone().into_iter().flatten().flatten().collect();
    let mut column_major_arvs_flat_dup = column_major_arvs_flat.clone();
    column_major_arvs_flat_dup.append(&mut column_major_arvs_flat_dup.clone());
    // column_major_arvs.iter().for_each(|x| println!("{:?}", open(x, ctx).unwrap()));

    let mut mins_concat = vec![];
    let mut maxes_concat = vec![];
    for _i in 0 .. fsv_amount {
        mins_concat.extend(&mins);
        maxes_concat.extend(&maxes);
    }
    mins_concat.append(&mut maxes_concat); //consolidate into mins, now mins represents both mins and maxes
    //let split_select = SystemTime::now();
    let mul_min_max = multiply(&column_major_arvs_flat_dup, &mins_concat, ctx)?;
    // println!("mul_min_max: {:?}", open(&mul_min_max, ctx));
    let selected_vals: Vec<Wrapping<u64>> = mul_min_max.chunks(attribute_count).map(|x| x.iter().fold(Wrapping(0), |acc, y| acc + y)).collect();
    // println!("selected_vals: {:?}", open(&selected_vals, ctx));




    // ADDED BY DAVID, NEEDS TO BE TESTED!

    // There are three options. Either (1) max,min are positive, (2) max is positive, min is negative, or (3) both max and min are negative.
    // Each operation requires a different way to calculate the range, so, make comparisons to determine which setting we are in
    
    let smallest_neg_value = if asym == 0 {u64::MAX} else {0}; // does this assume that Lambda = 64?

    let smallest_neg_array = vec![Wrapping(smallest_neg_value); selected_vals.len()];
    let pos_neg_minmaxes = z2_to_zq(&batch_geq(&selected_vals.clone(), &smallest_neg_array, ctx)?, ctx)?;

    let (selected_mins, selected_maxes) = selected_vals.split_at(selected_vals.len()/2);
    let (selected_mins_cmp_res, selected_maxes_cmp_res) = pos_neg_minmaxes.split_at(pos_neg_minmaxes.len()/2);
    let (selected_mins_cmp_res, selected_maxes_cmp_res) = (selected_mins_cmp_res.to_vec(), selected_maxes_cmp_res.to_vec());


    // println!("selected_mins:{:?}", util::ring_to_float(&open(&selected_mins.to_vec(), ctx)?, ctx.num.precision_int, ctx.num.precision_frac));
    // println!("selected_maxes:{:?}", util::ring_to_float(&open(&selected_maxes.to_vec(), ctx)?, ctx.num.precision_int, ctx.num.precision_frac));


    // correct way for (1)
    let selected_ranges_1: Vec<Wrapping<u64>> = selected_mins.iter().zip(selected_maxes.iter()).map(|(x, y)| y - x).collect();
    let option1_valid = multiply(&selected_mins_cmp_res.clone(), &selected_maxes_cmp_res.clone(), ctx)?;

    let selected_ranges_1 = multiply(&selected_ranges_1, &option1_valid, ctx)?;

    let bit_pos_msb = ctx.num.precision_int + ctx.num.precision_frac + 1;
    // eliminates the negative bit (if it exists, otherwise makes a number negative)
    let complement = Wrapping(asym * 2u64.pow(bit_pos_msb as u32)); // does this assume that Lambda = 64?

    // correct way for (2)
    let selected_ranges_2: Vec<Wrapping<u64>> = selected_mins.iter().zip(selected_maxes.iter()).map(|(x, y)| y + (complement - (x + complement))).collect();
    let selected_mins_cmp_res_neg: Vec<Wrapping<u64>> = selected_mins_cmp_res.iter().map(|x| -x + Wrapping(asym as u64)).collect();
    let option2_valid = multiply(&selected_mins_cmp_res_neg.clone(), &selected_maxes_cmp_res.clone(), ctx)?;

    let selected_ranges_2 = multiply(&selected_ranges_2, &option2_valid, ctx)?;

    // correct way for (3)
    let selected_ranges_3: Vec<Wrapping<u64>> = selected_mins.iter().zip(selected_maxes.iter()).map(|(x, y)| (y + complement) - (x + complement)).collect();
    let selected_maxes_cmp_res_neg: Vec<Wrapping<u64>> = selected_maxes_cmp_res.iter().map(|x| -x + Wrapping(asym as u64)).collect();
    let option3_valid = multiply(&selected_mins_cmp_res_neg, &selected_maxes_cmp_res_neg, ctx)?;

    let selected_ranges_3 = multiply(&selected_ranges_3, &option3_valid, ctx)?;


    let mut selected_ranges = vec![];
    
    for (x, y, z) in izip!(&selected_ranges_1, &selected_ranges_2, &selected_ranges_3) {
        selected_ranges.push(x + y + z);
    }

    selected_ranges.shrink_to_fit();
    
    // WORK BY DAVID FINISHED


    // println!("selected_ranges: {:?}", util::ring_to_float(&open(&selected_ranges, ctx)?, ctx.num.precision_int, ctx.num.precision_frac));

    let random_ratios =  create_random_ratios(selected_ranges.len(), seed, ctx)?;
    //println!("RANDOM RATIOS:{:?}", open(&random_ratios, ctx)?.iter().map(|x| x.0 as f64/2f64.powf(ctx.num.precision_frac as f64)).collect::<Vec<f64>>());
    
    let range_times_ratio = multiply(&selected_ranges, &random_ratios, ctx)?;
    // println!("range_times_ratio: {:?}", util::ring_to_float(&open(&range_times_ratio, ctx)?, ctx.num.precision_int, ctx.num.precision_frac));
    let selected_splits: Vec<Wrapping<u64>> = if use_pregenerated_splits_and_selections {load_splits_from_file(splits_path, ctx.num.asymm as usize, feature_count, tree_count, decimal_precision)?} else {range_times_ratio.iter().zip(selected_mins.iter()).map(|(x, y)| util::truncate(*x, decimal_precision, asym) + y).collect()};
    
    // println!("SELECTED SPLITS: {:?}", open(&selected_splits, ctx));
    //println!("split_select finished. Time taken: {:?}ms", split_select.elapsed().unwrap().as_millis());
    // column_major_arvs.iter().for_each(|x| println!("{:?}", open(x, ctx).unwrap()));
    let res = batch_matmul(&data, &row_major_arvs, ctx)?;
    let mut column_reduced_datasets = vec![];
    res.iter().for_each(|x| column_reduced_datasets.push(util::transpose(&x).unwrap()));

    //The sets are now column oriented. Next is to compare the contents to the chosen split point.
    let total_sets = tree_count;
    //println!("Binarizing sets.");
    //let set_compil = SystemTime::now();
    let asym = Wrapping(asym);
    let val_set = column_reduced_datasets.into_iter().flatten().flatten().collect();
    let mut split_set = vec![];
    for i in 0 .. total_sets {
        for j in 0 .. feature_count {
            split_set.append(&mut vec![selected_splits[i * feature_count + j]; instance_count]);
        }
    }

    split_set.shrink_to_fit();

    // println!("set_compil finished. Time taken: {:?}ms", set_compil.elapsed().unwrap().as_millis());
    //let comp_time = SystemTime::now();
    let cmp_res = z2_to_zq(&batch_geq(&val_set, &split_set, ctx)?, ctx)?;
    //println!("comp finished. Time taken: {:?}ms", comp_time.elapsed().unwrap().as_millis());
    let cmp_res_neg: Vec<Wrapping<u64>> = cmp_res.iter().map(|x| -x + asym).collect();
    
    // let comp_extract_time = SystemTime::now();
    let comparison_results: Vec<Vec<Vec<Wrapping<u64>>>> = cmp_res.chunks(instance_count * feature_count).map(|x| x.chunks(instance_count).map(|x| x.to_vec()).collect()).collect();
    let neg_comparison_results: Vec<Vec<Vec<Wrapping<u64>>>> = cmp_res_neg.chunks(instance_count * feature_count).map(|x| x.chunks(instance_count).map(|x| x.to_vec()).collect()).collect();
    let mut interleaved_complete_set = vec![];
    for i in 0 .. tree_count {
        let mut interleaved_set = vec![];
        for j in 0 .. feature_count {
            interleaved_set.push(neg_comparison_results[i][j].clone());
            interleaved_set.push(comparison_results[i][j].clone());
        }
        interleaved_set.shrink_to_fit();
        interleaved_complete_set.push(interleaved_set);
    }
    interleaved_complete_set.shrink_to_fit();
    // println!("comp_extract finished. Time taken: {:?}ms", comp_extract_time.elapsed().unwrap().as_millis());

    println!("Sets binarized.");

    //get the split points for each set of arvs


    //Doubled for proper ohe
    xtctx.tc.attribute_count = 2 * xtctx.feature_count;
    // println!("SELECTED SPLITS: {:?}", open(&selected_splits, ctx)?.iter().map(|x| x.0 as f64/2f64.powf(ctx.num.precision_frac as f64)).collect::<Vec<f64>>());

    let final_arv_splits = two_dim_to_3_dim(&selected_splits.iter().map(|x| vec![*x]).collect(), feature_count)?; 

    Ok((interleaved_complete_set, final_column_major_arvs, final_arv_splits))
}

pub fn two_dim_to_3_dim(data: &Vec<Vec<Wrapping<u64>>>, group_size: usize) -> Result<Vec<Vec<Vec<Wrapping<u64>>>>, Box<dyn Error>>{
    let mut result = vec![];
    for i in 0.. data.len() / group_size {
        let mut group = vec![];
        for j in 0 .. group_size {
            group.push(data[i * group_size + j].clone());
        }
        group.shrink_to_fit();
        result.push(group);
    }
    result.shrink_to_fit();
    return Ok(result);
}

//Not in ring
pub fn create_selection_vectors(quant: usize, size: usize, seed: usize, ctx: &mut Context) -> Result<Vec<Vec<Wrapping<u64>>>, Box<dyn Error> >{
    if ctx.num.asymm == 0 {
        return Ok(vec![vec![Wrapping(0); size]; quant]);
    }
    let mut rng = if seed > 0 {rand::StdRng::from_seed(&[seed])} else {rand::StdRng::new()?};

    let mut results: Vec<Vec<Wrapping<u64>>> = vec![];
    for i in 0 .. quant {
        let index: usize = rng.gen_range(0, size);
        let mut att_sel_vec = vec![Wrapping(0); size];
        att_sel_vec[index] = Wrapping(1);
        results.push(att_sel_vec);
    }
    results.shrink_to_fit();
    Ok(results)
}

//in ring
pub fn create_random_ratios(quant: usize, seed: usize, ctx: &mut Context) -> Result<Vec<Wrapping<u64>>, Box<dyn Error>> {
    if ctx.num.asymm == 0 {
        return Ok(vec![Wrapping(0); quant]);
    }
    let mut rng = if seed > 0 {rand::StdRng::from_seed(&[seed])} else {rand::StdRng::new()?};
    let upper_bound = 1 << ctx.num.precision_frac;

    let mut results = vec![];
    for i in 0 .. quant {
        let val = Wrapping(rng.gen_range(0, upper_bound));
        results.push(val);
    }
    results.shrink_to_fit();
    Ok(results)
}

pub fn load_arvs_from_file(path: &str, asym: usize, feature_count: usize, attribute_count: usize, tree_count: usize) -> Result<Vec<Vec<Wrapping<u64>>>, Box<dyn Error>>{
    if asym == 0 {
       return Ok(vec![vec![Wrapping(0u64); attribute_count]; feature_count * tree_count]);
    }
    let mut ret = vec![];
    let vals = io::matrix_csv_to_wrapping_vec(path)?;
    for i in 0 .. vals.len() {
        for j in 0 .. vals[i].len() {
            let index = vals[i][j].0 as usize;
            let mut sel_vec = vec![Wrapping(0); attribute_count];
            sel_vec[index] = Wrapping(1u64);
            ret.push(sel_vec);
        }
    }
    ret.shrink_to_fit();
    Ok(ret)
}

fn load_splits_from_file(path: &str, asym: usize, feature_count: usize, tree_count: usize, decimal_precision: usize) -> Result<Vec<Wrapping<u64>>, Box<dyn Error>>{
    let mut ratios = vec![Wrapping(0); feature_count * tree_count];
    if asym == 0 {
        return Ok(ratios);
    }
    let float_ratios = io::matrix_csv_to_float_vec(path)?;
    for i in 0 .. tree_count {
        for j in 0 .. feature_count {
            ratios[i * feature_count + j] = float_to_fixed(float_ratios[i][j], decimal_precision)?;
        }
    }
    Ok(ratios)
}

fn float_to_fixed(val: f64, decimal_precision: usize) -> Result<Wrapping<u64>, Box<dyn Error>> {
    let ringmod = 2f64.powi(64);
    let shift_val = 2f64.powi(decimal_precision as i32);
    let res = if val < 0f64 {Wrapping((ringmod - (-1f64 * val * shift_val).floor()) as u64)} else {Wrapping((val*shift_val).floor() as u64)};
    Ok(res)
}