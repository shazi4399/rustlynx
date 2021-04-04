use std::{error::Error};
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

    let USE_PREGENERATED_SPLITS_AND_SELECTIONS = true;
    let ARV_PATH = "custom_randomness/arvs.csv";
    let SPLITS_PATH = "custom_randomness/splits.csv";

    let minmax = minmax_batch(&util::transpose(data)?, ctx)?;

    let mins = minmax.0;
    let maxes = minmax.1;

    // println!("MINS:{:?}", open(&mins, ctx));
    // println!("MAXES:{:?}", open(&maxes, ctx));
    println!("maxes and mins found.");

    let fsv_amount = xtctx.tc.tree_count * xtctx.feature_count;
    let column_major_arvs = create_selection_vectors(fsv_amount, xtctx.tc.attribute_count, ctx)?;
    // column_major_arvs.iter().for_each(|x| println!("{:?}", open(x, ctx).unwrap()));
    let final_column_major_arvs = if !USE_PREGENERATED_SPLITS_AND_SELECTIONS{two_dim_to_3_dim(&column_major_arvs, feature_count)?} else {load_arvs_from_file(ARV_PATH, ctx.num.asymm as usize, feature_count, attribute_count, tree_count)?};
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

    let (selected_mins, selected_maxes)= selected_vals.split_at(selected_vals.len()/2);
    let selected_ranges: Vec<Wrapping<u64>> = selected_mins.iter().zip(selected_maxes.iter()).map(|(x, y)| y - x).collect();
    // println!("selected_ranges: {:?}", open(&selected_ranges, ctx));

    let random_ratios = create_random_ratios(selected_ranges.len(), ctx)?;
    // println!("RANDOM RATIOS:{:?}", open(&random_ratios, ctx));
    
    let range_times_ratio = multiply(&selected_ranges, &random_ratios, ctx)?;
    // println!("range_times_ratio: {:?}", open(&range_times_ratio, ctx));
    let selected_splits: Vec<Wrapping<u64>> = range_times_ratio.iter().zip(selected_mins.iter()).map(|(x, y)| util::truncate(*x, decimal_precision, asym) + y).collect();
    
    // println!("SELECTED SPLITS: {:?}", open(&selected_splits, ctx));
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
    // column_major_arvs.iter().for_each(|x| println!("{:?}", open(x, ctx).unwrap()));
    let column_reduced_datasets = util::transpose(&matmul(
        &data,
        &column_major_arvs,
        ctx,
    )?)?;

    // column_reduced_datasets.iter().for_each(|x| println!("{:?}", open(&x, ctx).unwrap()));
    
    //println!("Matmul finished. Time taken: {:?}ms", matmultime.elapsed().unwrap().as_millis());
    //The splits have been found. The discretized datasets must now be made.

    // the sets must be changed to column major.
    // let col_maj_time = SystemTime::now();
    let mut sets_col: Vec<Vec<Vec<Wrapping<u64>>>> = vec![];
    for i in 0 .. column_reduced_datasets.len() / feature_count {
        let mut set: Vec<Vec<Wrapping<u64>>> = vec![];
        for j in 0 .. feature_count {
            set.push(column_reduced_datasets[i * feature_count + j].clone());
        }
        sets_col.push(set);
    }
    //println!("col_maj finished. Time taken: {:?}ms", col_maj_time.elapsed().unwrap().as_millis());

    //The sets are now column oriented. Next is to compare the contents to the chosen split point.
    let total_sets = sets_col.len();
    //println!("Binarizing sets.");
    //let set_compil = SystemTime::now();
    let asym = Wrapping(asym);
    let val_set = column_reduced_datasets.into_iter().flatten().collect();
    let mut split_set = vec![];
    for i in 0 .. total_sets {
        for j in 0 .. feature_count {
            split_set.append(&mut vec![selected_splits[i * feature_count + j]; instance_count]);
        }
    }
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
        interleaved_complete_set.push(interleaved_set);
    }
    // println!("comp_extract finished. Time taken: {:?}ms", comp_extract_time.elapsed().unwrap().as_millis());

    println!("Sets binarized.");

    //get the split points for each set of arvs


    //Doubled for proper ohe
    xtctx.tc.attribute_count = 2 * xtctx.feature_count;

    let final_arv_splits = if USE_PREGENERATED_SPLITS_AND_SELECTIONS {load_splits_from_file(SPLITS_PATH, ctx.num.asymm as usize, feature_count, tree_count, decimal_precision)?} else {two_dim_to_3_dim(&selected_splits.iter().map(|x| vec![*x]).collect(), feature_count)?}; 

    Ok((interleaved_complete_set, final_column_major_arvs, final_arv_splits))
}

fn two_dim_to_3_dim(data: &Vec<Vec<Wrapping<u64>>>, group_size: usize) -> Result<Vec<Vec<Vec<Wrapping<u64>>>>, Box<dyn Error>>{
    let mut result = vec![];
    for i in 0.. data.len() / group_size {
        let mut group = vec![];
        for j in 0 .. group_size {
            group.push(data[i * group_size + j].clone());
        }
        result.push(group);
    }
    return Ok(result);
}

//Not in ring
pub fn create_selection_vectors(quant: usize, size: usize, ctx: &mut Context) -> Result<Vec<Vec<Wrapping<u64>>>, Box<dyn Error> >{
    let seed = [1234567usize];
    let mut rng = rand::StdRng::from_seed(&seed);

    let mut results: Vec<Vec<Wrapping<u64>>> = vec![];
    for i in 0 .. quant {
        let index: usize = rng.gen_range(0, size);
        let mut att_sel_vec = vec![];
        for j in 0 .. size {
            let val = if j == index && ctx.num.asymm == 1 {Wrapping(1)} else {Wrapping(0)};
            let p0: Wrapping<u64> = Wrapping(rng.gen());
            let p1: Wrapping<u64> = val - p0;
            let ret = if ctx.num.asymm == 1 {p1} else {p0};
            att_sel_vec.push(ret);
        }
        results.push(att_sel_vec);
    }
    Ok(results)
}
//in ring
pub fn create_random_ratios(quant: usize, ctx: &mut Context) -> Result<Vec<Wrapping<u64>>, Box<dyn Error> >{
    let seed = [1234567usize];
    let mut rng = rand::StdRng::from_seed(&seed);
    let upper_bound = 1 << ctx.num.precision_frac;

    let mut results = vec![];
    for i in 0 .. quant {
        let val = if ctx.num.asymm == 1 {Wrapping(rng.gen_range(0, upper_bound))} else {Wrapping(0)};
        results.push(val);
    }

    Ok(results)
}

fn load_arvs_from_file(path: &str, asym: usize, feature_count: usize, attribute_count: usize, tree_count: usize) -> Result<Vec<Vec<Vec<Wrapping<u64>>>>, Box<dyn Error>>{
    if asym != 1 {
       return Ok(vec![vec![vec![Wrapping(0); attribute_count]; feature_count]; tree_count]);
    }
    let mut ret = vec![];
    let vals = io::matrix_csv_to_wrapping_vec(path)?;
    for i in 0 .. vals.len() {
        let mut fsv = vec![];
        for j in 0 .. vals[i].len() {
            let index = vals[i][j].0 as usize;
            let mut sel_vec = vec![Wrapping(0); attribute_count];
            sel_vec[index] = Wrapping(1);
            fsv.push(sel_vec);
        }
        ret.push(fsv);
    }
    Ok(ret)
}

fn load_splits_from_file(path: &str, asym: usize, feature_count: usize, tree_count: usize, decimal_precision: usize) -> Result<Vec<Vec<Vec<Wrapping<u64>>>>, Box<dyn Error>>{
    let mut ratios = vec![vec![vec![Wrapping(0)]; feature_count]; tree_count];
    if asym != 1 {
        return Ok(ratios);
    }
    let float_ratios = io::matrix_csv_to_float_vec(path)?;
    for i in 0 .. tree_count {
        for j in 0 .. feature_count {
            ratios[i][j] = vec![float_to_fixed(float_ratios[i][j], decimal_precision)?];
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