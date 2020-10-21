use std::error::Error;
use std::num::Wrapping;
use std::time::SystemTime;
use super::super::super::Context;
use super::super::super::super::util;
use super::super::super::protocol;
use super::super::super::super::io;

#[derive(Default)]
struct NBInferenceContext {
    n_classes: usize,
    dict_len: usize,
    ex_len: usize,
    hash_len: usize,
    dict: Vec<u128>,
    example: Vec<u128>,
    l_probs: Vec<Vec<Wrapping<u64>>>,
    class_priors: Vec<Wrapping<u64>>,
    left_op: Vec<u128>,
    right_op: Vec<u128>,
}

pub fn run(ctx: &mut Context) -> Result<(), Box<dyn Error>> {

    let nb = init(&ctx.ml.cfg)?;

    println!("rustlynx::computing_party::ml::naive_bayes::inference::run: done parsing cfg");

    let membership = psi(&nb, ctx)?;    
    let class_probs = get_probabilities(&membership, &nb, ctx)?;
    let classification = protocol::argmax(&class_probs, ctx)?;

    println!("rustlynx::computing_party::ml::naive_bayes::inference::run: prediction: {:?}", protocol::open_z2( &classification, ctx )?);

    Ok(())
}


fn psi(nb: &NBInferenceContext, ctx: &mut Context) -> Result<Vec<Wrapping<u64>>, Box<dyn Error>> {

    let eq = protocol::equality_from_z2(&nb.left_op, &nb.right_op, nb.ex_len * nb.dict_len, nb.hash_len, ctx)?;
    let eq = protocol::z2_to_zq(&eq, ctx)?;

    let mut membership = vec![Wrapping(0u64) ; nb.dict_len];
    for i in 0..nb.dict_len {
        membership[i] = eq[nb.ex_len*i..nb.ex_len*(i+1)].into_iter().sum();
    }

    Ok(membership)
}

fn get_probabilities(vec: &Vec<Wrapping<u64>>, nb: &NBInferenceContext, ctx: &mut Context) -> Result<Vec<Wrapping<u64>>, Box<dyn Error>> {

    let mut left_op: Vec<Wrapping<u64>> = Vec::new();
    let mut right_op: Vec<Wrapping<u64>> = Vec::new();

    for i in 0..nb.n_classes {
        left_op.append(&mut vec.clone());
        let mut push = nb.l_probs[i].clone();
        right_op.append(&mut push);

    let probs = protocol::multiply(&left_op, &right_op, ctx)?;

    let mut class_probs: Vec<Wrapping<u64>> = Vec::new();
    for i in 0..nb.n_classes {
        let sum_of_probs = probs[i*nb.dict_len..(i+1)*nb.dict_len].into_iter().sum();
        class_probs.push( nb.class_priors[i] + util::truncate(sum_of_probs, ctx.num.precision_frac, ctx.num.asymm ));
    }

    Ok(class_probs)
}

fn init(cfg_file: &String) -> Result<NBInferenceContext, Box<dyn Error>> {
   
	let mut settings = config::Config::default();
    settings
        .merge(config::File::with_name(cfg_file.as_str())).unwrap()
        .merge(config::Environment::with_prefix("APP")).unwrap();

    let n_classes: usize = settings.get_int("n_classes")? as usize;
    let dict_len: usize = settings.get_int("dict_len")? as usize;
    let ex_len: usize = settings.get_int("ex_len")? as usize;
    let hash_len: usize = settings.get_int("hash_len")? as usize;

    let dict = io::single_col_csv_to_u128_vec(&settings.get_str("dict")?)?;    
    let example = io::single_col_csv_to_u128_vec(&settings.get_str("example")?)?;
    let l_probs = io::matrix_csv_to_wrapping_vec(&settings.get_str("l_probs")?)?;
    let class_priors = io::single_col_csv_to_wrapping_vec(&settings.get_str("class_priors")?)?;
       
    let l_probs = util::transpose(&l_probs)?;

    let mut left_op = Vec::<u128>::new();
    let mut right_op = Vec::<u128>::new();

    for i in 0..dict_len {
        left_op.append(&mut vec![ dict[i] ; ex_len ]);
        right_op.append(&mut example.clone());
    }

    let nb = NBInferenceContext {
        n_classes: n_classes,
        dict_len: dict_len,
        ex_len: ex_len,
        hash_len: hash_len,
        dict: dict,
        example: example,
        l_probs: l_probs,
        class_priors: class_priors,
        left_op: left_op,
        right_op: right_op,
    };

    Ok(nb)
}