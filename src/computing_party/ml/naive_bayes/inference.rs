use std::error::Error;
use std::num::Wrapping;
use std::time::SystemTime;
use super::super::super::Context;
use super::super::super::super::util;
use super::super::super::protocol;
use super::super::super::super::util::Bitset;

#[derive(Default)]
struct NBInferenceContext {
    n_classes: usize,
    dict_len: usize,
    ex_len: usize,
    hash_len: usize,
    dict: Vec<u128>,
    example: Vec<u128>,
    l_probs: Vec<Vec<Wrapping<u64>>>,
}

pub fn run(ctx: &mut Context) -> Result<(), Box<dyn Error>> {

    let n_tests = 10;
    for &dict_size in vec![369, 484, 688, 5200].iter() {
        for &ex_size in vec![5, 10, 25, 50, 100, 200].iter() {
            for &n_classes in vec![2].iter() {

                let mut runtimes: Vec<u128> = Vec::new();
                let nb = init(dict_size, ex_size, n_classes, ctx)?;
        
                for _test in 0..n_tests {
                    
                    let now = SystemTime::now();
        
                    let membership     = psi(&nb, ctx)?;
                    let class_probs    = get_probabilities(&membership, &nb, ctx)?;
                    let classification = protocol::argmax(&class_probs, ctx)?;
                
                    runtimes.push( now.elapsed().unwrap().as_millis() );

                }

                let avg_time = runtimes.iter().sum::<u128>() as f64 / (n_tests as f64);
                println!("dict_size: {:5}, ex_size: {:5}, n_classes: {:5} -- avg work time {:5.2} ms",
                dict_size, ex_size, n_classes, avg_time);
            }
        }
    }

    // let nb = init(dict_size, ex_size, n_classes, ctx)?;
    // let membership     = psi(&nb, ctx)?;
    // let class_probs    = get_probabilities(&membership, &nb, ctx)?;
    // let classification = protocol::argmax(&class_probs, ctx)?;

    Ok(())
}

fn psi(nb: &NBInferenceContext, ctx: &mut Context) -> Result<Vec<Wrapping<u64>>, Box<dyn Error>> {

    let mut left_op = Vec::<u128>::new();
    let mut right_op = Vec::<u128>::new();

    for i in 0..nb.dict_len {
        left_op.append(&mut vec![ nb.dict[i] ; nb.ex_len ]);
        right_op.append(&mut nb.example.clone());
    }

    let left_op = Bitset::new( left_op, nb.hash_len, ctx.num.asymm );
    let right_op = Bitset::new( right_op, nb.hash_len, ctx.num.asymm );
    
    let eq = protocol::equality_from_z2(&left_op, &right_op, ctx)?;
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
    }

    let probs = protocol::multiply(&left_op, &right_op, ctx)?;

    let mut class_probs: Vec<Wrapping<u64>> = Vec::new();
    for i in 0..nb.n_classes {
        let sum_of_probs = probs[i*nb.dict_len..(i+1)*nb.dict_len].into_iter().sum();
        class_probs.push( util::truncate(sum_of_probs, ctx.num.precision_frac, ctx.num.asymm ));
    }

    Ok(class_probs)

}

fn init(dict_size: usize, ex_size: usize, n_classes: usize, ctx: &mut Context) -> Result<NBInferenceContext, Box<dyn Error>> {

    
    let nb = NBInferenceContext {
        n_classes: n_classes,
        dict_len: dict_size,
        ex_len: ex_size,
        hash_len: 14,
        dict: vec![0u128 ; dict_size],
        example: vec![0u128 ; ex_size],
        l_probs: vec![ vec![ Wrapping(0u64) ; dict_size ] ; n_classes ],
    };

    Ok(nb)
}