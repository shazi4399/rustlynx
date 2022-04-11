use std::error::Error;
use std::num::Wrapping;
use std::time::SystemTime;
use super::super::super::Context;
use super::super::super::super::util;
use super::super::super::protocol;
use super::super::super::super::io;

#[derive(Default)]
struct LRLearningContext {
    learning_rate: Wrapping<u64>,
    data: Vec<Vec<Wrapping<u64>>>,
    class: Vec<Wrapping<u64>>,
    weights: Vec<Wrapping<u64>>,
    n_attributes: usize,
    n_instances: usize,
    n_iterations: usize,
}

pub fn run(ctx: &mut Context) -> Result<(), Box<dyn Error>> {
    
    let mut lr = init(ctx)?;
    
    println!("rustlynx::computing_party::ml::logistic_regression::learning::run: done parsing cfg");
    let runtime = SystemTime::now();

    // lr.data = protocol::normalize(&lr.data, ctx)?;
    let data_mask = vec![vec![Wrapping(0u64); lr.n_instances]; lr.n_attributes]; /* TODO: Get from TI instead */
    let masked_data = protocol::open(
        &lr.data.iter().flatten().zip(data_mask.iter().flatten()).map(|(a, u)| a - u).collect(), 
        ctx)?;
   
    println!("_____________________________________");
    for i in 0..lr.n_iterations {    
        
        let now = SystemTime::now();
        let predictions = protocol::batch_matmul(
            &data_mask, 
            &masked_data, 
            &vec![util::transpose(&vec![lr.weights.clone()])?], 
            ctx)?[0][0].to_owned();
        println!("[iter={}] matmul complete..: {} ms", i, now.elapsed()?.as_millis());
        
        let now = SystemTime::now();
        let diff = lr.class.iter().zip(protocol::clipped_relu(&predictions, ctx)?.iter())
            .map(|(y, o)| y - o).collect::<Vec<Wrapping<u64>>>();
        lr.weights = protocol::batch_matmul(
                &data_mask, 
                &masked_data, 
                &vec![util::transpose(&diff.iter().map(|x| vec![*x; lr.n_instances]).collect())?], 
                ctx)?[0]
            .iter()
            .zip(lr.weights.iter())
            .map(|(x, w)| w + util::truncate(lr.learning_rate * x.iter().sum::<Wrapping<u64>>(), 
                    ctx.num.precision_frac, 
                    ctx.num.asymm))
            .collect::<Vec<Wrapping<u64>>>();
        println!("[iter={}] gradient complete: {} ms", i, now.elapsed()?.as_millis());
        println!("_____________________________________");
    }
    
    println!("training complete.........: {} ms", runtime.elapsed()?.as_millis());
    let parameters = protocol::open(&vec![], ctx)?;

    println!("rustlynx::computing_party::ml::logistic_regression::learning::run: outputting model parameters to file");
    Ok(())
}


fn init(ctx: &mut Context) -> Result<LRLearningContext, Box<dyn Error>> {
   
	let mut settings = config::Config::default();
    settings
        .merge(config::File::with_name(ctx.ml.cfg.as_str())).unwrap()
        .merge(config::Environment::with_prefix("APP")).unwrap();

    let mut data = util::transpose(&io::matrix_csv_to_wrapping_vec(&settings.get_str("data")?)?)?;
    let class = io::single_col_csv_to_wrapping_vec(&settings.get_str("class")?)?;
    let n_attributes = data.len();
    let n_instances = data[0].len();
    let weights = vec![Wrapping(0u64) ; n_attributes];
    let learning_rate = util::float_to_ring(0.001, ctx.num.precision_frac);

    let lr = LRLearningContext {
        learning_rate: learning_rate,
        data: data,
        class: class,
        weights: weights,
        n_attributes: n_attributes,
        n_instances: n_instances,
        n_iterations: 300 
    };

    Ok(lr)
}
