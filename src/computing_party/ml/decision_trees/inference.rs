use std::error::Error;
use std::num::Wrapping;
use crate::util;
// use std::io::Read;
use super::super::super::Context;
use super::decision_tree::TreeNode;
#[derive(Default)]
pub struct InferenceContext {
    pub instance_count: usize,
    pub attribute_count: usize,
    pub bin_count: usize,
    pub class_label_count: usize,
    pub max_depth: usize,
}

pub fn run(ctx: &mut Context) -> Result<(), Box<dyn Error>> {

    Ok(())
}

pub fn classify_softvote(trees: &Vec<Vec<TreeNode>>, transactions_wrapping: &Vec<Vec<Wrapping<u64>>>, labels: &Vec<u64>,
                         infer_ctx: &InferenceContext, precision_int: usize, precision_frac: usize) -> Result<f64, Box<dyn Error>> {

    let mut transactions = vec![];

    for row in transactions_wrapping {
        transactions.push(util::ring_to_float(&row, precision_int, precision_frac)?)
    }


    let bin_count = infer_ctx.bin_count;

    let ensemble = trees;
    let mut correctly_classified = 0;
    let total_rows =  transactions.len();

    let depth = infer_ctx.max_depth - 1;

    let mut transaction_index = 0;

    for transaction in transactions {

        let mut votes = vec![0 as f64; infer_ctx.class_label_count];

        for tree in ensemble {

            let mut current_node = 0;

            for d in 0.. depth {
                let chosen_attr = tree[current_node].attribute_sel_vec[0].0 as usize;
                let splits = util::ring_to_float(&tree[current_node].split_point.clone(), precision_int, precision_frac)?;

                let val = transaction[chosen_attr];

                let mut bin = 1;
                for split in splits {
                    if val < split {break};
                    bin += 1;
                }

                current_node = bin_count * current_node + bin;

                if tree[current_node].frequencies[0].0 != 0 || tree[current_node].frequencies[1].0 != 0 {
                    let zero: f64 = tree[current_node].frequencies[0].0 as f64;
                    let one: f64 = tree[current_node].frequencies[1].0 as f64;
                    let total: f64 = zero + one;

                    votes[0] += zero/total;
                    votes[1] += one/total;
                }

            }

        }

        let mut largest_index = 0;
        let mut largest = votes[largest_index];

        // println!("{:?}", votes);

        for i in 1.. votes.len() {
            if largest < votes[i] {
                largest_index = i;
                largest = votes[i];
            }
        }

        if labels[transaction_index] as usize == largest_index {
            correctly_classified += 1;
        }

        transaction_index += 1;

    }

    Ok((correctly_classified as f64) / (total_rows as f64))
}



pub fn classify_argmax(trees: &Vec<Vec<TreeNode>>, transactions_wrapping: &Vec<Vec<Wrapping<u64>>>, labels: &Vec<u64>,
                       infer_ctx: &InferenceContext, precision_int: usize, precision_frac: usize) -> Result<f64, Box<dyn Error>> {

    let mut transactions = vec![];

    for row in transactions_wrapping {
        transactions.push(util::ring_to_float(&row, precision_int, precision_frac)?)
    }

    let bin_count = infer_ctx.bin_count;

    let ensemble = trees;
    let mut correctly_classified = 0;
    let total_rows =  transactions.len();

    let depth = infer_ctx.max_depth - 1;

    let mut transaction_index = 0;

    for transaction in transactions {

        let mut votes = vec![0; infer_ctx.class_label_count];

        for tree in ensemble {

            let mut valid_vote = false;

            let mut vote = 0;

            let mut current_node = 0;

            for d in 0.. depth {
                let chosen_attr = tree[current_node].attribute_sel_vec[0].0 as usize;
                let splits = util::ring_to_float(&tree[current_node].split_point.clone(), precision_int, precision_frac)?;

                let val = transaction[chosen_attr];

                let mut bin = 1;
                for split in splits {
                    if val < split {break};
                    bin += 1;
                }

                current_node = bin_count * current_node + bin;

                // if valid
                if tree[current_node].frequencies[0].0 != 0 || tree[current_node].frequencies[1].0 != 0 || d == infer_ctx.max_depth -1 {
                    // 'argmax'
                    if tree[current_node].frequencies[0] < tree[current_node].frequencies[1] {
                        votes[1] += 1;
                    } else {
                        votes[0] += 1;
                    }
                    break;
                }
            }
        }

        let mut largest_index = 0;
        let mut largest = votes[largest_index];

        for i in 1.. votes.len() {
            if largest < votes[i] {
                largest_index = i;
                largest = votes[i];
            }
        }

        if labels[transaction_index] as usize == largest_index {
            correctly_classified += 1;
        }

        transaction_index += 1;

    }

    Ok((correctly_classified as f64) / (total_rows as f64))
}
