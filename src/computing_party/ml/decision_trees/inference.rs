use std::error::Error;
use std::num::Wrapping;
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


pub fn classify_in_the_clear(trees: &Vec<Vec<TreeNode>>, transactions: &Vec<Vec<Wrapping<u64>>>, labels: &Vec<u64>, infer_ctx: InferenceContext) 
    -> Result<f64, Box<dyn Error>> {

        let bin_count = infer_ctx.bin_count;

        
        let ensemble = trees;
        let mut correctly_classified = 0;
        let total_rows =  transactions.len();

        let depth = infer_ctx.max_depth - 1;

        let mut transaction_index = 0;

        for transaction in transactions {

            let mut votes = vec![0; infer_ctx.class_label_count];

            for tree in ensemble {

                let mut vote = 0;
                
                let mut current_node = 1;

                for d in 0.. depth {

                    let chosen_attr = tree[current_node].attribute_sel_vec[0].0 as usize;
                    let splits = tree[current_node].split_point.clone();

                    let val = transaction[chosen_attr];

                    let mut bin = 0;
                    for split in splits {
                        if val < split {break};
                        bin += 1;
                    }

                    current_node = bin_count * current_node + bin;

                    vote += tree[current_node].classification.0 as usize;

                    if d + 1 == depth {
                        votes[vote] += 1;
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

//BIG problem for efficiency in RF rn, need to figure that out
//How do we know the first bin? double comparisons?
// pub fn batch_classify(transactions: &Vec<Vec<Wrapping<u64>>>, trees: &Vec<Vec<TreeNode>>, inf_ctx: InferenceContext, ctx: &mut Context) -> Result<Vec<Vec<Wrapping<u64>>>, Box<dyn Error>>{
//     let asym = ctx.num.asymm;
//     let max_depth = inf_ctx.max_depth;
//     let bin_count = inf_ctx.bin_count;
//     let mut result: Vec<Vec<Vec<Wrapping<u64>>>> = vec![vec![vec![]; trees.len()]; transactions.len()];
//     let mut currstart = 1;
//     let mut validbits = vec![asym as u128; trees.len() * transactions.len()]; //each root node for each transaction under consideration needs a valid bit
//     for layer in 0 .. max_depth {
//         let amountpertree = 2usize.pow(layer as u32);
//         let mut selection_vecs = vec![];
//         let mut splits = vec![];
//         let mut classes = vec![];
//         for _i in 0 .. transactions.len() {
//             for j in 0 .. trees.len() {
//                 for k in 0 .. amountpertree {
//                     splits.push(trees[j][currstart + k].split_point);
//                     classes.push(trees[j][currstart + k].classification);
//                     selection_vecs.append(&mut trees[j][currstart + k].attribute_sel_vec.clone());
//                 }
//             }
//         }

//         let validbits_add = protocol::z2_to_zq(&validbits, ctx)?;
//         let valid_classifications = protocol::multiply(&validbits_add, &classes, ctx)?; //the classification for the nodes

//         if layer == max_depth - 1 {
//             for i in 0 .. transactions.len() {
//                 for j in 0 .. trees.len() {
//                     for k in 0 .. amountpertree {
//                         result[i][j].push(valid_classifications[i * trees.len() * amountpertree + j * amountpertree  + k]);
//                     }
//                 }
//             }
//             continue;
//         }

//         let mut transactions_flat = vec![];
//         transactions.iter().for_each(|x| { 
//             for _i in 0 .. amountpertree * ctx.tree_count {
//                 transactions_flat.append(&mut x.clone())
//             }
//         });

//         let transactions_times_selection = batch_multiply(&transactions_flat, &selection_vecs, ctx);
//         let values: Vec<Wrapping<u64>> = transactions_times_selection.chunks(ctx.dt_data.attribute_count).map(|x| x.iter().fold(Wrapping(0), |acc, x| acc + x)).collect(); //for each multiplication of the ASV by the transaction, sum the results
//         let mut cmp_res = batch_compare(&values, &splits, ctx);
//         cmp_res.append(&mut cmp_res.iter().map(|x| x ^ ctx.asymmetric_bit as u64).collect()); //cmp_res now has the negation of the result of the comparison of the values and the split appended on the end
//         validbits.append(&mut validbits.clone()); //validbits duplicated
//         let newvalids = batch_bitwise_and(&validbits, &cmp_res, ctx, false);
//         let new_right_valids = newvalids[0 .. transactions.len() * trees.len() * amountpertree].to_vec();
//         let new_left_valids = newvalids[transactions.len() * trees.len() * amountpertree ..].to_vec();

//         validbits.clear();

//         for i in 0 .. transactions.len() {
//             for j in 0 .. trees.len() {
//                 for k in 0 .. amountpertree {
//                     result[i][j].push(valid_classifications[i * trees.len() * amountpertree + j * amountpertree  + k]);
//                     validbits.push(new_left_valids[i * trees.len() * amountpertree + j * amountpertree  + k]);
//                     validbits.push(new_right_valids[i * trees.len() * amountpertree + j * amountpertree  + k]);
//                 }
//             }
//         }
//         currstart += amountpertree;
//     }
//     let mut final_res = vec![];
//     for i in 0 .. transactions.len() {
//         final_res.push( result[i].iter().map(|x| truncate_local(x.iter().fold(Wrapping(0), |acc, x| x + acc), ctx.decimal_precision, ctx.asymmetric_bit)).collect());
//     }
//     Ok(final_res)
// }