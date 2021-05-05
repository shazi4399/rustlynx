use std::error::Error;
use super::super::super::Context;
use std::num::Wrapping;
use super::super::super::protocol;
use super::super::super::super::util;
use serde::{Serialize, Deserialize};
use std::fs;


#[derive(Default, Serialize, Deserialize)]
pub struct TreeNode {
    pub split_point: Vec<Wrapping<u64>>,
    pub attribute_sel_vec: Vec<Wrapping<u64>>,
    pub frequencies: Vec<Wrapping<u64>>,
}

impl Clone for TreeNode {
    fn clone(&self) -> Self {
        TreeNode {
            split_point: self.split_point.clone(),
            attribute_sel_vec: self.attribute_sel_vec.clone(),
            frequencies: self.frequencies.clone()
        }
    }
}

#[derive(Default)]
pub struct TrainingContext {
    pub instance_count: usize,
    pub class_label_count: usize,
    pub original_attr_count: usize,
    pub attribute_count: usize, //attribute count in training context
    pub bin_count: usize,
    pub tree_count: usize,
    pub max_depth: usize,
    pub epsilon: f64,
    pub save_location: String,
}

pub fn run(ctx: &mut Context) -> Result<(), Box<dyn Error>> {
    Ok(())
}

//Additive stares are Wrapping<u64>, binary are u128
pub fn sid3t(input: &Vec<Vec<Vec<Wrapping<u64>>>>, class: &Vec<Vec<Vec<Wrapping<u64>>>>, att_sel_vecs: &Vec<Vec<Vec<Wrapping<u64>>>>,
    split_points: &Vec<Vec<Vec<Wrapping<u64>>>>, ctx: &mut Context, train_ctx: &mut TrainingContext) -> Result<Vec<Vec<TreeNode>>, Box<dyn Error>>{

    let asymmetric_bit = ctx.num.asymm as u64;
    let original_attr_count = train_ctx.original_attr_count; //attribute count in orginal set (no bins)
    let attribute_count = train_ctx.attribute_count; //individual total dataset columns (including bins)
    let class_label_count = train_ctx.class_label_count;
    let instance_count = train_ctx.instance_count;
    let tree_count = train_ctx.tree_count;
    let decimal_precision = ctx.num.precision_frac;
    let bin_count = train_ctx.bin_count;
    let feat_count = attribute_count / bin_count; //How many features are represented within the subset
    let epsilon = train_ctx.epsilon;
    let mut treenodes: Vec<Vec<TreeNode>> = vec![vec![]; tree_count];
    for t in 0..tree_count {
        treenodes[t].push(TreeNode {
            attribute_sel_vec: vec![],
            frequencies: vec![Wrapping(0)],
            split_point: vec![],
        });
    }
    // input.iter().for_each(|x| x.iter().for_each(|y| println!("{:?}", protocol::open(&y, ctx).unwrap())));

    println!("Initialized ensemble structure.");

    let mut ances_class_bits = vec![Wrapping(0u64); tree_count];
    let mut layer_trans_bit_vecs = vec![vec![Wrapping(asymmetric_bit); instance_count]; tree_count];

    for layer in 0.. train_ctx.max_depth {
        let nodes_to_process_per_tree = bin_count.pow(layer as u32);

        let max_depth = layer == train_ctx.max_depth - 1; // Are we at the final layer?
        let number_of_nodes_to_process = nodes_to_process_per_tree * tree_count;

        // A way to calculate how many nodes must be processed
        // Access to these values from ctx
        // The n'th vector should be should run parallel to the data to zero out non-relevent rows.
        // Should probably be additively shared in Z_q right? Needs to be used for multiplicaiton
        // amount of rows in dataset (not active rows designated by the tbv, just rows)

        // AND each discritized vector per node with all of the classifications
        let mut classifications_flattened = vec![];
        let mut trans_bit_vecs_flattened = vec![];
        for t in 0 .. tree_count {
            for n in 0.. nodes_to_process_per_tree {
                for i in 0.. class_label_count {
                    classifications_flattened.append(&mut class[t][i].clone());
                    trans_bit_vecs_flattened.append(&mut layer_trans_bit_vecs[t * nodes_to_process_per_tree + n].clone());
                }
            }
        } 

        // STEP 1: find most frequent classification
        classifications_flattened = classifications_flattened.iter().map(|x| util::truncate(*x, decimal_precision, asymmetric_bit)).collect();
        // println!("CLASSIFICATIONS FLATTENED: {:?}", protocol::open(&classifications_flattened, ctx));

        let frequencies_flat_unsummed = protocol::multiply(&classifications_flattened, &trans_bit_vecs_flattened, ctx)?; 
        
        let mut frequencies_flat: Vec<Wrapping<u64>> = vec![];

        for n in 0.. number_of_nodes_to_process {
            for i in 0.. class_label_count {
                frequencies_flat.push(frequencies_flat_unsummed[(n * class_label_count + i) * instance_count.. 
                                        (n * class_label_count + i + 1) * instance_count].iter().sum());
            }
        }
       // println!("frequencies_flat{:?}", protocol::open(&frequencies_flat, ctx)?);

        // let frequencies_argmax = most_frequent_class(&frequencies_flat, number_of_nodes_to_process, ctx, train_ctx)?;
        // frequencies_argmax.iter().for_each(|x| println!("frequencies_argmax{:?}", protocol::open(&x, ctx).unwrap()));
        // let mut chosen_classifications: Vec<Wrapping<u64>> = vec![];
        // STEP 2: max_depth exit condition
        // let mut indices: Vec<Wrapping<u64>> = (0u64 .. class_label_count as u64).map(|x| Wrapping(x << decimal_precision)).collect(); //put into ring, as they aren't always either 0 or 1
        // let mut indices: Vec<Wrapping<u64>> = (0u64 .. class_label_count as u64).map(|x| Wrapping(x)).collect();
        // for n in 0 .. number_of_nodes_to_process {
        //     chosen_classifications.push(frequencies_argmax[n].iter().zip(indices.iter()).map(|(x, y)| x * y).sum());
        // }

        if max_depth {   
            // println!("chosen_classifications{:?}", protocol::open(&chosen_classifications, ctx)?);
            // println!("ances_class_bits{:?}", protocol::open(&ances_class_bits, ctx)?);

            let ances_class_bits_neg_exp = ances_class_bits.iter().map(|x| vec![-x + Wrapping(asymmetric_bit); class_label_count]).flatten().collect();
            // println!("ances_class_bits_neg{:?}", protocol::open(&ances_class_bits_neg, ctx)?);
            // println!("ances_xor_asym{:?}", reveal(&ances_xor_asym, ctx, ctx.decimal_precision, false, false));

            let chosen_classifications_corrected = protocol::multiply(&frequencies_flat, &ances_class_bits_neg_exp, ctx)?;
            //println!("chosen_classifications_corrected{:?}", protocol::open(&chosen_classifications_corrected, ctx)?);
            for t in 0 .. tree_count {
                for n in 0 .. nodes_to_process_per_tree {
                    treenodes[t].push(TreeNode {
                        attribute_sel_vec: vec![],
                        frequencies: chosen_classifications_corrected[t * nodes_to_process_per_tree * class_label_count + n * class_label_count .. t * nodes_to_process_per_tree * class_label_count + (n + 1) * class_label_count].to_vec(),
                        split_point: vec![Wrapping(0)],
                    });
                }
            }
            return Ok(treenodes);
        }

        // STEP 3: Apply class to node

        // let size_of_tbvs: Vec<Wrapping<u64>> = layer_trans_bit_vecs.iter().map(|x| Wrapping(dot_product(&x, &x, ctx, ctx.decimal_precision, false, false).0 << ctx.decimal_precision)).collect();
        //how many instances in each subset
        let size_of_tbvs: Vec<Wrapping<u64>> = layer_trans_bit_vecs.iter().map(|x| x.iter().fold(Wrapping(0u64), |acc, val| acc + val)).collect();
        let mut size_of_tbv_array = vec![];
        for v in 0 .. size_of_tbvs.len() {
            for _j in 0 .. class_label_count {
                size_of_tbv_array.push(size_of_tbvs[v]);
            }
        }
        let comparisons1 = protocol::batch_geq(&size_of_tbv_array, &frequencies_flat, ctx)?;
        let comparisons2 = protocol::batch_geq(&frequencies_flat, &size_of_tbv_array, ctx)?;
            
        let equality_array = protocol::multiply(
            &protocol::z2_to_zq(&comparisons1, ctx)?,
            &protocol::z2_to_zq(&comparisons2, ctx)?,
            ctx)?;

        // println!("equality_array: {:?}", protocol::open(&equality_array, ctx)?);
        let is_constant_class_sum: Vec<Wrapping<u64>> = equality_array.chunks(class_label_count).map(|x| x.into_iter().sum()).collect();
        let is_constant_class: Vec<Wrapping<u64>> =protocol::z2_to_zq(&protocol::batch_geq(&is_constant_class_sum, &vec![Wrapping(asymmetric_bit); is_constant_class_sum.len()], ctx)?, ctx)?;
        // println!("is_constant_class: {:?}", protocol::open(&is_constant_class, ctx)?);

        let total_size = Wrapping(
            ((epsilon * instance_count as f64)
                * 2.0f64.powf(decimal_precision as f64)) as u64,
        );
        let total_size_array: Vec<Wrapping<u64>>;
        if asymmetric_bit == 1 {
            total_size_array = vec![total_size; number_of_nodes_to_process];
        } else {
            total_size_array = vec![Wrapping(0); number_of_nodes_to_process];
        }
        //TODO CHECK POSSIBLE COMPARISON BETWEEN RING AND NON-RING VALUES
        let n_is_too_small = protocol::z2_to_zq(
            &protocol::batch_geq(&total_size_array, &size_of_tbv_array, ctx)?,
            ctx)?;
        // println!("n_is_too_small: {:?}", protocol::open(&n_is_too_small, ctx)?);
        let ances_class_bits_neg = ances_class_bits.iter().map(|x| -x + Wrapping(asymmetric_bit)).collect();
        // println!("ances_class_bits_neg: {:?}", protocol::open(&ances_class_bits_neg, ctx)?);

        // will be [0] if node does not classify data, [1] otherwise
        let this_layer_classifies = protocol::multiply(&protocol::or(&n_is_too_small, &is_constant_class, ctx)?, &ances_class_bits_neg, ctx)?;
        // println!("this_layer_classifies: {:?}", protocol::open(&this_layer_classifies, ctx)?);

        // STEP 4: GINI impurity
        //2d
        //nodes_to_process x feat_count

        let mut inputs_flattened = vec![];
        let mut trans_bit_vecs_flattened = vec![];
        for t in 0 .. tree_count {
            for m in 0 .. nodes_to_process_per_tree {
                inputs_flattened.append(&mut input[t].clone().into_iter().flatten().collect()); //POSSIBLE OPTIMIZATION
                for a in 0 .. attribute_count {
                    trans_bit_vecs_flattened.extend(&layer_trans_bit_vecs[t * nodes_to_process_per_tree + m]);
                }
            }
        } 

        let input_subsets_flattened = protocol::multiply(&inputs_flattened, &trans_bit_vecs_flattened, ctx)?; 

        // println!("{}", number_of_nodes_to_process);
        // println!("{}", attribute_count);

        let mut input_subsets = vec![vec![vec![]; attribute_count]; number_of_nodes_to_process];

        for n in 0 .. number_of_nodes_to_process {
            for a in 0 .. attribute_count {
                let indexer = a * instance_count + n * instance_count * attribute_count;
                input_subsets[n][a].extend(&input_subsets_flattened
                    [indexer .. indexer + instance_count]);

            }
        }

        let gini_argmax = gini_impurity(&input_subsets, &frequencies_flat_unsummed, number_of_nodes_to_process, ctx, train_ctx);

        // // ADDED
        // let gini_argmax_rev: Vec<Vec<Wrapping<u64>>> = gini_argmax.iter().map(|x| protocol::open(&x, ctx).unwrap()).collect();
        // let mut ans = vec![];
        // for i in 0 .. number_of_nodes_to_process {
        //     let mut index = 0;
        //     let gini = gini_argmax_rev[i].clone();
        //     for i in 0.. gini.len() {
        //         if gini[i] == Wrapping(1) {
        //             index = i;
        //             break;
        //         }
        //     }
        //     ans.push(index);
        // }
        //println!("GINI ARGMAX FOR NODE {:?}", ans);

        // STEP 5: Create data structures for next layer based on step 4

        let mut best_attributes: Vec<Vec<Wrapping<u64>>> = vec![]; //should be a vector containing selection vectors for the best attributes to split on for each node
        let mut chosen_split_points: Vec<Wrapping<u64>> = vec![]; //should be the associated value for the selection vector in best_attributes
        // let mut chosen_classifications: Vec<Wrapping<u64>>; //the most frequent classification at each node multiplied by the value calculated in this_layer_classification_bits\

        let mut indices_exp = (0u64 .. class_label_count as u64).map(|x| vec![Wrapping(x); number_of_nodes_to_process]).flatten().collect();
        if asymmetric_bit != 1 {
            indices_exp = vec![Wrapping(0u64); class_label_count * number_of_nodes_to_process];
        }

        // chosen_classifications = dot_product(&frequencies_argmax.into_iter().flatten().collect(), &indices_exp, class_label_count, ctx)?;
        // println!("chosen_classifications: {:?}", protocol::open(&chosen_classifications, ctx)?);
        // let chosen_classifications_corrected = protocol::multiply(&chosen_classifications, &this_layer_classifies, ctx)?;
        // println!("chosen_classifications_corrected: {:?}", protocol::open(&chosen_classifications_corrected, ctx)?);
        let next_layer_classification_bits = protocol::or(&this_layer_classifies, &ances_class_bits, ctx)?;
        // println!("next_layer_classification_bits: {:?}", protocol::open(&next_layer_classification_bits, ctx)?);

        //for each node to process
        //
        //consolidate gini argmax with this node's fsv, use as selection vector in the node
        //select the subset of the dataset using an expansion of the fsv
        //multiply (and) each column of the subset with the node's tbv to get each new tbv
        //choose the correct list of splits using the gini argmax vec
        //assemble the nodes

        // let gini_argmax_flat: Vec<Wrapping<u64>> = gini_argmax.iter().map(|x| x.iter().map(|y| vec![*y; instance_count]).flatten().collect::<Vec<Wrapping<u64>>>()).flatten().collect();
        let mut gini_argmax_flat_exp: Vec<Wrapping<u64>> = vec![];
        for i in 0 .. tree_count {
            for j in 0 .. nodes_to_process_per_tree {
                for k in 0 .. feat_count {
                    gini_argmax_flat_exp.extend(&vec![gini_argmax[i * nodes_to_process_per_tree + j][k]; original_attr_count]);
                }
            }
        }
        let this_layer_classifies_exp = this_layer_classifies.iter().map(|x| vec![*x; class_label_count]).flatten().collect(); 
        let corrected_frequencies = protocol::multiply(&this_layer_classifies_exp, &frequencies_flat, ctx)?;

        //let fsvs_flat: Vec<Wrapping<u64>> = att_sel_vecs.iter().map(|x| vec![x.clone(); nodes_to_process_per_tree]).flatten().flatten().flatten().collect();
        let mut fsvs_flat: Vec<Wrapping<u64>> = vec![];
        for i in 0 .. tree_count {
            for j in 0 .. nodes_to_process_per_tree {
                let mut fsv = vec![];
                for k in 0 .. feat_count {
                    fsv.extend(&att_sel_vecs[i][k]);
                }
                fsvs_flat.append(&mut fsv);
            }
        }
        let uncompressed_selected_fsvs = protocol::multiply(&gini_argmax_flat_exp, &fsvs_flat, ctx)?;

        let mut selected_fsvs: Vec<Vec<Wrapping<u64>>> = vec![];
        for i in 0 .. number_of_nodes_to_process {
            let att_times_feat = original_attr_count * feat_count;
            let mut fsvs_to_process = vec![];
            for j in 0 .. feat_count {
                fsvs_to_process.push(uncompressed_selected_fsvs[i * att_times_feat + j*(original_attr_count) .. i * att_times_feat + (j+1)*(original_attr_count)].to_vec());
            }
            let mut processed_fsv = vec![Wrapping(0); original_attr_count];
            for j in 0 .. feat_count {
                for k in 0 .. original_attr_count {
                    processed_fsv[k] += fsvs_to_process[j][k];
                }
            }
            selected_fsvs.push(processed_fsv); //the corresponding fsv to the index of the gini argmax
        }

        let mut select_best_feat_vec = vec![];
        for i in 0 .. number_of_nodes_to_process {
            for j in 0 .. feat_count {
                for k in 0 .. bin_count {
                    select_best_feat_vec.extend(&vec![gini_argmax[i][j]; instance_count]);
                }
            }
        }
        let mut flattened_dataset = vec![];
        for i in 0 .. tree_count {
            for j in 0 .. nodes_to_process_per_tree {
                for k in 0 .. attribute_count {
                    flattened_dataset.extend(&input[i][k]);
                }
            }
        }

        let selected_columns_flat = protocol::multiply(&select_best_feat_vec, &flattened_dataset, ctx)?; 
        let mut chosen_bins: Vec<Wrapping<u64>> = vec![];
        for i in 0 .. number_of_nodes_to_process {
            let nps = instance_count * attribute_count; //number per set
            let bin_indexer = instance_count * bin_count;
            let mut bins = vec![vec![Wrapping(0); instance_count]; bin_count];
            for j in 0 .. feat_count {
                for k in 0 .. bin_count {
                    for l in 0 .. instance_count {
                        bins[k][l] += selected_columns_flat[i*nps + j* bin_indexer + k*instance_count + l];
                    }
                }
            }
            chosen_bins.extend(bins.into_iter().flatten());
        }
        let mut tbv_exp = vec![];
        for i in 0 .. tree_count {
            for j in 0 .. nodes_to_process_per_tree {
                let tbv = layer_trans_bit_vecs[i * nodes_to_process_per_tree + j].clone();
                tbv_exp.extend(vec![tbv; bin_count].into_iter().flatten());
            }
        }
        let new_tbvs_flat = protocol::multiply(&chosen_bins, &tbv_exp, ctx)?;
        let new_tbvs : Vec<Vec<Wrapping<u64>>> = new_tbvs_flat.chunks(instance_count).map(|x| x.to_vec()).collect();
        // new_tbvs.iter().for_each(|x| println!("new_tbvs: {:?}", protocol::open(&x, ctx).unwrap()));
        
        let mut split_points_flat: Vec<Wrapping<u64>> = vec![];
        for i in 0 .. tree_count {
            for j in 0 .. nodes_to_process_per_tree {
                split_points_flat.extend(split_points[i].clone().into_iter().flatten());
            }
        }

        let gini_argmax_flat = gini_argmax.iter().map(|x| x.iter().map(|y| vec![*y; bin_count - 1]).flatten().collect::<Vec<Wrapping<u64>>>()).flatten().collect();
        let chosen_split_points_uncompressed = protocol::multiply(&gini_argmax_flat, &split_points_flat, ctx)?;
        let mut chosen_splits = vec![];
        for i in 0 .. number_of_nodes_to_process {
            let outer_iter = feat_count * (bin_count - 1);
            let mut node_splits = vec![Wrapping(0u64); bin_count - 1];
            for j in 0 .. feat_count {
                for k in 0 .. (bin_count - 1) {
                    node_splits[k] += chosen_split_points_uncompressed[i * outer_iter + j * (bin_count - 1) + k];
                }
            }
            chosen_splits.push(node_splits);
        }
        
        for t in 0 .. tree_count {
            for n in 0 .. nodes_to_process_per_tree {
                treenodes[t].push(TreeNode {
                    attribute_sel_vec: selected_fsvs[t * nodes_to_process_per_tree + n].clone(),
                    frequencies: corrected_frequencies[t * nodes_to_process_per_tree * class_label_count + n * class_label_count .. t * nodes_to_process_per_tree * class_label_count + (n + 1) * class_label_count].to_vec(),
                    split_point: chosen_splits[t * nodes_to_process_per_tree + n].clone(),
                });
            }
        }
        layer_trans_bit_vecs = new_tbvs;
        ances_class_bits.clear();
        for j in 0 .. number_of_nodes_to_process {
            for k in 0 .. bin_count {
                ances_class_bits.push(next_layer_classification_bits[j]);
            }
        }
    }
    Ok(treenodes)
}


pub fn most_frequent_class(frequencies_flat: &Vec<Wrapping<u64>>, 
    number_of_nodes_to_process: usize, ctx: &mut Context, train_ctx: &mut TrainingContext) -> Result<Vec<Vec<Wrapping<u64>>>, Box<dyn Error>>{


    let class_label_count = train_ctx.class_label_count;
    let asymmetric_bit = ctx.num.asymm;


    // for each set of frequencies logically partitioned by the nodes, find the most frequent classification.
    let mut current_length_freq = class_label_count;

    let mut logical_partition_lengths_freq = vec![];

    let mut new_values = frequencies_flat.clone();

    // this will allow us to calculate the arg_max
    let mut past_assignments_freq = vec![];

    // loop will continue until current_length_freq == 1
    loop {

        let odd_length = (current_length_freq % 2) == 1;

        let mut forgotten_values = vec![];

        let mut current_values = vec![];

        // if its of odd length, make current nums/dems even lengthed, and store the 'forgotten' values
        // it should be guarenteed that we have numerators/denonminators of even length
        if odd_length {
            for n in 0 .. number_of_nodes_to_process {
                current_values.append(&mut new_values[n * (current_length_freq).. 
                            (n + 1) * (current_length_freq - 1) + n].to_vec());

                forgotten_values.push(new_values[(n + 1) * (current_length_freq - 1) + n]);
            } 
        } else {
            current_values = new_values.clone();
        }

        let mut l_operands = vec![];
        let mut r_operands = vec![];

        for v in 0..current_values.len()/2 {
            l_operands.push(current_values[2 * v]);
            r_operands.push(current_values[2 * v + 1]);
        }

        // read this as "left is greater than or equal to right." value in array will be [1] if true, [0] if false.
        let l_geq_r = protocol::z2_to_zq(&protocol::batch_geq(&l_operands, &r_operands, ctx).unwrap(), ctx).unwrap();
        
        // read this is "left is less than right"
        let l_lt_r: Vec<Wrapping<u64>> = l_geq_r.iter().map(|x| -x + Wrapping(asymmetric_bit as u64)).collect();

        // grab the original values 
        let values = current_values.clone();

        let mut assignments = vec![];
        let mut freq_assignments = vec![];

        // alternate the left/right values to help to cancel out the original values
        // that 'lost' in their comparison
        for v in 0..l_geq_r.len() {
            assignments.push(l_geq_r[v]);
            assignments.push(l_lt_r[v]);

            freq_assignments.push(l_geq_r[v]);
            freq_assignments.push(l_lt_r[v]);

            let size = current_length_freq/2;
            if odd_length && ((v + 1) % size == 0) {
                freq_assignments.push(Wrapping(asymmetric_bit as u64));
            }
        }

        logical_partition_lengths_freq.push(current_length_freq);
        past_assignments_freq.push(freq_assignments);

        // EXIT CONDITION
        if (current_length_freq/2) + (current_length_freq % 2) == 1 {break;}

        let comparison_results = protocol::multiply(&values, &assignments, ctx).unwrap();

        new_values = vec![];

        // re-construct values
        for v in 0.. values.len()/2 {

            new_values.push(comparison_results[2 * v] + comparison_results[2 * v + 1]);

            if odd_length && ((v + 1) % current_length_freq/2) == 0 {
                new_values.push(current_values[2 * (v + 1)]);
            }
        }

        current_length_freq = (current_length_freq/2) + (current_length_freq % 2);
    }

    let mut frequencies_argmax = vec![];
    println!("Calculate arg_max for frequencies");

    // calculates flat arg_max in a tournament bracket style
    for v in (1..past_assignments_freq.len()).rev() {

        if past_assignments_freq[v].len() == past_assignments_freq[v - 1].len() {
            past_assignments_freq[v - 1] = protocol::multiply(&past_assignments_freq[v - 1], &past_assignments_freq[v], ctx).unwrap();
            continue;
        }

        let mut extended_past_assignment_v = vec![];
        for w in 0.. past_assignments_freq[v].len() {
            if ((w + 1) % logical_partition_lengths_freq[v] == 0) && ((logical_partition_lengths_freq[v - 1] % 2) == 1) {
                extended_past_assignment_v.push(past_assignments_freq[v][w]);
                continue;
            }
            extended_past_assignment_v.push(past_assignments_freq[v][w]);
            extended_past_assignment_v.push(past_assignments_freq[v][w]);
        }
        past_assignments_freq[v - 1] = protocol::multiply(&past_assignments_freq[v - 1], &extended_past_assignment_v, ctx).unwrap();
    }

    // un-flatten arg_max
    for n in 0.. number_of_nodes_to_process {
        frequencies_argmax.push(past_assignments_freq[0][n * class_label_count.. (n + 1) * class_label_count].to_vec());
    }

    Ok(frequencies_argmax)
}


pub fn gini_impurity(input: &Vec<Vec<Vec<Wrapping<u64>>>>, u_decimal: &Vec<Wrapping<u64>>, 
    number_of_nodes_to_process: usize, ctx: &mut Context, train_ctx: &mut TrainingContext) -> Vec<Vec<Wrapping<u64>>> {





        // let mut string: String = "".to_string();

        // for subset in input.clone() {
        //     for col in subset {
        //         string = [string, format!("{:?}", protocol::open(&col, ctx).unwrap())].join("\n");
        //     }
        // }

        // //println!("{}", string);

        // fs::write("input_res.txt", string).expect("Unable to write file");


    let class_label_count = train_ctx.class_label_count;
    let decimal_precision = ctx.num.precision_frac;
    let asymmetric_bit = ctx.num.asymm;
    let bin_count = train_ctx.bin_count;
    let feat_count = train_ctx.attribute_count/bin_count;

    let alpha = Wrapping(1); // Need this from ctx

    let data_instance_count = train_ctx.instance_count;

    // unary vector parrallel to feauture with best random split
    // will be a vector of n vectors with k unary values of [0] (except one value will be [1]).
    let mut gini_arg_max: Vec<Vec<Wrapping<u64>>> = vec![];

    let mut x_partitioned =
        vec![vec![vec![vec![Wrapping(0); bin_count]; class_label_count]; feat_count];number_of_nodes_to_process];
    let mut x2 = 
        vec![vec![vec![vec![Wrapping(0); bin_count]; class_label_count]; feat_count];number_of_nodes_to_process];

    let mut y_partitioned =
        vec![vec![vec![Wrapping(0); bin_count]; feat_count]; number_of_nodes_to_process];

    let mut gini_numerators = vec![Wrapping(0); feat_count * number_of_nodes_to_process];

    // Determine the number of transactions that are:
    // 1. in the current subset
    // 2. predict the i-th class value
    // 3. and have the j-th value of the k-th attribute for each node n


    // NOTE: Work on more meaningful names...
    let mut u_decimal_vectors = vec![];

    let mut u_decimal_extended = vec![];

    let mut discretized_sets_vectors = vec![];

    let mut discretized_sets = vec![vec![]; bin_count];
    //let mut discretized_sets_negation = vec![];

    println!("Calculating x values"); // STEP EIGHT (8) IN DE'HOOGHS ALGORITHM

    // make vectors of active rows and and discretized sets parrallel to prepare for
    // batch multiplication in order to find frequencies of classes
    for n in 0.. number_of_nodes_to_process {
        for k in 0.. feat_count {
            for i in 0.. class_label_count {

                let mut u_decimal_clone = u_decimal[(n * class_label_count + i) * data_instance_count.. 
                    (n * class_label_count + i + 1) * data_instance_count].to_vec();

                u_decimal_extended.append(&mut u_decimal_clone);


                for j in 0.. bin_count {
                    // grabs column of data, right?
                    discretized_sets[j].append(&mut input[n][k * bin_count + j].clone());
                }

            }
        }
    }

    let mut u_decimal_vectors_clone = u_decimal_vectors.clone();

    for j in 0.. bin_count {
        discretized_sets_vectors.append(&mut discretized_sets[j]);
        u_decimal_vectors.append(&mut u_decimal_extended.clone());

    }

    // Remenent of the past, helps me see how to code for j vals instead of 2.

    // // double the vector. First half will be for values >= split, second half will be < split
    // let mut u_decimal_vectors_clone = u_decimal_vectors.clone();
    // u_decimal_vectors.append(&mut u_decimal_vectors_clone);

    // discretized_sets.append(&mut discretized_sets_negation);

    //let u_decimal_vectors = u_decimal_vectors.iter().map(|x| Wrapping(x.0 >> decimal_precision)).collect();

    let batched_un_summed_frequencies =
        protocol::multiply(&u_decimal_vectors, &discretized_sets_vectors, ctx).unwrap();

    let total_number_of_rows = train_ctx.instance_count;

    let number_of_xs = batched_un_summed_frequencies.len() / (bin_count * total_number_of_rows);

    for v in 0..number_of_xs {
        
        for j in 0.. bin_count {

            // Sum up "total_number_of_rows" values to obtain frequency of classification for a particular subset
            // of data split a particular way dictated by things like the random feature chosen, and its split.
            let dp_result = batched_un_summed_frequencies
                [(v + number_of_xs * j) * total_number_of_rows.. 
                (v + 1 + number_of_xs * j) * total_number_of_rows].to_vec().iter().sum();

            x_partitioned[v / (feat_count * class_label_count)][(v / class_label_count) % feat_count]
                [v % class_label_count][j] = dp_result;

            y_partitioned[v / (feat_count * class_label_count)][(v / class_label_count) % feat_count][j] +=
                dp_result;

        }

    }

    let mut all_x_values = vec![];

    println!("Calculating y values and x^2 values"); // STEP THIRTEEN (13) OF ALGORITHM
    for n in 0..number_of_nodes_to_process {
        for k in 0..feat_count {
            y_partitioned[n][k][0] =
                alpha * y_partitioned[n][k][0] + Wrapping(asymmetric_bit as u64);
            y_partitioned[n][k][1] =
                alpha * y_partitioned[n][k][1] + Wrapping(asymmetric_bit as u64);

            // will be used to find x^2
            for i in 0..class_label_count {
                all_x_values.append(&mut x_partitioned[n][k][i]);
            }
        }
    }

    let all_x_values_squared: Vec<Wrapping<u64>> = protocol::multiply(&all_x_values, &all_x_values, ctx).unwrap();

    for v in 0..all_x_values_squared.len() / 2 {
        for j in 0.. bin_count {
            x2[v / (feat_count * class_label_count)][(v / class_label_count) % feat_count]
            [v % class_label_count][j] = all_x_values_squared[bin_count * v + j];
        }
    }

    // At this point we have all of our x, x^2 and y values. Now we can start calculation gini numerators/denominators
    let mut sum_of_x2_j =  vec![vec![vec![Wrapping(0); bin_count]; feat_count]; number_of_nodes_to_process];

    // let mut d_exclude_j = vec![vec![vec![Wrapping(0); bin_count]; feat_count]; number_of_nodes_to_process];
    // let mut d_include_j = vec![vec![]; number_of_nodes_to_process];

    let mut d_exclude_j = vec![];
    let mut d_include_j = vec![];

    // create vector of the 0 and 1 values for j to set us up for batch multiplicaiton.
    // also, sum all of the x^2 values over i, and push these sums over i to a vector
    // to batch multiply with the y_without_j values.
    for n in 0..number_of_nodes_to_process {

        let mut y_vals_include_j = vec![vec![]; feat_count];

        for k in 0..feat_count {

            let mut y_vals_exclude_j = vec![vec![]; bin_count];

            for j in 0.. bin_count {
                
                // at the j'th index of the vector, we need to 
                // push all values at indeces that are not equal to j onto it
                // not 100% sure about this..
                for not_j in 0.. bin_count {
                    if j != not_j {
                        y_vals_exclude_j[j].push(y_partitioned[n][k][not_j]);
                    }
                }

                y_vals_include_j[k].push(y_partitioned[n][k][j]);
                

                let mut sum_j_values = Wrapping(0);
    
                for i in 0..class_label_count {
                    sum_j_values += x2[n][k][i][j];
                }
                
                sum_of_x2_j[n][k][j] = sum_j_values;
            }
            d_exclude_j.extend(y_vals_exclude_j);
            // can be far better optimized. Named 'D' after De'Hooghs variable
            // d_exclude_j[n][k] = protocol::pairwise_mult_zq(&y_vals_exclude_j, ctx).unwrap();
            // println!("exclude {:?}", protocol::open(&d_exclude_j[n][k], ctx).unwrap()); //test
            // 
            // d_exclude_j.append(y_vals_exclude_j); what we should do?
        }
        d_include_j.extend(y_vals_include_j);
        //d_include_j[n] = protocol::pairwise_mult_zq(&y_vals_include_j, ctx).unwrap();
        // println!("include {:?}", protocol::open(&d_include_j[n], ctx).unwrap()); //test
        // d_include_j.append(y_vals_include_j); what we should do?
    }

    //let d_exclude_j_flattend = protocol::pairwise_mult_zq(&d_exclude_j, ctx).unwrap(); what we should do?
    //let d_include_j_flattend = protocol::pairwise_mult_zq(&d_include_j, ctx).unwrap(); what we should do?

    // let mut d_exclude_j_flattend = vec![];
    // let mut d_include_j_flattend = vec![];

    let mut sum_of_x2_j_flattend = vec![];

    for n in 0.. number_of_nodes_to_process {
        for k in 0.. feat_count {
            sum_of_x2_j_flattend.append(&mut sum_of_x2_j[n][k]);
        }
    } 

    let d_exclude_j = protocol::pairwise_mult_zq(&d_exclude_j, ctx).unwrap();
    let d_include_j = protocol::pairwise_mult_zq(&d_include_j, ctx).unwrap();

    // assert_eq!(protocol::open(&d, ctx).unwrap(), protocol::open(&d_exclude_j_flattend, ctx).unwrap());
    // assert_eq!(protocol::open(&d2, ctx).unwrap(), protocol::open(&d_include_j_flattend, ctx).unwrap());

    let gini_numerators_values_flat_unsummed = protocol::multiply(&d_exclude_j, &sum_of_x2_j_flattend, ctx).unwrap();

    // println!("{}", d_exclude_j_flattend.len()); //test
    // println!("{}", sum_of_x2_j_flattend.len()); //test

    for v in 0.. gini_numerators_values_flat_unsummed.len() / bin_count {
        for j in 0.. bin_count {
            gini_numerators[v] += gini_numerators_values_flat_unsummed[v * bin_count + j];
        }
    }

    // create denominators
    let gini_denominators: Vec<Wrapping<u64>> = d_include_j;

    // println!("{}: {:?}", gini_numerators.len(), protocol::open(&gini_numerators, ctx).unwrap()); //test
    // println!("{}: {:?}", gini_denominators.len(), protocol::open(&gini_denominators, ctx).unwrap()); //test

    /////////////////////////////////////////// COMPUTE ARGMAX ///////////////////////////////////////////

    let mut current_length = feat_count;

    let mut logical_partition_lengths = vec![];

    let mut new_numerators = gini_numerators.clone();
    let mut new_denominators = gini_denominators.clone();

    // this will allow us to calculate the arg_max
    let mut past_assignments = vec![];

    println!("Find largest gini ratios");

    // will run until current_length == 1
    loop {

        let odd_length = current_length % 2 == 1;
    
        let mut forgotten_numerators = vec![];
        let mut forgotten_denominators = vec![];

        let mut current_numerators = vec![];
        let mut current_denominators = vec![];

        // if its of odd length, make current nums/dems even lengthed, and store the 'forgotten' values
        // it should be guarenteed that we have numerators/denonminators of even length
        if odd_length {
            for n in 0 .. number_of_nodes_to_process {
                current_numerators.append(&mut new_numerators[n * (current_length).. (n + 1) * (current_length - 1) + n].to_vec());
                current_denominators.append(&mut new_denominators[n * (current_length).. (n + 1) * (current_length - 1) + n].to_vec());

                forgotten_numerators.push(new_numerators[(n + 1) * (current_length - 1) + n]);
                forgotten_denominators.push(new_denominators[(n + 1) * (current_length - 1) + n]);

            }
        } else {
            current_numerators = new_numerators.clone();
            current_denominators = new_denominators.clone();
        }

        // if denominators were originally indexed as d_1, d_2, d_3, d_4... then they are now represented
        // as d_2, d_1, d_4, d_3... This is helpful for comparisons
        let mut current_denominators_flipped = vec![];
        for v in 0 .. current_denominators.len()/2 {
            current_denominators_flipped.push(current_denominators[2 * v + 1]);
            current_denominators_flipped.push(current_denominators[2 * v]);
        }

        let product = protocol::multiply(&current_numerators, &current_denominators_flipped, ctx).unwrap();

        let mut l_operands = vec![];
        let mut r_operands = vec![];

        // left operands should be of the form n_1d_2, n_3d_4...
        // right operands should be of the form n_2d_1, n_4d_3...
        for v in 0..product.len()/2 {
            l_operands.push(product[2 * v]);
            r_operands.push(product[2 * v + 1]);
        }

        // read this as "left is greater than or equal to right." value in array will be [1] if true, [0] if false.
        let l_geq_r = protocol::z2_to_zq(&protocol::batch_geq(&l_operands, &r_operands, ctx).unwrap(), ctx).unwrap();  

        // read this is "left is less than right"
        let l_lt_r: Vec<Wrapping<u64>> = l_geq_r.iter().map(|x| -x + Wrapping(asymmetric_bit as u64)).collect();

        // grab the original values 
        let mut values = current_numerators.clone();
        values.append(&mut current_denominators.clone());

        // For the next iteration
        let mut assignments = vec![];
        // For record keeping
        let mut gini_assignments = vec![];

        // alternate the left/right values to help to cancel out the original values
        // that lost in their comparison
        for v in 0..l_geq_r.len() {
            assignments.push(l_geq_r[v]);
            assignments.push(l_lt_r[v]);

            gini_assignments.push(l_geq_r[v]);
            gini_assignments.push(l_lt_r[v]);

            let size = current_length/2;
            if odd_length && ((v + 1) % size == 0) {
                gini_assignments.push(Wrapping(asymmetric_bit));
            }
        }

        logical_partition_lengths.push(current_length);
        past_assignments.push(gini_assignments); 

        // EXIT CONDITION
        if 1 == (current_length/2) + (current_length % 2) {break;}

        assignments.append(&mut assignments.clone());

        let comparison_results = protocol::multiply(&values, &assignments, ctx).unwrap();

        new_numerators = vec![];
        new_denominators = vec![];

        // re-construct new_nums and new_dems
        for v in 0.. values.len()/4 {

            new_numerators.push(comparison_results[2 * v] + comparison_results[2 * v + 1]);
            new_denominators.push(comparison_results[values.len()/2 + 2 * v] + comparison_results[values.len()/2 + 2 * v + 1]);

            if odd_length && ((v + 1) % (current_length/2) == 0) {
                // ensures no division by 0
                let divisor = if current_length > 1 {current_length/2} else {1};
                new_numerators.push(forgotten_numerators[(v/divisor)]);
                new_denominators.push(forgotten_denominators[(v/divisor)]);
            }

        }

        current_length = (current_length/2) + (current_length % 2);
    }

    println!("Calculate arg_max for gini");

    // calculates flat arg_max in a tournament bracket style
    for v in (1..past_assignments.len()).rev() {

        if past_assignments[v].len() == past_assignments[v - 1].len() {
            past_assignments[v - 1] = protocol::multiply(&past_assignments[v - 1], &past_assignments[v], ctx).unwrap();
            continue;
        }

        let mut extended_past_assignment_v = vec![];
        for w in 0.. past_assignments[v].len() {
            if ((w + 1) % logical_partition_lengths[v] == 0) && ((logical_partition_lengths[v - 1] % 2) == 1) {
                extended_past_assignment_v.push(past_assignments[v][w]);
                continue;
            }
            extended_past_assignment_v.push(past_assignments[v][w]);
            extended_past_assignment_v.push(past_assignments[v][w]);
        }
        past_assignments[v - 1] = protocol::multiply(&past_assignments[v - 1], &extended_past_assignment_v, ctx).unwrap();
    }

    // un-flatten arg_max
    for n in 0.. number_of_nodes_to_process {
        if feat_count == 1 { // if there is only one attr count
            gini_arg_max.push(vec![Wrapping(asymmetric_bit)]);
        } else {
            gini_arg_max.push(past_assignments[0][n * feat_count.. (n + 1) * feat_count].to_vec());
        }
    }
    gini_arg_max
}

fn dot_product(x: &Vec<Wrapping<u64>>, y: &Vec<Wrapping<u64>>, sub_len: usize, ctx: &mut Context) -> Result<Vec<Wrapping<u64>>, Box<dyn Error>> {
    let mut mult_res:Vec<Wrapping<u64>> = protocol::multiply(x, y, ctx)?;
    let res = mult_res.chunks(sub_len).map(|subvec| subvec.iter().sum()).collect();
    Ok(res)
}

pub fn reveal_tree(nodes: &Vec<TreeNode>, ctx: &mut Context) -> Result<Vec<TreeNode>, Box<dyn Error>>{
    let mut freqs = vec![];
    let mut split_points = vec![];
    let mut att_sel_vecs = vec![];
    let mut rev_node = vec![];
    for i in 0..nodes.len() {
        //index changing is because dummy is in pos 0
        freqs.push(nodes[i].frequencies.clone());
        split_points.push(nodes[i].split_point.clone());
        att_sel_vecs.push(nodes[i].attribute_sel_vec.clone());
    }
    //println!("\n");
    for i in 0..nodes.len() {
        let att_sel_vecs_rev = protocol::open(&att_sel_vecs[i], ctx)?;
        let split_points_rev = protocol::open(&split_points[i], ctx)?;
        let freqs_rev = protocol::open(&freqs[i], ctx)?;
        let mut attr = att_sel_vecs_rev.iter().position(|x| *x == Wrapping(1u64));
        let attr_wrap: Wrapping<u64> = if attr.is_some() {Wrapping(attr.unwrap() as u64)} else {Wrapping(0)};
        let float_splits: Vec<f64> = split_points_rev.iter().map(|x| x.0 as f64 / 2f64.powf(ctx.num.precision_frac as f64)).collect();
        
        //println!("Node#{:?}, frequencies:{:?}, split_point:{:?}, att_sel_vec:{:?}", i , freqs_rev,  float_splits, att_sel_vecs_rev);
        rev_node.push(TreeNode {
            attribute_sel_vec: vec![attr_wrap],
            frequencies: freqs_rev,
            split_point: split_points_rev
        })
    }
    Ok(rev_node)
}