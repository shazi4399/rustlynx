use std::error::Error;
use super::super::super::Context;
use std::num::Wrapping;
use super::super::super::protocol;
use super::super::super::super::util;


#[derive(Default)]
pub struct TreeNode {
    pub split_point: Vec<Wrapping<u64>>,
    pub attribute_sel_vec: Vec<Wrapping<u64>>,
    pub classification: Wrapping<u64>,
}

impl Clone for TreeNode {
    fn clone(&self) -> Self {
        TreeNode {
            split_point: self.split_point.clone(),
            attribute_sel_vec: self.attribute_sel_vec.clone(),
            classification: self.classification.clone()
        }
    }
}

#[derive(Default)]
pub struct TrainingContext {
    pub instance_count: usize,
    pub class_label_count: usize,
    pub attribute_count: usize, //attribute count in training context
    pub bin_count: usize,
    pub tree_count: usize,
    pub max_depth: usize,
    pub epsilon: f64,
}

pub fn run(ctx: &mut Context) -> Result<(), Box<dyn Error>> {
    Ok(())
}

//Additive stares are Wrapping<u64>, binary are u128
pub fn sid3t(input: &Vec<Vec<Vec<Wrapping<u64>>>>, class: &Vec<Vec<Vec<Wrapping<u64>>>>, att_sel_vecs: &Vec<Vec<Vec<Wrapping<u64>>>>,
    split_points: &Vec<Vec<Vec<Wrapping<u64>>>>, ctx: &mut Context, train_ctx: &mut TrainingContext) -> Result<Vec<Vec<TreeNode>>, Box<dyn Error>>{

    // VALUES NEEDED

    let asymmetric_bit = ctx.num.asymm as u64;
    let attribute_count = train_ctx.attribute_count;
    let class_label_count = train_ctx.class_label_count;
    let instance_count = train_ctx.instance_count;
    let tree_count = train_ctx.tree_count;
    let max_depth = train_ctx.max_depth;
    let decimal_precision = ctx.num.precision_frac;
    let bin_count = train_ctx.bin_count;
    let feat_count = attribute_count / bin_count;
    //let mut layer_data: Vec<LayerData> = vec![];
    let mut treenodes: Vec<Vec<TreeNode>> = vec![vec![]; tree_count];
    for t in 0..tree_count {
        treenodes[t].push(TreeNode {
            attribute_sel_vec: vec![],
            classification: Wrapping(0),
            split_point: vec![],
        });
    }

    println!("Initialized ensemble structure.");

    let mut ances_class_bits = vec![Wrapping(0u64); tree_count];
    let mut layer_trans_bit_vecs = vec![vec![Wrapping(asymmetric_bit); instance_count]; tree_count];

    for layer in 0.. max_depth {
        let tree_count = tree_count;
        let nodes_to_process_per_tree = bin_count.pow(layer as u32);

        let max_depth = layer == train_ctx.max_depth - 1; // Are we at the final layer?
        let number_of_nodes_to_process = nodes_to_process_per_tree * tree_count;
        if !max_depth /*&& !ctx.dt_training.discretize_per_tree*/ {

            // for v in 0..layer_trans_bit_vecs.len() {
            //     layer_data.push(LayerData { // Do not have layer data yet
            //         trans_bit_vec: layer_trans_bit_vecs[v].clone(),
            //         split_data: data_subset[v].clone(),
            //     });
            // }
        }
        // else if ctx.dt_training.discretize_per_tree {
        //     for t in 0 .. ctx.tree_count {
        //         for n in 0 .. nodes_to_process_per_tree {
        //             data_subset.push(data[t].clone());
        //             layer_data.push(LayerData {
        //                 trans_bit_vec: layer_trans_bit_vecs[t*nodes_to_process_per_tree + n].clone(),
        //                 split_data: data[t].clone()
        //             });
        //         }
        //     }
        // }

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

        //---- SECTION TO BE TURNED INTO BITVECS
        //DELETE THIS IF CLASSIFICATIONS ARE IN THE RING
        // may or may not be needed
        classifications_flattened = classifications_flattened.iter().map(|x| util::truncate(*x, decimal_precision, asymmetric_bit)).collect();

        let and_results = protocol::multiply(&classifications_flattened, &trans_bit_vecs_flattened, ctx)?; 

        let tbv_and_classes_flat = &mut and_results.clone();

        let frequencies_flat_unsummed: Vec<Wrapping<u64>> = protocol::multiply(&tbv_and_classes_flat.clone(), &tbv_and_classes_flat.clone(), ctx)?;
        
        let mut frequencies_flat: Vec<Wrapping<u64>> = vec![];

        for n in 0.. number_of_nodes_to_process {
            for i in 0.. class_label_count {
                frequencies_flat.push(frequencies_flat_unsummed[(n * class_label_count + i) * instance_count.. 
                                        (n * class_label_count + i + 1) * instance_count].to_vec().iter().sum());
            }
        }

        // STEP 1: find most frequent classification
        let frequencies_argmax = most_frequent_class(&frequencies_flat.clone(), number_of_nodes_to_process, ctx, train_ctx)?;
        let mut chosen_classifications = vec![];
        // STEP 2: max_depth exit condition
        let mut indices: Vec<Wrapping<u64>> = (0u64 .. class_label_count as u64).map(|x| Wrapping(x << decimal_precision)).collect(); //put into ring, as they aren't always either 0 or 1
        for n in 0 .. number_of_nodes_to_process {
            let mfcsv = frequencies_argmax[n].clone();
            chosen_classifications.push(mfcsv.iter().zip(indices.iter()).map(|(x, y)| x * y).sum());
            // chosen_classifications.push();
            // chosen_classifications.push(dot_product(&mfcsv, &indices, ctx, ctx.decimal_precision, false, false)); //don;t need truncation because frequencies_argmax is additively shared non ring zeroes and ones
        }
        if max_depth {
            
            let mut chosen_classifications = vec![];
            
            // println!("ances_class_bits{:?}", reveal(&ances_class_bits, ctx, ctx.decimal_precision, false, false));

            let asym_exp = vec![Wrapping(asymmetric_bit); number_of_nodes_to_process];

            let ances_xor_asym = protocol::xor(&ances_class_bits, &asym_exp, ctx)?;
            // println!("ances_xor_asym{:?}", reveal(&ances_xor_asym, ctx, ctx.decimal_precision, false, false));

            let this_layer_classifies = protocol::multiply(&ances_xor_asym, &vec![Wrapping(asymmetric_bit); number_of_nodes_to_process], ctx)?; //because we always want to try to classify as a leaf
            // println!("this_layer_classifies{:?}", reveal(&this_layer_classifies, ctx, ctx.decimal_precision, false, false));

            let chosen_classifications_corrected = protocol::multiply(&chosen_classifications, &this_layer_classifies, ctx)?;
            // println!("chosen_classifications_corrected{:?}", reveal(&chosen_classifications_corrected, ctx, ctx.decimal_precision, true, false));
            for t in 0 .. tree_count {
                for n in 0 .. nodes_to_process_per_tree {
                    treenodes[t].push(TreeNode {
                        attribute_sel_vec: vec![Wrapping(0u64); feat_count],
                        classification: chosen_classifications_corrected[t * nodes_to_process_per_tree + n],
                        split_point: vec![Wrapping(0)],
                    });
                }
            }
            return Ok(treenodes);
        }
        // STEP 3: Apply class to node

        // let size_of_tbvs: Vec<Wrapping<u64>> = layer_trans_bit_vecs.iter().map(|x| Wrapping(dot_product(&x, &x, ctx, ctx.decimal_precision, false, false).0 << ctx.decimal_precision)).collect();
        //how many instances in each subset
        let size_of_tbvs: Vec<Wrapping<u64>> = layer_trans_bit_vecs.iter().map(|x| x.iter().fold(Wrapping(0u64), |acc, val| acc + val) << decimal_precision).collect();
        let mut size_of_tbv_array = vec![];
        for v in 0 .. size_of_tbvs.len() {
            for _j in 0 .. class_label_count {
                size_of_tbv_array.push(size_of_tbvs[v]);
            }
        }

        // first, find out if class is constant
        let comparisons1 = protocol::batch_geq(&size_of_tbv_array, &frequencies_flat, ctx)?;
        let comparisons2 = protocol::batch_geq(&frequencies_flat, &size_of_tbv_array, ctx)?;
            
        let equality_array = protocol::multiply(
            &protocol::z2_to_zq(&comparisons1, ctx)?,
            &protocol::z2_to_zq(&comparisons2, ctx)?,
            ctx)?;
        
        let is_constant_class: Vec<Wrapping<u64>> = equality_array.chunks(class_label_count).map(|x| x.into_iter().sum()).collect();
        // will come out to be [1] if it is, [0] otherwise
        // let mut contains_constant_class = vec![];
        // for v in 0 .. size_of_tbvs.len() {
        //     let mut acc = Wrapping(0u64);
        //     for i in 0 .. class_label_count {
        //         acc += equality_array[v * class_label_count + i];
        //     }
        //     contains_constant_class.push(acc); //should only ever be one or zero, as ony one class could ever be constant at a time
        // }

        // let is_constant_class = xor_share_to_additive(&batch_compare(&contains_constant_class, &vec![Wrapping(asymmetric_bit); contains_constant_class.len()], ctx), ctx, 1);

        let epsilon = train_ctx.epsilon;

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
        let n_is_too_small = protocol::z2_to_zq(
            &protocol::batch_geq(&total_size_array, &size_of_tbv_array, ctx)?,
            ctx)?;
        let ances_class_bits_neg = ances_class_bits.iter().map(|x| -x + Wrapping(asymmetric_bit)).collect();

        // will be [0] if node does not classify data, [1] otherwise
        let this_layer_classifies = protocol::multiply(&protocol::or(&n_is_too_small, &is_constant_class, ctx)?, &ances_class_bits_neg, ctx)?;

        let mut new_tbvs = vec![];

        // STEP 4: GINI impurity
        //2d
        //nodes_to_process x feat_count
        let gini_argmax = gini_impurity(&input, &and_results.clone(), number_of_nodes_to_process, ctx, train_ctx);

        // STEP 5: Create data structures for next layer based on step 4

        let mut best_attributes: Vec<Vec<Wrapping<u64>>> = vec![]; //should be a vector containing selection vectors for the best attributes to split on for each node
        let mut chosen_split_points: Vec<Wrapping<u64>> = vec![]; //should be the associated value for the selection vector in best_attributes
        let mut chosen_classifications: Vec<Wrapping<u64>> = vec![]; //the most frequent classification at each node multiplied by the value calculated in this_layer_classification_bits\

        let indices = (0u64 .. class_label_count as u64).map(|x| vec![Wrapping(x << decimal_precision); number_of_nodes_to_process]).flatten().collect();
        if asymmetric_bit != 1 {
            indices = vec![Wrapping(0u64); class_label_count];
        }

        chosen_classifications = dot_product(&gini_argmax.clone().into_iter().flatten().collect(), &indices, class_label_count, ctx)?;
        let chosen_classifications_corrected = protocol::multiply(&chosen_classifications, &this_layer_classifies, ctx)?;
        let next_layer_classification_bits = protocol::or(&this_layer_classifies, &ances_class_bits, ctx)?;

        //for each node to process
        //
        //consolidate gini argmax with this node's fsv, use as selection vector in the node
        //select the subset of the dataset using an expansion of the fsv
        //multiply (and) each column of the subset with the node's tbv to get each new tbv
        //choose the correct list of splits using the gini argmax vec
        //assemble the nodes

        let gini_argmax_flat: Vec<Wrapping<u64>> = gini_argmax.iter().map(|x| x.iter().map(|y| vec![y; feat_count]).flatten().collect()).flatten().collect();
        let fsvs_flat: Vec<Wrapping<u64>> = att_sel_vecs.iter().map(|x| x.iter().map(|y| vec![y; nodes_to_process_per_tree]).flatten().collect()).flatten().flatten().collect();
        let uncompressed_selected_fsvs = protocol::multiply(&gini_argmax_flat, &fsvs_flat, ctx)?;

        let mut selected_fsvs = vec![];
        for i in 0 .. nodes_to_process_per_tree {
            let att_times_feat = attribute_count * feat_count;
            let mut fsvs_to_process = vec![];
            for j in 0 .. feat_count {
                fsvs_to_process.push(uncompressed_selected_fsvs[i * att_times_feat + j*(attribute_count) .. i * att_times_feat + (j+1)*(attribute_count)].to_vec());
            }
            let mut processed_fsv = vec![Wrapping(0); attribute_count];
            for j in 0 .. feat_count {
                for k in 0 .. attribute_count {
                    processed_fsv[k] += fsvs_to_process[j][k];
                }
            }
            selected_fsvs.push(processed_fsv); //the corresponding fsv to the index of the gini argmax
        }

        let selected_fsvs_exp: Vec<Wrapping<u64>> = selected_fsvs.iter().flat_map(|x| x.iter().map(|y| vec![vec![y; instance_count]; bin_count]).flatten().collect()).collect();
        let mut flattened_dataset: Vec<Wrapping<u64>> = vec![];
        for i in 0 .. tree_count {
            for j in 0 .. nodes_to_process_per_tree {
                flattened_dataset.append(&mut input[i].clone().into_iter().flatten().collect());
            }
        }
        let selected_columns_flat = protocol::multiply(&selected_fsvs_exp, &flattened_dataset, ctx)?; 
        let mut chosen_bins: Vec<Wrapping<u64>> = vec![];
        for i in 0 .. number_of_nodes_to_process {
            let nps = instance_count * attribute_count; //number per set
            let bin_indexer = instance_count * bin_count;
            let mut bins = vec![];
            for j in 0 .. bin_count {
                let mut bin = vec![Wrapping(0); instance_count];
                for k in 0 .. feat_count {
                    for l in 0 .. instance_count {
                        bin[l] += selected_columns_flat[i*nps + k * bin_indexer + l];
                    }
                }
                bins.append(&mut bin);
            }
            chosen_bins.append(&mut bins);
        }
        let mut tbv_exp = vec![];
        for i in 0 .. tree_count {
            for j in 0 .. nodes_to_process_per_tree {
                tbv_exp.append(&mut layer_trans_bit_vecs[i * nodes_to_process_per_tree + j].clone());
            }
        }
        let new_tbvs_flat = protocol::multiply(&chosen_bins, &tbv_exp, ctx)?;

        let selected_fsvs_exp: Vec<Wrapping<u64>> = selected_fsvs.iter().flat_map(|x| x.iter().map(|y| vec![y; bin_count]).flatten().collect()).collect();
        
        let mut split_points_flat: Vec<Wrapping<u64>> = vec![];
        for i in 0 .. tree_count {
            for j in 0 .. nodes_to_process_per_tree {
                split_points_flat.append(&mut split_points[i].clone().into_iter().flatten().collect());
            }
        }

        let chosen_split_points_uncompressed = protocol::multiply(&selected_fsvs_exp, &split_points_flat, ctx)?;
        let mut splits = vec![];
        for i in 0 .. number_of_nodes_to_process {
            let mut bin = vec![Wrapping(0); bin_count];
            for j in 0 .. feat_count {
                for k in 0 .. bin_count {
                    bin[k] += chosen_split_points_uncompressed[(i * feat_count * bin_count) + (j * bin_count) + k];
                }
            }
            splits.push(bin);
        }
        
        
        for t in 0 .. tree_count {
            for n in 0 .. nodes_to_process_per_tree {
                treenodes[t].push(TreeNode {
                    attribute_sel_vec: selected_fsvs[t * nodes_to_process_per_tree + n].clone(),
                    classification: chosen_classifications_corrected[t * nodes_to_process_per_tree + n],
                    split_point: splits[t * nodes_to_process_per_tree + n],
                });
            }
        }
        layer_trans_bit_vecs.clear(); //making sure that the layer_trans_bit_vecs have all been used so the data structure can be reused
        layer_trans_bit_vecs = new_tbvs.clone();
        ances_class_bits.clear();
        for j in 0 .. next_layer_classification_bits.len() {
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
                            (n + 1) * (current_length_freq - 1) + n].to_vec().clone());

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

        let xor_l_geq_r = protocol::batch_geq(&l_operands, &r_operands, ctx).unwrap();
        // read this as "left is greater than or equal to right." value in array will be [1] if true, [0] if false.
        let l_geq_r = protocol::z2_to_zq(&xor_l_geq_r, ctx).unwrap();
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
            past_assignments_freq[v - 1] = protocol::multiply(&past_assignments_freq[v - 1].clone(), &past_assignments_freq[v], ctx).unwrap();
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
        past_assignments_freq[v - 1] = protocol::multiply(&past_assignments_freq[v - 1].clone(), &extended_past_assignment_v, ctx).unwrap();
    }

    // un-flatten arg_max
    for n in 0.. number_of_nodes_to_process {
        frequencies_argmax.push(past_assignments_freq[0][n * class_label_count.. (n + 1) * class_label_count].to_vec());
    }

    Ok(frequencies_argmax)
}


pub fn gini_impurity(input: &Vec<Vec<Vec<Wrapping<u64>>>>, u_decimal: &Vec<Wrapping<u64>>, 
    number_of_nodes_to_process: usize, ctx: &mut Context, train_ctx: &mut TrainingContext) -> Vec<Vec<Wrapping<u64>>> {

    // Access to these values from ctx
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
    // 3. and have the j-th value of the k-th attribute


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
        discretized_sets_vectors.append(&mut discretized_sets[j].clone());
        u_decimal_vectors.append(&mut u_decimal_extended.clone());

    }

    // Remenent of the past, helps me see how to code for j vals instead of 2.

    // // double the vector. First half will be for values >= split, second half will be < split
    // let mut u_decimal_vectors_clone = u_decimal_vectors.clone();
    // u_decimal_vectors.append(&mut u_decimal_vectors_clone);

    // discretized_sets.append(&mut discretized_sets_negation);

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
                all_x_values.append(&mut x_partitioned[n][k][i].clone());
            }
        }
    }

    let all_x_values_squared: Vec<Wrapping<u64>> = protocol::multiply(&all_x_values.clone(), &all_x_values.clone(), ctx).unwrap();

    for v in 0..all_x_values_squared.len() / 2 {
        for j in 0.. bin_count {
            x2[v / (feat_count * class_label_count)][(v / class_label_count) % feat_count]
            [v % class_label_count][j] = all_x_values_squared[bin_count * v + j];
        }
    }

    // very unsure about this section, have to debug

    // At this point we have all of our x, x^2 and y values. Now we can start calculation gini numerators/denominators
    let mut sum_of_x2_j =  vec![vec![vec![Wrapping(0); bin_count]; feat_count]; number_of_nodes_to_process];

    let mut D_exclude_j = vec![vec![vec![Wrapping(0); bin_count]; feat_count]; number_of_nodes_to_process];
    let mut D_include_j = vec![vec![Wrapping(0); bin_count]; number_of_nodes_to_process];

    // create vector of the 0 and 1 values for j to set us up for batch multiplicaiton.
    // also, sum all of the x^2 values over i, and push these sums over i to a vector
    // to batch multiply with the y_without_j values.
    for n in 0..number_of_nodes_to_process {

        let mut y_vals_include_j = vec![vec![]; bin_count];

        for k in 0..feat_count {

            let mut y_vals_exlude_j = vec![vec![]; bin_count];

            for j in 0.. bin_count {
                
                // at the j'th index of the vector, we need to 
                // push all values at indeces that are not equal to j onto it
                // not 100% sure about this..
                for not_j in 0.. bin_count {
                    if j != not_j {
                        y_vals_exlude_j[j].push(y_partitioned[n][k][not_j]);
                    }
                }

                y_vals_include_j[j].push(y_partitioned[n][k][j]);

                let mut sum_j_values = Wrapping(0);
    
                for i in 0..class_label_count {
                    sum_j_values += x2[n][k][i][j];
                }
                sum_of_x2_j[n][k][j] = sum_j_values;
            }
            // can be far better optimized. Named 'D' after De'Hooghs variable
            D_exclude_j[n][k] = protocol::pairwise_mult(&y_vals_exlude_j, ctx).unwrap();
        }
        D_include_j[n] = protocol::pairwise_mult(&y_vals_include_j, ctx).unwrap();
    }

    let mut D_exclude_j_flattend = vec![];
    let mut D_include_j_flattend = vec![];
    let mut sum_of_x2_j_flattend = vec![];

    for n in 0.. number_of_nodes_to_process {
        for k in 0.. feat_count {
            D_exclude_j_flattend.append(&mut D_exclude_j[n][k]);
            sum_of_x2_j_flattend.append(&mut sum_of_x2_j[n][k]);
        }
        D_include_j_flattend.append(&mut D_include_j[n]);
    } 

    let gini_numerators_values_flat_unsummed = protocol::multiply(&D_exclude_j_flattend, &sum_of_x2_j_flattend, ctx).unwrap();

    for v in 0.. gini_numerators_values_flat_unsummed.len() / bin_count {
        for j in 0.. bin_count {
            gini_numerators[v] += gini_numerators_values_flat_unsummed[v * bin_count + j];
        }
    }

    // create denominators
    let gini_denominators: Vec<Wrapping<u64>> = D_include_j_flattend.clone();

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
                current_numerators.append(&mut new_numerators[n * (current_length).. (n + 1) * (current_length - 1) + n].to_vec().clone());
                current_denominators.append(&mut new_denominators[n * (current_length).. (n + 1) * (current_length - 1) + n].to_vec().clone());

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

        let xor_l_geq_r = protocol::batch_geq(&l_operands, &r_operands, ctx).unwrap();  

        // read this as "left is greater than or equal to right." value in array will be 1 if true, 0 if false.
        let l_geq_r = protocol::z2_to_zq(&xor_l_geq_r, ctx).unwrap();
        // read this is "left is less than right"
        let l_lt_r: Vec<Wrapping<u64>> = l_geq_r.iter().map(|x| -x + Wrapping(asymmetric_bit as u64)).collect();

        // grab the original values 
        let mut values = current_numerators.clone();
        values.append(&mut current_denominators.clone());

        let mut assignments = vec![];
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
            past_assignments[v - 1] = protocol::multiply(&past_assignments[v - 1].clone(), &past_assignments[v], ctx).unwrap();
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
        past_assignments[v - 1] = protocol::multiply(&past_assignments[v - 1].clone(), &extended_past_assignment_v, ctx).unwrap();
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