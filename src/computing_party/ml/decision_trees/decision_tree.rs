use std::error::Error;
use super::super::super::Context;
use std::num::Wrapping;
use super::super::super::protocol;


// ------------- added by David, should probably be merge with TreeNode, but I didn't write that and don't wanna mess with it
pub struct SplitData {
    pub split_points: Vec<Wrapping<u64>>,
    pub feature_selection_vector: Vec<Vec<Wrapping<u64>>>,
    pub discritized_vec: Vec<Vec<Vec<Wrapping<u64>>>>,
}

impl Clone for SplitData {
    fn clone(&self) -> Self {
        SplitData {
            split_points: self.split_points.clone(),
            feature_selection_vector: self.feature_selection_vector.clone(),
            discritized_vec: self.discritized_vec.clone()
        }
    }
}
/// --------------

#[derive(Default)]
pub struct TreeNode {
    pub split_point: Wrapping<u64>,
    pub attribute_sel_vec: Vec<Wrapping<u64>>,
    pub classification: Wrapping<u64>,
}

#[derive(Default)]
pub struct TrainingContext {
    pub instance_count: usize,
    pub class_label_count: usize,
    pub attribute_count: usize, //attribute count in training context
    pub bin_count: usize,
    pub tree_count: usize,
    pub max_depth: usize,
}

pub fn run(ctx: &mut Context) -> Result<(), Box<dyn Error>> {
    Ok(())
}

//Additive stares are Wrapping<u64>, binary are u128
//TEMPORARY: BOTH INPUTS HERE SHOULD BE U128, TO GET THINGS RUNNING EARLY THEY'RE WRAPPING<U64>. THIS WILL CHANGE.
pub fn sid3t(input: Vec<Vec<Vec<Wrapping<u64>>>>, class: Vec<Vec<Wrapping<u64>>>, ctx: &mut Context, train_ctx: &mut TrainingContext) -> Result<Vec<Vec<TreeNode>>, Box<dyn Error>>{

    // VALUES NEEDED

    // A way to calculate how many nodes must be processed
    let number_of_nodes_to_process = 0;
    // Access to these values from ctx
    let class_label_count = train_ctx.class_label_count;
    let decimal_precision = 0;
    let asymmetric_bit = ctx.num.asymm;
    // The n'th vector should be should run parallel to the data to zero out non-relevent rows.
    // Should probably be additively shared in Z_q right? Needs to be used for multiplicaiton
    let mut layer_trans_bit_vecs: Vec<Vec<Wrapping<u64>>> = vec![];
    // amount of rows in dataset (not active rows designated by the tbv, just rows)
    let data_instance_count = train_ctx.instance_count;

    let mut classifications: Vec<Vec<Wrapping<u64>>> = vec![]; //ctx.dt_data.class_values.clone();

    // AND each discritized vector per node with all of the classifications
    let mut classifications_flattened = vec![];
    let mut trans_bit_vecs_flattened = vec![];

    for n in 0.. number_of_nodes_to_process {
        for i in 0.. class_label_count {
            classifications_flattened.append(&mut classifications[i].clone());
            trans_bit_vecs_flattened.append(&mut layer_trans_bit_vecs[n].clone());
        }
    }

    classifications_flattened = classifications_flattened.iter().map(|x| truncate_local(*x, decimal_precision, asymmetric_bit as u8)).collect();

    let and_results = protocol::multiply(&classifications_flattened, &trans_bit_vecs_flattened, ctx)?; 

    let tbv_and_classes_flat = &mut and_results.clone();

    let frequencies_flat_unsummed: Vec<Wrapping<u64>> = protocol::multiply(&tbv_and_classes_flat.clone(), &tbv_and_classes_flat.clone(), ctx)?;
    
    let mut frequencies_flat: Vec<Wrapping<u64>> = vec![];

    for n in 0.. number_of_nodes_to_process {
        for i in 0.. class_label_count {
            frequencies_flat.push(frequencies_flat_unsummed[(n * class_label_count + i) * data_instance_count.. 
                                    (n * class_label_count + i + 1) * data_instance_count].to_vec().iter().sum());
        }
    }

    // STEP 1: find most frequent classification
    let frequencies_argmax = most_frequent_class(&frequencies_flat.clone(), number_of_nodes_to_process, ctx, train_ctx);

    // STEP 2: max_depth exit condition

    // STEP 3: Apply class to node

    // STEP 4: GINI impurity

    let gini_argmax = gini_impurity(&and_results.clone(), ctx, train_ctx);

    // STEP 5: Create data structures for next layer based on step 4

    let placeholder = vec![vec![]];
    Ok(placeholder)
}


pub fn most_frequent_class(frequencies_flat: &Vec<Wrapping<u64>>, number_of_nodes_to_process: usize, ctx: &mut Context, train_ctx: &mut TrainingContext) -> Vec<Vec<Wrapping<u64>>> {


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

    frequencies_argmax
}


pub fn gini_impurity(u_decimal: &Vec<Wrapping<u64>>, ctx: &mut Context, train_ctx: &mut TrainingContext) -> Vec<Vec<Wrapping<u64>>> {

    // VALUES NEEDED

    // A way to calculate how many nodes must be processed
    let number_of_nodes_to_process = 0;
    // Access to these values from ctx
    let class_label_count = 0;
    let decimal_precision = 0;
    let asymmetric_bit = 0;
    let feat_count = 0;
    let bin_count = 0;
    let alpha = Wrapping(1);
    // amount of rows in dataset (not active rows designated by the tbv, just rows)
    let data_instance_count = 0;
    // subset of data containing OHE data
    let mut data_subset: Vec<SplitData<>> = vec![];

    // PHASE 3: FIND GINI INDEX OF FEATURE ON ITS SPLIT

    // unary vector parrallel to feauture with best random split
    // will be a vector of n vectors with k unary values of [0] (except one value will be [1].
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

    let mut discretized_sets = vec![vec![Wrapping(0)]; bin_count];
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
                                                // subset for the n'th tree, discretized by the k'th split,
                                                // where j is the j'th column of an OHE matrix
                    discretized_sets[j].append(&mut data_subset[n].discritized_vec[k][j].clone());

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


// Consider moving?
pub fn truncate_local(
    x: Wrapping<u64>,
    decimal_precision: u32,
    asymmetric_bit: u8,
) -> Wrapping<u64> {
    if asymmetric_bit == 0 {
        return -Wrapping((-x).0 >> decimal_precision);
    }

    Wrapping(x.0 >> decimal_precision)
}