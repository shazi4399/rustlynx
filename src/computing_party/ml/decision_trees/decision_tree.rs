use std::error::Error;
use super::super::super::Context;
use std::num::Wrapping;
use super::super::super::protocol;

#[derive(Default)]
pub struct TreeNode {
    split_point: Wrapping<u64>,
    attribute_sel_vec: Vec<Wrapping<u64>>,
    classification: Wrapping<u64>,
}

#[derive(Default)]
pub struct TrainingContext {
    instance_count: usize,
    attribute_count: usize, //attribute count in training context
    tree_count: usize,
    max_depth: usize,
}

pub fn run(ctx: &mut Context) -> Result<(), Box<dyn Error>> {

    Ok(())
}

//Additive stares are Wrapping<u64>, binary are u128
pub fn sid3t(input: Vec<Vec<u128>>, ctx: Context, train_ctx: TrainingContext) -> Result<Vec<Vec<TreeNode>>, Box<dyn Error>>{

    // VALUES NEEDED

    // A way to calculate how many nodes must be processed
    let number_of_nodes_to_process = 0;
    // Access to these values from ctx
    let class_label_count = 0;
    let decimal_precision = 0;
    let asymmetric_bit = 0;
    // The n'th vector should be should run parallel to the data to zero out non-relevent rows.
    // Should probably be additively shared in Z_q right? Needs to be used for multiplicaiton
    let layer_trans_bit_vecs: Vec<Vec<Wrapping<u64>>> = vec![];
    // amount of rows in dataset (not active rows designated by the tbv, just rows)
    let data_instance_count = 0;

    // STEP 1: find most frequent classification
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

    classifications_flattened = classifications_flattened.iter().map(|x| truncate_local(*x, decimal_precision, asymmetric_bit)).collect();

    let and_results = protocol::multiply(&classifications_flattened, &trans_bit_vecs_flattened, &mut ctx).unwrap(); 

    let tbv_and_classes_flat = &mut and_results.clone();

    let frequencies_flat_unsummed: Vec<Wrapping<u64>> = protocol::multiply(&tbv_and_classes_flat.clone(), &tbv_and_classes_flat.clone(), &mut ctx).unwrap();
    
    let mut frequencies_flat: Vec<Wrapping<u64>> = vec![];

    for n in 0.. number_of_nodes_to_process {
        for i in 0.. class_label_count {
            frequencies_flat.push(frequencies_flat_unsummed[(n * class_label_count + i) * data_instance_count.. 
                                    (n * class_label_count + i + 1) * data_instance_count].to_vec().iter().sum());
        }
    }

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

        let xor_l_geq_r = protocol::batch_geq(&l_operands, &r_operands, &mut ctx);
        // read this as "left is greater than or equal to right." value in array will be [1] if true, [0] if false.
        let l_geq_r = protocol::xor_share_to_additive(&xor_l_geq_r, ctx, 1);
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

        let comparison_results = protocol::multiply(&values, &assignments, &mut ctx);

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
            past_assignments_freq[v - 1] = protocol::multiply(&past_assignments_freq[v - 1].clone(), &past_assignments_freq[v], &mut ctx);
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
        past_assignments_freq[v - 1] = protocol::multiply(&past_assignments_freq[v - 1].clone(), &extended_past_assignment_v, &mut ctx);
    }

    // un-flatten arg_max
    for n in 0.. number_of_nodes_to_process {
        frequencies_argmax.push(past_assignments_freq[0][n * class_label_count.. (n + 1) * class_label_count].to_vec());
    }

    // STEP 2: max_depth exit condition

    // STEP 3: Apply class to node

    // STEP 4: GINI impurity

    // STEP 5: Create data structures for next layer based on step 4

    let placeholder = vec![vec![]];
    Ok(placeholder)
}



pub fn gini_impurity(u_decimal: Vec<Wrapping<u64>>, ctx: Context, train_ctx: TrainingContext) -> Vec<Vec<Wrapping<u64>>> {

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

    let mut u_decimal_vectors = vec![];

    let mut discretized_sets = vec![vec![Wrapping(0)]; bin_count];
    //let mut discretized_sets_negation = vec![];

    println!("Calculating x values"); // STEP EIGHT (8) IN DE'HOOGHS ALGORITHM

    // make vectors of active rows and and discretized sets parrallel to prepare for
    // batch multiplication in order to find frequencies of classes
    for n in 0..number_of_nodes_to_process {
        for k in 0..feat_count {
            for i in 0..class_label_count {
                for j in 0.. bin_count {

                    let mut u_decimal_clone = u_decimal[(n * class_label_count + i) * data_instance_count.. 
                    (n * class_label_count + i + 1) * data_instance_count].to_vec();
    
                    u_decimal_vectors.append(&mut u_decimal_clone);
    
                    discretized_sets[j].append(&mut data_subset[n].discritized_vec[k].val[j].clone());
                }
            }
        }
    }

    // double the vector. First half will be for values >= split, second half will be < split
    let mut u_decimal_vectors_clone = u_decimal_vectors.clone();
    u_decimal_vectors.append(&mut u_decimal_vectors_clone);

    discretized_sets.append(&mut discretized_sets_negation);

    let batched_un_summed_frequencies =
        batch_multiply(&u_decimal_vectors, &discretized_sets, ctx);

    let total_number_of_rows = ctx.dt_data.instance_count;

    let number_of_xs = batched_un_summed_frequencies.len() / (2 * total_number_of_rows);

    for v in 0..number_of_xs {
        let dp_result_geq_value: Wrapping<u64> = batched_un_summed_frequencies[v * total_number_of_rows..
        (v + 1) * total_number_of_rows]
        .to_vec().iter().sum();

        let dp_result_lt_value: Wrapping<u64> =
            batched_un_summed_frequencies[(v + number_of_xs) * total_number_of_rows..
            (v + 1 + number_of_xs) * total_number_of_rows]
            .to_vec().iter().sum();

        //x_partitioned[n][k][i][j == 1]
        x_partitioned[v / (attr_count * class_value_count)][(v / class_value_count) % attr_count]
            [v % class_value_count][1] = dp_result_geq_value;
        //y_partitioned[n][k][j == 1]
        y_partitioned[v / (attr_count * class_value_count)][(v / class_value_count) % attr_count][1] +=
            dp_result_geq_value;

        //x_partitioned[n][k][i][j == 0]
        x_partitioned[v / (attr_count * class_value_count)][(v / class_value_count) % attr_count]
            [v % class_value_count][0] = dp_result_lt_value;
        //y_partitioned[n][k][j == 0]
        y_partitioned[v / (attr_count * class_value_count)][(v / class_value_count) % attr_count][0] +=
            dp_result_lt_value;

        if dp_result_lt_value.0 == 0 {
            println!("WARNING! LEFT CHILD EMPTY")
        }

        if dp_result_geq_value.0 == 0 {
            println!("WARNING! RIGHT CHILD EMPTY")
        }

    }

    let mut all_x_values = vec![];

    println!("Calculating y values and x^2 values"); // STEP THIRTEEN (13) OF ALGORITHM
    for n in 0..number_of_nodes_to_process {
        for k in 0..attr_count {
            y_partitioned[n][k][0] =
                alpha * y_partitioned[n][k][0] + Wrapping(ctx.asymmetric_bit as u64);
            y_partitioned[n][k][1] =
                alpha * y_partitioned[n][k][1] + Wrapping(ctx.asymmetric_bit as u64);

            // will be used to find x^2
            for i in 0..class_value_count {
                all_x_values.append(&mut x_partitioned[n][k][i].clone());
            }
        }
    }

    let all_x_values_squared: Vec<Wrapping<u64>> = batch_multiply(&all_x_values.clone(), &all_x_values.clone(), ctx);

    for v in 0..all_x_values_squared.len() / 2 {
        x2[v / (attr_count * class_value_count)][(v / class_value_count) % attr_count]
            [v % class_value_count][0] = all_x_values_squared[2 * v];
        x2[v / (attr_count * class_value_count)][(v / class_value_count) % attr_count]
            [v % class_value_count][1] =
            all_x_values_squared[2 * v + 1];
    }

    // At this point we have all of our x, x^2 and y values
    let mut vector_of_sums_zero = vec![];
    let mut vector_of_sums_one = vec![];

    let mut y_value0 = vec![];
    let mut y_value1 = vec![];

    // create vector of the 0 and 1 values for j to set us up for batch multiplicaiton.
    // also, sum all of the x^2 values over i, and push these sums over i to a vector
    // to batch multiply with the y_without_j values.
    for n in 0..number_of_nodes_to_process {
        for k in 0..attr_count {
            let mut sum_lt_values = Wrapping(0);
            let mut sum_geq_values = Wrapping(0);

            y_value0.push(Wrapping(y_partitioned[n][k][0].0));
            y_value1.push(Wrapping(y_partitioned[n][k][1].0));

            for i in 0..class_value_count {
                sum_lt_values += x2[n][k][i][0];
                sum_geq_values += x2[n][k][i][1];
            }
            vector_of_sums_zero.push(sum_lt_values);
            vector_of_sums_one.push(sum_geq_values);
        }
    }

    // gini numerators corresponding to j = 0
    let gini_numerator_value_zero: Vec<Wrapping<u64>> = batch_multiply(&vector_of_sums_zero, &y_value1, ctx);
    // gini numerators corresponding to j = 1
    let gini_numerator_value_one: Vec<Wrapping<u64>> = batch_multiply(&vector_of_sums_one, &y_value0, ctx);

    // create numerators
    for v in 0..gini_numerator_value_one.len() {
        gini_numerators[v] = gini_numerator_value_one[v] + gini_numerator_value_zero[v];

    }
    
    // create denominators
    let gini_denominators: Vec<Wrapping<u64>> = batch_multiply(&y_value0, &y_value1, ctx);

    /////////////////////////////////////////// COMPUTE ARGMAX ///////////////////////////////////////////

    let mut current_length = attr_count;

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

        let product = batch_multiply(&current_numerators, &current_denominators_flipped, ctx);

        let mut l_operands = vec![];
        let mut r_operands = vec![];

        // left operands should be of the form n_1d_2, n_3d_4...
        // right operands should be of the form n_2d_1, n_4d_3...
        for v in 0..product.len()/2 {
            l_operands.push(product[2 * v]);
            r_operands.push(product[2 * v + 1]);
        }

        let xor_l_geq_r = batch_compare_integer(&l_operands, &r_operands, ctx);  

        // read this as "left is greater than or equal to right." value in array will be 1 if true, 0 if false.
        let l_geq_r = xor_share_to_additive(&xor_l_geq_r, ctx, 1);
        // read this is "left is less than right"
        let l_lt_r: Vec<Wrapping<u64>> = l_geq_r.iter().map(|x| -x + Wrapping(ctx.asymmetric_bit as u64)).collect();

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

        let comparison_results = batch_multiply(&values, &assignments, ctx);

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
            past_assignments[v - 1] = batch_multiply(&past_assignments[v - 1].clone(), &past_assignments[v], ctx);
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
        past_assignments[v - 1] = batch_multiply(&past_assignments[v - 1].clone(), &extended_past_assignment_v, ctx);
    }

    // un-flatten arg_max
    for n in 0.. number_of_nodes_to_process {
        if attr_count == 1 { // if there is only one attr count
            gini_arg_max.push(vec![Wrapping(asymmetric_bit)]);
        } else {
            gini_arg_max.push(past_assignments[0][n * attr_count.. (n + 1) * attr_count].to_vec());
        }
    }




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