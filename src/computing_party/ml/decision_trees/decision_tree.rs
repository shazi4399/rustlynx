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

    let and_results = multiply_single_thread(&classifications_flattened, &trans_bit_vecs_flattened, ctx); //This is a additively shared secret value.

    let tbv_and_classes_flat = and_results.clone();

    let frequencies_flat_unsummed = batch_multiply(&tbv_and_classes_flat.clone(), &tbv_and_classes_flat.clone(), ctx);
    
    let mut frequencies_flat: Vec<Wrapping<u64>> = vec![];

    for n in 0.. number_of_nodes_to_process {
        for i in 0.. class_value_count {
            frequencies_flat.push(frequencies_flat_unsummed[(n * class_value_count + i) * data_instance_count.. 
                                    (n * class_value_count + i + 1) * data_instance_count].to_vec().iter().sum());
        }
    }

    // for each set of frequencies logically partitioned by the nodes, find the most frequent classification.
    let mut current_length_freq = class_value_count;

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

        let xor_l_geq_r = batch_compare_integer(&l_operands, &r_operands, ctx);

        // read this as "left is greater than or equal to right." value in array will be 1 if true, 0 if false.
        let l_geq_r = xor_share_to_additive(&xor_l_geq_r, ctx, 1);
        // read this is "left is less than right"
        let l_lt_r: Vec<Wrapping<u64>> = l_geq_r.iter().map(|x| -x + Wrapping(ctx.asymmetric_bit as u64)).collect();

        // grab the original values 
        let values = current_values.clone();

        let mut assignments = vec![];
        let mut freq_assignments = vec![];

        // alternate the left/right values to help to cancel out the original values
        // that lost in their comparison
        for v in 0..l_geq_r.len() {
            assignments.push(l_geq_r[v]);
            assignments.push(l_lt_r[v]);

            freq_assignments.push(l_geq_r[v]);
            freq_assignments.push(l_lt_r[v]);

            let size = current_length_freq/2;
            if odd_length && ((v + 1) % size == 0) {
                freq_assignments.push(Wrapping(asymmetric_bit));
            }
        }

        logical_partition_lengths_freq.push(current_length_freq);
        past_assignments_freq.push(freq_assignments);

        // EXIT CONDITION
        if (current_length_freq/2) + (current_length_freq % 2) == 1 {break;}

        let comparison_results = batch_multiply(&values, &assignments, ctx);

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
            past_assignments_freq[v - 1] = batch_multiply(&past_assignments_freq[v - 1].clone(), &past_assignments_freq[v], ctx);
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
        past_assignments_freq[v - 1] = batch_multiply(&past_assignments_freq[v - 1].clone(), &extended_past_assignment_v, ctx);
    }

    // un-flatten arg_maxx
    for n in 0.. number_of_nodes_to_process {
        frequencies_argmax.push(past_assignments_freq[0][n * class_value_count.. (n + 1) * class_value_count].to_vec());
    }

    // STEP 2: max_depth exit condition

    // STEP 3: Apply class to node

    // STEP 4: GINI impurity

    // STEP 5: Create data structures for next layer based on step 4

    let placeholder = vec![vec![]];
    Ok(placeholder)
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