//TODO LIST
//Write init, setting up context
//port the preprocessing phase

use std::num::Wrapping;
use crate::computing_party::Context;
use crate::computing_party::ml::decision_trees::decision_tree::TrainingContext;
use std::error::Error;
use crate::computing_party::protocol::discretize_into_ohe_batch;
use crate::{util, io};
use crate::computing_party::ml::decision_trees::extra_trees::extra_trees::create_selection_vectors;
use rand::Rng;

#[derive(Default)]
pub struct RFContext {
    pub tc: TrainingContext,
    pub feature_count: usize,
}

pub const LOCAL_ADDITION: u8 = 0;
pub const LOCAL_SUBTRACTION: u8 = 1;
// never change
pub const MULT_ELEMS: usize = 2;
pub const SIZEOF_U64: usize = 8;

// can tweak batch size to optimize batch multiplication for different machines
pub const BATCH_SIZE: usize = 4096;
// how many mults can be done in one tx
pub const REVEAL_BATCH_SIZE: usize = 2 * BATCH_SIZE;
// how many reveals in one TX

pub const BUF_SIZE: usize = BATCH_SIZE * MULT_ELEMS * SIZEOF_U64;
pub const U64S_PER_TX: usize = MULT_ELEMS * BATCH_SIZE;
pub const U8S_PER_TX: usize = 8 * U64S_PER_TX;

pub const U64S_PER_MINI_TX: usize = 1024;
pub const U8S_PER_MINI_TX: usize = 8 * U64S_PER_MINI_TX;

pub const TI_BATCH_SIZE: usize = U64S_PER_TX / 3; // how many trplets in one tx

pub const BINARY_PRIME: usize = 2;

pub fn rf_preprocess(data: &Vec<Vec<Wrapping<u64>>>, rfctx: &mut RFContext, ctx: &mut Context)->Result<(Vec<Vec<Vec<Wrapping<u64>>>>, Vec<Vec<Vec<Wrapping<u64>>>>, Vec<Vec<Vec<Wrapping<u64>>>>), Box<dyn Error>> {
    let bucket_size = rfctx.tc.attribute_count;
    let processed_data_com = discretize_into_ohe_batch(&util::transpose(data)?,bucket_size,ctx );
    let processed_data = processed_data_com.0;
    let full_splits = processed_data_com.1;
    //hard coded shares
    let column_major_arvs = generate_rfs_share(rfctx.tc.tree_count, rfctx.tc.attribute_count,rfctx.feature_count, ctx)?;
    let mut final_arv_splits = Vec::new();
    //hard coded shares
    let matrix_shares = (vec![vec![Wrapping(0u64);1];1],vec![vec![Wrapping(0u64);1];1],vec![vec![Wrapping(0u64);1];1]);
    for t in 0..rfctx.tc.tree_count{
        final_arv_splits.push(matrix_multiplication_integer(column_major_arvs[t],&full_splits,ctx,0u64,&matrix_shares));
    }
    Ok((processed_data, column_major_arvs, final_arv_splits))

}

fn generate_rfs_share(tree_cnt:usize,feature_selected:usize,feature_cnt:usize,ctx: &mut Context) -> (Vec<Vec<Vec<Wrapping<u64>>>>) {
    //generate selection vector
    let mut feature_selected_remain = feature_selected;
    let mut feature_bit_vec = vec![0u8; feature_cnt];
        for item in [11, 13, 17, 23, 6].to_vec() {
            feature_bit_vec[item] = 1u8;
        }
    let mut fs_vec:Vec<Vec<Vec<Wrapping<u64>>>> = vec![vec![vec![Wrapping(0u64);feature_cnt];feature_selected];tree_cnt];

    for t in 0..tree_cnt{
        let mut cnt = 0usize;
        for i in 0..feature_bit_vec.len() {
            let item = feature_bit_vec[i];
            if item == 1 {
                fs_vec[t][cnt][i]= Wrapping(ctx.num.asymm);
                cnt+=1;
            }
        }
    }


    fs_vec
}

pub fn mod_subtraction(x: Wrapping<u64>, y: Wrapping<u64>, prime: u64) -> Wrapping<u64> {
    Wrapping((x.0 as i64 - y.0 as i64).mod_floor(&(prime as i64)) as u64)
}

fn local_matrix_computation(x: &Vec<Vec<Wrapping<u64>>>, y: &Vec<Vec<Wrapping<u64>>>, prime: u64, operation: u8) -> Vec<Vec<Wrapping<u64>>> {
    let mut result = Vec::new();
    for i in 0..x.len() {
        let mut row = Vec::new();
        for j in 0..x[0].len() {
            match operation {
                LOCAL_ADDITION => row.push(Wrapping((x[i][j] + y[i][j]).0.mod_floor(&prime))),
                LOCAL_SUBTRACTION => row.push(mod_subtraction(x[i][j], y[i][j], prime)),
                _ => {}
            }
        }
        result.push(row);
    }
    result
}

fn local_matrix_multiplication(x: &Vec<Vec<Wrapping<u64>>>, y: &Vec<Vec<Wrapping<u64>>>, prime: u64) -> Vec<Vec<Wrapping<u64>>> {
    let i = x.len();
    let k = x[0].len();
    let j = y[0].len();
    let mut result = Vec::new();
    for m in 0..i {
        let mut row = Vec::new();
        for n in 0..j {
            let mut multi_result = Wrapping(0);
            for p in 0..k {
                multi_result += (x[m][p] * y[p][n]);
            }
            multi_result = Wrapping(multi_result.0.mod_floor(&prime));
            row.push(multi_result);
        }
        result.push(row);
    }
    result
}

fn send_batch_message(ctx: &Context, data: &Vec<u8>) -> Xbuffer {
    let mut o_stream  = ctx.net.external.tcp.as_ref().unwrap()[i].ostream.try_clone()
        .expect("rustlynx::computing_party::protocol::multiply: failed cloning tcp ostream");
    let mut in_stream = ctx.net.external.tcp.as_ref().unwrap()[i].istream.try_clone()
        .expect("rustlynx::computing_party::protocol::multiply: failed cloning tcp istream");
    let mut recv_buf = Xbuffer { u8_buf: [0u8; U8S_PER_TX] };
    if ctx.num.asymm == 1 {
        let mut bytes_written = 0;
        while bytes_written < U8S_PER_TX {
            let current_bytes = unsafe {
                o_stream.write(&data[bytes_written..])
            };
            bytes_written += current_bytes.unwrap();
        }

        unsafe {
            let mut bytes_read = 0;
            while bytes_read < recv_buf.u8_buf.len() {
                let current_bytes = in_stream.read(&mut recv_buf.u8_buf[bytes_read..]).unwrap();
                bytes_read += current_bytes;
            }
        }
    } else {
        unsafe {
            let mut bytes_read = 0;
            while bytes_read < recv_buf.u8_buf.len() {
                let current_bytes = in_stream.read(&mut recv_buf.u8_buf[bytes_read..]).unwrap();
                bytes_read += current_bytes;
            }
        }

        let mut bytes_written = 0;
        while bytes_written < U8S_PER_TX {
            let current_bytes = unsafe {
                o_stream.write(&data[bytes_written..])
            };
            bytes_written += current_bytes.unwrap();
        }
    }
    recv_buf
}

pub fn send_u64_messages(ctx: &Context, data: &Vec<Wrapping<u64>>) -> Vec<Wrapping<u64>> {
    let mut batches: usize = 0;
    let mut data_len = data.len();
    let mut result: Vec<Wrapping<u64>> = Vec::new();
    let mut current_batch = 0;
    let mut push_buf = Xbuffer { u64_buf: [0u64; U64S_PER_TX] };
    batches = (data_len as f64 / U64S_PER_TX as f64).ceil() as usize;
    while current_batch < batches {
        for i in 0..U64S_PER_TX {
            unsafe {
                if current_batch * U64S_PER_TX + i < data_len {
                    push_buf.u64_buf[i] = data[current_batch * U64S_PER_TX + i].0;
                } else {
                    break;
                }
            }
        }
        unsafe {
            let buf_vec = push_buf.u8_buf.to_vec();
            let mut part_result = send_batch_message(ctx, &buf_vec);
            for item in part_result.u64_buf.to_vec() {
                result.push(Wrapping(item));
            }
        }

        current_batch += 1;
    }
    result[0..data_len].to_vec()
}

pub fn send_receive_u64_matrix(matrix_sent: &Vec<Vec<Wrapping<u64>>>, ctx: &Context) -> Vec<Vec<Wrapping<u64>>> {
    let mut list_sent = Vec::new();
    let mut matrix_received = Vec::new();
    for row in matrix_sent {
        for item in row {
            list_sent.push(item.clone());
        }
    }
    let list_received = send_u64_messages(ctx, &list_sent);
    let row_len = matrix_sent[0].len();
    let matrix_len = matrix_sent.len();
    for i in 0..matrix_len {
        let mut row = Vec::new();
        for j in 0..row_len {
            row.push(list_received[i * row_len + j]);
        }
        matrix_received.push(row);
    }
    matrix_received
}

pub fn matrix_multiplication_integer(x: &Vec<Vec<Wrapping<u64>>>, y: &Vec<Vec<Wrapping<u64>>>, ctx: &Context, prime: u64, matrix_mul_shares: &(Vec<Vec<Wrapping<u64>>>, Vec<Vec<Wrapping<u64>>>, Vec<Vec<Wrapping<u64>>>)) -> Vec<Vec<Wrapping<u64>>> {
    let mut d_matrix = Vec::new();
    let mut e_matrix = Vec::new();
    let u_shares = matrix_mul_shares.0.clone();
    let v_shares = matrix_mul_shares.1.clone();
    let w_shares = matrix_mul_shares.2.clone();
    d_matrix = local_matrix_computation(x, &u_shares, prime, LOCAL_SUBTRACTION);

    e_matrix = local_matrix_computation(y, &v_shares, prime, LOCAL_SUBTRACTION);

    let mut d_matrix_received = send_receive_u64_matrix(&d_matrix, ctx);

    let mut e_matrix_received = send_receive_u64_matrix(&e_matrix, ctx);

    let mut d = local_matrix_computation(&d_matrix, &d_matrix_received, prime, LOCAL_ADDITION);
    let mut e = local_matrix_computation(&e_matrix, &e_matrix_received, prime, LOCAL_ADDITION);
    let de = local_matrix_multiplication(&d, &e, prime);
    let eu = local_matrix_multiplication(&u_shares, &e, prime);
    let dv = local_matrix_multiplication(&d, &v_shares, prime);
    let w_eu = local_matrix_computation(&w_shares, &eu, prime, LOCAL_ADDITION);
    let w_eu_dv = local_matrix_computation(&w_eu, &dv, prime, LOCAL_ADDITION);
    let result = if ctx.num.asymm == 1 { local_matrix_computation(&w_eu_dv, &de, prime, LOCAL_ADDITION) } else { w_eu_dv.clone() };

    result
}

pub fn init(cfg_file: &String) -> Result<(RFContext, Vec<Vec<Wrapping<u64>>>, Vec<Vec<Vec<Wrapping<u64>>>>), Box<dyn Error>>{
    let mut settings = config::Config::default();
    settings
        .merge(config::File::with_name(cfg_file.as_str())).unwrap()
        .merge(config::Environment::with_prefix("APP")).unwrap();

    let class_label_count: usize = settings.get_int("class_label_count")? as usize;
    let attribute_count: usize = settings.get_int("attribute_count")? as usize;
    let instance_count: usize = settings.get_int("instance_count")? as usize;
    let feature_count: usize = settings.get_int("feature_count")? as usize;
    let tree_count: usize = settings.get_int("tree_count")? as usize;
    let max_depth: usize = settings.get_int("max_depth")? as usize;
    let epsilon: f64 = settings.get_int("epsilon")? as f64;
    let original_attr_count = attribute_count;
    let bin_count = 2usize;

    let data = io::matrix_csv_to_wrapping_vec(&settings.get_str("data")?)?;
    let mut classes = io::matrix_csv_to_wrapping_vec(&settings.get_str("classes")?)?;

    classes = util::transpose(&classes)?;
    let mut dup_classes = vec![];
    for i in 0 .. tree_count {
        dup_classes.push(classes.clone());
    }

    let tc = TrainingContext {
        instance_count,
        class_label_count,
        attribute_count,
        original_attr_count,
        bin_count,
        tree_count,
        max_depth,
        epsilon,
    };
    let rf = RFContext {
        tc,
        feature_count
    };

    Ok((rf, data, dup_classes))
}