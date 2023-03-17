use super::Context;
use super::super::constants;
use super::super::util;
use itertools::izip;
use std::io::{Read, Write};
use std::net::TcpStream;
use std::thread;
use std::error::Error;
use std::num::Wrapping;
use std::cmp;
use std::sync::{Arc, RwLock};

//pub fn share(x: &Vec<Wrapping<u64>>, ctx &mut Context) -> Result<Vec<Wrapping<u64>>, Box<dyn Error>> {}

pub fn clipped_relu(x: &Vec<Wrapping<u64>>, ctx: &mut Context) -> Result<Vec<Wrapping<u64>>, Box<dyn Error>> {
    let half = Wrapping(ctx.num.asymm) * util::float_to_ring(0.5, ctx.num.precision_frac);
    let mut l_op = vec![-half; x.len()];
    l_op.append(&mut x.clone());
    let mut r_op = x.clone();
    r_op.append(&mut vec![half; x.len()]);
    let l_op = z2_to_zq(&batch_geq(&l_op, &r_op, ctx)?, ctx)?;
    let mut r_op = x.iter().map(|x| -x - half).collect::<Vec<Wrapping<u64>>>();
    r_op.append(&mut x.iter().map(|x| -x + half).collect::<Vec<Wrapping<u64>>>());

    let thresholds = multiply(&l_op, &r_op, ctx)?;
    let lt_neg_half = thresholds[..x.len()].to_vec();
    let geq_half = thresholds[x.len()..].to_vec();

    // (x + 1/2) + x_lt_neg_one_hald * (-x - 1/2) + x_geq_one_half * (-x + 1/2)
    Ok(x.iter().zip(lt_neg_half.iter().zip(geq_half.iter()))
        .map(|(x, (lt, geq))| x + half + lt + geq)
        .collect()
    )

}

/*TODO: INSECURE*/
pub fn inverse_square_root(x: &Vec<Wrapping<u64>>, ctx: &mut Context) -> Result<Vec<Wrapping<u64>>, Box<dyn Error>> {
    let x: Vec<Wrapping<u64>> = open(&x, ctx)?;
    let x: Vec<f64> = x.iter().map(|x| util::ring_to_float(*x, ctx.num.precision_frac)).collect();
    let x: Vec<f64> = x.iter().map(|x| 1f64 / f64::sqrt(*x)).collect();
    let x: Vec<Wrapping<u64>> = x.iter().map(|x| util::float_to_ring(*x, ctx.num.precision_frac)).collect();

    if ctx.num.asymm == 1 {
        return Ok(vec![Wrapping(0u64) ; x.len()]);
    }

    Ok(x)
}



pub fn normalize(x: &Vec<Vec<Wrapping<u64>>>, ctx: &mut Context) -> Result<Vec<Vec<Wrapping<u64>>>, Box<dyn Error>> {

    let x_sum_of_squares: Vec<Wrapping<u64>> = x.iter().map(|c| multiply(&c, &c, ctx).unwrap().iter().sum()).collect();
    let x_inverse_square_roots: Vec<Wrapping<u64>> = inverse_square_root(&x_sum_of_squares, ctx)?;
    let x: Vec<Vec<Wrapping<u64>>> = x.iter()
        .zip(x_inverse_square_roots.iter())
        .map(|(c, s)| multiply(&c, &vec![*s ; c.len()], ctx).unwrap()).collect();

    Ok(x)
}

/*
    Input: two vectors x, y of n secret shares in Z_2^64
    Output: one vector z_i = {x_i * y_i} of n secret shares in Z_2^64
    CR: 2n Beaver Triples in Z_2^64
    Threads: local: ctx.sys.threads.online, online: 2*ctx.sys.threads.online
*/
pub fn multiply(x: &Vec<Wrapping<u64>>, y: &Vec<Wrapping<u64>>, ctx: &mut Context)
                -> Result<Vec<Wrapping<u64>>, Box<dyn Error>> {

    assert!(x.len() == y.len());
    let len = x.len();
    let asymm = Wrapping(ctx.num.asymm);

    let (u, v, w) = if ctx.num.asymm == 0 {
        constants::TEST_CR_WRAPPING_U64_0
    } else {
        constants::TEST_CR_WRAPPING_U64_1
    };
    let triples = vec![ (Wrapping(u), Wrapping(v), Wrapping(w)) ; len];

    let mut t_handles: Vec<thread::JoinHandle<Vec<Wrapping<u64>>>> = Vec::new();
    for i in 0..ctx.sys.threads.online {

        let lb = cmp::min((i * len) / ctx.sys.threads.online, (Wrapping(len) - Wrapping(1)).0);
        let ub = cmp::min(((i+1) * len) / ctx.sys.threads.online, len);
        let x_sub = x[lb..ub].to_vec();
        let y_sub = y[lb..ub].to_vec();
        let triple_sub = triples[lb..ub].to_vec();
        let len = ub - lb;

        let istream = ctx.net.external.tcp.as_ref().unwrap()[i].istream.try_clone()
            .expect("rustlynx::computing_party::protocol::multiply: failed cloning tcp istream");
        let ostream = ctx.net.external.tcp.as_ref().unwrap()[i].ostream.try_clone()
            .expect("rustlynx::computing_party::protocol::multiply: failed cloning tcp ostream");

        let t_handle = thread::spawn(move || {

            let mut d_share: Vec<Wrapping<u64>> =
                x_sub.iter().zip(&triple_sub).map(|(&x, (u, _v, _w))| x - u ).collect();
            let mut e_share: Vec<Wrapping<u64>> =
                y_sub.iter().zip(&triple_sub).map(|(&y, (_u, v, _w))| y - v ).collect();

            d_share.append(&mut e_share);
            let de_shares = d_share;

            let de = open_single_thread(&de_shares, istream, ostream).unwrap();

            triple_sub.iter().zip(de[..len].to_vec().iter().zip(&de[len..].to_vec()))
                .map(|((u, v, w), (d, e))| w + d*v + u*e + asymm*d*e)
                .collect()
        });

        t_handles.push(t_handle);
    }

    let mut subvecs: Vec<Vec<Wrapping<u64>>> = t_handles.into_iter().map(|t| t.join().unwrap()).collect();
    let mut result: Vec<Wrapping<u64>> = Vec::new();

    for i in 0..ctx.sys.threads.online {
        result.append(&mut subvecs[i]);
    }

    result.shrink_to_fit();

    Ok(result)
}

/*
    Input..........: two length n vectors of secret shares in Z_2^64
    Output.........: entry-wise product of inputs as secret shares in Z_2^64
    CR Requirements: n rank-0 beaver triples in Z_2^64
    Data Transfer..: 2n * sizeof(u64) bytes
    Comm Complexity: 1
    Threads........: local: 1, online: 2
    Ref............:
*/
#[allow(dead_code)]
fn multip3ly_single_thread(x: &Vec<Wrapping<u64>>, y: &Vec<Wrapping<u64>>, ctx: &mut Context)
                           -> Result<Vec<Wrapping<u64>>, Box<dyn Error>> {

    let len = x.len();
    let asymm = Wrapping(ctx.num.asymm);

    let (u, v, w) = if ctx.num.asymm == 0 {
        constants::TEST_CR_WRAPPING_U64_0
    } else {
        constants::TEST_CR_WRAPPING_U64_1
    };
    let triples = vec![ (Wrapping(u), Wrapping(v), Wrapping(w)) ; len];

    let mut d_share: Vec<Wrapping<u64>> = x.iter().zip(&triples).map(|(&x, (u, _v, _w))| x - u ).collect();
    let mut e_share: Vec<Wrapping<u64>> = y.iter().zip(&triples).map(|(&y, (_u, v, _w))| y - v ).collect();
    d_share.append(&mut e_share);
    let de_shares = d_share;

    let de = open(&de_shares, ctx)?;

    Ok(triples.iter().zip(de[..len].to_vec().iter().zip(&de[len..].to_vec()))
        .map(|((u, v, w), (d, e))| w + d*v + u*e + asymm*d*e)
        .collect())
}

/*
    Input: vector of n secret shares in Z_2^64
    Output: vector of n revealed secrets
    CR: None
    Threads: local: 1, online: 2 * ctx.sys.threads.online
*/
pub fn open(vec: &Vec<Wrapping<u64>>, ctx: &mut Context)
            -> Result<Vec<Wrapping<u64>>, Box<dyn Error>> {
    //println!("open");
    let len = vec.len();
    let mut t_handles: Vec<thread::JoinHandle<Vec<Wrapping<u64>>>> = Vec::new();

    for i in 0..ctx.sys.threads.online {
        //println!("thread {} in open", i);
        let lb = cmp::min((i * len) / ctx.sys.threads.online, (Wrapping(len) - Wrapping(1)).0);
        let ub = cmp::min(((i+1) * len) / ctx.sys.threads.online, len);
        let subvec = vec[lb..ub].to_vec();

        let mut istream = ctx.net.external.tcp.as_ref().unwrap()[i].istream.try_clone()
            .expect("rustlynx::computing_party::protocol::open: failed cloning tcp istream");
        let mut ostream = ctx.net.external.tcp.as_ref().unwrap()[i].ostream.try_clone()
            .expect("rustlynx::computing_party::protocol::open: failed cloning tcp ostream");

        let t_handle = thread::spawn(move || {

            let tx_buf: &[u8] = unsafe { subvec.align_to().1 };
            let msg_len = tx_buf.len();

            let rx_handle = thread::spawn(move || {

                let mut rx_buf = vec![0u8 ; msg_len];

                let mut bytes_read = 0;
                while bytes_read < msg_len {

                    bytes_read += match istream.read(&mut rx_buf[bytes_read..]) {
                        Ok(size) => size,
                        Err(_) => {println!("rustlynx::computing_party::protocol::open: std::IO:ErrorKind::Interrupted -- retrying"); 10},
                    };
                }
                rx_buf
            });

            let mut bytes_written = 0;
            while bytes_written < msg_len {

                bytes_written += match ostream.write(&tx_buf[bytes_written..]) {
                    Ok(size) => size,
                    Err(_) => {println!("rustlynx::computing_party::protocol::open: std::IO:ErrorKind::Interrupted -- retrying"); 10},
                };
            }

            //println!("performing operation for thread {}", i);
            let other: Vec<Wrapping<u64>> = unsafe { rx_handle.join().unwrap().align_to().1.to_vec() };
            //println!("operation for thread {} complete", i);

            subvec.iter().zip(&other).map(|(&x, &y)| x + y).collect()
        });

        //println!("pushing onto open");
        t_handles.push(t_handle);
    }

    //println!("joining in open");
    let mut subvecs: Vec<Vec<Wrapping<u64>>> = t_handles.into_iter().map(|t| t.join().unwrap()).collect();
    //println!("joined in open");
    let mut result: Vec<Wrapping<u64>> = Vec::new();

    for i in 0..ctx.sys.threads.online {
        result.append(&mut subvecs[i]);
    }

    result.shrink_to_fit();

    Ok(result)
}

/*
    Input: vector of n secret shares in Z_2^64
    Output: vector of n revealed secrets
    CR: None
    Threads: local: 1, online: 2
*/
fn open_single_thread(vec: &Vec<Wrapping<u64>>, mut istream: TcpStream, mut ostream: TcpStream)
                      -> Result<Vec<Wrapping<u64>>, Box<dyn Error>> {

    let tx_buf: &[u8] = unsafe { vec.align_to().1 };
    let msg_len = tx_buf.len();

    let rx_handle = thread::spawn(move || {

        let mut rx_buf = vec![0u8 ; msg_len];

        let mut bytes_read = 0;
        while bytes_read < msg_len {

            bytes_read += match istream.read(&mut rx_buf[bytes_read..]) {
                Ok(size) => size,
                Err(_) => {println!("rustlynx::computing_party::protocol::open: std::IO:ErrorKind::Interrupted -- retrying"); 10},
            };
        }
        rx_buf
    });

    let mut bytes_written = 0;
    while bytes_written < msg_len {

        bytes_written += match ostream.write(&tx_buf[bytes_written..]) {
            Ok(size) => size,
            Err(_) => {println!("rustlynx::computing_party::protocol::open: std::IO:ErrorKind::Interrupted -- retrying"); 10},
        };
    }

    let other: Vec<Wrapping<u64>> = unsafe { rx_handle.join().unwrap().align_to().1.to_vec() };

    Ok(vec.iter().zip(&other).map(|(&x, &y)| x + y).collect())
}

/*
    Input: vector of ceil(n / 128) u128's representing n secret shared in Z_2
    Output: vecttor of ceil(n / 128) u128's representing n revealed secret bits
    CR: None

*/
fn open_z2_single_thread(vec: &Vec<u128>, mut istream: TcpStream, mut ostream: TcpStream) -> Result<Vec<u128>, Box<dyn Error>> {

    let tx_buf: &[u8] = unsafe { vec.align_to().1 };
    let msg_len = tx_buf.len();

    let rx_handle = thread::spawn(move || {

        let mut rx_buf = vec![0u8 ; msg_len];
        let mut bytes_read = 0;
        while bytes_read < msg_len {

            bytes_read += match istream.read(&mut rx_buf[bytes_read..]) {
                Ok(size) => size,
                Err(_) => {println!("rustlynx::computing_party::protocol::open: std::IO:ErrorKind::Interrupted -- retrying"); 10},
            };
        }
        rx_buf
    });

    let mut bytes_written = 0;
    while bytes_written < msg_len {

        bytes_written += match ostream.write(&tx_buf[bytes_written..]) {
            Ok(size) => size,
            Err(_) => {println!("rustlynx::computing_party::protocol::open: std::IO:ErrorKind::Interrupted -- retrying"); 10},
        };
    }

    let other: Vec<u128> = unsafe { rx_handle.join().unwrap().align_to().1.to_vec() };

    Ok(vec.iter().zip(&other).map(|(&x, &y)| x ^ y).collect())
}

pub fn open_z2(vec: &Vec<u128>, ctx: &mut Context)
               -> Result<Vec<u128>, Box<dyn Error>> {

    let len = vec.len();
    let mut t_handles: Vec<thread::JoinHandle<Vec<u128>>> = Vec::new();

    for i in 0..ctx.sys.threads.online {

        let lb = cmp::min((i * len) / ctx.sys.threads.online, (Wrapping(len) - Wrapping(1)).0);
        let ub = cmp::min(((i+1) * len) / ctx.sys.threads.online, len);
        let subvec = vec[lb..ub].to_vec();

        let mut istream = ctx.net.external.tcp.as_ref().unwrap()[i].istream.try_clone()
            .expect("rustlynx::computing_party::protocol::open: failed cloning tcp istream");
        let mut ostream = ctx.net.external.tcp.as_ref().unwrap()[i].ostream.try_clone()
            .expect("rustlynx::computing_party::protocol::open: failed cloning tcp ostream");

        let t_handle = thread::spawn(move || {

            let tx_buf: &[u8] = unsafe { subvec.align_to().1 };
            let msg_len = tx_buf.len();

            let rx_handle = thread::spawn(move || {

                let mut rx_buf = vec![0u8 ; msg_len];

                let mut bytes_read = 0;
                while bytes_read < msg_len {

                    bytes_read += match istream.read(&mut rx_buf[bytes_read..]) {
                        Ok(size) => size,
                        Err(_) => {println!("rustlynx::computing_party::protocol::open: std::IO:ErrorKind::Interrupted -- retrying"); 10},
                    };
                }
                rx_buf
            });

            let mut bytes_written = 0;
            while bytes_written < msg_len {

                bytes_written += match ostream.write(&tx_buf[bytes_written..]) {
                    Ok(size) => size,
                    Err(_) => {println!("rustlynx::computing_party::protocol::open: std::IO:ErrorKind::Interrupted -- retrying"); 10},
                };
            }

            let other: Vec<u128> = unsafe { rx_handle.join().unwrap().align_to().1.to_vec() };

            subvec.iter().zip(&other).map(|(&x, &y)| x ^ y).collect()
        });

        t_handles.push(t_handle);
    }

    let mut subvecs: Vec<Vec<u128>> = t_handles.into_iter().map(|t| t.join().unwrap()).collect();
    let mut result: Vec<u128> = Vec::new();

    for i in 0..ctx.sys.threads.online {
        result.append(&mut subvecs[i]);
    }

    Ok(result)
}

pub fn multiply_z2(x: &Vec<u128>, y: &Vec<u128>, ctx: &mut Context)
                   -> Result<Vec<u128>, Box<dyn Error>> {

    let len = x.len();
    let asymm = (- Wrapping(ctx.num.asymm as u128)).0;

    let (u, v, w) = if ctx.num.asymm == 0 {
        constants::TEST_CR_XOR_U128_0
    } else {
        constants::TEST_CR_XOR_U128_1
    };
    let triples = vec![ (u, v, w) ; len];

    let mut t_handles: Vec<thread::JoinHandle<Vec<u128>>> = Vec::new();
    for i in 0..ctx.sys.threads.online {

        let lb = cmp::min((i * len) / ctx.sys.threads.online, (Wrapping(len) - Wrapping(1)).0);
        let ub = cmp::min(((i+1) * len) / ctx.sys.threads.online, len);
        let x_sub = x[lb..ub].to_vec();
        let y_sub = y[lb..ub].to_vec();
        let triple_sub = triples[lb..ub].to_vec();
        let len = ub - lb;

        let istream = ctx.net.external.tcp.as_ref().unwrap()[i].istream.try_clone()
            .expect("rustlynx::computing_party::protocol::multiply: failed cloning tcp istream");
        let ostream = ctx.net.external.tcp.as_ref().unwrap()[i].ostream.try_clone()
            .expect("rustlynx::computing_party::protocol::multiply: failed cloning tcp ostream");

        let t_handle = thread::spawn(move || {

            let mut d_share: Vec<u128> =
                x_sub.iter().zip(&triple_sub).map(|(&x, (u, _v, _w))| x ^ u ).collect();
            let mut e_share: Vec<u128> =
                y_sub.iter().zip(&triple_sub).map(|(&y, (_u, v, _w))| y ^ v ).collect();

            d_share.append(&mut e_share);
            let de_shares = d_share;

            let de = open_z2_single_thread(&de_shares, istream, ostream).unwrap();

            triple_sub.iter().zip(de[..len].to_vec().iter().zip(&de[len..].to_vec()))
                .map(|((u, v, w), (d, e))| w ^ d & v ^ u & e ^ asymm & d & e)
                .collect()
        });

        t_handles.push(t_handle);
    }

    let mut subvecs: Vec<Vec<u128>> = t_handles.into_iter().map(|t| t.join().unwrap()).collect();
    let mut result: Vec<u128> = Vec::new();

    for i in 0..ctx.sys.threads.online {
        result.append(&mut subvecs[i]);
    }

    Ok(result)
}

// TODO: remove unused n_elems and bitlen parameters
pub fn pairwise_mult_z2(bitset: &Vec<u128>, _n_elems: usize, _bitlen: usize, ctx: &mut Context) -> Result<Vec<u128>, Box<dyn Error>> {

    let len = bitset.len();
    let bitmask = 0xaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaau128;
    let asymm = (- Wrapping(ctx.num.asymm as u128)).0;

    /* note change w in constants to shifted down by 1*/
    let (uv, w) = if ctx.num.asymm == 0 {
        constants::TEST_CR_XOR_U128_PAIRWISE_0
    } else {
        constants::TEST_CR_XOR_U128_PAIRWISE_1
    };
    let triples = vec![(uv, w) ; len];

    let mut t_handles: Vec<thread::JoinHandle<Vec<u128>>> = Vec::new();
    for i in 0..ctx.sys.threads.online {

        let lb = cmp::min((i * len) / ctx.sys.threads.online, (Wrapping(len) - Wrapping(1)).0);
        let ub = cmp::min(((i+1) * len) / ctx.sys.threads.online, len);
        let vec_sub = bitset[lb..ub].to_vec();
        let triple_sub = triples[lb..ub].to_vec();

        let istream = ctx.net.external.tcp.as_ref().unwrap()[i].istream.try_clone()
            .expect("rustlynx::computing_party::protocol::pairswise_mult_z2: failed cloning tcp istream");
        let ostream = ctx.net.external.tcp.as_ref().unwrap()[i].ostream.try_clone()
            .expect("rustlynx::computing_party::protocol::pairswise_mult_z2: failed cloning tcp ostream");

        let t_handle = thread::spawn(move || {

            let de_share: Vec<u128> = vec_sub.iter().zip(&triple_sub).map(|(&xy, (uv, _w))| xy ^ uv ).collect();

            let de = open_z2_single_thread(&de_share, istream, ostream).unwrap();

            de.iter().zip(&triple_sub).map(|(&de, (uv, w))|
                (bitmask & (((w >> 1) ^ de & (uv >> 1) ^ (de >> 1) & uv ^ (de >> 1) & de & asymm) << 1)))
                .collect()

        });

        t_handles.push(t_handle);
    }

    let mut subvecs: Vec<Vec<u128>> = t_handles.into_iter().map(|t| t.join().unwrap()).collect();
    let mut result: Vec<u128> = Vec::new();

    for i in 0..ctx.sys.threads.online {
        result.append(&mut subvecs[i]);
    }

    Ok(result)

}

pub fn parallel_mult_z2(bitset: &Vec<u128>, n_elems: usize, bitlen: usize, ctx: &mut Context) -> Result<Vec<u128>, Box<dyn Error>> {

    if bitlen < 2 {
        return Ok(bitset.clone())
    }

    let mut bitlen = bitlen;
    let mut bitset = util::compress_bit_vector(&bitset, n_elems, bitlen, true, ctx.num.asymm, ctx.sys.threads.offline)?;

    while bitlen > 1 {

        bitset = pairwise_mult_z2(&bitset, n_elems, bitlen, ctx)?;

        bitlen = (bitlen + 1) >> 1;

        if bitlen == 1 {
            bitset = util::compress_from_tesselated_bit_vector(&bitset, n_elems, bitlen, false, ctx.num.asymm, ctx.sys.threads.offline)?;
        } else {
            bitset = util::compress_from_tesselated_bit_vector(&bitset, n_elems, bitlen, true, ctx.num.asymm, ctx.sys.threads.offline)?;
        }
    }

    let result = util::decompress_bit_vector(&bitset, n_elems, 1, false, ctx.num.asymm)?;

    Ok(result)
}

pub fn equality_from_z2(x: &Vec<u128>, y: &Vec<u128>, n_elems: usize, bitlen: usize, ctx: &mut Context) -> Result<Vec<u128>, Box<dyn Error>> {

    let bitmask = if ctx.num.asymm == 1 {(1u128 << bitlen) - 1} else {0u128};

    let bitset = x.iter().zip(y).map(|(xs, ys)| xs ^ ys ^ bitmask).collect::<Vec<u128>>();

    let eq_result = parallel_mult_z2(&bitset, n_elems, bitlen, ctx)?;

    Ok(eq_result)

}

/* right now only 1 bit */
pub fn z2_to_zq(vec: &Vec<u128>, ctx: &mut Context) -> Result<Vec<Wrapping<u64>>, Box<dyn Error>> {

    let s0 = if ctx.num.asymm == 0 {
        vec.iter().map(|&b| Wrapping(b as u64)).collect::<Vec<Wrapping<u64>>>()
    } else {
        vec![ Wrapping(0u64) ; vec.len() ]
    };

    let s1 = if ctx.num.asymm == 1 {
        vec.iter().map(|&b| Wrapping(b as u64)).collect::<Vec<Wrapping<u64>>>()
    } else {
        vec![ Wrapping(0u64) ; vec.len() ]
    };

    let prod = multiply(&s0, &s1, ctx)?;

    let xor = prod.iter().zip( s0.iter().zip( s1.iter() ) ).map( |(xy, (x, y))| x + y - Wrapping(2) * xy )
        .collect::<Vec<Wrapping<u64>>>();

    Ok(xor)
}


/* ToDo, make work for more than 2 inputs */
pub fn argmax(vec: &Vec<Wrapping<u64>>, ctx: &mut Context ) -> Result<Vec<u128>, Box<dyn Error>> {

    if vec.len() == 0 {
        return Ok(vec![])
    }

    let x_geq_y = geq(&vec[0], &vec[1], ctx)?;

    let agmx = vec![ x_geq_y, (ctx.num.asymm as u128) ^ x_geq_y ];

    Ok(agmx)

}


pub fn bit_extract(val: &Wrapping<u64>, bit_pos: usize, ctx: &mut Context) -> Result<u128, Box<dyn Error>> {

    assert!(bit_pos < 64);

    let propogate = val.0 as u128;

    if bit_pos == 0 {
        return Ok(1 & propogate);
    }

    let mut p_layer = propogate;
    let mut g_layer = if ctx.num.asymm == 0 {
        multiply_z2(&vec![propogate], &vec![0u128], ctx)?[0]
    } else {
        multiply_z2(&vec![0u128], &vec![propogate], ctx)?[0]
    };

    let mut matrices = bit_pos;
    while matrices > 1 {

        let pairs = matrices / 2;
        let remainder = matrices % 2;

        let mut p = 0u128;
        let mut p_next = 0u128;
        let mut g = 0u128;
        let mut g_next = 0u128;

        for i in 0..pairs {
            p |= ((p_layer >> (2*i)) & 1) << i;
            g |= ((g_layer >> (2*i)) & 1) << i;
            p_next |= ((p_layer >> (2*i+1)) & 1) << i;
            g_next |= ((g_layer >> (2*i+1)) & 1) << i;
        }

        let l_ops  = util::compress_bit_vector(&vec![ p_next ; 2 ], 2, pairs as usize, false, 0, ctx.sys.threads.offline)?;
        let r_ops  = util::compress_bit_vector(&vec![ p, g ], 2, pairs as usize, false, 0, ctx.sys.threads.offline)?;
        let matmul = multiply_z2(&l_ops, &r_ops, ctx)?;
        let matmul = util::decompress_bit_vector(&matmul, 2, pairs as usize, false, 0)?;

        let mut p_layer_next = 0u128;
        let mut g_layer_next = 0u128;

        for i in 0..pairs {
            p_layer_next |= ((matmul[0] >> i) & 1) << i;
            g_layer_next |= (((g_next >> i) ^ (matmul[1] >> i)) & 1) << i;
        }

        if remainder == 1 {
            p_layer_next |= ((p_layer >> (matrices-1)) & 1) << pairs;
            g_layer_next |= ((g_layer >> (matrices-1)) & 1) << pairs;
        }

        p_layer = p_layer_next;
        g_layer = g_layer_next;
        matrices = pairs + remainder;
    }

    Ok(1 & (g_layer ^ (propogate >> bit_pos)))

}

pub fn batch_bit_extract(vec: &Vec<Wrapping<u64>>, bit_pos: usize, ctx: &mut Context) -> Result<Vec<u128>, Box<dyn Error>> {

    assert!(bit_pos < 64);
    assert!(ctx.sys.threads.online > 1);

    if bit_pos == 0 {
        return Ok(vec.iter().map(|x| 1 & x.0 as u128).collect::<Vec<u128>>())
    }

    let n_elems = vec.len();
    let bit_len = bit_pos;
    let r_shift_mask = (1u128 << 127) - 1;
    let p_mask = (1u128 << bit_pos) - 1;

    let mut p_layer = util::compress_bit_vector(
        &vec.iter().map(|x| p_mask & x.0 as u128).collect::<Vec<u128>>(),
        n_elems,
        bit_len,
        bit_len > 1,
        ctx.num.asymm,
        ctx.sys.threads.online
    )?;

    let mut g_layer = if ctx.num.asymm == 0 {
        multiply_z2(&p_layer, &vec![0u128; p_layer.len()], ctx)?
    } else {
        multiply_z2(&vec![0u128; p_layer.len()], &p_layer, ctx)?
    };

    let mut bit_len = bit_pos;
    while bit_len > 1 {

        bit_len = (bit_len + 1) >> 1;

        let mut p_ctx = ctx.clone();
        let p_layer_1 = p_layer.clone();
        let p_layer_t_handle = thread::spawn(move || {

            p_ctx.sys.threads.online = cmp::max(1, p_ctx.sys.threads.online >> 1);
            p_ctx.sys.threads.offline = cmp::max(1, p_ctx.sys.threads.offline >> 1);
            p_ctx.net.external.tcp = Some(p_ctx.net.external.tcp.unwrap()[..p_ctx.sys.threads.online].to_vec());

            util::compress_from_tesselated_bit_vector(
                &pairwise_mult_z2(&p_layer_1, 0, 0, &mut p_ctx).unwrap(),
                n_elems,
                bit_len,
                bit_len > 1,
                p_ctx.num.asymm,
                p_ctx.sys.threads.offline
            ).unwrap()
        });

        let mut g_ctx = ctx.clone();
        let g_layer_t_handle = thread::spawn(move || {

            g_ctx.sys.threads.online = cmp::max(1, g_ctx.sys.threads.online >> 1);
            g_ctx.sys.threads.offline = cmp::max(1, g_ctx.sys.threads.offline >> 1);
            g_ctx.net.external.tcp = Some(g_ctx.net.external.tcp.unwrap()[g_ctx.sys.threads.online..].to_vec());

            util::compress_from_tesselated_bit_vector(
                &multiply_z2(&p_layer, &g_layer.iter().map(|g| (g & r_shift_mask) << 1).collect(), &mut g_ctx).unwrap()
                    .iter()
                    .zip(&g_layer)
                    .map(|(pp1_g, gp1)| pp1_g ^ gp1)
                    .collect(),
                n_elems,
                bit_len,
                bit_len > 1,
                g_ctx.num.asymm,
                g_ctx.sys.threads.online
            ).unwrap()
        });

        p_layer = p_layer_t_handle.join().unwrap();
        g_layer = g_layer_t_handle.join().unwrap();
    }

    Ok(util::decompress_bit_vector(&g_layer, n_elems, 1, false, ctx.num.asymm)?
        .iter()
        .zip(vec)
        .map(|(g, p)| 1 & (g ^ (p.0 as u128 >> bit_pos)))
        .collect()
    )
}

// x >= y <----> ~MSB(x - y)
pub fn geq(x: &Wrapping<u64>, y: &Wrapping<u64>, ctx: &mut Context) -> Result<u128, Box<dyn Error>> {

    let diff = x - y;
    let bit_pos = ctx.num.precision_int + ctx.num.precision_frac + 1;
    let msb = bit_extract(&diff, bit_pos, ctx)?;
    let x_geq_y = (ctx.num.asymm as u128) ^ msb;

    Ok(x_geq_y)
}

pub fn batch_geq(x: &Vec<Wrapping<u64>>, y: &Vec<Wrapping<u64>>, ctx: &mut Context) -> Result<Vec<u128>, Box<dyn Error>> {

    Ok(
        batch_bit_extract(
            &x.iter().zip(y).map(|(xx, yy)| xx - yy).collect(),
            ctx.num.precision_int + ctx.num.precision_frac + 1,
            ctx
        )?
            .iter()
            .map(|msb| (ctx.num.asymm as u128) ^ msb)
            .collect()
    )
}

//INSECURE PLACEHOLDER
//x row-wize, y column-wise
pub fn matmul(x: &Vec<Vec<Wrapping<u64>>>, y: &Vec<Vec<Wrapping<u64>>>, ctx: &mut Context) -> Result<Vec<Vec<Wrapping<u64>>>, Box<dyn Error>> {

    let rev_x: Vec<Vec<Wrapping<u64>>> = x.iter().map(|x_row| open(x_row, ctx).unwrap()).collect();
    let rev_y: Vec<Vec<Wrapping<u64>>> = y.iter().map(|y_col| open(y_col, ctx).unwrap()).collect();

    if ctx.num.asymm == 1 {
        let mut result = vec![];
        for i in 0 .. x.len() {
            let mut row = vec![];
            for j in 0 .. y.len() {
                let entry: Wrapping<u64> = rev_x[i].iter().zip(rev_y[j].iter()).map(|(x_ij, y_ji)| x_ij * y_ji).sum();
                row.push(entry);
            }
            result.push(row);
        }
        return Ok(result);
    } else {
        let result = vec![vec![Wrapping(0u64); rev_y.len()]; rev_x.len()];
        return Ok(result);
    }
}

pub fn minmax_batch(
    x_list: &Vec<Vec<Wrapping<u64>>>,
    ctx: &mut Context,
) -> Result<(Vec<Wrapping<u64>>, Vec<Wrapping<u64>>), Box<dyn Error>> {
    let asymmetric_bit = Wrapping(ctx.num.asymm as u64);

    // number of collums to process
    let n_star = x_list.len();

    // n is the number of elements in a single collumn
    let mut n = x_list[0].len();
    let mut pairs = n / 2;

    let mut l_operands: Vec<Wrapping<u64>> = Vec::new();
    let mut r_operands: Vec<Wrapping<u64>> = Vec::new();

    for col in x_list {
        for i in 0..col.len() / 2 {
            l_operands.push(col[2 * i]);
            r_operands.push(col[2 * i + 1]);
        }
    }

    let l_geq_r = batch_geq(&l_operands, &r_operands, ctx).unwrap();
    let l_geq_r = z2_to_zq(&l_geq_r, ctx).unwrap();
    let l_lt_r: Vec<Wrapping<u64>> = l_geq_r.iter().map(|x| -x + asymmetric_bit).collect();

    let mut values = l_operands.clone();
    values.append(&mut l_operands.clone());
    values.append(&mut r_operands.clone());
    values.append(&mut r_operands.clone());

    let mut assignments = l_geq_r.clone();
    assignments.append(&mut l_lt_r.clone());
    assignments.append(&mut l_geq_r.clone());
    assignments.append(&mut l_lt_r.clone());

    let min_max_pairs = multiply(&values, &assignments, ctx).unwrap();

    let mut mins: Vec<Wrapping<u64>> = Vec::new();
    let mut maxs: Vec<Wrapping<u64>> = Vec::new();

    for i in 0..pairs * n_star {
        if i % pairs == 0 && i != 0 {
            // i is a multiple of pairs
            if n % 2 == 1 {
                // at this point, we need to push onto max/min the neglected value from the
                // vector we just got done processing
                maxs.push(x_list[i / pairs - 1][n - 1]);
                mins.push(x_list[i / pairs - 1][n - 1]);
            }
        }

        maxs.push(min_max_pairs[i] + min_max_pairs[i + 3 * pairs * n_star]);
        mins.push(min_max_pairs[i + pairs * n_star] + min_max_pairs[i + 2 * pairs * n_star]);

        if i == pairs * n_star - 1 {
            // if we are the last value
            if n % 2 == 1 {
                // push last neglected value to max/min
                maxs.push(x_list[n_star - 1][n - 1]);
                mins.push(x_list[n_star - 1][n - 1]);
            }
        }
    }

    n = (n / 2) + (n % 2);
    pairs = n / 2;

    while n > 1 {

        let mut l_operands: Vec<Wrapping<u64>> = Vec::new();
        let mut r_operands: Vec<Wrapping<u64>> = Vec::new();

        let offset = (n % 2 == 1) as usize;

        for i in 0..pairs * n_star {
            l_operands.push(mins[offset * (i / pairs) + 2 * i]);
            r_operands.push(mins[offset * (i / pairs) + 2 * i + 1]);
        }

        for i in 0..pairs * n_star {
            l_operands.push(maxs[offset * (i / pairs) + 2 * i]);
            r_operands.push(maxs[offset * (i / pairs) + 2 * i + 1]);
        }

        let l_geq_r = batch_geq(&l_operands, &r_operands, ctx).unwrap();

        let l_geq_r = z2_to_zq(&l_geq_r, ctx).unwrap();

        let mut l_lt_r: Vec<Wrapping<u64>> = l_geq_r.iter().map(|x| -x + asymmetric_bit).collect();

        let mut values = r_operands[..(pairs * n_star)].to_vec();
        values.append(&mut l_operands[(pairs * n_star)..].to_vec());
        values.append(&mut l_operands[..(pairs * n_star)].to_vec());
        values.append(&mut r_operands[(pairs * n_star)..].to_vec());

        let mut assignments = l_geq_r;
        assignments.append(&mut l_lt_r);

        let min_max_pairs = multiply(&values, &assignments, ctx).unwrap();

        let mut new_mins: Vec<Wrapping<u64>> = Vec::new();
        let mut new_maxs: Vec<Wrapping<u64>> = Vec::new();

        for i in 0..pairs * n_star {
            if i % pairs == 0 && i != 0 {
                // i is a multiple of pairs
                if n % 2 == 1 {
                    // at this point, we need to push onto max/min the neglected value from the
                    // vector we just got done processing
                    new_mins.push(mins[(n - 1) * (i / pairs) + (offset * (i - pairs) / pairs)]);
                    new_maxs.push(maxs[(n - 1) * (i / pairs) + (offset * (i - pairs) / pairs)]);
                }
            }
            new_mins.push(min_max_pairs[i] + min_max_pairs[i + 2 * pairs * n_star]);
            new_maxs.push(
                min_max_pairs[i + pairs * n_star] + min_max_pairs[i + 3 * pairs * n_star],
            );

            if i == pairs * n_star - 1 && n % 2 == 1 {
                new_mins.push(mins[mins.len() - 1]);
                new_maxs.push(maxs[maxs.len() - 1]);
            }
        }

        mins = new_mins;
        maxs = new_maxs;

        // println!("min of pairs: {:5?}", reveal(&mins, ctx, ctx.decimal_precision, true ));
        // println!("max of pairs: {:5?}", reveal(&maxs, ctx, ctx.decimal_precision, true ));

        n = (n / 2) + (n % 2);
        pairs = n / 2;
    }

    mins.shrink_to_fit();
    maxs.shrink_to_fit();

    Ok((mins, maxs))
}


pub fn batch_compare(x_list : &Vec<Wrapping<u64>>,
                     y_list : &Vec<Wrapping<u64>>,
                     ctx    : &mut Context) -> Vec<u64> {

    let diff_dc  = batch_bit_extract(
        &x_list.iter().zip(y_list.iter()).map(|(&x, &y)| (x-y)).collect(),
        (ctx.num.precision_frac + ctx.num.precision_int + 1) as usize,
        ctx
    ).unwrap();

    let result=diff_dc.iter().map(|&z| z as u64 ^ ctx.num.asymm as u64).collect();
    result
}

pub fn truncate_local(x: Wrapping<u64>,
                      decimal_precision: u32,
                      asymmetric_bit: u8) -> Wrapping<u64> {
    if asymmetric_bit == 0 {
        return -Wrapping((-x).0 >> decimal_precision as u64);
    }

    Wrapping(x.0 >> decimal_precision as u64)
}

pub fn convert_integer_to_bits(x: u64, bit_length: usize) -> Vec<u8> {
    let mut result = Vec::new();
    let binary_str = format!("{:b}", x);
    let reversed_binary_vec: Vec<char> = binary_str.chars().rev().collect();
    for item in &reversed_binary_vec {
        let item_u8: u8 = format!("{}", item).parse().unwrap();
        result.push(item_u8);
    }
    if bit_length > reversed_binary_vec.len() {
        let mut temp = vec![0u8; bit_length - reversed_binary_vec.len()];
        result.append(&mut temp);
    } else {
        result = result[0..bit_length].to_vec();
    }
    result
}

pub fn discretize_into_ohe_batch(x_list: &Vec<Vec<Wrapping<u64>>>,
                                 bin_count: usize,
                                 ctx: &mut Context) -> (Vec<Vec<Wrapping<u64>>>,Vec<Vec<Wrapping<u64>>>) {

    // let asym = ctx.num.asymm;

    let cols = x_list.len();
    let rows = x_list[0].len();
    let minmax = minmax_batch(x_list, ctx).unwrap();
    let mins = minmax.0.clone();

    let mut mins_maxes = minmax.0;
    mins_maxes.extend(minmax.1);


    // println!("mins: {:?}", open(&mins, ctx).unwrap());
    // println!("maxs: {:?}", open(&maxes, ctx).unwrap());



    // let mut ranges:Vec<Wrapping<u64>> = Vec::new(); This is the original code, does not account for neg max or min ~ David
    // for i in 0.. cols {
    //     ranges.push(maxes[i] - mins[i]);
    // }



    // There are three options. Either (1) max,min are positive, (2) max is positive, min is negative, or (3) both max and min are negative.
    // Each operation requires a different way to calculate the range, so, make comparisons to determine which setting we are in

    // let smallest_neg_value = if asym == 0 {u64::MAX} else {0}; // This assumes Lambda = 64 TODO: Make more general

    // let smallest_neg_array = vec![Wrapping(smallest_neg_value); mins_maxes.len()];
    // let pos_neg_minmaxes = z2_to_zq(&batch_geq(&mins_maxes.clone(), &smallest_neg_array, ctx).unwrap(), ctx).unwrap();

    let (selected_mins, selected_maxes) = mins_maxes.split_at(mins_maxes.len()/2);
    // let (selected_mins_cmp_res, selected_maxes_cmp_res) = pos_neg_minmaxes.split_at(pos_neg_minmaxes.len()/2);
    // let (selected_mins_cmp_res, selected_maxes_cmp_res) = (selected_mins_cmp_res.to_vec(), selected_maxes_cmp_res.to_vec());

    let selected_ranges_1: Vec<Wrapping<u64>> = selected_mins.iter().zip(selected_maxes.iter()).map(|(x, y)| y - x).collect();

    let mut ranges = vec![];

    // for (x, y, z) in izip!(&selected_ranges_1, &selected_ranges_2, &selected_ranges_3) {
    //     selected_ranges.push(x + y + z);
    // }

    for x in izip!(&selected_ranges_1) {
        ranges.push(*x);
    }

    ranges.shrink_to_fit();


    let mut height_markers: Vec<Vec<Wrapping<u64>>> = Vec::new();
    let mut height_ratio_ring_vec:Vec<Wrapping<u64>> = Vec::new();
    for i in 1.. bin_count {
        let height_ratio = (i as f64) / (bin_count as f64);
        let height_ratio_ring = Wrapping((height_ratio * 2f64.powf(ctx.num.precision_frac as f64)) as u64);
        //	println!("height_ratio: {}, height_ratio_ring: {}", height_ratio,height_ratio_ring);
        height_ratio_ring_vec.push(height_ratio_ring);
    }
    for i in 0.. cols {
        let mut height_marker_vector:Vec<Wrapping<u64>> = Vec::new();
        for j in 0.. bin_count - 1 {
            height_marker_vector.push(
                mins[i] + truncate_local(
                    height_ratio_ring_vec[j] * ranges[i],
                    ctx.num.precision_frac as u32,
                    ctx.num.asymm as u8,
                ));
        }
        height_marker_vector.shrink_to_fit();
        height_markers.push(height_marker_vector);
    }

    // println!("height_markers");
    // height_markers.iter().for_each(|x| println!("{:?}", open(&x, ctx).unwrap()));

    let mut l_operands: Vec<Wrapping<u64>> = Vec::new();
    let mut r_operands: Vec<Wrapping<u64>> = Vec::new();

    for i in 0.. cols {
        for j in 0.. rows {
            for k in 0..(bin_count - 1) {
                l_operands.push(x_list[i][j]);
                r_operands.push(height_markers[i][k]);
            }
        }
    }

    let e = batch_geq(&l_operands, &r_operands, ctx).unwrap();
    // println!("e: {:?}", open_z2(&e, ctx).unwrap());

    let mut e_mat = vec![ vec![vec![0u128 ; bin_count-1]; rows] ; cols];
    for i in 0.. cols {
        for j in 0.. rows {
            for k in 0.. bin_count - 1 {
                e_mat[i][j][k] = e[rows * (bin_count-1) * i  + (bin_count - 1) * j + k]; // indices now parition e[] perfectly, i.e., no overlap/missing values
            }
        }

    }

    // ORIGINAL, BUGGED? ISSUE IS FREQUENT ZERO'ING OUT OF e[]. THIS RESULTS IN TAKING THE MULTIPLE VALUES THE SAME TIME, AND SKIPPING OTHER VALUES
    // let mut e_mat = vec![ vec![vec![0u64 ; bin_count-1]; rows] ; cols];
    // for i in 0.. cols {
    //     for j in 0.. rows {
    //         for k in 0.. bin_count - 1 {
    //             e_mat[i][j][k] = e[(bin_count-1)*i*j + k];
    //         }
    //     }

    // }

    let mut l_operands: Vec<u128> = Vec::new();
    let mut r_operands: Vec<u128> = Vec::new();

    for i in 0.. cols { // Looks good to me ~ David
        for j in 0.. rows {
            l_operands.push(ctx.num.asymm as u128);
            r_operands.push((ctx.num.asymm as u128) ^ e_mat[i][j][0] as u128);

            for k in 0..(bin_count - 2) {
                l_operands.push( e_mat[i][j][k] as u128 );
                r_operands.push( (ctx.num.asymm as u128) ^ e_mat[i][j][k+1] as u128 );
            }

            l_operands.push(ctx.num.asymm as u128);
            r_operands.push(e_mat[i][j][bin_count - 2] as u128);
        }
    }
    // println!("\nl_operands: {:?}", open_z2(&l_operands, ctx).unwrap());
    // println!("\nl_operands: {:?}", open_z2(&l_operands, ctx).unwrap());
    // println!("\nr_operands: {:?}", open_z2(&r_operands, ctx).unwrap());
    let tempf = &multiply_z2(&l_operands, &r_operands, ctx).unwrap().iter().map(|x| x & 1).collect(); //WHY DOES THIS WORK?????? ~Sam
    // println!("\n\ntempf: {:?}", open_z2(&tempf, ctx).unwrap());

    let f = z2_to_zq(&tempf, ctx).unwrap();
    // println!("\n\nf: {:?}", open(&f, ctx).unwrap());
    // println!("\n\nmaskedf: {:?}", open(&f).collect(), ctx).unwrap());

    // let mut x_discrete_ohe: Vec<Vec<Wrapping<u64>>> = Vec::new(); // need a different form
    // for i in 0.. cols {
    //     for j in 0.. rows {
    //         let mut col:Vec<Wrapping<u64>> = Vec::new();
    //         for k in 0.. bin_count {
    //             col.push(Wrapping(f[rows * (bin_count) * i  + (bin_count) * j + k] as u64));
    //             // col.push(Wrapping(f[i*j*bin_count + k] as u64)); // same issue as above, changed indexing
    //         }
    //         x_discrete_ohe.push(col);
    //     }
    // }

    let mut x_discrete_ohe: Vec<Vec<Wrapping<u64>>> = Vec::new();
    for i in 0.. cols {
        let mut x_discrete_ohe_tmp: Vec<Vec<Wrapping<u64>>> = Vec::new();
        for j in 0.. rows {
            let mut col:Vec<Wrapping<u64>> = vec![Wrapping(0); bin_count];
            for k in 0.. bin_count {
                col[k] = f[rows * (bin_count) * i  + (bin_count) * j + k];
                // col.push(Wrapping(f[i*j*bin_count + k] as u64)); // same issue as above, changed indexing
            }
            x_discrete_ohe_tmp.push(col);
        }

        x_discrete_ohe_tmp.shrink_to_fit();

        // makes column wise data
        if i == 0 {x_discrete_ohe = x_discrete_ohe_tmp.clone()}
        else {for j in 0.. rows { x_discrete_ohe[j].append(&mut x_discrete_ohe_tmp[j])}}

    }

    x_discrete_ohe.shrink_to_fit();
    height_markers.shrink_to_fit();

    (x_discrete_ohe, height_markers)
}

/** Multiplies a vector of a vectors values in a pairwise fashion, leading to log_2 communication complexity
* dependent on the inner vector size multiplied by the outer vector size. Still needs testing.
 */
pub fn pairwise_mult_zq(x: &Vec<Vec<Wrapping<u64>>>, ctx: &mut Context) -> Result<Vec<Wrapping<u64>>, Box<dyn Error>> {

    let vectors = x.clone();

    let num_of_vecs = x.len();

    if num_of_vecs < 1 {
        return Ok(vec![]);
    }

    let num_of_vals = x[0].len();
    let pairs = num_of_vals/2;

    // nothing to multiply
    if num_of_vals <= 1 {
        let mut result = vec![];

        for vec in x {
            for val in vec {
                result.push(*val);
            }
        }
        return Ok(result);
    }

    let mut l_operands: Vec<Wrapping<u64>> = Vec::new();
    let mut r_operands: Vec<Wrapping<u64>> = Vec::new();

    for vector in vectors.clone() {
        for i in 0.. pairs {
            l_operands.push(vector[2 * i]);
            r_operands.push(vector[2 * i + 1]);
        }
    }

    let products = multiply(&l_operands, &r_operands, ctx)?;

    let mut values_to_process = vec![];

    let mut offest = 0;

    // if true, values at the end of each vector were left behind (no one to pair with), add them back in
    if num_of_vals % 2 == 1 {
        for i in 0.. pairs * num_of_vecs {

            if i % pairs == 0 {
                values_to_process.push(vectors[offest][num_of_vals - 1]);
                offest += 1;
            }

            values_to_process.push(products[i]);
        }
    } else {
        values_to_process = products.clone();
    }

    let mut num_of_vals = (num_of_vals / 2) + (num_of_vals % 2);
    let mut pairs = num_of_vals / 2;

    while num_of_vals > 1 {

        let mut l_operands: Vec<Wrapping<u64>> = Vec::new();
        let mut r_operands: Vec<Wrapping<u64>> = Vec::new();

        let odd_length = num_of_vals % 2; // 1 if true 0 if false

        let mut offest = 0;

        // only used if length is odd
        let mut unprocessed_values = vec![];

        for i in 0.. num_of_vecs * pairs {

            // if odd length, we need to skip over the last element of the logically partioned vector
            l_operands.push(values_to_process[2 * i + offest * odd_length]);
            r_operands.push(values_to_process[2 * i + 1 + offest * odd_length]);

            if odd_length == 1 && (i + 1) % pairs == 0 {

                unprocessed_values.push(values_to_process[2 * (i + 1) + offest * odd_length]);
                offest += 1;

            }

        }

        // batch multiply the values that got paired up

        let products = multiply(&l_operands, &r_operands, ctx)?;

        //println!("\n\n\n {:?} \n\n\n", open(&products, ctx));

        values_to_process = vec![];

        // if true, tack on unprocessed values to the end of the logically partitioned vectors
        // that were not processed in previous round of multiplicaiton
        if odd_length == 1 {
            let mut offest = 0;
            for i in 0.. pairs * num_of_vecs {

                if i % pairs == 0 {
                    values_to_process.push(unprocessed_values[offest]);
                    offest += 1;
                }
                values_to_process.push(products[i])
            }
        } else {
            for val in products {
                values_to_process.push(val);
            }
        }

        num_of_vals = (num_of_vals / 2) + (num_of_vals % 2);
        pairs = num_of_vals / 2;
    }

    values_to_process.shrink_to_fit();

    Ok(values_to_process)
}

//ripped from our version of XT. Appears to work for additively shared values of 1 and 0.
pub fn xor(
    x_list: &Vec<Wrapping<u64>>,
    y_list: &Vec<Wrapping<u64>>,
    ctx: &mut Context,
) -> Result<Vec<Wrapping<u64>>, Box<dyn Error>>  {
    shared_or(x_list, y_list, ctx, Wrapping(2u64))
}

pub fn or(
    x_list: &Vec<Wrapping<u64>>,
    y_list: &Vec<Wrapping<u64>>,
    ctx: &mut Context,
) -> Result<Vec<Wrapping<u64>>, Box<dyn Error>>  {
    shared_or(x_list, y_list, ctx, Wrapping(1u64))
}

fn shared_or(
    x_list: &Vec<Wrapping<u64>>,
    y_list: &Vec<Wrapping<u64>>,
    ctx: &mut Context,
    mult: Wrapping<u64>,
) -> Result<Vec<Wrapping<u64>>, Box<dyn Error>> {
    let product = multiply(x_list, y_list, ctx)?;
    let mut res = vec![Wrapping(0u64); product.len()];
    // TODO: Put in threads
    for i in 0..product.len() {
        res[i] = x_list[i] + y_list[i] - (mult * product[i]); // mod prime % Wrapping(2u64);
    }
    Ok(res)
}

pub fn batch_matmul(u: &Vec<Vec<Wrapping<u64>>>, e: &Vec<Wrapping<u64>>, b: &Vec<Vec<Vec<Wrapping<u64>>>>, ctx: &mut Context) -> Result<Vec<Vec<Vec<Wrapping<u64>>>>, Box<dyn Error>> {
    let u = u.clone();

    let asymm = Wrapping(ctx.num.asymm);
    let m = u.len();
    let n = u[0].len();
    let r = b[0][0].len();
    let k = b.len();

    let v = vec![vec![vec![Wrapping(0u64); r]; n]; k];
    let z = vec![vec![vec![Wrapping(0u64); r]; m]; k];

    let mut f: Vec<Wrapping<u64>> = b.iter().flatten().flatten().zip(v.iter().flatten().flatten()).map(|(bb, vv)| bb - vv).collect();

    let mut ef = e.clone();
    f = open(&f, ctx)?;
    ef.append(&mut f);

    let lock = Arc::new(RwLock::new( (u, v, z, ef) ));

    let mut t_handles: Vec<thread::JoinHandle<Vec<Vec<Vec<Wrapping<u64>>>>>> = Vec::new();
    for i in 0..ctx.sys.threads.offline {

        let lb = cmp::min((i * k) / ctx.sys.threads.offline, (Wrapping(k) - Wrapping(1)).0);
        let ub = cmp::min(((i+1) * k) / ctx.sys.threads.offline, k);
        let lock = Arc::clone(&lock);

        let t_handle = thread::spawn(move || {

            let data = lock.read().unwrap();

            let mut mat_subset = vec![vec![vec![Wrapping(0u64); r]; m]; ub - lb];
            for kk in lb..ub {
                for mm in 0..m {
                    for rr in 0..r {
                        mat_subset[kk - lb][mm][rr] = (0..n)
                            .fold(Wrapping(0u64), |acc, nn| acc +
                                data.2[kk][mm][rr] +
                                data.0[mm][nn] * data.3[m * n + kk * n * r + nn * r + rr] +
                                data.3[n * mm + nn] * data.1[kk][nn][rr] +
                                asymm * data.3[n * mm + nn] * data.3[m * n + kk * n * r + nn * r + rr]
                            )
                    }
                }
            }

            mat_subset
        });
        t_handles.push(t_handle);
    }

    let result = t_handles.into_iter()
        .map(|t| t.join().unwrap())
        .flatten()
        .collect::<Vec<Vec<Vec<Wrapping<u64>>>>>();

    Ok(result)
}
