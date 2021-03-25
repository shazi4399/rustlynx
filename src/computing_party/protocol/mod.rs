use super::Context;
use super::super::constants;
use super::super::util;
use std::io::{Read, Write};
use std::net::TcpStream;
use std::thread;
use std::error::Error;
use std::num::Wrapping;
use std::cmp;
use std::time::SystemTime;

/*
    Input: two vectors x, y of n secret shares in Z_2^64
    Output: one vector z_i = {x_i * y_i} of n secret shares in Z_2^64
    CR: 2n Beaver Triples in Z_2^64
    Threads: local: ctx.sys.threads.online, online: 2*ctx.sys.threads.online 
*/
pub fn multiply(x: &Vec<Wrapping<u64>>, y: &Vec<Wrapping<u64>>, ctx: &mut Context) 
    -> Result<Vec<Wrapping<u64>>, Box<dyn Error>> {

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
fn multiply_single_thread(x: &Vec<Wrapping<u64>>, y: &Vec<Wrapping<u64>>, ctx: &mut Context) 
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

    let len = vec.len();
    let mut t_handles: Vec<thread::JoinHandle<Vec<Wrapping<u64>>>> = Vec::new();
    
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
                        Err(_) => {println!("rustlynx::computing_party::protocol::open: std::IO:ErrorKind::Interrupted -- retrying"); 0},
                    };
                }
                rx_buf
            });

            let mut bytes_written = 0;
            while bytes_written < msg_len {
        
                bytes_written += match ostream.write(&tx_buf[bytes_written..]) {
                    Ok(size) => size,
                    Err(_) => {println!("rustlynx::computing_party::protocol::open: std::IO:ErrorKind::Interrupted -- retrying"); 0},
                };
            }
        
            let other: Vec<Wrapping<u64>> = unsafe { rx_handle.join().unwrap().align_to().1.to_vec() };

            subvec.iter().zip(&other).map(|(&x, &y)| x + y).collect()
        });

        t_handles.push(t_handle);
    }

    let mut subvecs: Vec<Vec<Wrapping<u64>>> = t_handles.into_iter().map(|t| t.join().unwrap()).collect();
    let mut result: Vec<Wrapping<u64>> = Vec::new(); 
    
    for i in 0..ctx.sys.threads.online {
        result.append(&mut subvecs[i]); 
    }

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
				Err(_) => {println!("rustlynx::computing_party::protocol::open: std::IO:ErrorKind::Interrupted -- retrying"); 0},
			};
		}
		rx_buf
	});

	let mut bytes_written = 0;
	while bytes_written < msg_len {

		bytes_written += match ostream.write(&tx_buf[bytes_written..]) {
			Ok(size) => size,
			Err(_) => {println!("rustlynx::computing_party::protocol::open: std::IO:ErrorKind::Interrupted -- retrying"); 0},
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
                Err(_) => {println!("rustlynx::computing_party::protocol::open: std::IO:ErrorKind::Interrupted -- retrying"); 0},
            };
        }
        rx_buf
    });

    let mut bytes_written = 0;
    while bytes_written < msg_len {

        bytes_written += match ostream.write(&tx_buf[bytes_written..]) {
            Ok(size) => size,
            Err(_) => {println!("rustlynx::computing_party::protocol::open: std::IO:ErrorKind::Interrupted -- retrying"); 0},
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
                        Err(_) => {println!("rustlynx::computing_party::protocol::open: std::IO:ErrorKind::Interrupted -- retrying"); 0},
                    };
                }
                rx_buf
            });

            let mut bytes_written = 0;
            while bytes_written < msg_len {
        
                bytes_written += match ostream.write(&tx_buf[bytes_written..]) {
                    Ok(size) => size,
                    Err(_) => {println!("rustlynx::computing_party::protocol::open: std::IO:ErrorKind::Interrupted -- retrying"); 0},
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

pub fn pairwise_mult_z2(bitset: &Vec<u128>, n_elems: usize, bitlen: usize, ctx: &mut Context) -> Result<Vec<u128>, Box<dyn Error>> { 

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
        let len = ub - lb;

        let istream = ctx.net.external.tcp.as_ref().unwrap()[i].istream.try_clone()
		    .expect("rustlynx::computing_party::protocol::pairswise_mult_z2: failed cloning tcp istream");
	    let ostream = ctx.net.external.tcp.as_ref().unwrap()[i].ostream.try_clone()
		    .expect("rustlynx::computing_party::protocol::pairswise_mult_z2: failed cloning tcp ostream");

        let t_handle = thread::spawn(move || {

            let mut de_share: Vec<u128> = 
                vec_sub.iter().zip(&triple_sub).map(|(&xy, (uv, _w))| xy ^ uv ).collect();

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

    let total = SystemTime::now();

    if bitlen < 2 {
        return Ok(bitset.clone())
    }

    let now = SystemTime::now();
    let mut bitlen = bitlen;
    let mut bitset = util::compress_bit_vector(&bitset, n_elems, bitlen, true, ctx.num.asymm, ctx.sys.threads.offline)?;
    let mut compress_time = now.elapsed().unwrap().as_millis();   

    while bitlen > 1 {

        bitset = pairwise_mult_z2(&bitset, n_elems, bitlen, ctx)?;
        
        bitlen = (bitlen + 1) >> 1;

        let now = SystemTime::now();
        if bitlen == 1 {
            bitset = util::compress_from_tesselated_bit_vector(&bitset, n_elems, bitlen, false, ctx.num.asymm, ctx.sys.threads.offline)?;
        } else {
            bitset = util::compress_from_tesselated_bit_vector(&bitset, n_elems, bitlen, true, ctx.num.asymm, ctx.sys.threads.offline)?;
        }
        compress_time += now.elapsed().unwrap().as_millis();

    }

    let now = SystemTime::now();
    let result = util::decompress_bit_vector(&bitset, n_elems, 1, false, ctx.num.asymm)?;
    compress_time += now.elapsed().unwrap().as_millis();

    // println!("parallel_mult: compression {:5} ms, pairwise mult {:5} ms", 
    //     compress_time, total.elapsed().unwrap().as_millis() - compress_time);

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

    assert!(0 <= bit_pos && bit_pos < 64);

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

// x >= y <----> ~MSB(x - y)
pub fn geq(x: &Wrapping<u64>, y: &Wrapping<u64>, ctx: &mut Context) -> Result<u128, Box<dyn Error>> {

    let diff = (x - y);
    let bit_pos = ctx.num.precision_int + ctx.num.precision_frac + 1;
    let msb = bit_extract(&diff, bit_pos, ctx)?;
    let x_geq_y = (ctx.num.asymm as u128) ^ msb;

    Ok(x_geq_y)
}

//INSECURE PLACEHOLDER
pub fn batch_geq(x: &Vec<Wrapping<u64>>, y: &Vec<Wrapping<u64>>, ctx: &mut Context) -> Result<Vec<u128>, Box<dyn Error>> {

    let rev_x = open(x, ctx)?;
    let rev_y = open(y, ctx)?;

    if ctx.num.asymm == 1 {
        let result = rev_x.iter().zip(rev_y.iter()).map(|(x, y)| (x >= y) as u128).collect();
        return Ok(result);
    } else {
        let result = vec![0u128; rev_x.len()];
        return Ok(result);
    }
}

//INSECURE PLACEHOLDER
//x row-wize, y column-wise
pub fn matmul(x: &Vec<Vec<Wrapping<u64>>>, y: &Vec<Vec<Wrapping<u64>>>, ctx: &mut Context) -> Result<Vec<Vec<Wrapping<u64>>>, Box<dyn Error>> {

    let rev_x_box:Result<Vec<Vec<Wrapping<u64>>>, Box<dyn Error>>  = x.iter().map(|x_row| open(x_row, ctx)).collect();
    let rev_x = rev_x_box?;
    let rev_y_box:Result<Vec<Vec<Wrapping<u64>>>, Box<dyn Error>> = y.iter().map(|y_col| open(y_col, ctx)).collect();
    let rev_y = rev_y_box?;

    if ctx.num.asymm == 1 {
        let mut result = vec![];
        for i in 0 .. x.len() {
            let mut row = vec![];
            for j in 0 .. y.len() {
                let entry: Wrapping<u64> = x[i].iter().zip(y[j].iter()).map(|(x_ij, y_ji)| x_ij * y_ji).sum();
                row.push(entry);
            }
            result.push(row);
        }
        // let result = rev_x.iter().map(|x_row| y.iter().map(|y_col| x_row.iter().zip(y_col.iter()).fold(Wrapping(0u64), |acc, (x_j, y_i)|  acc + (x_j * y_i))).collect()).collect();
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

    let l_geq_r = z2_to_zq(&l_geq_r, ctx/*, 1*/).unwrap();

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

        let l_geq_r = z2_to_zq(&l_geq_r, ctx/*, 1*/).unwrap();

        let l_lt_r: Vec<Wrapping<u64>> = l_geq_r.iter().map(|x| -x + asymmetric_bit).collect();

        let mut values = r_operands[..(pairs * n_star)].to_vec();
        values.append(&mut l_operands[(pairs * n_star)..].to_vec());
        values.append(&mut l_operands[..(pairs * n_star)].to_vec());
        values.append(&mut r_operands[(pairs * n_star)..].to_vec());

        let mut assignments = l_geq_r.clone();
        assignments.append(&mut l_lt_r.clone());

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

        n = (n / 2) + (n % 2);
        pairs = n / 2;
    }

    Ok((mins, maxs))
}


/** Multiplies a vector of a vectors values in a pairwise fashion, leading to log_2 communication complexity
 * dependent on the inner vector size multiplied by the outer vector size. Still needs testing.
 */
pub fn pairwise_mult(x: &Vec<Vec<Wrapping<u64>>>, ctx: &mut Context) -> Result<Vec<Wrapping<u64>>, Box<dyn Error>> {

    let vectors = x.clone();

    let num_of_vecs = x.len();
    let num_of_vals = x[0].len();
    let pairs = num_of_vals/2;

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

            if i % pairs == 0 && i != 0 {
                values_to_process.push(vectors[offest][num_of_vals - 1]);
                offest += 1;
                continue;
            }

            values_to_process.push(products[i - offest]);
        }
    } else {
        values_to_process = products.clone();
    }

    let mut num_of_vals = (num_of_vals / 2) + (num_of_vals % 2);
    let mut pairs = num_of_vals / 2;

    while num_of_vals > 1 {

        let mut l_operands: Vec<Wrapping<u64>> = Vec::new();
        let mut r_operands: Vec<Wrapping<u64>> = Vec::new();

        let odd_length = ((num_of_vals % 2) == 1) as usize;

        let mut offest = 0;

        // only used if length is odd
        let mut unprocessed_values = vec![];

        for i in 0.. num_of_vecs * pairs {

            // if odd length, we need to skip over the last element of the logically partioned vector
            l_operands.push(values_to_process[2 * i + offest * odd_length]);
            r_operands.push(values_to_process[2 * i + 1 + offest * odd_length]);

            if (i + 1) % pairs == 0 && i != 0 && odd_length == 1 {

                unprocessed_values.push(values_to_process[2 * (i + 1) + offest * odd_length]);
                offest += 1;

            }

        }

        // batch multiply the values that got paired up
        let products = multiply(&l_operands, &r_operands, ctx)?;

        let mut values_to_process = vec![];

        // if true, tack on unprocessed values to the end of the logically partitioned vectors 
        // that were not processed in previous round of multiplicaiton
        if odd_length == 1 {
            let mut offest = 0;
            for i in 0.. pairs * num_of_vecs {

                if i % pairs == 0 && i != 0 {
                    values_to_process.push(unprocessed_values[offest]);
                    offest += 1;
                    continue;
                }
                values_to_process.push(products[i - offest])
            }
        } else {
            let values_to_process = products.clone(); // IDE says never used, but is used in next iter
        }

        num_of_vals = (num_of_vals / 2) + (num_of_vals % 2);
        pairs = num_of_vals / 2;
    }
    Ok(values_to_process)
}



 /* 
- at the lowest level (open), there are a fixed number of threads n.
- a protocol one level above can either
	- call open on the entire input set -- open splits input into n threads
	- call open w/ a specified port to run on a single thread
- have threads recombine pairwise


    1 u128 -> 64 products
    bitlen = 14 
    xyxyxyxyxyxyxy len 128
    uvuvuvuvuvuvuv len 128 + w0w0w0w0w0w0w0 len 128
    result: w + u*e + d*v + d*e
   (1) take xyxyxy...xy ^ uvuvuv...uv = dedede...de
   (2) open dedede...de
   (3) take u*e : (dedede..de) & ((uvuvuv..uv) >> 1) & 0xAAAAAAAAAAAAA = a0a0a0...a0
   (4) take d*v : ((dedede..de) >> 1) & (uvuvuv..uv) & 0x5555555555555 = 0b0b0b...0b
   (5) take d*e : ((dedede..de) & (dedede..de) << 1) & 0xAAAAAAAAAAAAA = 
   (6) xor between all and compactify 
    /* option to pad odds */
    bit0 = x
    bit1 = y
    triple = ()
    d = x - u;
    e = y - v;
    open(d, e)
*/

