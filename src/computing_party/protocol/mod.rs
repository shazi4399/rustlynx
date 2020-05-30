use super::Context;
use super::super::constants;
use super::super::util;
use super::super::util::Bitset;
use std::io::{Read, Write};
use std::net::TcpStream;
use std::thread;
use std::error::Error;
use std::num::Wrapping;
use std::cmp;
// use std::time::SystemTime;

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

pub fn pairwise_mult_z2(bitset: &Bitset, ctx: &mut Context) -> Result<Bitset, Box<dyn Error>> { 

    let len = bitset.size;
    let bitmask = 0xaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaau128;
    let asymm = (- Wrapping(ctx.num.asymm as u128)).0;

    let (uv, w) = if ctx.num.asymm == 0 {
        constants::TEST_CR_XOR_U128_PAIRWISE_0
    } else {
        constants::TEST_CR_XOR_U128_PAIRWISE_1
    };
	
    //let triples = vec![ (0u128, 0u128) ; len]; // let triples = vec![ (uv, w) ; len];
    let triples = if ctx.num.asymm == 0 {
        vec![ (0u128, 0x0u128) ; len ]
    } else {
        vec![ (0u128, 0u128) ; len]
    };

    let mut t_handles: Vec<thread::JoinHandle<Vec<u128>>> = Vec::new();
    for i in 0..ctx.sys.threads.online {

        let lb = cmp::min((i * len) / ctx.sys.threads.online, (Wrapping(len) - Wrapping(1)).0);
        let ub = cmp::min(((i+1) * len) / ctx.sys.threads.online, len);
        let vec_sub = bitset.bits[lb..ub].to_vec();
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
                    (bitmask & ((w ^ de & (uv >> 1) ^ (de >> 1) & uv ^ (de >> 1) & de & asymm) << 1))) 
                .collect()

        });

        t_handles.push(t_handle);
    }
    let mut subvecs: Vec<Vec<u128>> = t_handles.into_iter().map(|t| t.join().unwrap()).collect();
    let mut result: Vec<u128> = Vec::new(); 
    
    for i in 0..ctx.sys.threads.online {
        result.append(&mut subvecs[i]); 
    }
    
    let mut bitset_new = Bitset {
        bits: result,
        status: util::CompressionStatus::Tesselated,
        size: bitset.size,
        n_elems: bitset.n_elems, 
        bitlen: (bitset.bitlen >> 1),
        pad: ctx.num.asymm as u128,
        is_padded: false,
    };
    bitset_new.decompress()?;
    Ok(bitset_new)
}

pub fn parallel_mult_z2(bitset: &Bitset, ctx: &mut Context) -> Result<Bitset, Box<dyn Error>> {

    let mut bitset = bitset.clone();
    // if bitset.status != util::CompressionStatus::Compressed {
    //     bitset.compress(true);
    // }
    // println!("{:x?}", open_z2(&bitset.bits, ctx));
    while bitset.bitlen > 1 {

        if bitset.bitlen % 2 == 1 {
            for i in 0..bitset.n_elems {
                bitset.bits[i] = (bitset.bits[i] << 1) | (ctx.num.asymm as u128);
            }
            
            bitset.bitlen += 1;
        }

        bitset = pairwise_mult_z2(&bitset, ctx)?;
        
    //    println!("{:x?}", open_z2(&bitset.bits, ctx));
    }

    //bitset.decompress();

    Ok(bitset)
}

pub fn equality_from_z2(x: &Bitset, y: &Bitset, ctx: &mut Context) -> Result<Bitset, Box<dyn Error>> {

    let bitmask = (-Wrapping(ctx.num.asymm as u128)).0; 
    let mut bitset = Bitset::new( 
        x.bits[..14*x.n_elems/128].into_iter().zip(&y.bits[..14*x.n_elems/128].to_vec()).map(|(xb, yb)| bitmask ^ xb ^ yb).collect::<Vec<u128>>(),
        x.bitlen, ctx.num.asymm);
    

    let mut eq_result = parallel_mult_z2(&bitset, ctx)?;

    // Ok(eq_result)
    
    Ok(Bitset::new( vec![0u128 ; x.n_elems], 1, ctx.num.asymm ))
    /* verify */
    //let eq_result = open_z2(&bitset.bits, ctx)?;
    
    // Ok( 
    //     Bitset::new( 
    //         eq_result.iter().map(|&b| if b == bitmask {ctx.num.asymm as u128} else {0u128}).collect::<Vec<u128>>(),
    //         1,
    //         ctx.num.asymm,
    //     )
    // )
}

/* right now only 1 bit */
pub fn z2_to_zq(vec: &Bitset, ctx: &mut Context) -> Result<Vec<Wrapping<u64>>, Box<dyn Error>> {

    let s0 = if ctx.num.asymm == 0 {
        vec.bits.iter().map(|&b| Wrapping(b as u64)).collect::<Vec<Wrapping<u64>>>()
    } else {
        vec![ Wrapping(0u64) ; vec.n_elems ]
    }; 

    let s1 = if ctx.num.asymm == 1 {
        vec.bits.iter().map(|&b| Wrapping(b as u64)).collect::<Vec<Wrapping<u64>>>()
    } else {
        vec![ Wrapping(0u64) ; vec.n_elems ]
    }; 

    let prod = multiply(&s0, &s1, ctx)?;

    let xor = prod.iter().zip( s0.iter().zip( s1.iter() ) ).map( |(xy, (x, y))| x + y - Wrapping(2) * xy )
        .collect::<Vec<Wrapping<u64>>>();

    Ok(xor)
}

pub fn argmax(vec: &Vec<Wrapping<u64>>, ctx: &mut Context ) -> Result<Vec<u128>, Box<dyn Error>> {

    /**/
    multiply(&vec, &vec, ctx);
    multiply(&vec, &vec, ctx);
    multiply(&vec, &vec, ctx);


    let revealed = open(&vec, ctx)?;

    let mut argmax = vec![0u128 ; revealed.len()];

    let mut max = Wrapping(0u64);
    let mut i_max = 0;
    for (i, &elem) in revealed.iter().enumerate() {
        if elem > max {
            max = elem;
            i_max = i;
        }
    }

    argmax[i_max] = ctx.num.asymm as u128;
    
    Ok(argmax)
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