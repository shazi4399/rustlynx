use super::Context;
use super::constants;
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
    Input: two vectors x, y of n secret shares in Z_2^64
    Output: one vector z_i = {x_i * y_i} of n secret shares in Z_2^64
    CR: 2n Beaver Triples in Z_2^64
    Threads: local: 1, online: 2 
*/
pub fn multiply_single_thread(x: &Vec<Wrapping<u64>>, y: &Vec<Wrapping<u64>>, ctx: &mut Context) 
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
pub fn open_single_thread(vec: &Vec<Wrapping<u64>>, mut istream: TcpStream, mut ostream: TcpStream) 
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



// /* 
// - at the lowest level (open), there are a fixed number of threads n.
// - a protocol one level above can either
// 	- call open on the entire input set -- open splits input into n threads
// 	- call open w/ a specified port to run on a single thread
// - have threads recombine pairwise
//  */