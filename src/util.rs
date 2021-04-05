use std::error::Error;
use std::num::Wrapping;
// use std::fmt;
// use super::constants;
// use std::cmp::PartialEq;
use std::cmp;
use std::thread;

pub fn compress_bit_vector(vec: &Vec<u128>, n_elems: usize, bitlen: usize, pad_odd_bitlen: bool, asymm: u64, max_threads: usize) -> Result<Vec<u128>, Box<dyn Error>> {

    if bitlen < 1 || bitlen + if pad_odd_bitlen {1} else {0} > 128 {
        return Err("invalid bitlength".into())
    }

    let padded_bitlen = bitlen + if pad_odd_bitlen & (bitlen & 1 == 1) {1} else {0};

    let bits_per_chunk = lcm(128, padded_bitlen);
    let elems_per_chunk = bits_per_chunk / padded_bitlen;
    let n_chunks = n_elems / elems_per_chunk + if n_elems % elems_per_chunk > 0 {1} else {0};
    let n_threads = cmp::min(max_threads, n_chunks);

    let mut t_handles: Vec<thread::JoinHandle<Vec<u128>>> = Vec::new();

    for t in 0..n_threads {

        let base_lb = (t * n_elems) / n_threads;
        let segmented_lb = base_lb + if base_lb % elems_per_chunk > 0 { elems_per_chunk - (base_lb % elems_per_chunk)} else {0};
        let base_ub = ((t + 1) * n_elems) / n_threads;
        let segmented_ub = base_ub + if base_ub % elems_per_chunk > 0 { elems_per_chunk - (base_ub % elems_per_chunk)} else {0};
        let lb = cmp::min( segmented_lb, (Wrapping(n_elems) - Wrapping(1)).0);
        let ub = cmp::min( segmented_ub, n_elems);
        
        let subvec = vec[lb..ub].to_vec();
        let len = padded_bitlen * (ub - lb) / 128 + if padded_bitlen * (ub - lb) % 128 > 0 {1} else {0};

        let t_handle = thread::spawn(move || {

            let mut compressed_bits = vec![0u128 ; len];

            for (i, b) in subvec.iter().enumerate() {

                let bitset = if pad_odd_bitlen & (bitlen & 1 == 1) { (*b << 1) | (asymm as u128)} else {*b};
                let literal_shift = (1 + i) * (128 - padded_bitlen);
                let i_shift = literal_shift / 128;
                let bitshift = literal_shift % 128;
        
                compressed_bits[i - i_shift] |= bitset << bitshift;
        
                if bitshift + padded_bitlen > 128 {
                    compressed_bits[i - i_shift - 1] |= bitset >> (128 - bitshift);
                }
            }
            compressed_bits
        });

        t_handles.push(t_handle);
    }

    let mut subvecs: Vec<Vec<u128>> = t_handles.into_iter().map(|t| t.join().unwrap()).collect();
    let mut result: Vec<u128> = Vec::new(); 
    
    for i in 0..n_threads {
        result.append(&mut subvecs[i]); 
    }

    Ok(result)
}

pub fn compress_bit_vector_single_thread(vec: &Vec<u128>, n_elems: usize, bitlen: usize, pad_odd_bitlen: bool, asymm: u64) -> Result<Vec<u128>, Box<dyn Error>> {

    if bitlen < 1 || bitlen + if pad_odd_bitlen {1} else {0} > 128 {
        return Err("invalid bitlength".into())
    }

    let padded_bitlen = bitlen + if pad_odd_bitlen & (bitlen & 1 == 1) {1} else {0};
    let n_bits = n_elems * padded_bitlen;
    let compressed_size = n_bits / 128 + if n_bits % 128 > 0 {1} else {0};
    let mut compressed_bits = vec![0u128 ; compressed_size];

    for (i, b) in vec.iter().enumerate() {

        let bitset = if pad_odd_bitlen & (bitlen & 1 == 1) { (*b << 1) | (asymm as u128)} else {*b};
        let literal_shift = (1 + i) * (128 - padded_bitlen);
        let i_shift = literal_shift / 128;
        let bitshift = literal_shift % 128;

        compressed_bits[i - i_shift] |= bitset << bitshift;

        if bitshift + padded_bitlen > 128 {
            compressed_bits[i - i_shift - 1] |= bitset >> (128 - bitshift);
        }
    }

   Ok(compressed_bits)
}

// TODO: remove unused asymm parameter, add threading
pub fn decompress_bit_vector(vec: &Vec<u128>, n_elems: usize, original_bitlen: usize, pad_odd_bitlen: bool, _asymm: u64) -> Result<Vec<u128>, Box<dyn Error>> {

    if original_bitlen < 1 || original_bitlen + if pad_odd_bitlen {1} else {0} > 128 {
        return Err("invalid bitlength".into())
    }

    let padded_bitlen = original_bitlen + if pad_odd_bitlen && (original_bitlen & 1 == 1) {1} else {0};
    let bitmask = (-Wrapping(1u128)).0.checked_shr((128 - padded_bitlen) as u32).unwrap_or(0u128);
    let mut decompressed_bits = vec![0u128 ; n_elems];

    for i in 0..n_elems {

        let literal_shift = (1 + i) * (128 - padded_bitlen);
        let i_shift = literal_shift / 128;
        let bitshift = literal_shift % 128;        
        
        let mut bitset = (vec[i - i_shift] >> bitshift) & bitmask;

        if bitshift + padded_bitlen > 128 {
            bitset |= (vec[i-i_shift-1] << (128 - bitshift)) & bitmask;
        }

        bitset >>= if pad_odd_bitlen & (original_bitlen & 1 == 1) {1} else {0};
        decompressed_bits[i] = bitset;
    }

    Ok(decompressed_bits)
}

pub fn compress_from_tesselated_bit_vector(vec: &Vec<u128>, n_elems: usize, original_bitlen: usize, pad_odd_bitlen: bool, asymm: u64, max_threads: usize) -> Result<Vec<u128>, Box<dyn Error>> {

    if original_bitlen < 1 || original_bitlen + if pad_odd_bitlen && (original_bitlen & 1 == 1) {1} else {0} > 64 {
        return Err("invalid bitlength".into())
    }

    let padded_bitlen = original_bitlen + if pad_odd_bitlen && (original_bitlen & 1 == 1) {1} else {0};
    let tesselated_bitlen = 2 * original_bitlen;                

    let bitmask = (-Wrapping(1u128)).0.checked_shr((128 - tesselated_bitlen) as u32).unwrap_or(0u128);
    let step_size = 128 - tesselated_bitlen;

   /* need an input size that lines up with 128 that will yield an output size that lines up with 128 */
    let mut bits_per_chunk = lcm(128, lcm(padded_bitlen, tesselated_bitlen));
    if (bits_per_chunk / 128) % 2 == 1 {
        bits_per_chunk *= 2;
    }
    let elems_per_chunk = bits_per_chunk / tesselated_bitlen;
    let chunk_size = bits_per_chunk / 128;
    let output_bits_per_chunk = elems_per_chunk * padded_bitlen;
    let output_chunk_size = output_bits_per_chunk as f64 / 128 as f64;

    let n_chunks = n_elems / elems_per_chunk + if n_elems % elems_per_chunk > 0 {1} else {0};

    let n_threads = cmp::min(max_threads, n_chunks);

    let mut t_handles: Vec<thread::JoinHandle<Vec<u128>>> = Vec::new();
    let mut remaining_elems = n_elems;

    for t in 0..n_threads {

        let base_lb = (t * vec.len()) / n_threads;
        let segmented_lb = base_lb + if base_lb % chunk_size > 0 { chunk_size - (base_lb % chunk_size)} else {0};
        let base_ub = ((t + 1) * vec.len()) / n_threads;
        let segmented_ub = base_ub + if base_ub % chunk_size > 0 { chunk_size - (base_ub % chunk_size)} else {0};
        let lb = cmp::min( segmented_lb, (Wrapping(vec.len()) - Wrapping(1)).0);
        let ub = cmp::min( segmented_ub, vec.len());

        let subvec = vec[lb..ub].to_vec();
        let len = output_chunk_size as usize * (ub - lb) / chunk_size + if (output_chunk_size as usize * (ub - lb)) % chunk_size > 0 {1} else {0};
        
        let sub_n_elems;
        
        if t == n_threads - 1 {
          sub_n_elems = remaining_elems;  
        } else {
            sub_n_elems = len * 128 / padded_bitlen;
            remaining_elems -= sub_n_elems;
        }

        let elems = sub_n_elems; 
        let t_handle = thread::spawn(move || {

            let mut compressed_bits = vec![0u128 ; len];

            for i in 0..elems {

                let literal_shift = (1 + i) * step_size;             
                let i_shift = literal_shift / 128;
                let bitshift = literal_shift % 128;
                let mut bitset = (subvec[i - i_shift] >> bitshift) & bitmask;

                if bitshift + tesselated_bitlen > 128 {
                    bitset |= (subvec[i-i_shift-1] << (128 - bitshift)) & bitmask;
                }

                let mut compact_bitset = 0u128;
                for j in 0..original_bitlen {
                    compact_bitset |= ((bitset >> (2*j + 1)) & 1) << j;           
                }

                if original_bitlen & 1 == 1 && pad_odd_bitlen {
                    compact_bitset = (compact_bitset << 1) | (asymm as u128);
                }

                let bitset = compact_bitset;
                let literal_shift = (1 + i) * (128 - padded_bitlen);
                let i_shift = literal_shift / 128;
                let bitshift = literal_shift % 128;

                compressed_bits[i - i_shift] |= bitset << bitshift;

                if bitshift + padded_bitlen > 128 {
                    compressed_bits[i - i_shift - 1] |= bitset >> (128 - bitshift);
                }         
            }

            compressed_bits
        });

        t_handles.push(t_handle);
    }

    let mut subvecs: Vec<Vec<u128>> = t_handles.into_iter().map(|t| t.join().unwrap()).collect();
    let mut result: Vec<u128> = Vec::new(); 
    
    for i in 0..n_threads {
        result.append(&mut subvecs[i]); 
    }

    Ok(result)
}


pub fn compress_from_tesselated_bit_vector_single_thread(vec: &Vec<u128>, n_elems: usize, original_bitlen: usize, pad_odd_bitlen: bool, asymm: u64) -> Result<Vec<u128>, Box<dyn Error>> {
    
    if original_bitlen < 1 || original_bitlen + if pad_odd_bitlen && (original_bitlen & 1 == 1) {1} else {0} > 64 {
        return Err("invalid bitlength".into())
    }

    let padded_bitlen = original_bitlen + if pad_odd_bitlen && (original_bitlen & 1 == 1) {1} else {0};
    let n_bits = n_elems * padded_bitlen;
    let tesselated_bitlen = 2 * original_bitlen;                
    let compressed_size = n_bits / 128 + if n_bits % 128 > 0 {1} else {0};
    let mut compressed_bits = vec![0u128 ; compressed_size];

    let bitmask = (-Wrapping(1u128)).0.checked_shr((128 - tesselated_bitlen) as u32).unwrap_or(0u128);
    let step_size = 128 - tesselated_bitlen;

    for i in 0..n_elems {

        let literal_shift = (1 + i) * step_size;             
        let i_shift = literal_shift / 128;
        let bitshift = literal_shift % 128;
        let mut bitset = (vec[i - i_shift] >> bitshift) & bitmask;

        if bitshift + tesselated_bitlen > 128 {
            bitset |= (vec[i-i_shift-1] << (128 - bitshift)) & bitmask;
        }

        let mut compact_bitset = 0u128;
        for j in 0..original_bitlen {
            compact_bitset |= ((bitset >> (2*j + 1)) & 1) << j;           
        }

        if original_bitlen & 1 == 1 && pad_odd_bitlen {
            compact_bitset = (compact_bitset << 1) | (asymm as u128);
        }

        let bitset = compact_bitset;
        let literal_shift = (1 + i) * (128 - padded_bitlen);
        let i_shift = literal_shift / 128;
        let bitshift = literal_shift % 128;

        compressed_bits[i - i_shift] |= bitset << bitshift;

        if bitshift + padded_bitlen > 128 {
            compressed_bits[i - i_shift - 1] |= bitset >> (128 - bitshift);
        }         
    }


    Ok(compressed_bits)

}


pub fn gcd(a: usize, b: usize) -> usize {

    if b == 0 {a} else {gcd(b, a % b)}
}

pub fn lcm(a: usize, b: usize) -> usize {

    a * b / gcd(a, b)

}

pub fn hex(val: u128) -> String {

    let mut val = val;
    let mut hex_string = String::new();
    for _i in 0..32 {
        hex_string = format!("{:x}{}", val & 0xf, hex_string);
        val >>= 4;
    }
    hex_string
}


pub fn truncate(x: Wrapping<u64>, decimal_precision : usize, asymm: u64) -> Wrapping<u64> {

    if asymm == 0 {

        return  - Wrapping( ( - x).0 >> decimal_precision )  
    }

    Wrapping( x.0 >> decimal_precision )
}


pub fn transpose(mat: &Vec<Vec<Wrapping<u64>>>) -> Result<Vec<Vec<Wrapping<u64>>>, Box<dyn Error>> {

    let mut mat_t = vec![vec![ Wrapping(0u64) ; mat.len() ] ; mat[0].len()];

    for i in 0..mat.len() {
        for j in 0..mat[0].len() {
            mat_t[j][i] = mat[i][j] 
        }
    }

    Ok(mat_t)

}