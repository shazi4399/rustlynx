use std::error::Error;
use std::num::Wrapping;
use std::fmt;
use super::constants;
use std::cmp::PartialEq;

#[derive(Debug, Clone, PartialEq)]
pub enum CompressionStatus {
    Compressed,
    Decompressed,
    Tesselated,
    NotAllocated, 
}

#[derive(Debug, Clone, PartialEq)]
pub struct Bitset {
    pub bits: Vec<u128>,
    pub status: CompressionStatus,
    pub size: usize,
    pub n_elems: usize, 
    pub bitlen: usize,
    pub pad: u128, // 0 or 1
    pub is_padded: bool,
}

impl fmt::Display for Bitset {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {

        let mut hex_string = String::from("|");
        for subset in &self.bits {
            hex_string = format!("{}{}|", hex_string, hex(*subset));
        }

        write!(f, "Bitset {{ bits: {}, status: {:?}, size: {}, n_elems: {}, bitlen: {}, pad: {}, is_padded: {} }}", 
        hex_string, self.status, self.size, self.n_elems, self.bitlen, self.pad, self.is_padded)
    }
}

impl Bitset {

    pub fn new(bits: Vec<u128>, bitlen: usize, asymm: u64) -> Bitset {

        /* bound check bitlen, asymm == 1 or 0 */
        let len = bits.len();
        Bitset {
            bits: bits,
            status: CompressionStatus::Decompressed,
            size: len,
            n_elems: len,
            bitlen: bitlen,
            pad: asymm as u128,
            is_padded: false,
        }
    }

    pub fn compress(&mut self, with_padding: bool) -> Result<(), Box<dyn Error>> {

        match self.status {
            CompressionStatus::Tesselated => {

                /* doesn't work for bitlengths > 64 */
                
                let padded_bitlen = self.bitlen + if with_padding & (self.bitlen & 1 == 1) {1} else {0};
                let n_bits = self.n_elems * padded_bitlen;
                let tesselated_bitlen = 2 * padded_bitlen;                
                let compressed_size = n_bits / 128 + if n_bits % 128 > 0 {1} else {0};
                let mut compressed_bits = vec![0u128 ; compressed_size];

                if tesselated_bitlen < 129 {
                    let bitmask = (-Wrapping(1u128)).0 >> (128 - tesselated_bitlen);
                    let step_size = 128 - tesselated_bitlen;

                    for i in 0..self.n_elems {

                        let literal_shift = (1 + i) * step_size;             
                        let i_shift = literal_shift / 128;
                        let bitshift = literal_shift % 128;
                        let mut bitset = (self.bits[i - i_shift] >> bitshift) & bitmask;
                
                        if bitshift + tesselated_bitlen > 128 {
                            bitset |= (self.bits[i-i_shift-1] << (128 - bitshift)) & bitmask;
                        }

                        let mut compact_bitset = 0u128;
                        for j in 0..padded_bitlen {
                            compact_bitset |= ((bitset >> (2*j + 1)) & 1) << j;           
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
                } else {
                    println!("not implemented for bitlength > 64 yet");
                }

                self.bits = compressed_bits;
                self.size = compressed_size;
                self.is_padded = with_padding & (self.bitlen & 1 == 1);
                self.status = CompressionStatus::Compressed;
                Ok(())
            },
            CompressionStatus::Decompressed => {
       
                let padded_bitlen = self.bitlen + if with_padding & (self.bitlen & 1 == 1) {1} else {0};
                let n_bits = self.n_elems * padded_bitlen;
                let compressed_size = n_bits / 128 + if n_bits % 128 > 0 {1} else {0};
                let mut compressed_bits = vec![0u128 ; compressed_size];

                for (i, b) in self.bits.iter().enumerate() {

                    let bitset = if with_padding & (self.bitlen & 1 == 1) {(*b << 1)|self.pad} else {*b};
                    let literal_shift = (1 + i) * (128 - padded_bitlen);
                    let i_shift = literal_shift / 128;
                    let bitshift = literal_shift % 128;
        
                    compressed_bits[i - i_shift] |= bitset << bitshift;
        
                    if bitshift + padded_bitlen > 128 {
                        compressed_bits[i - i_shift - 1] |= bitset >> (128 - bitshift);
                    }
                }

                // WHY does this fail? compressed_bits[compressed_size - 1] |= (-(Wrapping(self.pad))).0 >> (n_bits % 128); 

                self.bits = compressed_bits;
                self.size = compressed_size;
                self.is_padded = with_padding & (self.bitlen & 1 == 1);
                self.status = CompressionStatus::Compressed;
                Ok(())
            },
            _ => Ok(())
        }
    }
    
    pub fn decompress(&mut self) -> Result<(), Box<dyn Error>> {

        match self.status {
            CompressionStatus::Decompressed => return Ok(()),
            CompressionStatus::NotAllocated => return Ok(()),
            CompressionStatus::Tesselated => return Ok(()),
            CompressionStatus::Compressed => {

                let padded_bitlen = self.bitlen + if self.is_padded {1} else {0};
                let bitmask = (-Wrapping(1u128)).0 >> (128 - padded_bitlen);
                let mut decompressed_bits = vec![0u128 ; self.n_elems];

                for i in 0..self.n_elems {

                    let literal_shift = (1 + i) * (128 - padded_bitlen);
                    let i_shift = literal_shift / 128;
                    let bitshift = literal_shift % 128;        
                    
                    let mut bitset = (self.bits[i - i_shift] >> bitshift) & bitmask;
            
                    if bitshift + padded_bitlen > 128 {
                        bitset |= (self.bits[i-i_shift-1] << (128 - bitshift)) & bitmask;
                    }
            
                    bitset >>= if self.is_padded {1} else {0};
                    decompressed_bits[i] = bitset;
                }

                self.is_padded = false;
                self.bits = decompressed_bits;
                self.size = self.n_elems;
                self.status = CompressionStatus::Decompressed;
                Ok(())
            },
        }
    }  
    
    pub fn tesselate(&mut self) -> Result<(), Box<dyn Error>> {

        match self.status {
            CompressionStatus::Compressed => {
                
                let padded_bitlen = self.bitlen + if self.is_padded {1} else {0};
                let n_bits = self.n_elems * padded_bitlen; 
                let n_u128s_complete = n_bits / 128;
                let n_bits_partial = n_bits % 128;
                let bitmask = (-Wrapping(1u128)).0 >> 64;
                let mut tesselated_bits = Vec::<u128>::new();
            
                let pad = (-Wrapping(self.pad)).0 & 0x55555555555555555555555555555555;

                for i in 0..n_u128s_complete {

                    let upper = self.bits[i] >> 64;
                    let lower = self.bits[i] & bitmask;

                    let mut upper_tesselated = pad;
                    let mut lower_tesselated = pad;

                    for j in 0..64 {
                        upper_tesselated |= ((upper >> j) & 1) << (2*j + 1);
                        lower_tesselated |= ((lower >> j) & 1) << (2*j + 1);
                    }

                    tesselated_bits.push(upper_tesselated);
                    tesselated_bits.push(lower_tesselated);
                }

                if n_bits_partial > 64 {
                    let partial_pad = pad & !((1 << (128 - n_bits_partial)) -1 );
                    let upper = self.bits[self.size-1] >> 64;
                    let lower = self.bits[self.size-1] & bitmask;

                    let mut upper_tesselated = pad;
                    let mut lower_tesselated = partial_pad;                    

                    for j in 0..64 {
                        upper_tesselated |= ((upper >> j) & 1) << (2*j + 1);
                        lower_tesselated |= ((lower >> j) & 1) << (2*j + 1);
                    }

                    tesselated_bits.push(upper_tesselated);
                    tesselated_bits.push(lower_tesselated);

                } else if n_bits_partial > 0 {

                    let partial_pad = pad & !((1 << (128 - n_bits_partial)) -1 );
                    let upper = self.bits[self.size-1] >> 64;
                    let mut upper_tesselated = partial_pad;

                    for j in 0..64 {
                        upper_tesselated |= ((upper >> j) & 1) << (2*j + 1);
                    }

                    tesselated_bits.push(upper_tesselated);
                }

                let new_size = tesselated_bits.len();
                self.bits = tesselated_bits;
                self.size = new_size;
                self.status = CompressionStatus::Tesselated;

                Ok(())
            },
            CompressionStatus::Decompressed => {

                self.compress(true);
                self.tesselate();

                Ok(())
            }
            _ => Ok(()),
        }

    }
}

pub fn compress_bit_vector(vec: &Vec<u128>, len: usize, bitlen: usize, pad_odd_bitlen: bool, asymm: u64) -> Result<Vec<u128>, Box<dyn Error>> {

    if bitlen < 1 || bitlen + if pad_odd_bitlen {1} else {0} > 128 {
        return Err("invalid bitlength".into())
    }

    let asymm = asymm as u128;
    let bitlen_u128 = 8 * constants::SIZEOF_U128;     
    let bitlen = bitlen + if pad_odd_bitlen {1} else {0};
    let compressed_len = (len * bitlen) / bitlen_u128 
        + if (len * bitlen) % bitlen_u128 > 0 {1} else {0}; 
    let mut compressed_vec = vec![0u128 ; compressed_len];


    /* make # of threads: min( len / lcm(128, bitlen), max_local_threads ) */

    for i in 0..len {

        let bitset = if pad_odd_bitlen { (vec[i] << 1) | asymm } else { vec[i] };
        let literal_shift = (1 + i) * (bitlen_u128 - bitlen);
        let i_shift = literal_shift / bitlen_u128;
        let bitshift = literal_shift % bitlen_u128;

        compressed_vec[i - i_shift] |= bitset << bitshift;
        
        if bitshift + bitlen > bitlen_u128 {
            compressed_vec[i - i_shift - 1] |= bitset >> (bitlen_u128 - bitshift);
        }
    }
    
    Ok(compressed_vec)
}

pub fn decompress_bit_vector(vec: &Vec<u128>, decompressed_len: usize, original_bitlen: usize, is_padded: bool, asymm: u64) -> Result<Vec<u128>, Box<dyn Error>> {

    if original_bitlen < 1 || original_bitlen + if is_padded {1} else {0} > 128 {
        return Err("invalid bitlength".into())
    }

    let asymm = asymm as u128;
    let bitlen_u128 = 8 * constants::SIZEOF_U128; 
    let bitlen = original_bitlen + if is_padded {1} else {0};
    let mut decompressed_vec = vec![0u128 ; decompressed_len];
    let bitmask = (-Wrapping(1u128)).0 >> (bitlen_u128 - bitlen);

    for i in 0..decompressed_len {

        let literal_shift = (1 + i) * (bitlen_u128 - bitlen);
        let i_shift = literal_shift / bitlen_u128;
        let bitshift = literal_shift % bitlen_u128;        

        let mut bitset = 0u128;

        bitset |= (vec[i - i_shift] >> bitshift) & bitmask;

        if bitshift + bitlen > bitlen_u128 {
            bitset |= (vec[i - i_shift - 1] << (bitlen_u128 - bitshift)) & bitmask;
        }

        bitset >>= if is_padded {1} else {0};
        decompressed_vec[i] = bitset;
    }

    Ok(decompressed_vec)
}

pub fn remove_tesselated_padding(vec: &Vec<u128>, len: usize) -> Result<Vec<u128>, Box<dyn Error>> {

    let mut compressed_vec = vec![0u128 ; (len >> 1) + (len & 1)];

    for i in (0..len >> 1).step_by(2) {

        let mut upper = 0u128;
        let mut lower = 0u128;

        for j in 0..64 {

            upper |= ((vec[i] >> (127 - (j << 1))) & 1) << (63 - j); 
            lower |= ((vec[i + 1] >> (127 - (j << 1))) & 1) << (63 - j);

        }

        compressed_vec[i >> 1] = (upper << 64) | lower;

    }

    if len & 1 == 1 {

        let mut upper = 0u128;

        for j in 0..64 {
            upper |= ((vec[len-1] >> (128 - 1 - 2*j)) & 1) << (64 - 1 - j) 
        }

        compressed_vec[len >> 1] = upper << 64;
    }

    Ok(compressed_vec)

}

pub fn compress_from_tesselated(vec: &Vec<u128>, len: usize, bitlen: usize, pad_odd_bitlen: bool, asymm: u64) -> Result<Vec<u128>, Box<dyn Error>> {
    
    if bitlen < 1 || bitlen + if pad_odd_bitlen {1} else {0} > 128 {
        return Err("invalid bitlength".into())
    }

    let asymm = asymm as u128;
    let bitlen_u128 = 8 * constants::SIZEOF_U128;     
    let tesselated_bitlen = 2 * bitlen;
    let bitlen = bitlen + if pad_odd_bitlen {1} else {0};
    let bitmask = (-Wrapping(1u128)).0 >> (bitlen_u128 - tesselated_bitlen);

    let compressed_len = (len * bitlen) / bitlen_u128 
        + if (len * bitlen) % bitlen_u128 > 0 {1} else {0}; 

    let mut compressed_vec = vec![0u128 ; compressed_len];

    // println!("bitlen: {}, tesselated_bitlen: {}, ")

    for i in 0..len {

        let literal_shift_compressed = (1 + i) * (bitlen_u128 - bitlen);
        let i_shift_compressed = literal_shift_compressed / bitlen_u128;
        let bitshift_compressed = literal_shift_compressed % bitlen_u128;      
        
        let literal_shift = (1 + i) * (bitlen_u128 - tesselated_bitlen);
        let i_shift = literal_shift / bitlen_u128;
        let bitshift = literal_shift % bitlen_u128;        

        let mut bitset = 0u128;

        bitset |= (vec[i - i_shift] >> bitshift) & bitmask;

        // println!("vec[i-i_shift]: {:x}", &vec[i-i_shift]);

        if bitshift + bitlen > bitlen_u128 {
            bitset |= (vec[i - i_shift - 1] << (bitlen_u128 - bitshift)) & bitmask;
        }

        //println!("bitset: {:x}", bitset);

        let mut bitset_compressed = 0u128;
        for j in 0..bitlen {
            bitset_compressed |= (bitset & (1 << (2*j + 1))) >> (j+1);
        }

        //println!("bitset compressed: {:x}", bitset_compressed);

        if pad_odd_bitlen {
            bitset_compressed = (bitset_compressed << 1) | asymm;
        }

        //println!("bitset compressed w/ pad: {:x}", bitset_compressed);

        compressed_vec[i - i_shift_compressed] |= bitset_compressed << bitshift_compressed;
        
        if bitshift_compressed + bitlen > bitlen_u128 {
            compressed_vec[i - i_shift_compressed - 1] |= bitset_compressed >> (bitlen_u128 - bitshift_compressed);
        }        

    }
    
    Ok(compressed_vec)
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

/*
len: length of tesselated vec (in values represented):
bitlen: bitlength of compressed bitset: half of bitlen from pairwise mult
pad: add a 1 after compressed bitset: true if half of bitlen from pairwise mult is odd

size of chunk to collect bitset from: 2 * bitlen
-> extract chunk from vec
-> remove tesselated padding
-> if 'pad' is true, pad with a 1 on lower end
-> push into new compressed vec in same way as compress_bit_vec

equivalent to

vec = remove_tesselated_padding(vec, len)
vec = decompress(vec, original_len, original_bitlen / 2, padded=false, asymm )
vec = compress(vec, original_len, original_bitlen / 2, padded=true, asymm)

*/