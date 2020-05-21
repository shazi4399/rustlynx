extern crate csv;
use std::error::Error;
use std::num::Wrapping;
use std::fs::File;

pub fn single_col_csv_to_wrapping_vec(filename: &str) 
    -> Result<Vec<Wrapping<u64>>, Box<dyn Error>> {

    let file = File::open(filename)?;
    let mut rdr = csv::ReaderBuilder::new()
        .has_headers(false)    
        .from_reader(file);

    let mut vec: Vec<Wrapping<u64>> = Vec::new(); 

    for entry in rdr.records() {
        let val = Wrapping(entry?[0].parse::<u64>()?);
        vec.push(val);
    }
    Ok(vec)
}

