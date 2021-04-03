extern crate csv;

use std::error::Error;
use std::num::Wrapping;
use std::fs::File;
use std::fs;

pub fn single_col_csv_to_wrapping_vec(filename: &str) -> Result<Vec<Wrapping<u64>>, Box<dyn Error>> {

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

pub fn matrix_csv_to_wrapping_vec(filename: &str) -> Result<Vec<Vec<Wrapping<u64>>>, Box<dyn Error>> {

    let file = File::open(filename)?;
    let mut rdr = csv::ReaderBuilder::new()
        .has_headers(false)    
        .from_reader(file);

    let mut vec: Vec<Vec<Wrapping<u64>>> = Vec::new(); 

    for entry in rdr.records() {

        let row: Vec<String> = entry?.deserialize(None)?;
        let row = row.iter().map(|e| Wrapping(e.parse::<u64>().unwrap())).collect::<Vec<Wrapping<u64>>>();
        vec.push(row);

    }
    Ok(vec)

}

pub fn matrix_csv_to_float_vec(filename: &str) -> Result<Vec<Vec<f64>>, Box<dyn Error>> {

    let file = File::open(filename)?;
    let mut rdr = csv::ReaderBuilder::new()
        .has_headers(false)    
        .from_reader(file);

    let mut vec: Vec<Vec<f64>> = Vec::new(); 

    for entry in rdr.records() {

        let row: Vec<String> = entry?.deserialize(None)?;
        let row = row.iter().map(|e| e.parse::<f64>().unwrap()).collect::<Vec<f64>>();
        vec.push(row);

    }
    Ok(vec)

}

pub fn single_col_csv_to_u128_vec(filename: &str) -> Result<Vec<u128>, Box<dyn Error>> {

    let file = File::open(filename)?;
    let mut rdr = csv::ReaderBuilder::new()
        .has_headers(false)    
        .from_reader(file);

    let mut vec: Vec<u128> = Vec::new(); 

    for entry in rdr.records() {
        let val = u128::from_str_radix(&entry?[0][2..], 16)?;
        vec.push(val);
    }
    Ok(vec)
}

pub fn write_to_file(filename: &str, data: &str) -> Result<(), Box<dyn Error>> {
    fs::write(filename, data).expect("Problem Writing File");
    Ok(())
}