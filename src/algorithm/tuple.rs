use std::fs::File;
use std::path::Path;
use serde::{Deserialize, Serialize};


#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct Tuple {
    pub id: i32,
    pub label: String,
    pub room: String,
    pub teacher: String,
}


pub fn tuples_from_csv(path: impl AsRef<Path>) -> std::io::Result<Vec<Tuple>> {
    let file = File::open(path)?;
    let mut reader = csv::Reader::from_reader(file);

    let mut tuples = Vec::new();

    for result in reader.records() {
        let record = result?;
        let tuple = Tuple {
            id: record[0].parse().unwrap(),
            label: record[1].to_string(),
            room: record[2].to_string(),
            teacher: record[3].to_string(),
        };
        tuples.push(tuple);
    }

    Ok(tuples)
}