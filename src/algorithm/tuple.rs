use std::fs::File;
use std::path::Path;
use serde::Deserialize;

#[derive(Debug, Deserialize)]
struct Tuple {
    id: i32,
    label: String,
    room: String,
    teacher: String,
}

pub fn read_csv(path: impl AsRef<Path>) -> std::io::Result<Tuple> {
    let file = File::open(path)?;
    let config = serde_json::from_reader(file)?;
    Ok(config)
}