use std::error::Error;
use std::fs::File;
use std::path::Path;

#[derive(Debug, serde::Deserialize)]
struct Tuple {
    id: i32,
    label: String,
    room: String,
    teacher: String,
}

fn read_csv(path: impl AsRef<Path>) -> Result<String, dyn Error> {
    let mut file = File::open(path);
    let config = serde_json::from_reader(&mut file?);
    config
}