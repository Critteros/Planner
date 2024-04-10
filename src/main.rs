mod algorithm;

use clap::{Arg, ArgAction, Command};

use crate::algorithm::data::AlgorithmConfig;

fn main() {
    let args = Command::new("Genetic Algorithm")
        .arg(
            Arg::new("config")
                .short('c')
                .value_name("FILE")
                .help("Sets a custom config file")
                .action(ArgAction::Set)
                .required(false),
        )
        .get_matches();

    let path = args
        .get_one::<String>("config")
        .map(String::as_str)
        .unwrap_or("config.json");

    let config = AlgorithmConfig::from_file(path).unwrap_or_default();

    println!("{:?}", config);
}
