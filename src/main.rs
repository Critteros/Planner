use clap::{Arg, ArgAction, Command};
use mpi::{Rank, Threading, traits::*};
use rand::{Rng, SeedableRng};
use serde::{Deserialize, Serialize};

use crate::algorithm::datatypes::Tuple;

use self::{algorithm::config::AlgorithmConfig, mpi_utils::mpi_execute_and_synchronize_at};

mod algorithm;
mod mpi_utils;

const ROOT_RANK: Rank = 0;

fn root_init() -> (AlgorithmConfig, Vec<Tuple>) {
    let args = Command::new("Genetic Algorithm")
        .arg(
            Arg::new("config")
                .short('c')
                .value_name("FILE")
                .help("Sets a custom config file")
                .action(ArgAction::Set)
                .required(false),
        )
        .arg(
            Arg::new("tuples")
                .short('t')
                .value_name("FILE")
                .help("Custom location of tuples")
                .action(ArgAction::Set)
                .required(false),
        )
        .get_matches();

    let config_path = args
        .get_one::<String>("config")
        .map(String::as_str)
        .unwrap_or("config.json");

    let tuples_path = args
        .get_one::<String>("tuples")
        .map(String::as_str)
        .unwrap_or("tuples.csv");

    let config = AlgorithmConfig::from_json(config_path).unwrap_or_default();
    let tuples = Tuple::from_csv(tuples_path).expect("Tuples could not be loaded");

    return (config, tuples);
}

fn main() {
    let (universe, threading) = mpi::initialize_with_threading(Threading::Multiple).unwrap();
    assert_eq!(threading, mpi::environment::threading_support());
    let world = universe.world();
    let root_process = world.process_at_rank(ROOT_RANK);

    let size = world.size();
    let rank = world.rank();

    let (config, tuples) = mpi_execute_and_synchronize_at(root_init, &world, ROOT_RANK);
    println!("{:?}", config);
}
