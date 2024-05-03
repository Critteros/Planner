use clap::{Arg, ArgAction, Command};
use mpi::{traits::*, Rank, Threading};
use rayon::iter::IntoParallelIterator;

use crate::algorithm::datatypes::{Individual, Tuple};
use crate::algorithm::{calculate_fitness, crossover, mutate};

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
use rayon::prelude::*;
fn main() {
    let (universe, threading) = mpi::initialize_with_threading(Threading::Multiple).unwrap();
    assert_eq!(threading, mpi::environment::threading_support());
    let world = universe.world();
    let root_process = world.process_at_rank(ROOT_RANK);

    let size = world.size();
    let rank = world.rank();

    let (config, tuples) = mpi_execute_and_synchronize_at(root_init, &world, ROOT_RANK);
    println!("{:?}", config);

    let mut population = algorithm::create_first_population(&config, &tuples);

    for generation_number in 0..config.max_generations {
        println!("Generation: {}", generation_number + 1);

        population = population
            .par_iter()
            .map(|_| crossover(&config, &population))
            .map(|mut individual| {
                mutate(&config, &mut individual);
                individual
            })
            .map(|mut individual| {
                individual.adaptation = calculate_fitness(&individual, &tuples, false);
                individual
            })
            .collect();

        population.sort_by(|a, b| b.adaptation.partial_cmp(&a.adaptation).unwrap());

        println!("Best adaptation: {}", population[0].adaptation);

        if population[0].adaptation == 0 {
            break;
        }
    }

    calculate_fitness(&population[0], &tuples, true);
}
