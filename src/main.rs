#![allow(unused_variables)]

use clap::{Arg, ArgAction, Command};
use mpi::{traits::*, Rank, Threading};
use rayon::prelude::*;

use self::{
    algorithm::config::AlgorithmConfig,
    mpi_utils::{mpi_execute_and_synchronize_at, ROOT_RANK},
};
use crate::algorithm::datatypes::Chromosome;
use crate::algorithm::{calculate_fitness, crossover, mutate};
use crate::mpi_utils::{mpi_gather_and_synchronize, MPITransferable};
use crate::{algorithm::datatypes::Tuple, mpi_utils::mpi_split_data_across_nodes};

mod algorithm;
mod mpi_utils;

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

fn adapt_population_size_to_worker_number(population_size: usize, rank: Rank, size: Rank) -> usize {
    let mut new_population_size = population_size;

    if population_size % size as usize != 0 {
        new_population_size = population_size + size as usize - (population_size % size as usize);

        if rank == ROOT_RANK {
            println!(
                "Changing population size from {} to {}, to match node number",
                population_size, new_population_size
            )
        }
    }

    new_population_size
}

fn main() {
    let (universe, threading) = mpi::initialize_with_threading(Threading::Multiple).unwrap();
    assert_eq!(threading, mpi::environment::threading_support());

    let world = universe.world();

    let size = world.size();
    let rank = world.rank();

    let (mut config, tuples) = mpi_execute_and_synchronize_at(root_init, &world, ROOT_RANK);

    config.population_size =
        adapt_population_size_to_worker_number(config.population_size, rank, size);

    println!("{:?}", config);

    let mut population = algorithm::create_first_population(&config, &tuples);

    // let mut population_to_be_processed =
    //     mpi_split_data_across_nodes(&population, &world, ROOT_RANK);
    //
    // if rank == ROOT_RANK {
    //     println!(
    //         "Population len: {}, Worker population shard len: {}",
    //         population.len(),
    //         population_to_be_processed.len()
    //     );
    // }

    // population = mpi_gather_and_synchronize(&population_to_be_processed, &world, ROOT_RANK);
    // if rank == ROOT_RANK {
    //     println!("Population size: {}", population.len());
    // }
    //
    // let d_c_e = Chromosome {
    //     id: i32::MAX - 1,
    //     genes: Vec::new(),
    // };
    // let d_c_f = Chromosome {
    //     id: 1,
    //     genes: Vec::new(),
    // };
    //
    // println!(
    //     "ULALA: d_c_e {}, d_c_f {}",
    //     d_c_e.into_bytes().len(),
    //     d_c_f.into_bytes().len()
    // );

    for generation_number in 0..config.max_generations {
        let mut population_to_be_processed =
            mpi_split_data_across_nodes(&population, &world, ROOT_RANK);
        //
        // println!("Individual sizes before crossover: ");
        //
        // for individual in &population_to_be_processed {
        //     println!("{:?}", individual.clone().into_bytes().len());
        // }
        //
        // if rank == ROOT_RANK {
        //     println!(
        //         "Population len: {}, Worker population shard len: {}",
        //         population.len(),
        //         population_to_be_processed.len()
        //     );
        // }

        if rank == ROOT_RANK {
            println!("Generation: {}", generation_number + 1);
        }

        population_to_be_processed = population_to_be_processed
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
        //
        // println!("Individual sizes after crossover: ");
        //
        // for individual in &population_to_be_processed {
        //     println!("{:?}", individual.clone().into_bytes().len());
        // }

        population = mpi_gather_and_synchronize(&population_to_be_processed, &world, ROOT_RANK);

        population.sort_by(|a, b| b.adaptation.partial_cmp(&a.adaptation).unwrap());

        // early stop, print results
        if rank == ROOT_RANK {
            println!(
                "Best adaptation: {}",
                population_to_be_processed[0].adaptation
            );

            if population_to_be_processed[0].adaptation == 0 {
                break;
            }
        }
    }
}
