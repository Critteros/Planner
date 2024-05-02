mod algorithm;
mod mpi_utils;

use std::cmp::max;
use mpi::{Rank, Threading};
use mpi::traits::*;
use algorithm::tuple;

use rand::{SeedableRng, Rng};
use rand::rngs::StdRng;

use clap::{Arg, ArgAction, Command};
use serde::{Deserialize, Serialize};
use algorithm::tuple::Tuple;
use algorithm::config::AlgorithmConfig;
use crate::mpi_utils::mpi_synchronize_ref;

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
                .required(false)
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

    let config = AlgorithmConfig::from_file(config_path).unwrap_or_default();
    let tuples = tuple::tuples_from_csv(tuples_path).expect("Tuples could not be loaded");

    return (config, tuples);
}

type Gene = i32;

#[derive(Debug, Clone, Serialize, Deserialize)]
struct Chromosome {
    id: i32,
    genes: Vec<Gene>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct Individual {
    adaptation: f32,
    chromosomes: Vec<Chromosome>,
}

impl Individual {
    fn new(chromosomes: impl IntoIterator<Item=Chromosome>) -> Self {
        Individual {
            adaptation: 0.0,
            chromosomes: chromosomes.into_iter().collect(),
        }
    }
}

type Population = Vec::<Individual>;


fn get_random_generator() -> StdRng {
    let seed: [u8; 32] = [42; 32];
    let mut rng = StdRng::from_seed(seed);
    rng
}


// for each individual (list of periods) in population size
// for tuple in tuples
// assign tuple to a random period from individual
fn create_first_population(config: &AlgorithmConfig, tuples: &Vec<Tuple>) -> Population {
    let population_size = usize::try_from(config.population_size).unwrap();
    let number_of_periods = usize::try_from(config.number_of_periods).unwrap();

    let mut population = Vec::<Individual>::with_capacity(population_size);

    let mut rng = get_random_generator();

    for _ in 0..population_size {
        let mut individual: Individual = Individual {
            adaptation: 0.0,
            chromosomes: Vec::<Chromosome>::with_capacity(number_of_periods),
        };

        // create periods
        for period_id in 0..number_of_periods {
            let period = Chromosome {
                id: i32::try_from(period_id).unwrap(),
                genes: Vec::<Gene>::new(),
            };

            individual.chromosomes.push(period);
        }

        // assign tuple to a random period from individual
        for tuple in tuples {
            let random_period_index = rng.gen_range(0..number_of_periods);
            individual.chromosomes[random_period_index].genes.push(tuple.id);
        }

        population.push(individual)
    }

    population
}

fn crossover(config: &AlgorithmConfig, population: &Population) -> Individual {
    let population_size = usize::try_from(config.population_size).unwrap();
    let number_of_periods = usize::try_from(config.number_of_periods).unwrap();

    let mut rng = get_random_generator();

    // ToDo: add check that parent is alive
    let mother_index = rng.gen_range(0..population_size);
    let father_index = rng.gen_range(0..number_of_periods);

    let mother = &population[mother_index];
    let father = &population[father_index];

    let mut child: Individual = Individual {
        adaptation: 0.0,
        chromosomes: Vec::<Chromosome>::with_capacity(number_of_periods),
    };

    // mutate genes
    for period_id in 0..number_of_periods {
        let mother_genes = &mother.chromosomes[period_id].genes;
        let father_genes = &father.chromosomes[period_id].genes;

        let mating_point_upper_bound = max(mother_genes.len(), father_genes.len());
        let mating_point = rng.gen_range(0..mating_point_upper_bound);

        let (mother_left, _) = mother_genes.split_at(mating_point);
        let (_, father_right) = father_genes.split_at(mating_point);
        let child_genes = mother_left.iter().chain(father_right.iter()).cloned().collect();

        child.chromosomes[period_id] = Chromosome {
            id: i32::try_from(period_id).unwrap(),
            genes: child_genes,
        };
    }

    // at this point there could be duplicated and missing genes, so we want to fix this

    // repair lost
    let all_genes: Vec<i32> = mother.chromosomes.iter().flat_map(|g| g.genes.iter().cloned()).collect();
    let lost_genes: Vec<i32> = all_genes.iter().filter(|g| !child.chromosomes.iter().any(|c| c.genes.contains(g))).cloned().collect();

    for gene in lost_genes {
        let period_id = rng.gen_range(0..number_of_periods);
        child.chromosomes[period_id].genes.push(gene);
    }

    // remove duplicates
    let mut seen = std::collections::HashSet::new();

    for period in &mut child.chromosomes {
        period.genes.retain(|x| seen.insert(x.clone()));
    }

    child
}

fn mutate(config: &AlgorithmConfig, individual: &mut Individual) {
    let mutation_probability = config.mutation_probability;
    let number_of_periods = usize::try_from(config.number_of_periods).unwrap();

    let mut rng = get_random_generator();

    for period_id in 0..number_of_periods {
        if rng.gen_bool(mutation_probability.into()) {
            let period = individual.chromosomes[period_id].clone();
            let gene_index = rng.gen_range(0..period.genes.len());
            let gene = period.genes[gene_index];

            let new_gene = rng.gen_range(0..100);
            period.genes[gene_index] = new_gene;
        }
    }
}

fn main() {
    let (universe, threading) = mpi::initialize_with_threading(Threading::Multiple).unwrap();
    assert_eq!(threading, mpi::environment::threading_support());
    let world = universe.world();
    let root_process = world.process_at_rank(ROOT_RANK);

    let size = world.size();
    let rank = world.rank();

    let (mut config, mut tuples) = (AlgorithmConfig::default(), vec![]);

    if rank == ROOT_RANK {
        (config, tuples) = root_init();
    }

    mpi_synchronize_ref(&mut config, &world, ROOT_RANK);
    println!("{:?}", config);

    mpi_synchronize_ref(&mut tuples, &world, ROOT_RANK);

    // let data_size = serialized_config.len();
    // let mut data_size_buf = vec![0; world.size()];
    // world.all_gather_into(&data_size, &mut data_size_buf[..]);

    // let first_population = create_first_population(config, tuples);

    // println!("Supported level of threading: {:?}", threading);
    //
    // let next_rank = (rank + 1) % size;
    // let previous_rank = (rank - 1 + size) % size;
    //
    // if rank == ROOT_RANK {
    //     println!("ROOT");
    // }

    // let msg = vec![rank, 2 * rank, 4 * rank];
    // mpi::request::scope(|scope| {
    //     let _sreq = WaitGuard::from(
    //         world
    //             .process_at_rank(next_rank)
    //             .immediate_send(scope, &msg[..]),
    //     );
    //
    //     let (msg, status) = world.any_process().receive_vec::<Rank>();
    //
    //     println!(
    //         "Process {} got message {:?}.\nStatus is: {:?}",
    //         rank, msg, status
    //     );
    //     let x = status.source_rank();
    //     assert_eq!(x, previous_rank);
    //     assert_eq!(vec![x, 2 * x, 4 * x], msg);
    //
    //     let root_rank = 0;
    //     let root_process = world.process_at_rank(root_rank);
    //
    //     let mut a;
    //     if world.rank() == root_rank {
    //         a = vec![2, 4, 8, 16];
    //         println!("Root broadcasting value: {:?}.", &a[..]);
    //     } else {
    //         a = vec![0; 4];
    //     }
    //     root_process.broadcast_into(&mut a[..]);
    //     println!("Rank {} received value: {:?}.", world.rank(), &a[..]);
    //     assert_eq!(&a[..], &[2, 4, 8, 16]);
    // });
}
