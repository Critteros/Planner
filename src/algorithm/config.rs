use std::fs::File;
use std::path::Path;
use mpi::traits::Equivalence;
use serde::{Deserialize, Serialize};
use crate::algorithm::data::MPITransferable;

// Individual - list of periods
// Chromosome - a period
// Gene - a class
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Equivalence)]
#[serde(default)]
pub struct AlgorithmConfig {
    // How many generations maximum to run
    pub max_generations: u32,

    // How many individuals to have in the population
    pub population_size: u32,

    // How many genes are in each chromosome. Chromosome length
    pub number_of_periods: u32,

    // The probability of crossover occurring
    pub crossover_probability: f32,

    // The probability of mutation occurring
    pub mutation_probability: f32,

    // If child adaptation is worse than threshold - child is dead
    pub dead_threshold: f32,
}


impl AlgorithmConfig {
    pub fn from_file(path: impl AsRef<Path>) -> anyhow::Result<Self> {
        let mut file = File::open(path)?;
        let config = serde_json::from_reader(&mut file)?;
        Ok(config)
    }
}

impl<'a> Default for AlgorithmConfig {
    fn default() -> Self {
        AlgorithmConfig {
            max_generations: 100,
            population_size: 100,
            number_of_periods: 10,
            crossover_probability: 0.6,
            mutation_probability: 0.01,
            dead_threshold: 0.1,
        }
    }
}
