use std::{fs::File, path::Path};

use mpi::traits::Equivalence;
use serde::{Deserialize, Serialize};
use thiserror::Error;

#[derive(Error, Debug)]
pub enum ConfigLoadError {
    #[error("Configuration file not found")]
    FileNotFound(#[from] std::io::Error),
    #[error(transparent)]
    JsonError(#[from] serde_json::Error),
}

/// Configuration for the genetic algorithm
/// * Individual - list of periods
/// * Chromosome - a period of time with a list of genes (classes that are
///   happening at that time)
/// * Gene - a tuple consisting of teacher, subject, room and class
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Equivalence)]
#[serde(default)]
pub struct AlgorithmConfig {
    /// How many generations maximum to run
    pub max_generations: usize,

    /// How many individuals to have in the population
    pub population_size: usize,

    /// How many genes are in each chromosome. Chromosome length
    pub number_of_periods: usize,

    /// The probability of crossover occurring
    pub crossover_probability: f32,

    /// The probability of mutation occurring
    pub mutation_probability: f32,

    /// If child adaptation is worse than threshold - child is dead
    pub dead_threshold: f32,
}

impl AlgorithmConfig {
    /// Load the configuration from a JSON file
    pub fn from_json(path: impl AsRef<Path>) -> Result<AlgorithmConfig, ConfigLoadError> {
        let mut file = File::open(path)?;
        let config = serde_json::from_reader(&mut file)?;
        Ok(config)
    }
}

impl Default for AlgorithmConfig {
    /// Default configuration
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
