use std::{fs::File, path::Path};

use anyhow::Result;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(default)]
pub struct AlgorithmConfig {
    // How many generations maximum to run
    pub max_generations: u32,

    // How many individuals to have in the population
    pub population_size: u32,

    // How many genes are in each chromosome (how many classes to schedule)
    pub chromosome_length: u32,

    // The probability of crossover occurring
    pub crossover_probability: f32,

    // The probability of mutation occurring
    pub mutation_probability: f32,

    // How many genes can be mutated at once in a chromosome
    pub mutation_count: u32,

    // How many individuals to select for the next generation
    pub selection_size: u32,
}

impl AlgorithmConfig {
    pub fn from_file(path: impl AsRef<Path>) -> Result<Self> {
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
            chromosome_length: 10,
            crossover_probability: 0.6,
            mutation_probability: 0.01,
            mutation_count: 1,
            selection_size: 10,
        }
    }
}
