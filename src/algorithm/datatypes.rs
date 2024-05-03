use std::{fs::File, path::Path};

use serde::{Deserialize, Serialize};
use thiserror::Error;

#[derive(Debug, Error)]
pub enum TuplesLoadError {
    #[error("Configuration file not found")]
    Io(#[from] std::io::Error),
    #[error(transparent)]
    Csv(#[from] csv::Error),
}

#[derive(Debug, Serialize, Deserialize, Clone, Default, Eq, PartialEq, Hash)]
pub struct Tuple {
    pub id: i32,
    pub label: String,
    pub room: String,
    pub teacher: String,
}

impl Tuple {
    pub fn from_csv(path: impl AsRef<Path>) -> Result<Vec<Tuple>, TuplesLoadError> {
        let file = File::open(path)?;
        let mut reader = csv::Reader::from_reader(file);

        let mut tuples = Vec::new();

        for result in reader.records() {
            let record = result?;
            let tuple = Tuple {
                id: record[0].parse().unwrap(),
                label: record[1].to_string(),
                room: record[2].to_string(),
                teacher: record[3].to_string(),
            };
            tuples.push(tuple);
        }

        Ok(tuples)
    }
}

pub type Gene = i32;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Individual {
    pub adaptation: i32,
    pub chromosomes: Vec<Chromosome>,
}

impl Individual {
    pub fn new(num_chromosomes: usize) -> Self {
        Individual {
            chromosomes: Vec::with_capacity(num_chromosomes),
            ..Self::default()
        }
    }

    pub fn with_chromosomes(chromosomes: Vec<Chromosome>) -> Self {
        Individual {
            chromosomes,
            ..Self::default()
        }
    }
}

impl Default for Individual {
    fn default() -> Self {
        Individual {
            adaptation: -1000,
            chromosomes: Vec::new(),
        }
    }
}

pub type Population = Vec<Individual>;

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct Chromosome {
    pub id: i32,
    pub genes: Vec<Gene>,
}

impl Chromosome {
    pub fn new(id: i32) -> Self {
        Chromosome {
            id,
            genes: Vec::new(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_value_of_individuals() {
        let invidual = Individual::default();
        assert_eq!(invidual.adaptation, -1000);
        assert_eq!(invidual.chromosomes.len(), 0);
    }

    #[test]
    fn test_individual_with_chromosomes() {
        let chromosomes = vec![Chromosome {
            id: 1,
            genes: vec![1, 2, 3],
        }];
        let individual = Individual::with_chromosomes(chromosomes);
        assert_eq!(individual.adaptation, -1000);
        assert_eq!(individual.chromosomes.len(), 1);
    }
}
