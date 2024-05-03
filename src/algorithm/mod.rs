use std::cmp::max;

use rand::Rng;

use self::{
    config::AlgorithmConfig,
    datatypes::{Chromosome, Individual, Population, Tuple},
    random::get_random_generator,
};

pub mod config;
pub mod datatypes;
mod random;

/// for each individual (list of periods) in population size
/// for tuple in tuples
/// assign tuple to a random period from individual
fn create_first_population(config: &AlgorithmConfig, tuples: &[Tuple]) -> Population {
    let AlgorithmConfig {
        population_size,
        number_of_periods,
        ..
    } = config.to_owned();

    let mut population = Population::with_capacity(population_size);

    let mut rng = get_random_generator();

    for _ in 0..population_size {
        let mut individual: Individual = Individual::new(number_of_periods);

        // create periods
        for period_id in 0..number_of_periods {
            let period = Chromosome::new(period_id.try_into().unwrap());

            individual.chromosomes.push(period);
        }

        // assign tuple to a random period from individual
        for tuple in tuples {
            let random_period_index = rng.gen_range(0..number_of_periods);
            individual.chromosomes[random_period_index]
                .genes
                .push(tuple.id);
        }

        population.push(individual)
    }

    population
}

fn crossover(config: &AlgorithmConfig, population: &Population) -> Individual {
    let AlgorithmConfig {
        population_size,
        number_of_periods,
        ..
    } = config.to_owned();

    let mut rng = get_random_generator();

    // ToDo: add check that parent is alive, otherwise select another parent
    let mother_index = rng.gen_range(0..population_size);
    let father_index = rng.gen_range(0..number_of_periods);

    let mother = &population[mother_index];
    let father = &population[father_index];

    let mut child: Individual = Individual::new(number_of_periods);

    // mutate genes
    for period_id in 0..number_of_periods {
        let mother_genes = &mother.chromosomes[period_id].genes;
        let father_genes = &father.chromosomes[period_id].genes;

        let mating_point_upper_bound = max(mother_genes.len(), father_genes.len());
        let mating_point = rng.gen_range(0..mating_point_upper_bound);

        let (mother_left, _) = mother_genes.split_at(mating_point);
        let (_, father_right) = father_genes.split_at(mating_point);
        let child_genes = mother_left
            .iter()
            .chain(father_right.iter())
            .cloned()
            .collect();

        child.chromosomes[period_id] = Chromosome {
            id: i32::try_from(period_id).unwrap(),
            genes: child_genes,
        };
    }

    // at this point there could be duplicated and missing genes, so we want to fix this

    // repair lost
    let all_genes: Vec<i32> = mother
        .chromosomes
        .iter()
        .flat_map(|g| g.genes.iter().cloned())
        .collect();

    let lost_genes: Vec<i32> = all_genes
        .iter()
        .filter(|g| !child.chromosomes.iter().any(|c| c.genes.contains(g)))
        .cloned()
        .collect();

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
            let period = &mut individual.chromosomes[period_id];
            let gene_index = rng.gen_range(0..period.genes.len());
            let gene = period.genes[gene_index];

            // remove gene from period
            period.genes.retain(|g| g != &gene);

            // get random period. ToDo: make sure it's not the same period
            let new_period_id = rng.gen_range(0..number_of_periods);

            // add gene to new period
            individual.chromosomes[new_period_id].genes.push(gene);
        }
    }
}

fn calculate_fitness(individual: &Individual, tuples: &Vec<Tuple>) -> f32 {
    let mut fitness = 0.0;

    for period in &individual.chromosomes {
        let mut period_fitness = 0.0;
        // if teacher is teaching more than one class at the same time decrease fitness by 10

        for gene in &period.genes {
            // if the same teacher is teaching more than one class at the same time decrease fitness by 10
            // if different teachers occupy the same room at the same time decrease fitness by 20
            // ToDo: split in tuples lecture type, so CWL and LAB can be in the same room at the same time

            let tuple = tuples
                .iter()
                .find(|t| t.id == *gene)
                .expect(format!("Tuple with id {gene} not found").as_str());
        }
    }

    0.0
}
