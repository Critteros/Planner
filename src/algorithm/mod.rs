use itertools::Itertools;
use rand::distributions::WeightedIndex;
use rand::prelude::*;
use rand::seq::IteratorRandom;
use rayon::prelude::*;
use std::cmp::min;

use rand::Rng;

use self::{
    config::AlgorithmConfig,
    datatypes::{Chromosome, Individual, Population, Tuple},
    random::get_random_generator,
};

pub mod config;
pub mod datatypes;
mod random;

/// Create a first population
///
/// Create a population of size `population_size` with each individual having `number_of_periods`
/// periods.
/// Then assign tuple to a random period of individual
pub fn create_first_population(config: &AlgorithmConfig, tuples: &[Tuple]) -> Population {
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

/// Get parents from the current population
///
/// Can't use roulette wheel selection because the population is big but
/// wheel selections sums up all the adaptation function values and calculates the probability
/// of each individual being selected as adaptation / sum of all adaptations.
/// When population is big the sum of all adaptations is big and the probability of
/// each individual being selected is very small. In practise this means that
/// less adapted individuals are selected with relatively high probability.
///
/// Instead, we sort the population by adaptation descending.
/// Then we apply exponent function (a * e^x + b) to the index of the individual in the sorted population.
/// Controlling the a and b parameters we can control the probability of selecting the individual.
/// Current values are selected by trial and error.
/// Then we apply roulette wheel selection to select the parents making sure that the parents are different.
pub fn rand_parents(parents: &Population) -> (&Individual, &Individual) {
    assert!(parents.len() > 1);

    let mut rng = get_random_generator();

    let sorted_parents = parents
        .into_iter()
        .sorted_by(|a, b| b.adaptation.partial_cmp(&a.adaptation).unwrap())
        .collect::<Vec<_>>();

    let weights = (0..sorted_parents.len())
        .map(|x| f64::exp((-0.3f64 * x as f64) + 2f64))
        .collect::<Vec<_>>();

    let dist = WeightedIndex::new(weights.clone()).unwrap();

    let idx1 = dist.sample(&mut rng);

    // Sample the second index ensuring its different from the first
    let idx2 = loop {
        let idx = dist.sample(&mut rng);
        if idx != idx1 {
            break idx;
        }
    };

    // println!(
    //     "Min: {}, Max: {}, Parent 1 weights: {}, Parent 2 weights: {}, Parent 1 weight: {}, Parent 2 weight: {}",
    //     min_adaptation, max_adaptation, p[idx1].adaptation, p[idx2].adaptation, weights[idx1], weights[idx2]
    // );

    return (
        sorted_parents.get(idx1).unwrap(),
        sorted_parents.get(idx2).unwrap(),
    );
}

/// Crossover two parents to create a child
///
/// We are choosing random parents from the readonly current population. Then for each corresponding
/// period we are choosing a gene mating point and creating a child by combining the genes from the parents.
/// Then we need to solve 2 potential problems:
/// 1. Missing genes. To solve it we are adding missing genes to the random period.
/// 2. Duplicated genes. To solve it we are removing duplicated genes from the periods.
///
/// There is most likely a bug in Rust or Rayon as when we use par_bridge instead of (collect, par_iter)
/// the assert fails meaning it selects items from `mother` and `father` in different order.
pub fn crossover(config: &AlgorithmConfig, population: &Population) -> Individual {
    let AlgorithmConfig {
        number_of_periods, ..
    } = config.to_owned();

    let (mother, father) = rand_parents(population);

    let mut child: Individual = Individual::with_chromosomes(
        std::iter::zip(mother.chromosomes.iter(), father.chromosomes.iter())
            .collect::<Vec<_>>()
            .par_iter()
            // .par_bridge()
            .map(|(mother_chromosome, father_chromosome)| {
                assert_eq!(mother_chromosome.id, father_chromosome.id);
                let mut rng = get_random_generator();

                let id = mother_chromosome.id;

                let mother_genes = &father_chromosome.genes;
                let father_genes = &mother_chromosome.genes;

                let mating_point_upper_bound = min(mother_genes.len(), father_genes.len());

                let mating_point = rng.gen_range(0..=mating_point_upper_bound);

                let (mother_left, _) = mother_genes.split_at(mating_point);
                let (_, father_right) = father_genes.split_at(mating_point);
                let child_genes = mother_left
                    .iter()
                    .chain(father_right.iter())
                    .cloned()
                    .collect::<Vec<_>>();

                Chromosome {
                    id,
                    genes: child_genes,
                }
            })
            .collect(),
    );

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

    let mut rng = get_random_generator();

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

/// Mutate the individual
///
/// Typically, mutation probability determines the probability of individual mutation.
/// In this case, mutation probability determines the probability of chromosome mutation, so it is
/// good idea to keep it small.
///
/// For each period, we are checking if the mutation should occur. If it should, we are removing
/// a random gene from the period and adding it to a random period.
pub fn mutate(config: &AlgorithmConfig, individual: &mut Individual) {
    let mutation_probability = config.mutation_probability;
    let number_of_periods = usize::try_from(config.number_of_periods).unwrap();

    let mut rng = get_random_generator();

    for period_id in 0..number_of_periods {
        if rng.gen_bool(mutation_probability.into()) {
            let gene_count = individual.chromosomes[period_id].genes.len();

            if gene_count == 0 {
                continue;
            }

            let gene_index = rng.gen_range(0..gene_count);

            let gene = individual.chromosomes[period_id].genes.remove(gene_index);

            // remove gene from current period
            individual.chromosomes[period_id]
                .genes
                .retain(|g| g != &gene);

            // add gene to random period
            individual
                .chromosomes
                .iter_mut()
                .filter(|target| target.id != i32::try_from(period_id).unwrap())
                .choose(&mut rng)
                .unwrap()
                .genes
                .push(gene);
        }
    }
}

/// Calculate fitness of the individual
///
/// For every period in individual we are checking 2 rules:
/// 1) If the same teacher is teaching more than one class at the same time decrease fitness by 10
/// 2) If different teachers occupy the same room at the same time decrease fitness by 20
pub fn calculate_fitness(individual: &Individual, tuples: &Vec<Tuple>, debug: bool) -> i32 {
    let mut individual_fitness = 0;

    for period in &individual.chromosomes {
        // if teacher is teaching more than one class at the same time decrease fitness by 10

        let genes = &period.genes;

        for gene_id in genes {
            // if the same teacher is teaching more than one class at the same time decrease fitness by 10,
            // if different teachers occupy the same room at the same time decrease fitness by 20

            // additional rules may be added, for example,
            // the division of lectures by type of classes, if the types of classes differ for the
            // same lecture, reduce the suitability by a smaller value

            let tuple = tuples
                .iter()
                .find(|t| t.id == *gene_id)
                .expect(format!("Tuple with id {} not found", *gene_id).as_str());

            let other_classes = tuples
                .iter()
                .filter(|t| genes.contains(&t.id))
                .filter(|t| t.id != tuple.id);

            // get count of tuples with the same teacher
            let same_teacher_different_classes_count = other_classes
                .clone()
                .filter(|t| t.room == tuple.room)
                .filter(|t| t.teacher == tuple.teacher)
                .count();

            individual_fitness -= (same_teacher_different_classes_count as i32) * 10;

            let same_room_different_teacher_count = other_classes
                .clone()
                .filter(|t| t.room == tuple.room)
                .filter(|t| t.teacher != tuple.teacher)
                .count();

            individual_fitness -= (same_room_different_teacher_count as i32) * 20;

            let same_teacher_same_subject_count = other_classes
                .clone()
                .filter(|t| t.teacher == tuple.teacher)
                .filter(|t| t.label == tuple.label)
                .count();

            individual_fitness -= (same_teacher_same_subject_count as i32) * 10;

            let same_teacher_different_subject_count = other_classes
                .clone()
                .filter(|t| t.teacher == tuple.teacher)
                .filter(|t| t.label != tuple.label)
                .count();

            individual_fitness -= (same_teacher_different_subject_count as i32) * 20;

            if debug {
                println!(
                    "same_teacher_different_classes_count: {}, same_room_different_teacher_count: {}",
                    same_teacher_different_classes_count, same_room_different_teacher_count
                );
            }
        }
    }

    if debug {
        println!("Individual fitness: {}", individual_fitness);
    }

    individual_fitness
}
