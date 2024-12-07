import numpy as np
import random
import matplotlib.pyplot as plt
import pandas as pd

# -----------------------
# Problem Setup Functions
# -----------------------
def generate_cities(num_cities=20, x_range=(0, 100), y_range=(0, 100)):
    return np.random.uniform(low=[x_range[0], y_range[0]], 
                             high=[x_range[1], y_range[1]], 
                             size=(num_cities, 2))

def calculate_distance(cities, route):
    dist = 0
    for i in range(len(route)):
        dist += np.linalg.norm(cities[route[i]] - cities[route[(i + 1) % len(route)]])
    return dist

def fitness_function(cities, population):
    return [1 / calculate_distance(cities, individual) for individual in population]

# -----------------------
# GA Operators
# -----------------------
def initialize_population(pop_size, num_cities):
    return [random.sample(range(num_cities), num_cities) for _ in range(pop_size)]

def roulette_selection(population, fitness):
    total_fit = sum(fitness)
    pick = random.uniform(0, total_fit)
    current = 0
    for i, f in enumerate(fitness):
        current += f
        if current > pick:
            return population[i]

def ordered_crossover(parent1, parent2):
    size = len(parent1)
    start, end = sorted(random.sample(range(size), 2))
    child = [-1]*size
    # Copy segment from parent1
    for i in range(start, end):
        child[i] = parent1[i]
    # Fill remaining positions with genes from parent2
    p2_elems = [gene for gene in parent2 if gene not in child]
    p2_index = 0
    for i in range(size):
        if child[i] == -1:
            child[i] = p2_elems[p2_index]
            p2_index += 1
    return child

def swap_mutation(individual, mutation_rate=0.1, num_swaps=1):
    for _ in range(num_swaps):
        if random.random() < mutation_rate:
            idx1, idx2 = random.sample(range(len(individual)), 2)
            individual[idx1], individual[idx2] = individual[idx2], individual[idx1]
    return individual

def inject_diversity(population, num_to_replace, num_cities):
    for _ in range(num_to_replace):
        idx = random.randrange(len(population))
        population[idx] = random.sample(range(num_cities), num_cities)
    return population

# -----------------------
# Genetic Algorithm
# -----------------------
def genetic_algorithm(cities, pop_size=100, generations=500, mutation_rate=0.1, num_swaps=2, diversity_injection_rate=0.05):
    num_cities = len(cities)
    population = initialize_population(pop_size, num_cities)
    fitness = fitness_function(cities, population)
    best_idx = np.argmax(fitness)
    best_individual = population[best_idx]
    best_fitness = fitness[best_idx]

    best_fitness_over_time = [best_fitness]

    # Continuous diversity injection
    num_to_replace_each_gen = int(pop_size * diversity_injection_rate)

    for _ in range(1, generations + 1):
        new_population = [best_individual]  # Elitism
        while len(new_population) < pop_size:
            parent1 = roulette_selection(population, fitness)
            parent2 = roulette_selection(population, fitness)
            child1 = ordered_crossover(parent1, parent2)
            child2 = ordered_crossover(parent2, parent1)
            child1 = swap_mutation(child1, mutation_rate, num_swaps)
            child2 = swap_mutation(child2, mutation_rate, num_swaps)
            new_population.append(child1)
            if len(new_population) < pop_size:
                new_population.append(child2)

        # Inject diversity every generation
        new_population = inject_diversity(new_population, num_to_replace_each_gen, num_cities)

        population = new_population
        fitness = fitness_function(cities, population)
        gen_best_idx = np.argmax(fitness)
        gen_best_fitness = fitness[gen_best_idx]
        if gen_best_fitness > best_fitness:
            best_fitness = gen_best_fitness
            best_individual = population[gen_best_idx]

        best_fitness_over_time.append(best_fitness)

    best_distance = calculate_distance(cities, best_individual)
    return best_individual, best_fitness_over_time, best_distance

# -----------------------
# Hyperparameter Optimization
# -----------------------
def hyperparameter_optimization(cities):
    # Example parameters for optimization:
    # Three parameters, three values each:
    pop_sizes = [50, 100, 200]
    mutation_rates = [0.05, 0.1, 0.2]
    generations_list = [100, 300, 500]

    results = []
    # Evaluate each combination with 10 runs and store mean fitness
    for pop_size in pop_sizes:
        for mutation_rate in mutation_rates:
            for generations in generations_list:
                # Conduct 10 runs
                final_fitnesses = []
                for _ in range(10):
                    _, fitness_over_time, distance = genetic_algorithm(
                        cities, 
                        pop_size=pop_size, 
                        generations=generations, 
                        mutation_rate=mutation_rate,
                        num_swaps=2,
                        diversity_injection_rate=0.05
                    )
                    # Use final best fitness = 1/distance for performance measure
                    final_fitnesses.append(1/distance)

                mean_perf = np.mean(final_fitnesses)
                results.append((pop_size, mutation_rate, generations, mean_perf))
    
    # Results as a DataFrame
    results_df = pd.DataFrame(results, columns=["Pop_Size", "Mutation_Rate", "Generations", "Mean_Fitness"])
    return results_df

# -----------------------
# Final Evaluation of Best Configuration
# -----------------------
def evaluate_best_config(cities, pop_size, mutation_rate, generations):
    # Run the best configuration 10 times
    distances = []
    all_runs_fitness_curves = []
    for _ in range(10):
        _, fitness_over_time, distance = genetic_algorithm(cities, pop_size=pop_size, generations=generations, 
                                                           mutation_rate=mutation_rate, num_swaps=2, 
                                                           diversity_injection_rate=0.05)
        distances.append(distance)
        all_runs_fitness_curves.append(fitness_over_time)

    distances = np.array(distances)
    mean_dist = np.mean(distances)
    std_dist = np.std(distances)
    best_dist = np.min(distances)
    worst_dist = np.max(distances)

    # Box plot of final distances
    plt.figure()
    plt.boxplot(distances)
    plt.title("Box Plot of Distances (Best Configuration)")
    plt.ylabel("Distance")
    plt.show()

    # Line plot for convergence (mean fitness over time across runs)
    # First, convert all_runs_fitness_curves to a numpy array
    arr = np.array(all_runs_fitness_curves)  # shape: (10, generations+1)
    mean_fitness_over_time = np.mean(arr, axis=0)
    plt.figure()
    plt.plot(mean_fitness_over_time)
    plt.title("Convergence (Mean Fitness Over 10 Runs)")
    plt.xlabel("Generation")
    plt.ylabel("Fitness")
    plt.show()

    # Print the stats
    print("Final Evaluation (Best Configuration):")
    print("Mean Distance:", mean_dist)
    print("Std Distance:", std_dist)
    print("Best Distance:", best_dist)
    print("Worst Distance:", worst_dist)

    # Create a table of all runs for the final configuration
    df_final = pd.DataFrame({"Run": range(1,11), "Distance": distances})
    print("\nFinal Runs Raw Data:")
    print(df_final)

    return mean_dist, std_dist, best_dist, worst_dist

# -----------------------
# Main Execution
# -----------------------
if __name__ == "__main__":
    # Generate cities
    num_cities = 20
    cities = generate_cities(num_cities)

    # Step 1: Hyperparameter Optimization
    # Run 27 combinations (3x3x3), 10 runs each
    results_df = hyperparameter_optimization(cities)
    print("Hyperparameter Optimization Results (Mean Fitness):")
    print(results_df)

    # Identify the best configuration based on highest mean fitness
    best_row = results_df.iloc[results_df["Mean_Fitness"].idxmax()]
    best_pop_size = int(best_row["Pop_Size"])
    best_mutation_rate = float(best_row["Mutation_Rate"])
    best_generations = int(best_row["Generations"])

    print("\nBest Configuration Found:")
    print("Population Size:", best_pop_size)
    print("Mutation Rate:", best_mutation_rate)
    print("Generations:", best_generations)

    # Step 2: Final Evaluation with Best Configuration
    mean_dist, std_dist, best_dist, worst_dist = evaluate_best_config(
        cities, best_pop_size, best_mutation_rate, best_generations
    )
