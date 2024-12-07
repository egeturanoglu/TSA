# TSA
Solution to "Travelling Salesman Problem"(TSP) using generational algorithm. 

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

Hyperparameter Optimization (27 configurations, 10 runs each):
  The function hyperparameter_optimization() iterates over three parameters (population size, mutation rate, generations) each having three values, creating 27 total configurations.
  For each configuration, it runs the genetic_algorithm() function 10 times and records the mean performance (mean final best fitness = 1/distance) across runs.
  These mean results are stored in a DataFrame and printed out for comparison.
  
Selection of Best Configuration:
  After generating the results, the code identifies the best configuration as the one yielding the highest mean fitness.
  
Final Evaluation:
  Once the best configuration is known, evaluate_best_config() runs the GA 10 more times with these parameters.
  It calculates and prints the mean, standard deviation, best, and worst distances of these 10 runs.
  A box plot of distances and a line plot of mean convergence (average fitness over generations) are produced.
  
Statistics and Plots:
  For the hyperparameter optimization phase, only mean fitness values are reported.
  For the final best configuration evaluation, detailed statistics (mean, std, best, worst) and plots (box plot, convergence line plot) are generated.
  
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

