import numpy as np
from math import log, sqrt
from scipy.stats import norm
from causallearn.utils.cit import CIT, CIT_Base, register_ci_test, NO_SPECIFIED_PARAMETERS_MSG
from causallearn.search.ConstraintBased.PC import pc
import time

# Import our modified modules
# Assuming the modified code is saved in a file called modified_cit.py
#from modified_cit import CIT, CIT_Base, register_ci_test, NO_SPECIFIED_PARAMETERS_MSG

from causallearn.utils.cit import CIT, CIT_Base, register_ci_test, NO_SPECIFIED_PARAMETERS_MSG

# Define a custom implementation of Fisher Z test

class CustomFisherZ(CIT_Base):
    def __init__(self, data, **kwargs):
        super().__init__(data, **kwargs)
        self.check_cache_method_consistent('custom_fisherz', NO_SPECIFIED_PARAMETERS_MSG)
        self.assert_input_data_is_valid()
        # Calculate the correlation matrix just like the original FisherZ
        self.correlation_matrix = np.corrcoef(data.T)
        print("Initialized CustomFisherZ test")

    def __call__(self, X, Y, condition_set=None):
        print("Using the CI test")
        '''
        Custom implementation of Fisher-Z's test that mirrors the original.
        
        Parameters
        ----------
        X, Y and condition_set : column indices of data

        Returns
        -------
        p : the p-value of the test
        '''
        Xs, Ys, condition_set, cache_key = self.get_formatted_XYZ_and_cachekey(X, Y, condition_set)
        if cache_key in self.pvalue_cache: 
           # print(f"Using cached result for {cache_key}")
            return self.pvalue_cache[cache_key]
            
#        print(f"Computing new result for {cache_key}")
        var = Xs + Ys + condition_set
        sub_corr_matrix = self.correlation_matrix[np.ix_(var, var)]
        
        try:
            inv = np.linalg.inv(sub_corr_matrix)
        except np.linalg.LinAlgError:
            raise ValueError('Data correlation matrix is singular. Cannot run fisherz test. Please check your data.')
            
        r = -inv[0, 1] / sqrt(abs(inv[0, 0] * inv[1, 1]))
        if abs(r) >= 1: 
            r = (1. - np.finfo(float).eps) * np.sign(r)
            
        Z = 0.5 * log((1 + r) / (1 - r))
        X = sqrt(self.sample_size - len(condition_set) - 3) * abs(Z)
        p = 2 * (1 - norm.cdf(abs(X)))
        
        self.pvalue_cache[cache_key] = p
        return p

register_ci_test("custom_fisherz", CustomFisherZ)
def run_test():
    # Generate some random data
    np.random.seed(42)  # For reproducibility
    n_samples = 100
    n_features = 5
    data = np.random.randn(n_samples, n_features)
    
    print("=== Testing with original FisherZ ===")
    # Create a CI test with the original method
    original_ci_test = CIT(data, method="fisherz")
    
    # Run some tests
    original_p1 = original_ci_test(0, 1)
    original_p2 = original_ci_test(0, 1, [2])
    original_p3 = original_ci_test(0, 1, [2, 3])
    
    print(f"Original FisherZ p-values:")
    print(f"  X=0, Y=1: {original_p1}")
    print(f"  X=0, Y=1, Z=[2]: {original_p2}")
    print(f"  X=0, Y=1, Z=[2, 3]: {original_p3}")
    
    print("\n=== Testing with custom FisherZ ===")
    # Create a CI test with our custom method
    custom_ci_test = CIT(data, method="custom_fisherz")
    
    # Run the same tests
    custom_p1 = custom_ci_test(0, 1)
    custom_p2 = custom_ci_test(0, 1, [2])
    custom_p3 = custom_ci_test(0, 1, [2, 3])
    
    print(f"Custom FisherZ p-values:")
    print(f"  X=0, Y=1: {custom_p1}")
    print(f"  X=0, Y=1, Z=[2]: {custom_p2}")
    print(f"  X=0, Y=1, Z=[2, 3]: {custom_p3}")
    
    # Compare results
    print("\n=== Comparing results ===")
    print(f"P-value match for X=0, Y=1: {original_p1 == custom_p1}")
    print(f"P-value match for X=0, Y=1, Z=[2]: {original_p2 == custom_p2}")
    print(f"P-value match for X=0, Y=1, Z=[2, 3]: {original_p3 == custom_p3}")
    
    # Test caching mechanism by running the same test again
    print("\n=== Testing caching mechanism ===")
    custom_p1_cached = custom_ci_test(0, 1)  # Should use cached result
    print(f"Cached result matches: {custom_p1 == custom_p1_cached}")
# Register the custom class


def generate_synthetic_data(n_samples=500):
    """
    Generate synthetic data with a known causal structure:
    X1 -> X3 <- X2
    X4 -> X5 -> X6
    """
    np.random.seed(42)
    
    # X1, X2, X4 are exogenous
    X1 = np.random.normal(0, 1, n_samples)
    X2 = np.random.normal(0, 1, n_samples)
    X4 = np.random.normal(0, 1, n_samples)
    
    # X3 depends on X1 and X2
    X3 = 0.7 * X1 + 0.8 * X2 + np.random.normal(0, 1, n_samples)
    
    # X5 depends on X4
    X5 = 0.9 * X4 + np.random.normal(0, 0.5, n_samples)
    
    # X6 depends on X5
    X6 = 0.8 * X5 + np.random.normal(0, 0.5, n_samples)
    
    # Combine all variables
    data = np.column_stack([X1, X2, X3, X4, X5, X6])
    
    # Ground truth DAG adjacency matrix (1 if i->j)
    true_dag = np.zeros((6, 6))
    true_dag[0, 2] = 1  # X1 -> X3
    true_dag[1, 2] = 1  # X2 -> X3
    true_dag[3, 4] = 1  # X4 -> X5
    true_dag[4, 5] = 1  # X5 -> X6
    
    return data, true_dag

def print_graph_edges(adj_matrix, title):
    """Print edges from an adjacency matrix"""
    print(f"\n{title}:")
    edge_count = 0
    for i in range(adj_matrix.shape[0]):
        for j in range(adj_matrix.shape[1]):
            if adj_matrix[i, j] != 0:
                edge_count += 1
                print(f"  X{i+1} -> X{j+1}")
    if edge_count == 0:
        print("  No edges found")
    else:
        print(f"  Total: {edge_count} edges")

def compare_results(adj1, adj2):
    """Compare two adjacency matrices and return metrics"""
    # Check if the matrices have the same shape
    if adj1.shape != adj2.shape:
        raise ValueError("Adjacency matrices must have the same shape")
    
    # Convert to binary (just in case)
    adj1_bin = (adj1 != 0).astype(int)
    adj2_bin = (adj2 != 0).astype(int)
    
    # Count matches and mismatches
    matches = np.sum(adj1_bin == adj2_bin)
    total = adj1.shape[0] * adj1.shape[1]
    
    # Calculate edge presence match
    match_percentage = (matches / total) * 100
    
    # Count missing and extra edges
    in_1_not_2 = np.sum((adj1_bin == 1) & (adj2_bin == 0))
    in_2_not_1 = np.sum((adj1_bin == 0) & (adj2_bin == 1))
    
    return {
        'match_percentage': match_percentage,
        'edges_in_first_not_second': in_1_not_2,
        'edges_in_second_not_first': in_2_not_1
    }

def run_pc_algorithm_test():
    """Run the PC algorithm with both built-in and custom Fisher-Z tests"""
    print("\n=== Testing PC Algorithm with Custom CI Test ===")
    
    # Generate synthetic data
    data, true_dag = generate_synthetic_data(n_samples=500)
    
    print("Data shape:", data.shape)
    
    # Print true DAG edges
    print_graph_edges(true_dag, "True DAG Edges")
    
    # Run PC with built-in Fisher-Z
    print("\nRunning PC with built-in Fisher-Z...")
    start_time = time.time()
    pc_result_built_in = pc(data, 0.05, indep_test="fisherz")
    built_in_time = time.time() - start_time
    print(f"Built-in Fisher-Z test took {built_in_time:.4f} seconds")
    
    # Run PC with custom Fisher-Z
    print("\nRunning PC with custom Fisher-Z...")
    start_time = time.time()
    pc_result_custom = pc(data, 0.05, indep_test="custom_fisherz")
    custom_time = time.time() - start_time
    print(f"Custom Fisher-Z test took {custom_time:.4f} seconds")
    
    # Get the adjacency matrices
    adj_built_in = pc_result_built_in.G.graph
    adj_custom = pc_result_custom.G.graph
    
    # Print edges from both results
    print_graph_edges(adj_built_in, "PC with Built-in Fisher-Z")
    print_graph_edges(adj_custom, "PC with Custom Fisher-Z")
    
    # Compare built-in and custom results
    comparison = compare_results(adj_built_in, adj_custom)
    print(f"\nComparison of built-in vs custom CI test:")
    print(f"Match percentage: {comparison['match_percentage']:.2f}%")
    print(f"Edges in built-in result not in custom: {comparison['edges_in_first_not_second']}")
    print(f"Edges in custom result not in built-in: {comparison['edges_in_second_not_first']}")
    
    
if __name__ == "__main__":
    run_test()
    run_pc_algorithm_test()

