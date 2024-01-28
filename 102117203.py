import sys
import pandas as pd
import numpy as np

def topsis(input_file, weights, impacts, output_file):
    try:
        # Read the input CSV file
        data = pd.read_csv(input_file)

        # Validate the input data
        validate_input(data, weights, impacts)

        # Normalize the data
        normalized_data = normalize_data(data.iloc[:, 1:])

        # Weighted normalized decision matrix
        weighted_normalized_matrix = apply_weights(normalized_data, weights)

        # Ideal best and worst solutions
        ideal_best = find_ideal(weighted_normalized_matrix, np.max)
        ideal_worst = find_ideal(weighted_normalized_matrix, np.min)

        impacts_array = np.array(impacts.split(','))
        for i, impact in enumerate(impacts_array):
            if impact == "-":
                temp = ideal_best.iloc[i]
                ideal_best.iloc[i] = ideal_worst.iloc[i]
                ideal_worst.iloc[i] = temp
        # Calculate separation measures
        separation_pos = calculate_separation(weighted_normalized_matrix, ideal_best)
        separation_neg = calculate_separation(weighted_normalized_matrix, ideal_worst)

        # Calculate TOPSIS score
        topsis_score = separation_neg / (separation_pos + separation_neg)

        # Add TOPSIS score and rank to the original data
        data['Topsis Score'] = topsis_score
        data['Rank'] = data['Topsis Score'].rank(ascending=False)

        # Save the result to a new CSV file
        data.to_csv(output_file, index=False)

        print("TOPSIS analysis completed. Results saved to", output_file)

    except Exception as e:
        print("Error:", str(e))

def validate_input(data, weights, impacts):
    # Validate the number of parameters
    if len(sys.argv) != 5:
        raise ValueError("Usage: python <program.py> <InputDataFile> <Weights> <Impacts> <ResultFileName>")

    # Validate the number of weights, impacts, and columns
    num_weights = len(weights.split(','))
    num_impacts = len(impacts.split(','))
    num_columns = data.shape[1] - 1

    if num_weights != num_impacts or num_weights != num_columns:
        raise ValueError("Number of weights, impacts, and columns must be the same.")

def normalize_data(data):
    return data / np.linalg.norm(data, axis=0)

def apply_weights(data, weights):
    weights_array = np.array([float(w) for w in weights.split(',')])
    return data * weights_array

def find_ideal(data, func):
    return func(data, axis=0)

def calculate_separation(data, ideal_solution):
    return np.sqrt(np.sum((data - ideal_solution) ** 2, axis=1))

if __name__ == "__main__":
    # Get command-line arguments
    input_file = sys.argv[1]
    weights = sys.argv[2]
    impacts = sys.argv[3]
    output_file = sys.argv[4]

    topsis(input_file, weights, impacts, output_file)
