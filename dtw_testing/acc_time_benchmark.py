import sys
import math
import pandas as pd
import numpy as np
from load_data import load_dataset
import time
from dtw import dtw, sakoeChibaWindow
from fastdtw import fastdtw
from tqdm import tqdm
from pathlib import Path
from au_dtw import sakoe_chiba
from au_dtw import fastdtw as fastdtw_own
from au_dtw import cdtw as cdtw_own

MEASURE_ROUNDS = 1

# Measures time and accuracy (measured as weighted k-NN) for a certain method and window sizes ranging from start-->stop with step-size
def _main(dataset, method, start, stop, step, kNN, pruning_method = ...):    
    if kNN < 1:
        raise ValueError("Only kNN larger than 0 accepted")
    
    # Load dataset and separate labels and series
    train_data, test_data = load_dataset(dataset)

    # Methods
    methods = ["FastDTW_lib", "cDTW_lib", "Filtering", "cDTW_own", "FastDTW_own"]
    
    columns = ["Method", "Window size"] + [f'Round {r} time (s)' for r in range(1, MEASURE_ROUNDS+1)] + ['Average time (s)', f'Error Rate ({kNN}NN)']
    results = pd.DataFrame([], columns=columns)

    for i in tqdm(range(start, stop+1, step)): #For all the different warping windows calculate average running time
        # Add method to row
        full_method_name = methods[method]
        if method == 2 and pruning_method is not Ellipsis:
            full_method_name += f"_{pruning_method}"
        
        full_method_name += "-Univar"
        row = [full_method_name, i]


        # Calculate window based on method
        window_size = math.ceil(len(test_data[0]) * i / 100.0) #For cdtw i is interpreted as %, so we calculate what the percentage is of the time-series length
        if method == 0 or method == 4:
            window_size = i; # Use exact number for FastDTW's radius instead of percentage - the two are not equivalent 

        # Total time for all queries
        total_time = 0

        # Error rate for given window size
        error_rate = 0

        measured_rounds = 0
        profiled_measured_rounds = MEASURE_ROUNDS

        while (measured_rounds < MEASURE_ROUNDS and 
               measured_rounds < profiled_measured_rounds): #Amount of rounds where for each round we choose each row of query
            measured_rounds += 1

            correct_predictions = 0
            round_time = 0

            for query in tqdm(test_data, total=len(test_data), desc=f"Round {str(measured_rounds).rjust(3)}", leave=False): #Picks one row at a time as query and finds bet candidates of all rows below - OK since all pairs will be computed
                best_distances = np.full(kNN, np.inf)
                best_matches = np.full(kNN, None)

                for match in train_data:
                    if method == 0:  # python FastDTW = 0
                        start_time = time.time_ns()
                        distance, _ = fastdtw(query[1:], match[1:], radius = window_size)
                        elapsed_time = time.time_ns() - start_time
                        
                    elif method == 1:           # python cDtW = 1
                        start_time = time.time_ns()
                        try:
                            distance = dtw(query[1:], match[1:], step_pattern = 'symmetric1', dist_method="sqeuclidean",
                                        window_type = sakoeChibaWindow, window_args={'window_size': window_size}).distance #We use sqeuclidean as this is the measure used in fastdtw and it also results in a better accuracy for 1-NN
                        except:
                            pass
                        elapsed_time = time.time_ns() - start_time
                        
                    elif method == 2: #cDTW our implementation
                        start_time = time.time_ns()
                        distance = cdtw_own(query[1:], match[1:], sakoe_chiba, window_size).distance
                        elapsed_time = time.time_ns() - start_time
                        
                    elif method == 3: #FastDTW  our implementation
                        start_time = time.time_ns()
                        distance = fastdtw_own(query[1:], match[1:], window_size).distance
                        elapsed_time = time.time_ns() - start_time

                    round_time += elapsed_time
                    if measured_rounds == 1: # Only calculate accuracy in first round
                        any_smaller = np.argwhere(distance<best_distances)
                        if (distance and #None handling
                            any_smaller.size>0): #Exists larger in best distances larger than current distance
                            index_larger = np.argmax(best_distances) #Choose the largest distance and replace
                            best_distances[index_larger] = distance
                            best_matches[index_larger] = match
                
                if (measured_rounds == 1): #Calculate error rate
                    #Majority vote of best matches:
                    try:
                        best_matches_labels = [m[0] for m in best_matches]
                    except:
                        print(f"best_matches are: \n {best_matches}")
                        raise ValueError("Size of k in k-NN is larger than dataset size \n \
                                         or no candidates found with the given pruning_max")
                        
                    
                    #We use weighted k-NN:
                    inv = 1/best_distances
                    inv_sum = np.sum(inv)
                    weights = inv/inv_sum
                    
                    #Multiply each label with the given weight
                    
                    no_of_labels = len(set(test_data[:, 0]))
                    weighted = np.zeros((no_of_labels))

                    
                    for i in range(len(set(test_data[:, 0]))):
                        for k in range(kNN):
                            if best_matches_labels[k] == list(set(test_data[:, 0]))[i]:
                                weighted[i] += weights[k]
                        
                    prediction_index = np.argmax(weighted)
                    prediction = list(set(test_data[:, 0]))[prediction_index]
                    
                    if(prediction == query[0]):
                        #print("Guessed correctly")
                        correct_predictions += 1
                    else:
                        pass #print(f"Guessed incorrectly {prediction}, correct was: {query[0]}")
                
            total_time += round_time
            row.append(round_time / 1e9)
            
            if measured_rounds == 1:
                error_rate = (len(test_data) - correct_predictions) / len(test_data)

            current_average = total_time / measured_rounds / 1e9
            #To avoid excessive time usage of testing:
            if current_average < 3600:
                profiled_measured_rounds = max(3, int(math.ceil(3600 / current_average)))
            else:
                profiled_measured_rounds = 2

        row += [np.nan] * (MEASURE_ROUNDS - measured_rounds)
        row.append(total_time / measured_rounds / 1e9)

        row.append(error_rate)
        
        results.loc[len(results)] = row
        
    output_folder = Path(__file__).parent.parent / "output"
    output_folder.mkdir(exist_ok=True)
    
    output = output_folder / f'{dataset}-{full_method_name}-{start}-{stop}-{step}-{kNN}.csv'

    results.to_csv(output, index=False)

if __name__ == '__main__':
    # Extract arguments from command line
    try:
        dataset = sys.argv[1]
        method = int(sys.argv[2])
        start = int(sys.argv[3])
        print(start)
        stop = int(sys.argv[4])
        step = int(sys.argv[5])
        kNN = int(sys.argv[6])
        if len(sys.argv) > 7:
            pruning_method = int(sys.argv[7])
        else:
            pruning_method = Ellipsis
    except:
        raise ValueError("Wrong number of arguments")

    # Call main function with arguments
    _main(dataset=dataset, method=method, start=start, stop=stop, step=step, kNN=kNN, pruning_method=pruning_method)
