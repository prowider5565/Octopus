import concurrent.futures
import time


def my_function(arg):
    # Function logic
    result = arg * 2  # Example: multiply the argument by 2
    return result


# Using ThreadPoolExecutor for concurrent execution
with concurrent.futures.ThreadPoolExecutor(max_workers=100) as executor:
    # Submitting tasks with arguments (e.g., range(100) as arguments)
    futures = [executor.submit(my_function, i) for i in range(100)]

    # Retrieving the results
    results = [future.result() for future in concurrent.futures.as_completed(futures)]

# Use the results
print(results)
