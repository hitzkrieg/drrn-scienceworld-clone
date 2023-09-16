import subprocess

# Get a list of all queues
queues = subprocess.check_output("sinfo -h -o %P", shell=True, universal_newlines=True).split()

# Initialize a dictionary to store the available node count for each queue
queue_node_counts = {}

# Iterate over each queue and count available nodes
for queue in queues:
    # Use squeue to get the number of running jobs in the queue
    running_jobs = subprocess.check_output(f"squeue -p {queue} -h | wc -l", shell=True, universal_newlines=True)
    running_jobs = int(running_jobs.strip())

    # Use sinfo to get the total number of nodes and subtract running jobs to get available nodes
    total_nodes = subprocess.check_output(f"sinfo -N -p {queue} -h -o %D", shell=True, universal_newlines=True)
    total_nodes = int(total_nodes.strip())
    available_nodes = total_nodes - running_jobs

    # Store the available node count in the dictionary
    queue_node_counts[queue] = available_nodes

# Print the results
print("Queue           Available Nodes")
print("-------------------------------")
for queue in queues:
    print(f"{queue}           {queue_node_counts[queue]}")
