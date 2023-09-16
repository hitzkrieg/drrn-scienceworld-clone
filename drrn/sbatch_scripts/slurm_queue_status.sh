#!/bin/bash

# Get a list of all queues
queues=$(sinfo -h -o "%P")

# Initialize an associative array to store the available node count for each queue
declare -A queue_node_counts

# Iterate over each queue and count available nodes
for queue in $queues; do
    # Use squeue to get the number of running jobs in the queue
    running_jobs=$(squeue -p $queue -h | wc -l)

    # Use sinfo to get the total number of nodes and subtract running jobs to get available nodes
    total_nodes=$(sinfo -N -p $queue -h -o "%D")
    available_nodes=$((total_nodes - running_jobs))

    # Store the available node count in the associative array
    queue_node_counts[$queue]=$available_nodes
done

# Print the results
echo "Queue           Available Nodes"
echo "-----------------------------"
for queue in $queues; do
    echo "$queue           ${queue_node_counts[$queue]}"
done