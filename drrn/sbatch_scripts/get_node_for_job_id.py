import subprocess

def get_node_for_job_id(job_id):
    try:
        # Run the squeue command and capture the output
        output = subprocess.check_output(["squeue", "--noheader", "-j", str(job_id)])

        # Decode the byte-string output into a normal string
        output = output.decode("utf-8").strip()

        if not output:
            print(f"Job ID {job_id} not found in the queue.")
            return None

        # Split the output by whitespace and extract the node from the relevant column
        columns = output.split()
        node_index = 7  # Adjust this index based on the specific output format of squeue

        return columns[node_index]

    except subprocess.CalledProcessError:
        print("Error occurred while running squeue.")
        return None

if __name__ == "__main__":
    job_id = "8738359"  # Replace this with your actual job ID
    node = get_node_for_job_id(job_id)
    if node:
        print(f"The job {job_id} is assigned to node {node}.")
