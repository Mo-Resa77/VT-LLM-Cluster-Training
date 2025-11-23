import subprocess

def get_node_info():
    """
    Executes `sinfo` to retrieve node-level information from SLURM.

    Returns:
        list[str]: List of sinfo output lines (excluding header).
    """
    result = subprocess.run(
        ['sinfo', '-N', '-o', '%P %n %C %t'],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )
    if result.returncode != 0:
        print("Error running sinfo:", result.stderr)
        return []

    lines = result.stdout.strip().split('\n')[1:]  # Skip the header
    return lines

def parse_node_info(lines):
    """
    Parses sinfo output into a dictionary of partitions and their idle CPU nodes,
    filtering out GPU and down nodes.

    Args:
        lines (list[str]): Output lines from sinfo -N.

    Returns:
        dict: {partition_name: [node_info, ...]}
    """
    partition_nodes = {}
    for line in lines:
        parts = line.split()
        if len(parts) < 4:
            continue

        partition = parts[0].strip('*').lower()
        node = parts[1]
        cpus_str = parts[2]  # Format: alloc/idle/other/total (e.g., 0/4/0/4)
        state = parts[3].lower()

        try:
            alloc, idle, other, total = map(int, cpus_str.split('/'))
        except ValueError:
            continue  # Skip malformed lines

        if 'down' in state:
            continue  # Skip unavailable nodes

        if idle > 0 and 'gpu' not in partition:  # Filter out GPU partitions
            if partition not in partition_nodes:
                partition_nodes[partition] = []
            partition_nodes[partition].append({
                'node': node,
                'idle': idle,
                'total': total
            })
    return partition_nodes

def recommend_partition(partition_nodes, max_nodes=20):
    """
    Selects the best CPU partition with the most idle CPUs (capped to max_nodes - 1).

    Args:
        partition_nodes (dict): Dictionary of available nodes per partition.
        max_nodes (int): Upper limit of how many nodes to request.

    Returns:
        tuple: (partition_name, list_of_selected_nodes) or None
    """
    if not partition_nodes:
        print("❌ No CPU partitions with idle nodes found.")
        return None

    # Sort partitions by total idle CPUs (descending)
    sorted_partitions = sorted(
        partition_nodes.items(),
        key=lambda kv: sum(n['idle'] for n in kv[1]),
        reverse=True
    )

    for partition, nodes in sorted_partitions:
        if len(nodes) >= 1:
            # Use all available - 1 nodes, but no more than max_nodes
            count = min(max(len(nodes) - 1, 1), max_nodes)
            return partition, nodes[:count]

    return None

def build_salloc_cmd(partition, nodes=1, ntasks=1, cpus=4, time="00:30:00", use_nodelist=False, nodelist=None):
    """
    Constructs a SLURM salloc command string.

    Args:
        partition (str): Partition name to request.
        nodes (int): Number of nodes to allocate.
        ntasks (int): Tasks per node.
        cpus (int): CPUs per task.
        time (str): Time limit in HH:MM:SS.
        use_nodelist (bool): Whether to include a specific --nodelist.
        nodelist (list[str]): List of node names (only if use_nodelist=True).

    Returns:
        str: Complete `salloc` command.
    """
    cmd = f"salloc --partition={partition} --nodes={nodes} --ntasks-per-node={ntasks} --cpus-per-task={cpus} --time={time}"
    if use_nodelist and nodelist:
        cmd += f" --nodelist={','.join(nodelist)}"
    return cmd

def main():
    """
    Main function to analyze SLURM cluster, pick best CPU partition,
    and generate the corresponding `salloc` command suggestion.
    """
    # Resource configuration
    cpus_per_task = 4
    ntasks_per_node = 1
    wall_time = "00:30:00"
    use_nodelist = False      # Set True to pin exact nodes (not recommended at large scale)
    max_nodes = 20            # Maximum number of CPU nodes to allocate

    print("🔍 Checking node-level SLURM CPU availability...\n")
    lines = get_node_info()
    partition_nodes = parse_node_info(lines)

    print("📊 CPU Partitions with idle nodes:")
    for p, nodes in partition_nodes.items():
        print(f"  {p:<10}: {len(nodes)} node(s) with idle CPUs")
        for n in nodes:
            print(f"    - {n['node']} | idle: {n['idle']} / total: {n['total']}")

    recommendation = recommend_partition(partition_nodes, max_nodes=max_nodes)
    if recommendation:
        partition, nodes = recommendation
        node_names = [n['node'] for n in nodes]

        print(f"\n✅ Recommended: Partition '{partition}' with {len(nodes)} node(s) (CPU only, capped at {max_nodes})")
        print(f"🧾 Nodes selected: {', '.join(node_names)}")

        print("\n💡 Suggested salloc:")
        print(build_salloc_cmd(
            partition=partition,
            nodes=len(nodes),
            ntasks=ntasks_per_node,
            cpus=cpus_per_task,
            time=wall_time,
            use_nodelist=use_nodelist,
            nodelist=node_names
        ))
    else:
        print("❌ No suitable CPU partition found with idle nodes.")

if __name__ == "__main__":
    main()
