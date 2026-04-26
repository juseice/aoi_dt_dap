

def compute_task_delay(task, node):
    return task.workload / node.compute_power


def compute_chain_delay(task_chain, node):
    """All tasks on the same node (legacy / single-node placement)."""
    return sum(compute_task_delay(task, node) for task in task_chain.tasks)


def compute_chain_delay_distributed(task_chain, node_list):
    """Each task in the chain runs on its corresponding node in node_list."""
    return sum(
        compute_task_delay(task, node)
        for task, node in zip(task_chain.tasks, node_list)
    )
