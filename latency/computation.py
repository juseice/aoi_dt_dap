

def compute_task_delay(task, node):
    return task.workload / node.compute_power


def compute_chain_delay(task_chain, node):
    total_delay = 0

    for task in task_chain.tasks:
        total_delay += compute_task_delay(task, node)

    return total_delay
