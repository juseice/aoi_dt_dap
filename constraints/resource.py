# constraint/resource.py


def check_memory_constraint(node, task, task_chain_id, task_index, simulator):
    """
    校验节点是否有足够的内存运行任务链中的第 task_index 个任务。
    公式 C2: m_{j,n,k} <= M_i

    若该任务上一次就部署在同一节点，无需额外分配内存，直接通过。
    """
    prev_info = simulator.dt_placements.get(task_chain_id)
    if prev_info:
        prev_node_ids = prev_info['node_ids']
        if task_index < len(prev_node_ids) and prev_node_ids[task_index] == node.id:
            return True

    return node.available_memory >= task.memory_requirement
