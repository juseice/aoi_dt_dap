# constraint/resource.py

def check_memory_constraint(node, task_chain, simulator):
    """
    校验节点是否有足够的内存/资源来运行这条任务链
    公式 C2: \sum (m_j^{DT} + \sum m_{j,n,k}) <= M_i
    """
    prev_info = simulator.dt_placements.get(task_chain.id)
    current_deployed_node_id = prev_info['node_id'] if prev_info else None
    if current_deployed_node_id == node.id:
        return True
    # 假设 task_chain 有一个整体的内存需求，或者累加所有 task 的需求
    total_memory_required = sum(task.memory_requirement for task in task_chain.tasks)

    # TODO: 还可以加上 DT 实例本身的常驻内存
    # total_memory_required += task_chain.dt_base_memory

    return node.available_memory >= total_memory_required
