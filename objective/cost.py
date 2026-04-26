def compute_cost(node_list, task_chain, migration=False):
    """
    计算分布式部署的单步总成本。

    node_list: list[EdgeNode]，长度等于 task_chain.tasks 的数量。
    每个任务按其所在节点的单价和计算时长独立计费，迁移时叠加整条链的部署成本。
    """
    cost = sum(
        node.cost * (task.workload / node.compute_power)
        for node, task in zip(node_list, task_chain.tasks)
    )

    if migration:
        cost += sum(task.deployment_cost for task in task_chain.tasks)

    return cost
