def compute_cost(node, compute_time, task_chain, migration=False):
    """
    计算部署在节点 node 上的单步总成本
    """
    # 基础运行成本
    cost = node.cost * compute_time

    # 如果发生迁移，需要加上数字孪生服务（即整条任务链）的状态迁移/重新部署成本
    if migration:
        # 将 task_chain 中所有子任务的 deployment_cost 累加起来
        migration_cost = sum(task.deployment_cost for task in task_chain.tasks)
        cost += migration_cost

    return cost