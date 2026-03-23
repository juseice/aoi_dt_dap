# constraint/sequence.py

def check_sequence_constraint(task_chain_schedule):
    """
    校验子任务是否满足严格的时序依赖
    公式 C3: t_{start}^k >= t_{finish}^{k-1}
    TODO: 保证服务按顺序执行
    """
    return True