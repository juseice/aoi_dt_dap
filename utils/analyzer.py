# utils/analyzer.py
import json
import os
from utils.logger import logger
import math


def save_simulation_results(histories, filename="results/latest_simulation.json"):
    """
    将仿真结果持久化保存为 JSON 文件，方便后续单独分析或导入 Excel/Origin 画图。
    """
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    with open(filename, 'w', encoding='utf-8') as f:
        # 结果全是基础数据类型 (int, float, bool)，完美适配 JSON
        json.dump(histories, f, indent=4)
    logger.info(f"仿真原始数据已持久化至: {filename}")


def generate_summary_report(histories, total_requests):
    """
    生成并打印极其清晰的终端汇总报表
    """
    logger.info("\n" + "=" * 70)
    logger.info(f"仿真结果宏观汇总报告 (Total Requests: {total_requests})")
    logger.info("=" * 70)

    # 表头排版
    header = f"{'Algorithm':<20} | {'Success Rate':<15} | {'Avg AoI (s)':<12} | {'Avg Cost':<12} | {'Total Migrations':<15}"
    logger.info(header)
    logger.info("-" * 70)

    for algo_name, history in histories.items():
        if not history:
            logger.info(f"{algo_name:<20} | {'Failed/0%':<15} | {'-':<12} | {'-':<12} | {'-':<15}")
            continue

        # 1. 成功率 (部署成功的请求 / 总请求)
        successful_reqs = [h for h in history if not math.isinf(h['aoi'])]
        success_count = len(successful_reqs)

        success_rate = (success_count / total_requests) * 100 if total_requests > 0 else 0

        # 2. 平均指标
        if success_count > 0:
            avg_aoi = sum(h['aoi'] for h in successful_reqs) / success_count
            avg_cost = sum(h['cost'] for h in successful_reqs) / success_count
        else:
            avg_aoi = float('inf')
            avg_cost = float('inf')

        # 3. 总迁移次数
        total_mig = sum(1 for h in successful_reqs if h.get('migrated', False))

        logger.info(
            f"{algo_name:<22} | {success_rate:>6.1f}% ({success_count:<4}/{total_requests:<2}) | {avg_aoi:>10.3f} | {avg_cost:>10.3f} | {total_mig:>15}")

    logger.info("=" * 70 + "\n")
