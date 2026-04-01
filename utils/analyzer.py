# utils/analyzer.py
import json
import os
from utils.logger import logger


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
        success_count = len(history)
        success_rate = (success_count / total_requests) * 100

        # 2. 平均指标
        avg_aoi = sum(item['aoi'] for item in history) / success_count
        avg_cost = sum(item['cost'] for item in history) / success_count

        # 3. 总迁移次数
        total_migrations = sum(1 for item in history if item.get('migrated', False))

        # 格式化输出对齐
        row = f"{algo_name:<20} | {success_rate:>6.1f}% ({success_count:<4}) | {avg_aoi:>12.3f} | {avg_cost:>12.3f} | {total_migrations:>15}"
        logger.info(row)

    logger.info("=" * 70 + "\n")
