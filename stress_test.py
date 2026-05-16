"""
压力测试脚本 - 测试并发上传和 API 调用
"""
import asyncio
import aiohttp
import time
from pathlib import Path

# 测试配置
BASE_URL = "http://localhost:8000"
TEST_FILE = "test_resume.docx"  # 测试简历文件
TEST_JD = """
职位：Python 后端工程师
要求：
1. 3年以上 Python 开发经验
2. 熟悉 FastAPI/Django 等 Web 框架
3. 熟悉 MySQL/PostgreSQL 数据库
4. 了解 Docker 容器化部署
5. 有高并发系统开发经验优先
"""

async def upload_resume(session, file_path, jd_text, test_id):
    """异步上传简历"""
    start_time = time.time()

    try:
        # 转换为字符串
        file_path_str = str(file_path)

        # 准备表单数据
        data = aiohttp.FormData()

        # 根据文件类型设置 content_type
        content_type = 'application/vnd.openxmlformats-officedocument.wordprocessingml.document' if file_path_str.endswith('.docx') else 'application/pdf'
        filename = f'test_{test_id}' + ('.docx' if file_path_str.endswith('.docx') else '.pdf')

        data.add_field('file',
                      open(file_path_str, 'rb'),
                      filename=filename,
                      content_type=content_type)
        data.add_field('jd_text', jd_text)

        # 发送请求
        async with session.post(f"{BASE_URL}/upload", data=data) as response:
            result = await response.json()
            elapsed = time.time() - start_time

            return {
                'test_id': test_id,
                'status': 'success' if response.status == 200 else 'failed',
                'status_code': response.status,
                'elapsed': elapsed,
                'upload_id': result.get('upload_id'),
                'error': result.get('error')
            }
    except Exception as e:
        elapsed = time.time() - start_time
        return {
            'test_id': test_id,
            'status': 'error',
            'elapsed': elapsed,
            'error': str(e)
        }

async def run_concurrent_test(num_requests, file_path, jd_text):
    """运行并发测试"""
    print(f"\n{'='*60}")
    print(f"开始压力测试：{num_requests} 个并发请求")
    print(f"{'='*60}\n")

    start_time = time.time()

    # 创建异步 HTTP 会话
    async with aiohttp.ClientSession() as session:
        # 创建并发任务
        tasks = [
            upload_resume(session, file_path, jd_text, i+1)
            for i in range(num_requests)
        ]

        # 等待所有任务完成
        results = await asyncio.gather(*tasks)

    total_time = time.time() - start_time

    # 统计结果
    success_count = sum(1 for r in results if r['status'] == 'success')
    failed_count = sum(1 for r in results if r['status'] == 'failed')
    error_count = sum(1 for r in results if r['status'] == 'error')

    response_times = [r['elapsed'] for r in results]
    avg_time = sum(response_times) / len(response_times)
    min_time = min(response_times)
    max_time = max(response_times)

    # 打印结果
    print(f"\n{'='*60}")
    print(f"测试完成")
    print(f"{'='*60}")
    print(f"总请求数：{num_requests}")
    print(f"成功：{success_count} | 失败：{failed_count} | 错误：{error_count}")
    print(f"总耗时：{total_time:.2f}秒")
    print(f"平均响应时间：{avg_time:.2f}秒")
    print(f"最快响应：{min_time:.2f}秒")
    print(f"最慢响应：{max_time:.2f}秒")
    print(f"QPS（每秒请求数）：{num_requests/total_time:.2f}")

    # 显示失败详情
    if failed_count > 0 or error_count > 0:
        print(f"\n{'='*60}")
        print("失败详情：")
        print(f"{'='*60}")
        for r in results:
            if r['status'] != 'success':
                print(f"Test #{r['test_id']}: {r['status']} - {r.get('error', 'Unknown')}")

    return results

async def main():
    """主函数"""
    # 检查测试文件是否存在
    test_file = Path(TEST_FILE)
    if not test_file.exists():
        print(f"错误：测试文件 {TEST_FILE} 不存在")
        print("请在当前目录放置一个测试简历文件")
        return

    # 测试不同并发数
    test_cases = [
        {'num': 1, 'desc': '单个请求（基准测试）'},
        {'num': 3, 'desc': '3个并发请求'},
        {'num': 5, 'desc': '5个并发请求'},
        {'num': 10, 'desc': '10个并发请求（压力测试）'},
    ]

    all_results = {}

    for case in test_cases:
        print(f"\n\n{'#'*60}")
        print(f"测试场景：{case['desc']}")
        print(f"{'#'*60}")

        results = await run_concurrent_test(case['num'], test_file, TEST_JD)
        all_results[case['num']] = results

        # 等待一段时间再进行下一轮测试
        if case != test_cases[-1]:
            print("\n等待 5 秒后进行下一轮测试...")
            await asyncio.sleep(5)

    # 总结
    print(f"\n\n{'#'*60}")
    print("压力测试总结")
    print(f"{'#'*60}")
    print(f"{'并发数':<10} {'成功率':<10} {'平均响应时间':<15} {'QPS':<10}")
    print(f"{'-'*60}")

    for num, results in all_results.items():
        success_rate = sum(1 for r in results if r['status'] == 'success') / len(results) * 100
        avg_time = sum(r['elapsed'] for r in results) / len(results)
        qps = len(results) / sum(r['elapsed'] for r in results)
        print(f"{num:<10} {success_rate:.1f}%{'':<5} {avg_time:.2f}秒{'':<10} {qps:.2f}")

if __name__ == "__main__":
    print("="*60)
    print("简历评估系统 - 压力测试工具")
    print("="*60)
    print(f"目标服务器：{BASE_URL}")
    print(f"测试文件：{TEST_FILE}")
    print("\n注意：请确保服务器正在运行，且有足够的 OpenAI API 配额\n")

    asyncio.run(main())
