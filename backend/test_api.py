import time
import urllib.request
import json
import numpy as np


def test_api():
    print("=" * 60)
    print("  高炉温度反演系统 - 集成测试")
    print("=" * 60)

    print("\n[1/5] 测试健康检查接口...")
    resp = urllib.request.urlopen("http://localhost:8000/api/health")
    data = json.loads(resp.read().decode())
    print(f"  Status: {data['status']}")
    print(f"  TCP Server: {data['tcp_server']}")
    assert data["status"] == "ok", "健康检查失败"
    print("  ✓ 通过")

    time.sleep(2)

    print("\n[2/5] 测试统计数据接口...")
    resp = urllib.request.urlopen("http://localhost:8000/api/stats")
    data = json.loads(resp.read().decode())
    print(f"  帧数: {data['frame_count']}")
    print(f"  FPS: {data['fps']}")
    print(f"  最低温度: {data.get('min_temp', 'N/A')}")
    print(f"  最高温度: {data.get('max_temp', 'N/A')}")
    assert data["frame_count"] > 0, "没有接收到帧数据"
    print("  ✓ 通过")

    print("\n[3/5] 测试温度反演接口...")
    resp = urllib.request.urlopen("http://localhost:8000/api/temperature-frame")
    data = json.loads(resp.read().decode())
    print(f"  形状: {data['shape']}")
    print(f"  最低温度: {data['min_temp']:.2f} °C")
    print(f"  最高温度: {data['max_temp']:.2f} °C")
    print(f"  平均温度: {data['avg_temp']:.2f} °C")
    assert data["shape"] == [128, 128], "温度矩阵形状不正确"
    assert data["min_temp"] < data["max_temp"], "温度范围异常"
    print("  ✓ 通过")

    print("\n[4/5] 测试RBF插值接口...")
    resp = urllib.request.urlopen("http://localhost:8000/api/interpolated-temp")
    data = json.loads(resp.read().decode())
    print(f"  形状: {data['shape']}")
    print(f"  网格大小: {data['grid_size']}")
    print(f"  最低温度: {data['min_temp']:.2f} °C")
    print(f"  最高温度: {data['max_temp']:.2f} °C")
    assert data["shape"] == [256, 256], "插值矩阵形状不正确"
    print("  ✓ 通过")

    print("\n[5/5] 测试校准数据接口...")
    resp = urllib.request.urlopen("http://localhost:8000/api/calibration")
    data = json.loads(resp.read().decode())
    print(f"  形状: {data['shape']}")
    print(f"  最小吸收系数: {data['min']:.4f}")
    print(f"  最大吸收系数: {data['max']:.4f}")
    assert data["shape"] == [128, 128], "校准矩阵形状不正确"
    print("  ✓ 通过")

    print("\n" + "=" * 60)
    print("  所有测试通过！ ✓")
    print("=" * 60)


if __name__ == "__main__":
    test_api()
