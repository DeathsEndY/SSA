# TLE数据获取&SGP4方法轨道递推

## 代码描述

包含基于Python的卫星轨道计算工具，使用TLE（Two-Line Element）数据和SGP4算法计算卫星在指定时刻的位置和速度，并将其转换为地球固定坐标系（ECF）。

主要功能包括：
- 从CelesTrak或Space-Track下载TLE数据
- 使用SGP4算法计算卫星在TEME坐标系下的位置和速度，并将TEME坐标（地球惯性系）转换为ECF坐标系（地球固定系）

## 依赖项

- Python 3.6+
- sgp4 (用于SGP4算法)
- requests (用于下载TLE数据)

## 使用方法

### 运行主程序

运行 `main.py` 来计算单个卫星的位置：

```bash
python main.py
```

这将输出指定卫星在给定UTC时刻的ECF坐标。

### 下载TLE数据

运行 `loadTLE.py` 来下载TLE数据：

```bash
python loadTLE.py
```

根据提示选择下载方式：
- 1: 批量获取全部轨道目标的TLE（需要Space-Track账号）
- 2: 批量获取在轨存活卫星的TLE（从CelesTrak）
- 3: 获取指定NORAD ID的TLE


### 批量处理TLE数据

使用 `load_tle_from_txt` 函数从TXT文件中加载TLE数据：

```python
from loadTLE import load_tle_from_txt

tle_data = load_tle_from_txt("data/active_tle.txt")
```

## 文件结构

- `main.py`: 主程序，演示单个卫星位置计算
- `sgp_wrapper.py`: SGP4算法封装，计算卫星位置和速度
- `frame.py`: 坐标系转换函数（TEME到ECF）
- `loadTLE.py`: TLE数据下载和处理函数
- `data/`: 存放TLE数据文件的目录，已经存放2026年3月17日获取的TLE数据，包含
  - `active_tle.txt`: 在轨存活卫星TLE数据
  - `all_tle.txt`: 全部轨道目标TLE数据
  - `part_tle.txt`: 指定卫星TLE数据（48274 68110 68103 68115 68116）

## 注意！

- 下载全部TLE数据需要Space-Track账号，地址https://www.space-track.org/，请在`loadTLE.py`中配置用户名和密码。目前函数中我已经隐去账号密码，需要注册后添加
- 时间使用UTC格式。
- 坐标单位：位置为km，速度为km/s。