"""
碰撞概率计算函数。基础计算部分由Matlab程序改写  
v1.0 Foster方法碰撞概率-- YSH 202604
v2.0 加入协方差估计
v2.1 加入时间范围内的最大概率搜索
"""

import numpy as np
import datetime
import time
from datetime import timezone
from numpy.linalg import norm, eig
from scipy.integrate import dblquad
from scipy.optimize import minimize_scalar
from skyfield.api import load, EarthSatellite

# RTN坐标系下的经验位置误差（km），后续可调整
EMPIRICAL_ERRORS = [1.0, 5.0, 1.0]
# 硬体半径（Hard Body Region, HBR），单位 km，后续可调整
HBR = 0.15
# 相对容差（Relative Tolerance）用于数值积分的收敛判断，后续可调整
RELTOL = 1e-10
# 硬体区域类型，'circle'、'square'、'squareEquArea'
HBR_TYPE = "square" 

# 对应 CovRemEigValClip.m
def cov_rem_eig_val_clip(Araw, Lclip=0.0, Lraw=None, Vraw=None):
    """
    功能：
    检测协方差是否非正定(NPD)，并通过特征值裁剪(Eigenvalue Clipping)
    将其修复为 PSD 或 PD。该方法来自：
    Hall et al., 2017 (NASA Conjunction SDK 标准做法)

    输入：
        Araw  : 原始协方差矩阵 (NxN)
        Lclip : 特征值裁剪下限 (通常用于Pc计算取 (1e-4*HBR)^2 )
        Lraw  : (可选) Araw 的特征值
        Vraw  : (可选) Araw 的特征向量

    输出：
        Lrem          : 修复后的特征值
        Lraw          : 原始特征值
        Vraw          : 特征向量
        pos_def_status: 原协方差正定性状态
        clip_status   : 是否发生裁剪
        Adet          : 修复后协方差行列式
        Ainv          : 修复后协方差逆矩阵
        Arem          : 修复后协方差矩阵
    """

    if Lclip < 0:
        raise ValueError("Lclip cannot be negative")

    # 若未提供特征分解，则自行计算
    if (Lraw is None) != (Vraw is None):
        raise ValueError("Lraw and Vraw must both be provided or both None")

    if Lraw is None:
        Lraw, Vraw = eig(Araw)
        Lraw = np.real(Lraw)
        Vraw = np.real(Vraw)

    # 判断原始协方差正定性
    pos_def_status = np.sign(np.min(Lraw))

    # 特征值裁剪
    Lrem = Lraw.copy()
    clip_status = np.min(Lraw) < Lclip
    if clip_status:
        Lrem[Lraw < Lclip] = Lclip

    # 计算修复后协方差的行列式与逆
    Adet = np.prod(Lrem)
    Ainv = Vraw @ np.diag(1.0 / Lrem) @ Vraw.T

    # 重构修复后的协方差矩阵
    if clip_status:
        Arem = Vraw @ np.diag(Lrem) @ Vraw.T
    else:
        Arem = Araw.copy()

    return Lrem, Lraw, Vraw, pos_def_status, clip_status, Adet, Ainv, Arem

# 对应 Pc2D_Foster.m
def pc2d_foster(r1, v1, cov1, r2, v2, cov2, HBR, RelTol=1e-8, HBRType="circle"):
    """
    按照 Foster 方法计算二维碰撞概率（2D Pc）。

    该函数支持三种不同的硬体区域（Hard Body Region, HBR）形状：
    - 'circle'           ：圆形
    - 'square'           ：正方形
    - 'squareEquArea'    ：与圆形面积等效的正方形

    函数既可以处理 3×3，也可以处理 6×6 协方差矩阵，
    但根据 2D Pc 的定义，实际只使用其中的 3×3 位置协方差部分。


    输入（Input）
    ------------
    r1 : 主目标在 ECI 坐标系下的位置向量 [1x3]，单位 km
    v1 : 主目标在 ECI 坐标系下的速度向量 [1x3]，单位 km/s
    cov1 : 主目标在 ECI 坐标系下的协方差矩阵 [3x3] 或 [6x6]

    r2 : 次目标在 ECI 坐标系下的位置向量 [1x3]，单位 km
    v2 : 次目标在 ECI 坐标系下的速度向量 [1x3]，单位 km/s
    cov2 : 次目标在 ECI 坐标系下的协方差矩阵 [3x3] 或 [6x6]

    HBR : 硬体半径（Hard Body Region）
    RelTol : 双重积分收敛的相对误差容限（通常设为 1e-08）
    HBRType : 硬体区域类型，'circle'、'square'、'squareEquArea'


    输出（Output）
    -------------
    Pc : 碰撞概率（Probability of Collision）

    Arem : 在相对遭遇坐标系中，投影到 x-z 遭遇平面后的组合协方差矩阵（也称 Cp）

    IsPosDef : 标志位，表示组合、降维及修复后的协方差是否仍存在负特征值
            若检测失败（存在负特征值），则不计算 Pc
            成功 = True，失败 = False

    IsRemediated : 标志位，表示组合并降维后的协方差是否经过特征值裁剪修复
    """

    # 联合位置协方差
    covcomb = cov1[:3, :3] + cov2[:3, :3]

    # 相对交会坐标系
    r = r1 - r2
    v = v1 - v2
    h = np.cross(r, v)

    y = v / norm(v)
    z = h / norm(h)
    x = np.cross(y, z)

    # 从 ECI 坐标系转换至相对交会平面
    eci2xyz = np.vstack((x, y, z))
    covcombxyz = eci2xyz @ covcomb @ eci2xyz.T

    # Project onto encounter plane (x-z)
    Cp = np.array([[1, 0, 0],
                   [0, 0, 1]]) @ covcombxyz @ np.array([[1, 0],
                                                         [0, 0],
                                                         [0, 1]])

    # Eigenvalue clipping remediation (NASA requirement)
    Lclip = (1e-4 * HBR) ** 2
    Lrem, _, _, _, is_remediated, Adet, Ainv, Arem = \
        cov_rem_eig_val_clip(Cp, Lclip)

    if np.min(Lrem) <= 0:
        raise RuntimeError("Non positive definite covariance in encounter plane")

    C = Ainv
    x0 = norm(r)
    z0 = 0.0

    # Integrand
    def integrand(z, x):
        return np.exp(-0.5 * (
            C[0, 0] * x * x +
            (C[0, 1] + C[1, 0]) * x * z +
            C[1, 1] * z * z
        ))

    AbsTol = 1e-13

    # Depending on the type of hard body region, compute Pc
    if HBRType.lower() == "circle":
        def z_upper(x):
            dx = x - x0
            if abs(dx) > HBR:
                return 0.0
            return np.sqrt(HBR ** 2 - dx ** 2)

        def z_lower(x):
            return -z_upper(x)

        integral = dblquad(
            integrand,
            x0 - HBR, x0 + HBR,
            lambda x: z_lower(x),
            lambda x: z_upper(x),
            epsabs=AbsTol,
            epsrel=RelTol
        )[0]

    elif HBRType.lower() == "square":
        integral = dblquad(
            integrand,
            x0 - HBR, x0 + HBR,
            lambda x: z0 - HBR,
            lambda x: z0 + HBR,
            epsabs=AbsTol,
            epsrel=RelTol
        )[0]

    elif HBRType.lower() == "squareequarea":
        half = np.sqrt(np.pi) / 2 * HBR
        integral = dblquad(
            integrand,
            x0 - half, x0 + half,
            lambda x: z0 - half,
            lambda x: z0 + half,
            epsabs=AbsTol,
            epsrel=RelTol
        )[0]

    else:
        raise ValueError("Unsupported HBRType")

    Pc = (1.0 / (2.0 * np.pi)) * (1.0 / np.sqrt(Adet)) * integral

    return Pc, Arem, True, is_remediated

def calculate_rtn_to_eci_rotation(r, v):
    """
    根据ECI坐标系下的位置（r）和速度（v）
    计算从RTN（径向、横向、法向）坐标系到ECI坐标系的旋转矩阵.
    """
    # 径向单位向量（从地心指向卫星的方向）
    u_R = r / np.linalg.norm(r)
    
    # 法向单位向量（垂直于轨道平面）
    h = np.cross(r, v)
    u_N = h / np.linalg.norm(h)
    
    # 横向单位向量（沿轨道方向，右手坐标系）
    u_T = np.cross(u_N, u_R)
    
    # 得到从 RTN系到 ECI系的旋转矩阵
    R = np.column_stack((u_R, u_T, u_N))
    return R

def generate_covariance_matrix(r1_eci, v1_eci, r2_eci, v2_eci, rtn_errors_km=[1.0, 5.0, 1.0]):
    """
    在 ECI 坐标系下生成3x3的位置协方差矩阵.
    """
    # 1. 在RTN坐标系下定义经验协方差矩阵，采用给定的固定经验值
    # 假设误差是以 km 为单位的标准差（sigma）
    # 方差 = sigma^2。假设没有交叉相关性。
    sigma_r, sigma_t, sigma_n = rtn_errors_km
    
    cov_RTN = np.array([
        [sigma_r**2, 0,          0],
        [0,          sigma_t**2, 0],
        [0,          0,          sigma_n**2]
    ])
    
    # 2. 计算从RTN到ECI的旋转矩阵
    R1 = calculate_rtn_to_eci_rotation(r1_eci, v1_eci)
    R2 = calculate_rtn_to_eci_rotation(r2_eci, v2_eci)
    
    # 3. 将RTN协方差矩阵旋转到ECI坐标系：P_ECI = R * P_RTN * R^T
    cov_eci1 = R1 @ cov_RTN @ R1.T
    cov_eci2 = R2 @ cov_RTN @ R2.T

    return cov_eci1, cov_eci2

def tle_to_pc_at_time(sat1_tle, sat2_tle, t, rtn_errors_km=[1.0, 5.0, 1.0], hbr=0.03, reltol=1e-8, hbr_type="circle"):
    """
    计算在给定时刻t，两颗卫星（sat1和sat2）的碰撞概率Pc。
    输入：
    sat1_tle, sat2_tle : 两颗卫星的TLE数据，格式为字典，包含 "name", "tle1", "tle2"
    t : 目标时刻，datetime对象
    rtn_errors_km : RTN坐标系下的经验位置误差（km），列表或数组，格式为 [sigma_r, sigma_t, sigma_n]
    输出：
    Pc : 碰撞概率
    """
    # # 先转成TEME系的方法，暂时不使用
    # # 1. 获取两卫星在 TEME 坐标系下的位置和速度
    # r1_teme = satellite_rv_at_time(sat1_tle, t)
    # v1_teme = satellite_rv_at_time(sat1_tle, t)
    # r2_teme = satellite_rv_at_time(sat2_tle, t)
    # v2_teme = satellite_rv_at_time(sat2_tle, t)
    # # 2. 将 TEME 坐标系下的位置和速度转换为 ECI 坐标系

    # 1. 使用skyfield库计算两卫星在 ECI (Earth-Centered Inertial，地球固定坐标系) 下的位置和速度
    
    # 加载 Skyfield 的时间系统
    ts = load.timescale()

    # 初始化 EarthSatellite 对象
    satellite1 = EarthSatellite(sat1_tle["tle1"], sat1_tle["tle2"], sat1_tle["name"], ts)
    satellite2 = EarthSatellite(sat2_tle["tle1"], sat2_tle["tle2"], sat2_tle["name"], ts)

    # 获取目标时刻的 Skyfield 时间对象
    t_sf = ts.utc(t.year, t.month, t.day, t.hour, t.minute, t.second)

    # 计算两卫星在目标时刻的 ECI 坐标和速度
    geocentric1 = satellite1.at(t_sf)
    geocentric2 = satellite2.at(t_sf)
    r1_eci = geocentric1.position.km
    v1_eci = geocentric1.velocity.km_per_s
    r2_eci = geocentric2.position.km
    v2_eci = geocentric2.velocity.km_per_s

    # 2. 计算两卫星在 ECI 坐标系下的位置协方差矩阵cov1和cov2
    cov1, cov2 = generate_covariance_matrix(r1_eci, v1_eci, r2_eci, v2_eci, rtn_errors_km)

    # 3. 计算碰撞概率 Pc
    Pc, _, is_pos_def, is_remediated = pc2d_foster(r1_eci, v1_eci, cov1, r2_eci, v2_eci, cov2, HBR=hbr, RelTol=reltol, HBRType=hbr_type)

    return Pc

def find_tca_and_max_pc(sat1_tle, sat2_tle, start_time, end_time, step_sec=10.0, 
                        rtn_errors_km=[1.0, 5.0, 1.0], hbr=0.15, reltol=1e-8, hbr_type="square"):
    """
    在给定的时间窗口内，以较低的时间复杂度寻找两颗卫星的最大碰撞概率及其发生时刻。
    三阶段精确寻优算法
    
    参数:
    sat1_tle, sat2_tle: TLE 字典数据
    start_time: 开始时刻 (datetime 对象)
    end_time: 结束时刻 (datetime 对象)
    step_sec: 粗搜步长(秒)，如10-30秒，足以捕捉低轨卫星交会。
    """
    ts = load.timescale()
    sat1 = EarthSatellite(sat1_tle["tle1"], sat1_tle["tle2"], sat1_tle["name"], ts)
    sat2 = EarthSatellite(sat2_tle["tle1"], sat2_tle["tle2"], sat2_tle["name"], ts)

    # 确保时间带有 UTC 时区，以便 Skyfield 批量处理
    if start_time.tzinfo is None:
        start_time = start_time.replace(tzinfo=timezone.utc)
    if end_time.tzinfo is None:
        end_time = end_time.replace(tzinfo=timezone.utc)

    total_seconds = (end_time - start_time).total_seconds()
    if total_seconds <= 0:
        raise ValueError("结束时间必须晚于开始时间")

    # ==========================================
    # 步骤 1: 向量化粗搜 (寻找最近距离粗略极小值)
    # ==========================================
    # 生成时间数组
    offsets = np.arange(0, total_seconds, step_sec)
    dt_list = [start_time + datetime.timedelta(seconds=float(s)) for s in offsets]
    t_array = ts.from_datetimes(dt_list)

    # 批量计算位置并求距离
    r1_array = sat1.at(t_array).position.km
    r2_array = sat2.at(t_array).position.km
    distances = np.linalg.norm(r1_array - r2_array, axis=0)

    # 找到距离最近的索引
    min_idx = np.argmin(distances)
    coarse_tca = dt_list[min_idx]
    coarse_min_dist = distances[min_idx]

    # 安全距离阈值提前阻断
    # LEO 极限相对速度不到 15km/s，使用 20km/s 作为安全裕度
    threshold_dist = 20.0 * step_sec
    if coarse_min_dist > threshold_dist:
        # 直接返回粗搜结果，此时碰撞概率在物理上视为 0
        coarse_tca_naive = coarse_tca.replace(tzinfo=None)
        return coarse_tca_naive, 0.0, coarse_min_dist

    # ==========================================
    # 步骤 2: 局部精细寻优 (寻找精确的 TCA 时刻)
    # ==========================================
    def distance_at_offset(offset):
        """目标函数：输入相对于 coarse_tca 的时间偏移(秒)，返回两星距离"""
        t_eval = coarse_tca + datetime.timedelta(seconds=offset)
        t_sf = ts.from_datetime(t_eval)
        r1 = sat1.at(t_sf).position.km
        r2 = sat2.at(t_sf).position.km
        return np.linalg.norm(r1 - r2)

    # 在粗搜点前后各扩展一个步长的范围内，寻找精确极小值点
    res_dist = minimize_scalar(
        distance_at_offset, 
        bounds=(-step_sec, step_sec), 
        method='bounded',
        options={'xatol': 1e-1} # 精度达到 0.1 秒即可
    )

    exact_tca = coarse_tca + datetime.timedelta(seconds=res_dist.x)
    exact_min_dist = res_dist.fun

    # ==========================================
    # 阶段 3: 以 TCA 为中心，精细寻优最大碰撞概率 (Max Pc)
    # ==========================================
    # 定义搜索窗口：在 TCA 前后 1.5 秒内搜索
    pc_search_window = 1.5
    
    def negative_pc_at_offset(offset):
            """目标函数：返回负的碰撞概率，以便使用极小值优化算法"""
            t_eval = exact_tca + datetime.timedelta(seconds=offset)
            t_eval_naive = t_eval.replace(tzinfo=None)
            try:
                # 调用你原有的计算单点Pc的函数
                pc = tle_to_pc_at_time(
                    sat1_tle, sat2_tle, t_eval_naive, 
                    rtn_errors_km=rtn_errors_km, hbr=hbr, reltol=reltol, hbr_type=hbr_type
                )
                return -pc  # scipy.optimize 只找最小值，所以加负号
            except Exception as e:
                # 如果某时刻协方差非正定抛出异常，视为概率0
                return 0.0

    # 寻找负Pc的极小值（即Pc的极大值）
    res_pc = minimize_scalar(
        negative_pc_at_offset,
        bounds=(-pc_search_window, pc_search_window),
        method='bounded',
        options={'xatol': 1e-1}  # 寻找概率峰值，0.1秒的精度已足够
    )

    max_pc_time = exact_tca + datetime.timedelta(seconds=res_pc.x)
    max_pc_val = -res_pc.fun
    max_pc_time_naive = max_pc_time.replace(tzinfo=None)
    exact_tca_naive = exact_tca.replace(tzinfo=None)

    # 顺便计算 Max Pc 发生时刻的几何距离，供参考
    dist_at_max_pc = distance_at_offset(res_dist.x + res_pc.x)

    return max_pc_time_naive, max_pc_val, dist_at_max_pc, exact_tca_naive, exact_min_dist


if __name__ == "__main__":
    
    """
    示例1：单碰撞概率计算（r1,v1,cov1,r2,v2,cov2，HBR代入自己算例的数据）
    参数：sat1_tle, sat2_tle, 目标时刻
    输出：该时刻碰撞概率
    """

    # 示例计算
    # 目标1：STARLINK-3809 - 52851
    sat1_tle = {
        "name": "STARLINK-3809",
        "tle1": "1 52851U 22062X   26089.96624807  .00000045  00000-0  20931-4 0  9998",
        "tle2": "2 52851  53.2169 288.6068 0001274  89.8700 270.2438 15.08839245209139"
    }

    # 目标2：OBJECT T - 43775
    sat2_tle = {
        "name": "OBJECT T - 43775",
        "tle1": "1 43775U 18099T   26090.08321991  .00005708  00000-0  35881-3 0  9991",
        "tle2": "2 43775  97.4298 136.7548 0004620 223.1343 136.9525 15.09611681400532"
    }

    # 目标时刻
    t = datetime.datetime(2026, 4, 1, 13, 35, 58)
    # 计算碰撞概率函数
    Pc = tle_to_pc_at_time(sat1_tle, sat2_tle, t, EMPIRICAL_ERRORS, HBR, RELTOL, HBR_TYPE)

    print("="*50 )
    print(f"在时刻 {t} 的碰撞概率 Pc = {Pc:.12e}")
    print("="*50)

    """
    示例2：在给定时间窗口内寻找最大碰撞概率及其发生时刻
    参数：sat1_tle, sat2_tle, start_time, end_time, step_sec, rtn_errors_km, hbr, reltol, hbr_type
    输出：最大碰撞概率发生的时刻，最大碰撞概率值，以及该时刻的最短相对距离
    """
    # 定义我们要搜索的时间窗口 (例如：2026年4月1日 13:00 到 14:00 这一个小时内)
    # 记录程序用时
    t1 = time.time()
    start_time = datetime.datetime(2026, 4, 1, 8, 0, 0)
    end_time = datetime.datetime(2026, 4, 1, 20, 0, 0)

    print(f"开始搜索分析窗口: {start_time} 到 {end_time} ...")
    
    # 调用优化算法寻找最大概率
    max_pc_time, max_pc, dist_at_max_pc, exact_tca, exact_min_dist = find_tca_and_max_pc(
        sat1_tle, sat2_tle, start_time, end_time, 
        step_sec=10.0, 
        rtn_errors_km=EMPIRICAL_ERRORS, 
        hbr=HBR, 
        reltol=RELTOL, 
        hbr_type=HBR_TYPE
    )

    t2 = time.time()
    print("="*50)
    print(f"分析结果完成！总用时: {t2 - t1:.2f} 秒")
    print(f"▶ 纯几何最短距离时刻 (TCA) : {exact_tca}")
    print(f"  -> 此时几何距离        : {exact_min_dist:.5f} km")
    print("-" * 60)
    print(f"▶ 真正的最大碰撞概率时刻   : {max_pc_time}")
    print(f"  -> 偏离几何 TCA 时间带   : {(max_pc_time - exact_tca).total_seconds():.4f} 秒")
    print(f"  -> 该时刻的几何距离      : {dist_at_max_pc:.5f} km")
    print(f"  -> ⭐ 最大碰撞概率 (Max Pc): {max_pc:.12e}")
    print("="*50)