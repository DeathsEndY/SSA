"""
碰撞概率计算函数。由Matlab程序改写  
v1.0 -- YSH 202604
v2.0 加入协方差估计 -- 202605
"""

import numpy as np
import datetime
from numpy.linalg import norm, eig
from scipy.integrate import dblquad
from sgp4.api import Satrec
from sgp4.api import jday
from sgp_wrapper import satellite_rv_at_time
from skyfield.api import load, EarthSatellite
import matplotlib.pyplot as plt

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

    # Combined position covariance
    covcomb = cov1[:3, :3] + cov2[:3, :3]

    # Relative encounter frame
    r = r1 - r2
    v = v1 - v2
    h = np.cross(r, v)

    y = v / norm(v)
    z = h / norm(h)
    x = np.cross(y, z)

    # Transform from ECI to relative encounter plane
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

def tle_to_rv(tle):
    """
    将 TLE 转换为位置和速度向量（r, v）。
    使用了 SGP4 库来解析 TLE 数据。
    """
    # 解析 TLE
    sat = Satrec.twoline2rv(tle[0], tle[1])

    # 获取当前时间的 Julian Date
    jd, fr = jday(2024, 6, 1, 0, 0, 0)

    # 计算位置和速度
    e, r, v = sat.sgp4(jd, fr)

    if e != 0:
        raise RuntimeError("Error in SGP4 propagation")

    return np.array(r), np.array(v)

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

if __name__ == "__main__":
    """
    示例：碰撞概率计算（r1,v1,cov1,r2,v2,cov2，HBR代入自己算例的数据）
    参数：sat1_tle, sat2_tle, 目标时刻
    输出：该时刻碰撞概率
    """

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

    print(f"在时刻 {t} 的Pc = {Pc:.12e}")
    print()

    time_offsets = np.arange(-1, 1.1, 0.1)  # 从-1到1，每0.1s
    pc_values = []

    for dt in time_offsets:
        current_t = t + datetime.timedelta(seconds=dt)
        pc = tle_to_pc_at_time(sat1_tle, sat2_tle, current_t, EMPIRICAL_ERRORS, HBR, RELTOL, HBR_TYPE)
        pc_values.append(pc)

    # 画图
    plt.figure(figsize=(10, 6))
    plt.plot(time_offsets, pc_values, marker='o')
    plt.xlabel('Time Offset (seconds)')
    plt.ylabel('Collision Probability (Pc)')
    plt.title('Collision Probability vs Time Offset')
    plt.grid(True)
    plt.show()