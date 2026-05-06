"""
碰撞概率计算函数。由Matlab程序改写
v1.0 -- YSH
"""

import numpy as np
from numpy.linalg import norm, eig
from scipy.integrate import dblquad

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
    说明（Description）
    -------------------
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

# -------------------------------------------------------------------
# 碰撞概率计算（r1,v1,cov1,r2,v2,cov2，HBR代入自己算例的数据）
r1 = np.array([2134.70569701, -5160.26704941, 3811.54902792])
v1 = np.array([7.13392701, 0.97049, -2.67128116])

cov1 = np.array([
    [1.9760,  0.2348, -0.6463],
    [0.2348,  0.2819, -0.0879],
    [-0.6463, -0.0879, 0.4920]
])

r2 = np.array([2124.70569701, -5160.26704941, 3811.54902792])
v2 = 1.0e+04 * np.array([1.2323, -3.4336, -0.6110]) / 1e3
cov2 = np.array([
    [1.9760,  0.2348, -0.6463],
    [0.2348,  0.2819, -0.0879],
    [-0.6463, -0.0879, 0.4920]
])

HBR = 0.03
RelTol = 1e-8
HBRType = "circle"

Pc, Cp, isPD, isRem = pc2d_foster(
    r1, v1, cov1,
    r2, v2, cov2,
    HBR=HBR,
    RelTol=RelTol,
    HBRType=HBRType
)

print(f"Pc = {Pc:.12e}")