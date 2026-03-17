"""
坐标系转换函数
"""

import math

def teme_to_ecf(r_teme, jd, fr):
    """
    将 TEME (True Equator Mean Equinox) 坐标转换为 ECF (Earth-Centered Fixed) 坐标
    即考虑地球自转效果
    """
    # 计算 GMST (Greenwich Mean Sidereal Time) in degrees
    t = (jd + fr - 2451545.0) / 36525.0  # Julian centuries since J2000
    gmst_sec = 67310.54841 + (876600 * 3600 + 8640184.812866) * t + 0.093104 * t**2 - 6.2e-6 * t**3
    tau = 2 * math.pi
    gst = (gmst_sec * tau / 86400.0) % tau
    
    # 旋转矩阵应用 (绕 Z 轴旋转)
    x_ecf = r_teme[0] * math.cos(gst) + r_teme[1] * math.sin(gst)
    y_ecf = - r_teme[0] * math.sin(gst) + r_teme[1] * math.cos(gst)
    z_ecf = r_teme[2]
    
    return [x_ecf, y_ecf, z_ecf]