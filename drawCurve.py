"""
计算卫星私有指标，绘制曲线
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from sgp4.api import Satrec, jday
from datetime import datetime, timedelta

# 地球半径 (km)
R_EARTH = 6378.137
# 地球引力常数 (km^3/s^2)
MU = 3.986004418e5
# 绘制的时间长度 (hours)
TIME_SPAN_HOURS = 8
# 绘制的时间步长 (minutes)
TIME_STEP_MINUTES = 20

def calculate_orbit_params(tle_line1, tle_line2, start_time, end_time, step_minutes=20):
    """
    根据TLE和时间范围，计算卫星轨道参数随时间的变化。
    
    参数:
    - tle_line1: TLE第一行
    - tle_line2: TLE第二行
    - start_time: 开始时间 (datetime对象)
    - end_time: 结束时间 (datetime对象)
    - step_minutes: 时间步长 (min)
    
    返回:
    - times: 时间列表 (datetime)
    - peri_heights: 近地点高度列表 (km)
    - apo_heights: 远地点高度列表 (km)
    - periods: 轨道周期列表 (min)
    - inclinations: 倾角列表 (deg)
    - RAANs: 升交点赤经列表 (deg)
    - arg_peris: 近地点幅角列表 (deg)
    - longitude: 星下点经度 (deg)
    - latitude: 星下点纬度 (deg)
    """
    # 创建卫星对象
    satellite = Satrec.twoline2rv(tle_line1, tle_line2)
    
    times = []
    peri_heights = []
    apo_heights = []
    inclinations = []
    periods = []
    RAANs = []
    arg_peris = []
    longitudes = []
    latitudes = []
    
    current_time = start_time
    while current_time <= end_time:
        # 计算儒略日
        jd, fr = jday(current_time.year, current_time.month, current_time.day,
                      current_time.hour, current_time.minute, current_time.second)
        
        # 递推位置和速度 (km, km/s)
        e, r, v = satellite.sgp4(jd, fr)
        if e != 0:  # 错误码不为0时跳过
            current_time += timedelta(minutes=step_minutes)
            continue
        
        r_vec = np.array(r)
        v_vec = np.array(v)
        
        # 半长轴 a (km)
        r_norm = np.linalg.norm(r_vec)
        v_norm = np.linalg.norm(v_vec)
        a = 1 / (2 / r_norm - v_norm**2 / MU)
        
        # 角动量 h 与偏心率 e
        h_vec = np.cross(r_vec, v_vec)
        e_vec = (np.cross(v_vec, h_vec) / MU - r_vec / r_norm)
        e = np.linalg.norm(e_vec)
        
        # 倾角 i (deg)
        i = np.arccos(h_vec[2] / np.linalg.norm(h_vec)) * 180 / np.pi
        
        # 近地点高度和远地点高度 (km)
        peri_height = a * (1 - e) - R_EARTH
        apo_height = a * (1 + e) - R_EARTH
        
        # 轨道周期 (min)
        period_sec = 2 * np.pi * np.sqrt(a**3 / MU)
        period_min = period_sec / 60

        # 升交点向量 n
        k_vec = np.array([0.0, 0.0, 1.0])
        if i > 1e-6 and i < 180 - 1e-6:  # 非赤道轨道
            n_vec = np.cross(k_vec, h_vec)
        else:
            n_vec = np.array([1.0, 0.0, 0.0])

        # 升交点赤经 RAAN (deg)
        raan = np.arccos(n_vec[0] / np.linalg.norm(n_vec)) * 180 / np.pi
        if n_vec[1] < 0:
            raan = 360 - raan

        # 近地点幅角 arg_peri (deg)
        if e > 1e-10:
            dot_n_e = np.dot(n_vec, e_vec)
            # 使用clip避免误差超出[-1,1]
            arg_peri = np.arccos(np.clip(dot_n_e / (np.linalg.norm(n_vec) * e), -1.0, 1.0)) * 180 / np.pi
            # 判断象限
            if np.dot(np.cross(n_vec, e_vec), h_vec) < 0:
                arg_peri = 360 - arg_peri
        else:
            arg_peri = 0

        # 规范化RAAN和arg_peri到[0,360)
        raan = raan % 360
        arg_peri = arg_peri % 360

        # 计算星下点经纬度
        x_ecf, y_ecf, z_ecf = teme_to_ecf(r_vec, jd, fr)
        lon = np.arctan2(y_ecf, x_ecf) * 180 / np.pi
        lon = (lon + 360) % 360  # 规范化到[0,360)

        p = np.sqrt(x_ecf**2 + y_ecf**2)
        a_wgs84 = 6378.137
        f = 1 / 298.257223563
        e2 = f * (2 - f)

        lat = np.arctan2(z_ecf, p * (1 - e2))
        for _ in range(5):
            sin_lat = np.sin(lat)
            N = a_wgs84 / np.sqrt(1 - e2 * sin_lat**2)
            lat = np.arctan2(z_ecf + N * e2 * sin_lat, p)

        lat = lat * 180 / np.pi

        times.append(current_time)
        peri_heights.append(peri_height)
        apo_heights.append(apo_height)
        inclinations.append(i)
        periods.append(period_min)
        RAANs.append(raan)
        arg_peris.append(arg_peri)
        longitudes.append(lon)
        latitudes.append(lat)
        
        current_time += timedelta(minutes=step_minutes)
    
    return times, peri_heights, apo_heights, periods, inclinations, RAANs, arg_peris, longitudes, latitudes


def teme_to_ecf(r_teme, jd, fr):
    """
    将 TEME (True Equator Mean Equinox) 坐标转换为 ECF (Earth-Centered Fixed) 坐标
    """
    # 计算 GMST (Greenwich Mean Sidereal Time)
    t = (jd + fr - 2451545.0) / 36525.0
    gmst_sec = 67310.54841 + (876600 * 3600 + 8640184.812866) * t + 0.093104 * t**2 - 6.2e-6 * t**3
    tau = 2 * np.pi
    gst = (gmst_sec * tau / 86400.0) % tau
    
    # 旋转矩阵应用 (绕 Z 轴旋转)
    x_ecf = r_teme[0] * np.cos(gst) + r_teme[1] * np.sin(gst)
    y_ecf = - r_teme[0] * np.sin(gst) + r_teme[1] * np.cos(gst)
    z_ecf = r_teme[2]
    
    return [x_ecf, y_ecf, z_ecf]

def plot_curves(times, peri, apo, period, inc, raan, arg_peri, lon, lat):
    """
    绘制轨道参数随时间变化的曲线，并保存为图片
    
    参数:
    - times: 时间列表
    - peri: 近地点高度列表 (km)
    - apo: 远地点高度列表 (km)
    - period: 轨道周期列表 (min)
    - inc: 倾角列表 (deg)
    - raan: 升交点赤经列表 (deg)
    - arg_peri: 近地点幅角列表 (deg)
    - lon: 星下点经度列表 (deg)
    - lat: 星下点纬度列表 (deg)
    """
    fig, axs = plt.subplots(2, 4, figsize=(20, 8))
    locator = mdates.HourLocator(interval=1)
    formatter = mdates.DateFormatter('%m/%d %H:%M')

    for ax in axs.flatten():
        ax.xaxis.set_major_locator(locator)
        ax.xaxis.set_major_formatter(formatter)
        ax.tick_params(axis='x', rotation=45)

    # 近地点高度
    axs[0, 0].plot(times, peri, label='Perigee Altitude(km)',
                   marker='o', markerfacecolor='none')
    axs[0, 0].set_title('Periapsis Height vs Time')
    axs[0, 0].set_xlabel('Time')
    axs[0, 0].set_ylabel('Height (km)')
    axs[0, 0].legend()
    
    # 远地点高度
    axs[0, 1].plot(times, apo, label='Apogee Altitude(km)', color='orange',
                   marker='o', markerfacecolor='none')
    axs[0, 1].set_title('Apoapsis Height vs Time')
    axs[0, 1].set_xlabel('Time')
    axs[0, 1].set_ylabel('Height (km)')
    axs[0, 1].legend()
    
    # 轨道周期
    axs[0, 2].plot(times, period, label='Orbital Period (min)', color='red',
                   marker='o', markerfacecolor='none')
    axs[0, 2].set_title('Orbital Period vs Time')
    axs[0, 2].set_xlabel('Time')
    axs[0, 2].set_ylabel('Period (min)')
    axs[0, 2].legend()
    
    # 倾角
    axs[0, 3].plot(times, inc, label='Inclination (deg)', color='green',
                   marker='o', markerfacecolor='none')
    axs[0, 3].set_title('Inclination vs Time')
    axs[0, 3].set_xlabel('Time')
    axs[0, 3].set_ylabel('Inclination (deg)')
    axs[0, 3].legend()

    # 升交点赤经
    axs[1, 0].plot(times, raan, label='RAAN (deg)', color='purple',
                   marker='o', markerfacecolor='none')
    axs[1, 0].set_title('RAAN vs Time')
    axs[1, 0].set_xlabel('Time')
    axs[1, 0].set_ylabel('RAAN (deg)')
    axs[1, 0].legend()

    # 近地点幅角
    axs[1, 1].plot(times, arg_peri, label='Argument of Perigee (deg)', color='brown',
                   marker='o', markerfacecolor='none')
    axs[1, 1].set_title('Argument of Perigee vs Time')
    axs[1, 1].set_xlabel('Time')
    axs[1, 1].set_ylabel('Argument of Perigee (deg)')
    axs[1, 1].legend()

    # 星下点经度
    axs[1, 2].plot(times, lon, label='Longitude (deg)', color='pink',
                   marker='o', markerfacecolor='none')
    axs[1, 2].set_title('Longitude vs Time')
    axs[1, 2].set_xlabel('Time')
    axs[1, 2].set_ylabel('Longitude (deg)')
    axs[1, 2].legend()

    # 星下点纬度
    axs[1, 3].plot(times, lat, label='Latitude (deg)', color='gray',
                   marker='o', markerfacecolor='none')
    axs[1, 3].set_title('Latitude vs Time')
    axs[1, 3].set_xlabel('Time')
    axs[1, 3].set_ylabel('Latitude (deg)')
    axs[1, 3].legend()
    
    plt.tight_layout()
    plt.savefig('output/orbit_parameters.png')
    plt.show()

def tle_epoch_to_datetime(epoch_str):
    """
    将TLE中的历元字符串转换为datetime对象。
    
    参数:
    - epoch_str: TLE历元字符串，如 "26097.58010940"
    
    返回:
    - datetime对象
    """
    # 解析历元：YYDDD.DDDDDD
    yy = int(epoch_str[:2])
    if yy >= 57:
        year = 1900 + yy
    else:
        year = 2000 + yy
    day_of_year = int(epoch_str[2:5])
    fractional_day = float(epoch_str[5:])
    
    # 计算日期
    start_of_year = datetime(year, 1, 1)
    days_delta = timedelta(days=day_of_year - 1 + fractional_day)
    epoch_datetime = start_of_year + days_delta
    
    return epoch_datetime

def extract_epoch_from_tle1(tle_line1: str) -> str:
    """
    从TLE第一行提取历元字符串（第19-32位，1-based）。

    例：
    "1 67718U 26024E   26097.58010940  .00420466 ..."
    返回 "26097.58010940"
    """
    return tle_line1[18:32].strip()

# 示例使用
if __name__ == "__main__":

    # 示例TLE数据（ISS）
    tle1 = "1 67718U 26024E   26097.58010940  .00420466  00000-0  46745-2 0  9994"
    tle2 = "2 67718  50.3141 234.6074 0168678  35.6225 325.5865 15.45852152  9078"

    # 从TLE第一行提取绘制的初始时间，并确定结束时间
    epoch_str = extract_epoch_from_tle1(tle1)
    epoch_datetime = tle_epoch_to_datetime(epoch_str)
    # print(f"Extracted epoch from TLE: {epoch_str} -> {epoch_datetime}")
    start_time = epoch_datetime
    end_time = start_time + timedelta(hours=TIME_SPAN_HOURS)

    # 按照设置的时间步长计算每个时刻的卫星参数
    times, peri, apo, period, inc, raan, arg_peri, lon, lat  = calculate_orbit_params(tle1, tle2, start_time, end_time, step_minutes=TIME_STEP_MINUTES)
    
    # 绘制图像并保存（可选择）
    # plot_curves(times, peri, apo, period, inc, raan, arg_peri, lon, lat)