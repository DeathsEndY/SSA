"""
比较 TEME 坐标系下的卫星位置和速度与 ECI 坐标系下的结果的差别是否大，确定是否可以视为一样
结果：不能视为一样，暂时使用skyfield库计算 ECI 坐标系下的位置和速度
"""

import numpy as np
from skyfield.api import load, EarthSatellite
from sgp4.api import Satrec

def compare_teme_eci():
    # 1. 选取不同轨道的典型卫星 TLE 数据 (以某历史历元为例)
    tles = {
        "ISS (LEO - 低轨)": (
            "1 25544U 98067A   23298.66567130  .00015509  00000+0  28373-3 0  9997",
            "2 25544  51.6416 295.3423 0005085 241.6702 249.2078 15.49845353421832"
        ),
        "NAVSTAR 72 (MEO - 中轨)": (
            "1 40534U 15013A   23298.53755498 -.00000045  00000-0  00000-0 0  9999",
            "2 40534  55.3054  18.6644 0002875  25.4057 334.6133  2.00557457 62450"
        ),
        "GOES 16 (GEO - 高轨)": (
            "1 41866U 16071A   23298.66699411 -.00000201  00000-0  00000-0 0  9998",
            "2 41866   0.0163  86.1105 0002660  47.8860 357.2148  1.00271501 25301"
        )
    }

    # 加载 Skyfield 的时间系统
    ts = load.timescale()

    for name, (line1, line2) in tles.items():
        print("="*60)
        print(f"卫星: {name}")
        print("="*60)
        
        # 初始化 sgp4 底层对象 (输出 TEME)
        satrec = Satrec.twoline2rv(line1, line2)
        
        # 初始化 skyfield 高层对象 (输出 ECI/GCRS)
        satellite = EarthSatellite(line1, line2, name, ts)
        
        # 获取 TLE 的历元时间，并以此为基准生成 3 个时间点：+0小时, +12小时, +24小时
        epoch = satellite.epoch
        times_to_eval = [
            ts.utc(epoch.utc.year, epoch.utc.month, epoch.utc.day, epoch.utc.hour + h)
            for h in [0, 12, 24]
        ]
        
        for t in times_to_eval:
            # ---------------------------------------------------
            # 1. 使用 sgp4 计算 TEME 系坐标
            # 关键点：SGP4 算法期望的时间输入是 UT1 对应的 Julian Date
            # Skyfield 的 t.whole 和 t.ut1_fraction 正好提供了最高精度的输入
            # ---------------------------------------------------
            err_code, r_teme, v_teme = satrec.sgp4(t.whole, t.ut1_fraction)
            
            if err_code != 0:
                print(f"SGP4 计算错误，错误码: {err_code}")
                continue
                
            r_teme = np.array(r_teme)
            v_teme = np.array(v_teme)
            
            # ---------------------------------------------------
            # 2. 使用 skyfield 计算 ECI 系坐标 (GCRS/J2000)
            # Skyfield 内部会先调 SGP4 算出 TEME，再应用岁差和章动矩阵转至 GCRS
            # ---------------------------------------------------
            geocentric = satellite.at(t)
            r_eci = geocentric.position.km
            v_eci = geocentric.velocity.km_per_s
            
            # ---------------------------------------------------
            # 3. 计算三维空间中的欧氏距离偏差
            # ---------------------------------------------------
            pos_diff_km = np.linalg.norm(r_eci - r_teme)
            vel_diff_kms = np.linalg.norm(v_eci - v_teme)
            
            # 打印结果
            print(f"▶ 时间点: {t.utc_strftime('%Y-%m-%d %H:%M:%S UTC')}")
            print(f"  [TEME] 位置: [{r_teme[0]:10.3f}, {r_teme[1]:10.3f}, {r_teme[2]:10.3f}] km")
            print(f"  [ECI]  位置: [{r_eci[0]:10.3f}, {r_eci[1]:10.3f}, {r_eci[2]:10.3f}] km")
            print(f"  -> 绝对位置偏差: {pos_diff_km:7.3f} km")
            print(f"  -> 绝对速度偏差: {vel_diff_kms:7.6f} km/s\n")

if __name__ == "__main__":
    compare_teme_eci()