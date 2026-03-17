from sgp4.api import Satrec, jday
import datetime

def satellite_rv_at_time(sat_tle, time: datetime.datetime):
    """
    输入：
        sat_tle: 字典，包含satellite TLE信息：
            {
                "line1": "1 25544U ...",
                "line2": "2 25544 ...",
            }
        time: 一个 datetime.datetime 对象（UTC）
    返回：
        (r, v) - TEME (True Equator Mean Equinox) 坐标系: 
            r: 列表 [x, y, z] 表示位置（单位：km）
            v: 列表 [vx, vy, vz] 表示速度（单位：km/s）
    """
    # 用 TLE 行生成一个卫星记录对象
    satrec = Satrec.twoline2rv(sat_tle["line1"], sat_tle["line2"])

    # 将 datetime 转换为儒略日（Julian Date）格式
    jd, fr = jday(
        time.year, time.month, time.day,
        time.hour, time.minute, time.second + time.microsecond * 1e-6
    )

    # 调用sgp4算法获得状态
    error_code, r, v = satrec.sgp4(jd, fr)

    # 检查错误
    if error_code != 0:
        raise RuntimeError(f"SGP4 propagation failed with code {error_code}")

    return r, v, jd, fr