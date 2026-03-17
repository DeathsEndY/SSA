import sys
import datetime
from loadTLE import load_tle_from_txt
from frame import teme_to_ecf
from sgp_wrapper import satellite_rv_at_time
from sgp4.api import Satrec, jday

def main():
    """
    主函数, 给定单个卫星的tle数据，输出在ECF (Earth-Centered Fixed)地球固定坐标系的坐标
    """
    # 给定卫星tle
    sat_tle = {
        "name": "STARLINK-31278",
        "line1": "1 59163U 24044M   26076.63872110  .00003782  00000+0  14691-3 0  9990",
        "line2": "2 59163  42.9979  61.9278 0002262 277.0070  83.0524 15.27577766114780"
    }

    # 给定目标时刻，UTC 时间
    t = datetime.datetime(2026, 3, 20, 0, 0, 0)
    r_teme, _, jd, fr = satellite_rv_at_time(sat_tle, t)

    # 转换到地球固定坐标系
    r_ecf = teme_to_ecf(r_teme, jd, fr)
    print("地固系下卫星位置为 r (km):", r_ecf)


if __name__ == "__main__":
    sys.exit(main())