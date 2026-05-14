# 电磁部分计算函数 -ljh
# pip install -U cdasws cdflib xarray pandas matplotlib

import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.gridspec import GridSpec

from cdasws import CdasWs
from cdasws.datarepresentation import DataRepresentation as dr


def fetch_omni_series(start_time,
                      end_time,
                      plot=False,
                      limit_to_last_3_months=True):
    """
    获取 OMNI 小时级数据，并返回 Kp、IMF 场强、密度、速度四个时间序列。

    Parameters
    ----------
    start_time : str or pandas.Timestamp
        用户请求的起始时间，例如 "2025-01-01 00:00:00" 或 "2025-01-01T00:00:00Z"
    end_time : str or pandas.Timestamp
        用户请求的结束时间
    plot : bool, optional
        是否绘图。默认 False，不绘图。
    limit_to_last_3_months : bool, optional
        是否在时间跨度超过 3 个月时，自动只保留结束时间前 3 个月的数据。
        默认 True。

    Returns
    -------
    kp : pandas.Series
        Kp 指数序列，索引为 UTC 时间。
        返回的是常见 Kp 数值形式，即 OMNI 的 KP1800 / 10.0。
    imf_strength : pandas.Series
        IMF 场强序列，单位 nT。
        采用 sqrt(Bx^2 + By^2 + Bz^2) 近似计算。
    density : pandas.Series
        太阳风数密度序列，单位 cm^-3。
        OMNI 中对应 N1800。
    speed : pandas.Series
        太阳风速度序列，单位 km/s。
        OMNI 中对应 V1800。

    Notes
    -----
    1. 数据集使用 OMNI2_H0_MRG1HR。
    2. 若 start_time 到 end_time 超过 3 个月，则默认只抓取 end_time 往前 3 个月的数据。
    3. 返回的四个序列具有相同时间索引，便于后续直接拼接或输入模型。
    """

    # -------------------------------------------------------------
    # 1) 屏蔽 xarray / cdflib 的时间精度提示
    # -------------------------------------------------------------
    warnings.filterwarnings(
        "ignore",
        message="Converting non-nanosecond precision datetime values to nanosecond precision.*",
        category=UserWarning
    )

    # -------------------------------------------------------------
    # 2) 时间解析与范围控制
    # -------------------------------------------------------------
    start_time = pd.to_datetime(start_time, utc=True)
    end_time = pd.to_datetime(end_time, utc=True)

    if end_time <= start_time:
        raise ValueError("end_time must be later than start_time.")

    # 若跨度超过 3 个月，则仅保留结束时间前 3 个月
    actual_start_time = start_time
    if limit_to_last_3_months:
        three_months_before_end = end_time - pd.DateOffset(months=3)
        if start_time < three_months_before_end:
            actual_start_time = three_months_before_end
            print(f"[INFO] Requested range > 3 months. "
                  f"Using only the last 3 months: "
                  f"{actual_start_time} to {end_time}")

    # CDAWeb 接口使用 ISO 字符串
    start_str = actual_start_time.strftime("%Y-%m-%dT%H:%M:%SZ")
    end_str = end_time.strftime("%Y-%m-%dT%H:%M:%SZ")

    # -------------------------------------------------------------
    # 3) 数据集与变量设置
    # -------------------------------------------------------------
    dataset = "OMNI2_H0_MRG1HR"
    variables = [
        "BX_GSE1800",
        "BY_GSM1800",
        "BZ_GSM1800",
        "V1800",
        "N1800",
        "KP1800",
    ]

    # -------------------------------------------------------------
    # 4) 下载数据
    # -------------------------------------------------------------
    cdas = CdasWs()
    status, ds = cdas.get_data(
        dataset,
        variables,
        start_str,
        end_str,
        dataRepresentation=dr.XARRAY
    )

    print("HTTP status:", status["http"]["status_code"])
    print("CDAS warnings:", len(status["cdas"]["warning"]))
    print("CDAS errors:", status["cdas"]["error"])

    if status["http"]["status_code"] != 200:
        raise RuntimeError(f"Download failed: HTTP {status['http']['status_code']}")

    if status["cdas"]["error"]:
        raise RuntimeError(f"CDAS returned errors: {status['cdas']['error']}")

    # -------------------------------------------------------------
    # 5) 转为 DataFrame
    # -------------------------------------------------------------
    df = ds[variables].to_dataframe().reset_index()

    # 自动寻找时间列
    time_candidates = [c for c in df.columns if c.lower().startswith("epoch")]
    if not time_candidates:
        raise KeyError(f"Cannot find time column. Existing columns: {df.columns.tolist()}")

    time_col = time_candidates[0]
    df = df.rename(columns={time_col: "time"})
    df["time"] = pd.to_datetime(df["time"], utc=True)

    # 排序并去重
    df = df.sort_values("time").drop_duplicates(subset="time").reset_index(drop=True)

    # -------------------------------------------------------------
    # 6) 构造输出序列
    # -------------------------------------------------------------
    # Kp：转成常见显示形式，如 0.7, 1.0, 1.3, ...
    kp = pd.Series(df["KP1800"].astype(float).values / 10.0,
                   index=df["time"],
                   name="Kp")

    # IMF 场强（近似）
    imf_strength = pd.Series(
        np.sqrt(
            df["BX_GSE1800"].astype(float).values**2 +
            df["BY_GSM1800"].astype(float).values**2 +
            df["BZ_GSM1800"].astype(float).values**2
        ),
        index=df["time"],
        name="IMF_strength"
    )

    # 密度
    density = pd.Series(df["N1800"].astype(float).values,
                        index=df["time"],
                        name="Density")

    # 速度
    speed = pd.Series(df["V1800"].astype(float).values,
                      index=df["time"],
                      name="Speed")

    # -------------------------------------------------------------
    # 7) 可选绘图
    # -------------------------------------------------------------
    if plot:
        fig = plt.figure(figsize=(16, 9), constrained_layout=True)
        gs = GridSpec(2, 3, figure=fig, height_ratios=[1.2, 1.0])

        ax_kp = fig.add_subplot(gs[0, :])
        ax_b  = fig.add_subplot(gs[1, 0], sharex=ax_kp)
        ax_n  = fig.add_subplot(gs[1, 1], sharex=ax_kp)
        ax_v  = fig.add_subplot(gs[1, 2], sharex=ax_kp)

        # Kp
        ax_kp.step(kp.index, kp.values, where="mid", linewidth=1.2)
        ax_kp.set_title("Kp Index", fontsize=13)
        ax_kp.set_ylabel("Kp")
        ax_kp.set_ylim(bottom=0)
        ax_kp.grid(True, alpha=0.3)

        # IMF 场强
        ax_b.plot(imf_strength.index, imf_strength.values, linewidth=1.0)
        ax_b.set_title("IMF Field Strength (approx.)", fontsize=12)
        ax_b.set_ylabel("|B| [nT]")
        ax_b.grid(True, alpha=0.3)

        # 密度
        ax_n.plot(density.index, density.values, linewidth=1.0)
        ax_n.set_title("Solar Wind Density", fontsize=12)
        ax_n.set_ylabel("N [cm$^{-3}$]")
        ax_n.grid(True, alpha=0.3)

        # 速度
        ax_v.plot(speed.index, speed.values, linewidth=1.0)
        ax_v.set_title("Solar Wind Speed", fontsize=12)
        ax_v.set_ylabel("V [km/s]")
        ax_v.grid(True, alpha=0.3)

        locator = mdates.AutoDateLocator(minticks=6, maxticks=12)
        formatter = mdates.ConciseDateFormatter(locator)

        for ax in [ax_kp, ax_b, ax_n, ax_v]:
            ax.xaxis.set_major_locator(locator)
            ax.xaxis.set_major_formatter(formatter)

        plt.setp(ax_kp.get_xticklabels(), visible=False)
        ax_b.set_xlabel("Time")
        ax_n.set_xlabel("Time")
        ax_v.set_xlabel("Time")

        fig.suptitle("OMNI Hourly Data Overview", fontsize=15)
        plt.show()

    return kp, imf_strength, density, speed