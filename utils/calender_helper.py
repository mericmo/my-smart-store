import chinese_calendar as cal
import pandas as pd
from datetime import date, datetime, timedelta
from typing import Optional, Union, List, Dict, Any
import numpy as np


def create_china_holidays_from_library(
        start_year: int = 2020,
        end_year: int = 2025,
        include_weekends_in_holidays: bool = False
) -> pd.DataFrame:
    """
    使用chinese_calendar库创建节假日数据并生成完整的时间序列DataFrame

    参数:
    ----------
    start_year : int, 默认 2020
        起始年份
    end_year : int, 默认 2025
        结束年份
    include_weekends_in_holidays : bool, 默认 False
        是否在法定节假日中包含周末

    返回:
    ----------
    pandas.DataFrame
        包含日期扩展信息和节假日数据的DataFrame
    """

    # 验证年份范围
    if start_year > end_year:
        raise ValueError("起始年份不能大于结束年份")

    # 创建日期范围
    start_date = date(start_year, 1, 1)
    end_date = date(end_year, 12, 31)
    date_range = pd.date_range(start=start_date, end=end_date, freq='D')

    # 初始化DataFrame
    df = pd.DataFrame({
        'date': date_range.strftime('%Y%m%d').astype(int),
        'ds': date_range
    })

    # 添加基本时间特征
    df['year'] = df['ds'].dt.year
    df['month'] = (df['ds'].dt.year * 100 + df['ds'].dt.month).astype(int)

    # 处理yearweek - 使用isocalendar
    iso_calendar = df['ds'].dt.isocalendar()
    df['yearweek'] = (iso_calendar.year * 100 + iso_calendar.week).astype(int)

    df['yearday'] = df['ds'].dt.dayofyear
    df['week'] = df['ds'].dt.dayofweek  # 0=Monday, 6=Sunday

    # 初始化标记列
    df['weekend'] = 0
    df['workday'] = 0
    df['holiday_legal'] = 0  # 法定节假日标记
    df['holiday_recess'] = 0  # 节假日调休标记
    df['holiday_name'] = ''

    # 获取所有法定节假日
    all_holidays = cal.get_holidays(start_date, end_date, include_weekends=include_weekends_in_holidays)

    # 标记法定节假日
    holiday_dates = [d for d in all_holidays]
    df.loc[df['ds'].dt.date.isin(holiday_dates), 'holiday_legal'] = 1

    # 标记周末
    df.loc[df['week'] >= 5, 'weekend'] = 1  # 周六(5)和周日(6)为周末

    # 标记工作日：使用chinese_calendar库判断是否是工作日
    df['workday'] = df['ds'].apply(lambda x: 1 if cal.is_workday(x.date()) else 0)

    # 节假日名称映射和窗口配置
    holiday_config = {
        "春节": {"window": range(-3, 4), "name": "春节"},
        "国庆": {"window": range(-2, 3), "name": "国庆节"},
        "劳动": {"window": range(-1, 2), "name": "劳动节"},
        "元旦": {"window": range(0, 2), "name": "元旦"},
        "清明": {"window": range(0, 2), "name": "清明节"},
        "端午": {"window": range(0, 2), "name": "端午节"},
        "中秋": {"window": range(0, 2), "name": "中秋节"},
    }

    # 添加节假日名称和窗口标记
    for holiday_date in all_holidays:
        holiday_name = cal.get_holiday_detail(holiday_date)[1]

        # 根据假期类型设置影响窗口
        window_range = None
        for key, config in holiday_config.items():
            if key in holiday_name:
                window_range = config["window"]
                display_name = config["name"]
                break

        if window_range is None:
            window_range = range(0, 2)
            display_name = holiday_name

        # 标记节假日窗口期
        for offset in window_range:
            window_date = holiday_date + timedelta(days=offset)
            idx = df[df['ds'].dt.date == window_date].index
            if len(idx) > 0:
                # 如果已经有节假日名称，不覆盖（避免多个节假日重叠）
                if df.loc[idx, 'holiday_name'].iloc[0] == '':
                    df.loc[idx, 'holiday_name'] = display_name
                df.loc[idx, 'holiday_recess'] = 2  # 节假日期间标记为2

    # 标记节假日调休：在节假日窗口期之外，但被调整为工作日的情况
    # 情况1：工作日但是周末（正常的周末上班调休）
    mask = (df['workday'] == 1) & (df['weekend'] == 1) & (df['holiday_recess'] == 0)
    df.loc[mask, 'holiday_recess'] = 1

    # 情况2：如果已经有节假日名称，但holiday_recess是0，设为2
    mask = (df['holiday_name'] != '') & (df['holiday_recess'] == 0)
    df.loc[mask, 'holiday_recess'] = 2

    # 添加节假日类型标记
    df['holiday_type'] = 'normal'
    df.loc[df['holiday_recess'] == 1, 'holiday_type'] = 'recess'  # 调休
    df.loc[df['holiday_recess'] == 2, 'holiday_type'] = 'holiday'  # 节假日

    # 重新排列列的顺序
    df = df[['date', 'ds', 'year', 'month', 'yearweek', 'yearday',
             'week', 'weekend', 'workday', 'holiday_legal',
             'holiday_recess', 'holiday_type', 'holiday_name']]

    # 重置索引
    df.reset_index(drop=True, inplace=True)

    return df


def create_china_holidays_from_date_list(
        date_series: Optional[Union[pd.Series, pd.DatetimeIndex, List]] = None,
        include_weekends_in_holidays: bool = False
) -> pd.DataFrame:
    """
    从日期列表生成中国节假日相关的特征数据

    Args:
        date_series: 日期序列，默认为None则生成一年的数据
        include_weekends_in_holidays: 是否将周末也标记为节假日

    Returns:
        DataFrame: 包含日期特征的数据框
    """
    # 如果没有提供日期序列，则生成默认的一年数据
    if date_series is None:
        raise ValueError("日期 date_series参数不能为None")
        # 保存原始输入，用于最后的结果合并
    if isinstance(date_series, (pd.Series, pd.DatetimeIndex)):
        date_range = pd.to_datetime(date_series)
    else:
        date_range = pd.Series(date_series)
    dates = []
    years = []
    months = []
    yearweeks = []
    yeardays = []
    weeks = []
    weekends = []
    workdays = []
    holiday_legals = []
    holiday_recesses = []
    holiday_names = []
    holiday_type = []
    for dt in date_range:
        current_date = dt.date()

        # 基础时间特征
        dates.append(current_date)
        years.append(current_date.year)
        months.append(current_date.month)
        yearweeks.append(int(dt.strftime('%U')))
        yeardays.append(current_date.timetuple().tm_yday)
        weeks.append(current_date.weekday())

        # 周末判断 (0=Monday, 6=Sunday)
        is_weekend = int(current_date.weekday() >= 5)
        weekends.append(is_weekend)

        # 使用chinese_calendar获取节假日信息
        try:
            is_workday, holiday_name = cal.get_holiday_detail(current_date)
        except Exception:
            # 如果获取不到节假日详情，设为默认值
            is_workday = None
            holiday_name = None

        # 工作日判断
        workdays.append(int(is_workday))

        # 检查是否是法定节假日
        is_legal_holiday = int(cal.is_holiday(current_date))
        is_legal_workday = int(cal.is_workday(current_date))

        # 法定节假日标记
        if is_legal_holiday:
            holiday_legals.append(1)
        else:
            holiday_legals.append(0)

        # 假期标记
        if holiday_name is not None:
            holiday_recesses.append(1)
            holiday_type.append(None)
            holiday_names.append(holiday_name.name if hasattr(holiday_name, 'name') else str(holiday_name))
        elif is_legal_holiday:
            holiday_recesses.append(1)
            holiday_type.append("holiday") # 节假日
            holiday_names.append('Public Holiday')
        elif include_weekends_in_holidays and is_weekend:
            holiday_recesses.append(1)
            holiday_type.append("recess")  # 调休
            holiday_names.append('Weekend')
        else:
            holiday_recesses.append(0)
            holiday_type.append(None)
            holiday_names.append(None)

    df = pd.DataFrame({
        'date': dates,
        'year': years,
        'month': months,
        'yearweek': yearweeks,
        'yearday': yeardays,
        'week': weeks,
        'weekend': weekends,
        'workday': workdays,
        'holiday_type': holiday_type,
        "is_legal_workday": is_legal_workday,
        'holiday_legal': holiday_legals,
        'holiday_recess': holiday_recesses,
        'holiday_name': holiday_names
    })
    # df.to_csv("china_calender_2024_2025_1.csv", encoding="utf-8-sig")
    return df







# 示例使用和测试
# if __name__ == "__main__":
#     print("=== 测试 create_china_holidays_from_library ===")
#     # 生成2024-2025年的数据
#     df_full = create_china_holidays_from_library(2024, 2025)
#     print(f"数据形状: {df_full.shape}")
#     print(f"列名: {list(df_full.columns)}")
#
#     # 查看2024年春节数据
#     spring_festival_2024 = df_full[
#         (df_full['year'] == 2024) &
#         (df_full['holiday_name'].str.contains('春节', na=False))
#         ].head(10)
#     print("\n2024年春节相关日期:")
#     print(spring_festival_2024[['date', 'holiday_name', 'holiday_type', 'workday', 'weekend']].to_string(index=False))
#
#     # 获取节假日统计
#     summary_2024 = get_holiday_summary(df_full, 2024)
#     print(f"\n2024年节假日统计:")
#     print(f"总天数: {summary_2024['total_days']}")
#     print(f"工作日: {summary_2024['workdays']}")
#     print(f"周末: {summary_2024['weekends']}")
#     print(f"法定节假日: {summary_2024['legal_holidays']}")
#     print(f"调休日: {summary_2024['recess_days']}")
#     print(f"节假日期间: {summary_2024['holiday_period_days']}")
#
#     print("\n=== 测试 create_china_holidays_from_date_list ===")
#     # 测试不同的输入类型
#     test_dates = pd.date_range('2024-01-01', periods=10)
#     result = create_china_holidays_from_date_list(test_dates)
#     print(f"输入日期数: {len(test_dates)}")
#     print(f"输出形状: {result.shape}")
#     print("\n结果示例:")
#     print(result.head())
#
#     # 测试列表输入
#     date_list = ['2024-05-01', '2024-05-02', '2024-05-03', '2024-05-04', '2024-05-05']
#     result_list = create_china_holidays_from_date_list(date_list)
#     print(f"\n列表输入结果:")
#     print(result_list[['date', 'holiday_name', 'workday', 'weekend']])
#
#     print("\n=== 测试 add_festival_features ===")
#     df_with_festival = add_festival_features(df_full[df_full['year'] == 2024].head(30).copy())
#     print(f"添加节日特征后列数: {len(df_with_festival.columns)}")
#     festival_cols = [col for col in df_with_festival.columns if 'festival' in col]
#     print(f"节日特征列: {festival_cols}")
#
#     # 保存示例数据
#     try:
#         df_full.to_csv('china_calendar_2024_2025.csv', index=False, encoding='utf-8-sig')
#         print("\n数据已保存到: china_calendar_2024_2025.csv")
#     except Exception as e:
#         print(f"\n保存文件时出错: {e}")