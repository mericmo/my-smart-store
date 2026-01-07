import pandas as pd
import numpy as np
from typing import List, Optional
from utils.common import save_to_csv, init_logger

logger = init_logger("DailyFeatureStore")


class DailyFeatureStore:
    """
    特征工程核心类：时间特征、滞后特征、滚动特征、趋势特征、交互特征等
    核心优化：向量化计算、防除0、分类特征编码、移除冗余特征
    """

    def __init__(self, config: dict):
        self.config = config
        self.lag_days = config["feature"]["lag_days"]
        self.rolling_windows = config["feature"]["rolling_windows"]
        self.discount_rate_min = config["feature"]["discount_rate_min"]

    def create_time_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """创建时间特征（移除季节冗余特征，保留核心时间维度）"""
        df = df.copy()
        df["日期"] = pd.to_datetime(df["日期"])
        time_features = {
            "星期": df["日期"].dt.dayofweek,
            "月份": df["日期"].dt.month,
            "是否周末": (df["日期"].dt.dayofweek >= 5).astype(int),
            "是否月末": (df["日期"].dt.day >= 25).astype(int),
            "季度": df["日期"].dt.quarter,
            "年份": df["日期"].dt.year,
            "日期序号": (df["日期"] - df["日期"].min()).dt.days
        }
        df = df.assign(**time_features)
        logger.info("时间特征创建完成")
        return df

    def create_lag_features(self, df, lags=[1, 3, 7, 14, 30]):
        """创建滞后特征 - 修复索引安全版本"""
        df = df.sort_values(['商品编码', '日期']).reset_index(drop=True)

        # 生成前面第x天的销量信息（用transform自动对齐索引）
        for lag in lags:
            df[f'销量_滞后{lag}天'] = (
                df.groupby('商品编码')['销售数量']
                .transform(lambda x: x.shift(lag))
            )
            df[f'销量_滞后{lag}天'] = df[f'销量_滞后{lag}天'].fillna(0)

        return df

    def create_rolling_features(self, df, windows=[7, 14, 30]):
        """创建滚动特征 - 修复索引不匹配版本"""
        # 确保按正确顺序排序并重置索引（关键：避免索引混乱）
        df = df.sort_values(['商品编码', '日期']).reset_index(drop=True)

        # 对每个窗口大小进行计算
        for window in windows:
            # 使用transform自动保持索引对齐（核心修复）
            df[f'销量_{window}天均值'] = (
                df.groupby('商品编码')['销售数量']
                .transform(lambda x: x.rolling(window, min_periods=1).mean())
            )

            df[f'销量_{window}天标准差'] = (
                df.groupby('商品编码')['销售数量']
                .transform(lambda x: x.rolling(window, min_periods=1).std())
            )

            df[f'销量_{window}天最大值'] = (
                df.groupby('商品编码')['销售数量']
                .transform(lambda x: x.rolling(window, min_periods=1).max())
            )

            df[f'销量_{window}天最小值'] = (
                df.groupby('商品编码')['销售数量']
                .transform(lambda x: x.rolling(window, min_periods=1).min())
            )

        # 填充缺失值（标准差可能产生NaN，需特殊处理）
        rolling_cols = [
            col for col in df.columns
            if any(f'{window}天' in col for window in windows)
        ]
        for col in rolling_cols:
            if '标准差' in col:
                df[col] = df[col].fillna(0)  # 标准差NaN填充0
            else:
                df[col] = df[col].fillna(0)

        return df

    def create_trend_features(self, df):
        """创建趋势特征 - 索引安全版本"""
        # 1. 首先按商品编码和日期排序并重置索引
        df = df.sort_values(['商品编码', '日期']).reset_index(drop=True)

        # 2. 7天趋势特征（用transform替代循环）
        df['销量_7天趋势'] = (
            df.groupby('商品编码')['销量_7天均值']
            .transform(lambda x: x.pct_change(periods=7))
        )

        # 3. 同比上周特征
        df['销量_同比上周'] = (
            df.groupby('商品编码')['销售数量']
            .transform(lambda x: x.shift(7))
        )

        # 填充缺失值
        df['销量_7天趋势'] = df['销量_7天趋势'].fillna(0)
        df['销量_同比上周'] = df['销量_同比上周'].fillna(0)

        return df

    def create_weather_enhanced_features(self, df: pd.DataFrame, weather_df: pd.DataFrame) -> pd.DataFrame:
        """创建天气增强特征（含分类编码）"""
        df = df.copy().merge(weather_df, on="日期", how="left")
        # 温度差
        df["温度差"] = df["最高温度"] - df["最低温度"]
        # 温度等级（分类特征，后续编码）
        df["温度等级"] = pd.cut(
            df["温度"],
            bins=[-np.inf, 0, 10, 20, 30, np.inf],
            labels=["严寒", "寒冷", "凉爽", "温暖", "炎热"],
            include_lowest=True
        )
        # 恶劣天气标记
        df["是否恶劣天气"] = (
                (df["温度"] < 10) | (df["温度"] > 35) | (df["降雨量"] > 25)
        ).astype(int)
        logger.info("天气增强特征创建完成")
        return df

    def create_calendar_enhanced_features(self, df: pd.DataFrame, calendar_df: pd.DataFrame) -> pd.DataFrame:
        """创建日历增强特征（含节假日连续天数）"""
        df = df.copy().merge(calendar_df, on="日期", how="left")
        # 节假日前后标记
        df["节假日前1天"] = df["是否节假日"].shift(1).fillna(0)
        df["节假日后1天"] = df["是否节假日"].shift(-1).fillna(0)
        # 节假日连续天数
        df = self._calculate_holiday_streak(df)
        logger.info("日历增强特征创建完成")
        return df

    def create_interaction_features(self, df):
        """创建交互特征 - 完整鲁棒版本"""
        # 前置检查：确保关键列存在且类型正确
        required_cols = ['实际折扣率', '是否周末', '是否月末', '销售数量']
        for col in required_cols:
            if col not in df.columns:
                raise ValueError(f"缺失必要列：{col}")

        # 1. 先清理并转换「实际折扣率」为浮点型，处理异常值
        df['实际折扣率'] = pd.to_numeric(df['实际折扣率'], errors='coerce')  # 转数值，非数值转NaN
        df['实际折扣率'] = df['实际折扣率'].fillna(0)  # NaN填充为0
        df['实际折扣率'] = df['实际折扣率'].clip(lower=0, upper=1)  # 限制折扣率范围0~1

        # 2. 安全折扣率（防止除以0，且确保类型一致）
        safe_discount = np.maximum(df["实际折扣率"].astype(float), float(self.discount_rate_min))

        # 3. 促销与时间的交互（修复除以0问题）
        df['促销_周末交互'] = np.where(
            df['实际折扣率'] != 0,  # 有折扣时计算
            df['是否周末'] * df['销售数量'] / safe_discount,  # 用安全折扣率避免除以0
            0  # 无折扣时赋值0
        )
        df['促销_月末交互'] = np.where(
            df['实际折扣率'] != 0,
            df['是否月末'] * df['销售数量'] / safe_discount,
            0
        )

        # 4. 天气与促销的交互（同理修复）
        if '是否恶劣天气' in df.columns:
            df['天气_促销交互'] = np.where(
                df['实际折扣率'] != 0,
                df['是否恶劣天气'] * df['销售数量'] / safe_discount,
                0
            )

        # 5. 确保交互特征是数值类型（兜底）
        interaction_cols = ['促销_周末交互', '促销_月末交互']
        if '天气_促销交互' in df.columns:
            interaction_cols.append('天气_促销交互')
        df[interaction_cols] = df[interaction_cols].astype(float).fillna(0)

        return df

    def encode_categorical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """分类特征编码（LabelEncoder，适配Prophet）"""
        df = df.copy()
        # 需要编码的分类列
        cat_cols = ["温度等级", "节假日类型"]
        for col in cat_cols:
            if col in df.columns:
                # 处理缺失值
                df[col] = df[col].fillna("未知").astype(str)
                # Label编码
                le = pd.factorize(df[col])[0]
                df[f"{col}_编码"] = le
                # 移除原分类列
                df = df.drop(columns=[col])
        logger.info("分类特征编码完成")
        return df

    def _calculate_holiday_streak(self, df: pd.DataFrame) -> pd.DataFrame:
        """计算节假日连续天数（优化版）"""
        if "是否节假日" not in df.columns:
            raise ValueError("DataFrame必须包含'是否节假日'列")

        # 按日期去重，保证计算准确性
        date_unique = df.drop_duplicates("日期序号")[["日期序号", "是否节假日"]].sort_values("日期序号")
        # 分组标记（状态变化时新建分组）
        date_unique["group"] = (date_unique["是否节假日"] != date_unique["是否节假日"].shift()).cumsum()
        # 计算每个节假日分组的长度
        holiday_group_size = date_unique[date_unique["是否节假日"] == 1].groupby("group").size()
        # 映射连续天数
        streak_map = {}
        for group_id, size in holiday_group_size.items():
            dates = date_unique[date_unique["group"] == group_id]["日期序号"].tolist()
            streak_map.update({d: size for d in dates})
        # 赋值到原数据
        df["节假日连续天数"] = df["日期序号"].map(streak_map).fillna(0).astype(int)
        return df

    def build_all_features(self, sales_df: pd.DataFrame, weather_df: pd.DataFrame,
                           calendar_df: pd.DataFrame) -> pd.DataFrame:
        """一站式构建所有特征（简化调用）"""
        try:
            logger.info("开始全流程特征构建")
            # 按顺序构建特征
            df = self.create_time_features(sales_df)
            df = self.create_lag_features(df)
            df = self.create_rolling_features(df)
            df = self.create_trend_features(df)
            df = self.create_weather_enhanced_features(df, weather_df)
            df = self.create_calendar_enhanced_features(df, calendar_df)
            df = self.create_interaction_features(df)
            df = self.encode_categorical_features(df)

            # 保存特征结果
            save_to_csv(df, filename="final_features", sub_dir="features")
            logger.info("全流程特征构建完成")
            return df

        except Exception as e:
            logger.error(f"特征构建失败: {str(e)}", exc_info=True)
            raise