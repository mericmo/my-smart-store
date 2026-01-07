import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from utils.common import save_to_csv, init_logger

logger = init_logger("DailyPreprocessor")


class DailyPreprocessor:
    """
    负责原始数据的清洗、格式标准化、基础字段处理
    核心：数据过滤、每日聚合、缺失日期填充、字段标准化
    """

    def __init__(self, product_code: str, store_code: str, config: dict):
        self.product_code = product_code
        self.store_code = store_code
        self.config = config
        self.min_data_days = config["data"]["min_data_days"]
        self.sales_qty_min = config["data"]["sales_qty_min"]

    def preprocess_sales_data(self, hsr_df: pd.DataFrame) -> pd.DataFrame:
        """销售数据全流程预处理"""
        try:
            logger.info(f"开始预处理商品[{self.product_code}]门店[{self.store_code}]销售数据")

            # 1. 基础过滤（门店/商品/销量/渠道）
            filtered_df = self._filter_store_product(hsr_df)

            # 2. 处理退货（移除负值）
            cleaned_df = self._handle_negative_sales(filtered_df)

            # 3. 每日聚合（商品×日期级别）
            daily_agg_df = self._create_daily_aggregation(cleaned_df)

            # 4. 基础字段处理（折扣率、售价等）
            basic_processed_df = self._process_basic_sales_fields(daily_agg_df)

            # 5. 填补缺失日期（关键：保证时间序列连续）
            final_df = self._fill_missing_dates(basic_processed_df)

            # 保存预处理结果
            save_to_csv(
                final_df,
                filename=f"sales_preprocessed_{self.product_code}_{self.store_code}",
                sub_dir="preprocessed"
            )
            logger.info(f"完成销售数据预处理 | 最终数据行数: {len(final_df)}")
            return final_df

        except Exception as e:
            logger.error(f"销售数据预处理失败: {str(e)}", exc_info=True)
            raise

    def _filter_store_product(self, df: pd.DataFrame) -> pd.DataFrame:
        """过滤指定门店/商品/有效销量/线下渠道"""
        # 类型转换（避免编码匹配问题）
        df["商品编码"] = df["商品编码"].astype(str)
        df["门店编码"] = df["门店编码"].astype(str)

        # 过滤条件
        filters = (
                (df["门店编码"] == self.store_code) &
                (df["商品编码"] == self.product_code) &
                (df["销售数量"] > self.sales_qty_min) &
                (df["渠道名称"] == "线下销售")
        )
        filtered_df = df[filters].drop(
            columns=["会员id", "流水单号", "平台触点名称", "渠道名称", "小类编码"],
            errors="ignore"
        )

        # 数据量校验
        if len(filtered_df) < self.min_data_days:
            raise ValueError(
                f"商品[{self.product_code}]门店[{self.store_code}]有效数据不足{self.min_data_days}天 | 实际: {len(filtered_df)}条"
            )

        # 售价类型标准化
        filtered_df["售价"] = pd.to_numeric(filtered_df["售价"], errors="coerce").fillna(0)
        # 促销标记
        filtered_df["是否促销"] = (filtered_df["折扣类型"] != "n-无折扣促销").astype(int)

        logger.info(f"基础过滤完成 | 过滤后数据行数: {len(filtered_df)}")
        return filtered_df

    def _handle_negative_sales(self, df: pd.DataFrame) -> pd.DataFrame:
        """处理退货（负值销量）"""
        negative_qty_count = len(df[df["销售数量"] < 0])
        if negative_qty_count > 0:
            logger.warning(f"发现{negative_qty_count}条退货记录，已移除")
        return df[df["销售数量"] >= 0].copy()

    def _create_daily_aggregation(self, df: pd.DataFrame) -> pd.DataFrame:
        """按商品+日期聚合数据"""
        agg_config = {
            "销售数量": "sum",
            "销售金额": "sum",
            "销售净额": "sum",
            "折扣金额": "sum",
            "售价": "mean",
            "是否促销": "max"  # 当日是否有促销（0/1）
        }

        daily_agg_df = df.groupby(["商品编码", "商品名称", "日期"]).agg(agg_config).reset_index()

        # 平均售价（避免除0）
        daily_agg_df["平均售价"] = np.where(
            daily_agg_df["销售数量"] > 0,
            (daily_agg_df["销售金额"] - daily_agg_df["折扣金额"]) / daily_agg_df["销售数量"],
            0
        )

        logger.info(f"每日聚合完成 | 聚合后数据行数: {len(daily_agg_df)}")
        return daily_agg_df

    def _process_basic_sales_fields(self, df: pd.DataFrame) -> pd.DataFrame:
        """处理折扣率等基础字段"""
        # 折扣率（避免除0）
        df["实际折扣率"] = np.where(
            (df["销售金额"] > 0) & (df["折扣金额"] >= 0),
            1 - (df["折扣金额"] / df["销售金额"]),
            1.0  # 无折扣时默认1.0
        )
        # 字段类型校准
        for col in ["销售数量", "销售金额", "销售净额", "折扣金额"]:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

        return df

    def _fill_missing_dates(self, daily_sales):
        """填补缺失日期，创建完整的时间序列"""
        # 确保日期格式正确
        daily_sales['日期'] = pd.to_datetime(daily_sales['日期'])

        date_range = pd.date_range(
            start=daily_sales['日期'].min(),
            end=daily_sales['日期'].max(),
            freq='D'
        )

        # 为每个商品创建完整的时间序列
        all_products = daily_sales['商品编码'].unique()
        full_index = pd.MultiIndex.from_product(
            [all_products, date_range],
            names=['商品编码', '日期']
        )

        daily_sales_full = daily_sales.set_index(['商品编码', '日期']).reindex(full_index).reset_index()

        # ========== 修复核心：调整填充逻辑 ==========
        # 1. 简单数值列直接填充0
        simple_fill_cols = ['销售数量', '销售金额', '是否促销', '实际折扣率', '促销次数']
        for col in simple_fill_cols:
            if col in daily_sales_full.columns:
                daily_sales_full[col] = daily_sales_full[col].fillna(0)

        # 2. 按商品编码分组填充售价（基于完整DataFrame操作，而非Series）
        if '售价' in daily_sales_full.columns:
            # 先按商品编码分组，用每组的均值填充缺失值
            daily_sales_full['售价'] = daily_sales_full.groupby('商品编码')['售价'].transform(
                lambda x: x.fillna(x.mean())
            )
            # 如果还有缺失值（比如某商品所有售价都是NaN），用全局均值填充
            daily_sales_full['售价'] = daily_sales_full['售价'].fillna(daily_sales_full['售价'].mean())

        # 3. 填充商品名称（向前/向后填充）
        if '商品名称' in daily_sales_full.columns:
            daily_sales_full['商品名称'] = daily_sales_full.groupby('商品编码')['商品名称'].transform(
                lambda x: x.fillna(method='ffill').fillna(method='bfill')
            )

        return daily_sales_full

    def preprocess_weather_data(self, weather_df: pd.DataFrame) -> pd.DataFrame:
        """天气数据预处理"""
        try:
            logger.info("开始预处理天气数据")
            # 列名标准化
            weather_df = weather_df.rename(columns={
                "date": "日期",
                "high": "最高温度",
                "low": "最低温度"
            })
            # 日期类型转换
            weather_df["日期"] = pd.to_datetime(weather_df["日期"], errors="coerce")
            # 数值字段处理
            for col in ["最高温度", "最低温度"]:
                weather_df[col] = pd.to_numeric(weather_df[col], errors="coerce").fillna(method="ffill")
            # 计算平均温度
            weather_df["温度"] = (weather_df["最高温度"] + weather_df["最低温度"]) / 2
            # 降雨量默认0（实际场景建议对接真实API）
            weather_df["降雨量"] = weather_df.get("降雨量", 0)

            save_to_csv(weather_df, filename="weather_preprocessed", sub_dir="preprocessed")
            return weather_df[["日期", "温度", "最高温度", "最低温度", "降雨量"]]

        except Exception as e:
            logger.error(f"天气数据预处理失败: {str(e)}", exc_info=True)
            raise

    def preprocess_calendar_data(self, calendar_df: pd.DataFrame) -> pd.DataFrame:
        """日历数据预处理"""
        try:
            logger.info("开始预处理日历数据")
            # 列名标准化
            calendar_df = calendar_df.rename(columns={
                "date": "日期",
                "holiday_legal": "是否节假日"
            })
            # 日期类型转换
            calendar_df["日期"] = pd.to_datetime(calendar_df["日期"], errors="coerce")
            # 节假日字段校准
            calendar_df["是否节假日"] = calendar_df["是否节假日"].fillna(0).astype(int)
            # 节假日类型
            calendar_df["节假日类型"] = np.where(
                calendar_df["是否节假日"] == 1,
                "法定节假日",
                "普通日"
            )

            save_to_csv(calendar_df, filename="calendar_preprocessed", sub_dir="preprocessed")
            return calendar_df[["日期", "是否节假日", "节假日类型"]]

        except Exception as e:
            logger.error(f"日历数据预处理失败: {str(e)}", exc_info=True)
            raise