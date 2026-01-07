import pandas as pd
from utils.common import load_config, init_logger
from data.daily_preprocessor import DailyPreprocessor
from data.daily_feature_store import DailyFeatureStore
from models.prophet_model import ProphetModel
from utils import calender_helper

# 初始化日志
logger = init_logger("main")


def main():
    """主流程：数据预处理 → 特征工程 → 模型训练 → 预测"""
    try:
        # 1. 加载配置
        config = load_config("config.yaml")
        logger.info("配置文件加载完成")

        # 2. 加载原始数据
        sales_raw_df = pd.read_csv(
            config["data"]["historical_transactions_path"],
            encoding="utf-8",
            parse_dates=["日期", "交易时间"],
            dtype={"商品编码": str, "门店编码": str, "商品小类": str}
        )
        weather_raw_df = pd.read_csv(config["data"]["weather_info_path"], encoding="utf-8")
        logger.info("原始数据加载完成")

        # 3. 批量处理多商品/门店
        for pair in config["data"]["product_store_pairs"]:
            product_code = pair["product_code"]
            store_code = pair["store_code"]
            logger.info(f"========== 开始处理商品[{product_code}]门店[{store_code}] ==========")

            # 4. 数据预处理
            preprocessor = DailyPreprocessor(product_code, store_code, config)
            # 销售数据预处理
            sales_processed_df = preprocessor.preprocess_sales_data(sales_raw_df)
            # 天气数据预处理
            weather_processed_df = preprocessor.preprocess_weather_data(weather_raw_df)
            # 日历数据预处理
            date_series = sales_processed_df["日期"].unique()
            calendar_raw_df = calender_helper.create_china_holidays_from_date_list(date_series=date_series)
            calendar_processed_df = preprocessor.preprocess_calendar_data(calendar_raw_df)

            # 5. 特征工程
            feature_store = DailyFeatureStore(config)
            final_features_df = feature_store.build_all_features(
                sales_processed_df,
                weather_processed_df,
                calendar_processed_df
            )

            # 6. 模型训练
            prophet_model = ProphetModel(config)
            model, prophet_df = prophet_model.train_prophet(final_features_df, target_col="销售数量")

            # 7. 预测
            forecast_df = prophet_model.predict_prophet(final_features_df)
            logger.info(f"预测结果:\n{forecast_df[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].round(2)}")

            logger.info(f"========== 完成处理商品[{product_code}]门店[{store_code}] ==========\n")

    except Exception as e:
        logger.error(f"主流程执行失败: {str(e)}", exc_info=True)
        raise


if __name__ == "__main__":
    main()