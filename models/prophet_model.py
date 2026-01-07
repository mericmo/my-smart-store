from prophet import Prophet
from prophet.plot import plot_plotly, plot_components_plotly
from prophet.diagnostics import cross_validation, performance_metrics
import plotly.io as pio
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from lunar_python import Lunar, Solar  # 需安装：pip install lunar-python
from datetime import datetime
from utils.common import save_to_csv, save_model, init_logger

logger = init_logger("ProphetModel")
pio.templates.default = "plotly_white"


class ProphetModel:
    """
    Prophet模型封装：训练、预测、评估、可视化、持久化
    核心优化：动态节假日、完整回归因子、时间序列CV、模型评估
    """

    def __init__(self, config: dict):
        self.config = config
        self.model_params = config["prophet"]
        self.model = None
        self.scaler = StandardScaler()
        self.scaler_columns = config["feature"]["scaler_columns"]
        self.evaluation_metrics = None

    def train_prophet(self, features_df: pd.DataFrame, target_col: str = "销售数量") -> tuple:
        """训练Prophet模型（含时间序列交叉验证）"""
        try:
            logger.info("开始训练Prophet模型")
            # 1. 数据准备（适配Prophet格式）
            prophet_df = self._prepare_prophet_data(features_df, target_col)
            if len(prophet_df) < self.config["data"]["min_data_days"]:
                raise ValueError(
                    f"训练数据不足 | 仅{len(prophet_df)}天（需至少{self.config['data']['min_data_days']}天）")

            # 2. 动态调整模型参数（根据数据时长）
            data_duration = (prophet_df["ds"].max() - prophet_df["ds"].min()).days
            adjusted_params = self._adjust_params_by_data_duration(data_duration)

            # 3. 初始化模型 + 添加节假日 + 添加回归因子
            self.model = Prophet(**adjusted_params)
            self._add_chinese_holidays(self.model, prophet_df["ds"].min().year, prophet_df["ds"].max().year)
            self._add_regressors(self.model, features_df)

            # 4. 训练模型
            self.model.fit(prophet_df)
            logger.info("Prophet模型训练完成")

            # 5. 时间序列交叉验证（评估模型）
            self._cross_validate_model()

            # 6. 保存模型
            save_model(self.model, f"prophet_model_{data_duration}days")

            return self.model, prophet_df

        except Exception as e:
            logger.error(f"模型训练失败: {str(e)}", exc_info=True)
            raise

    def _prepare_prophet_data(self, features_df, target_col):
        """
        准备Prophet标准格式数据（包含所有回归因子）
        """
        df = features_df.copy()

        # 1. 基础格式转换
        df['ds'] = pd.to_datetime(df['日期'])
        df['y'] = df[target_col].clip(lower=0)  # 销量非负

        # 2. 标准化数值特征
        df = self._standardize_numeric_features(df)

        # 3. 收集所有需要的列（时间列+目标列+所有回归因子）
        # 先获取已添加的回归因子列表
        regressors = [
            '温度', '温度差', '是否恶劣天气',
            '是否周末', '是否节假日', '是否促销',
            '销量_7天均值', '促销_周末交互'
        ]
        # 筛选出数据中存在的列
        available_cols = ['ds', 'y'] + [r for r in regressors if r in df.columns]

        # 4. 最终数据（去重+去空）
        prophet_df = df[available_cols].drop_duplicates(subset=['ds']).dropna(subset=['ds', 'y'])

        if len(prophet_df) == 0:
            raise ValueError("Prophet数据为空，请检查输入数据")

        logger.info(
            f"Prophet数据准备完成 | 数据行数: {len(prophet_df)} | 包含回归因子: {[c for c in available_cols if c not in ['ds', 'y']]}")
        return prophet_df

    def _adjust_params_by_data_duration(self, duration_days: int) -> dict:
        """根据数据时长动态调整参数"""
        adjusted_params = self.model_params.copy()
        if duration_days < 365:
            adjusted_params["yearly_seasonality"] = False  # 不足1年禁用年季节性
            adjusted_params["changepoint_prior_scale"] = 0.01  # 降低趋势灵敏度
        return adjusted_params

    def _add_chinese_holidays(self, model: Prophet, start_year: int, end_year: int) -> None:
        """添加中国节假日（动态计算农历节日，如春节/端午/中秋）"""
        holidays = []
        for year in range(start_year, end_year + 1):
            # 公历节日
            holidays.extend([
                (f"{year}-01-01", "元旦"),
                (f"{year}-05-01", "劳动节"),
                (f"{year}-10-01", "国庆节"),
                (f"{year}-10-02", "国庆节"),
                (f"{year}-10-03", "国庆节"),
            ])
            # 农历节日（动态转换）
            holidays.extend([
                (self._lunar_to_solar(year, 1, 1), "春节"),  # 正月初一
                (self._lunar_to_solar(year, 5, 5), "端午节"),  # 五月初五
                (self._lunar_to_solar(year, 8, 15), "中秋节")  # 八月十五
            ])

        # 构建节假日DataFrame
        holiday_df = pd.DataFrame({
            "holiday": [h[1] for h in holidays],
            "ds": pd.to_datetime([h[0] for h in holidays]),
            "lower_window": -2,  # 节前2天
            "upper_window": 2  # 节后2天
        }).dropna()

        model.holidays = holiday_df
        logger.info(f"添加{len(holiday_df)}个节假日到模型")

    def _lunar_to_solar(self, year: int, month: int, day: int) -> str:
        """农历转公历"""
        lunar = Lunar(year, month, day, 12, 30, 30)
        solar = lunar.getSolar()
        return f"{solar.getYear()}-{solar.getMonth():02d}-{solar.getDay():02d}"

    def _add_regressors(self, model: Prophet, features_df: pd.DataFrame) -> None:
        """添加回归因子（天气、促销、时间特征等）"""
        # 核心回归因子列表
        regressors = [
            "温度", "温度差", "是否恶劣天气", "是否周末", "是否节假日",
            "是否促销", "销量_7天均值", "促销_周末交互"
        ]
        # 批量添加回归因子
        for reg in regressors:
            if reg in features_df.columns:
                model.add_regressor(reg)
        logger.info(f"添加{len(regressors)}个回归因子到模型")

    def _standardize_numeric_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """标准化数值特征（避免量纲影响）"""
        df = df.copy()
        for col in self.scaler_columns:
            if col in df.columns:
                df[col] = self.scaler.fit_transform(df[[col]])
        return df

    def _cross_validate_model(self):
        """
        时间序列交叉验证
        """
        if not self.model:
            raise ValueError("模型未训练，无法交叉验证")

        try:
            # 交叉验证配置（适配短数据）
            data_length = len(self.model.history)
            initial_days = max(30, data_length // 4)  # 初始窗口至少30天
            period_days = min(30, data_length // 10)  # 滚动步长
            horizon_days = min(14, data_length // 20)  # 预测窗口

            # 执行交叉验证
            cv_results = cross_validation(
                self.model,
                initial=f"{initial_days} days",
                period=f"{period_days} days",
                horizon=f"{horizon_days} days",
                parallel="processes"  # 加速计算
            )

            # 计算评估指标（自动跳过无效指标）
            self.evaluation_metrics = performance_metrics(cv_results)

            # ========== 核心修复：指标存在性检查 + 鲁棒指标计算 ==========
            metrics_summary = {}
            # 1. 基础指标（必存在）
            metrics_summary["mae"] = self.evaluation_metrics.get("mae", 0).mean()
            metrics_summary["rmse"] = self.evaluation_metrics.get("rmse", 0).mean()
            metrics_summary["mse"] = self.evaluation_metrics.get("mse", 0).mean()

            # 2. 百分比类指标（容错处理）
            if "mape" in self.evaluation_metrics.columns:
                metrics_summary["mape"] = self.evaluation_metrics["mape"].mean()
            else:
                # 替换为SMAPE（对称MAPE，避免除以0）
                cv_results["smape"] = 2 * np.abs(cv_results["yhat"] - cv_results["y"]) / (
                            np.abs(cv_results["yhat"]) + np.abs(cv_results["y"]) + 1e-6)
                metrics_summary["smape"] = cv_results["smape"].mean() * 100  # 转百分比

            # 3. R2分数（补充）
            metrics_summary["r2"] = r2_score(cv_results["y"], cv_results["yhat"])

            # 打印友好的评估结果
            logger.info("===== 模型交叉验证结果 =====")
            for metric, value in metrics_summary.items():
                if "%" in metric or metric in ["mape", "smape"]:
                    logger.info(f"{metric.upper()}: {value:.2f}%")
                else:
                    logger.info(f"{metric.upper()}: {value:.2f}")
            logger.info("============================")

            # 保存评估结果
            save_to_csv(self.evaluation_metrics, filename="cv_evaluation_metrics", sub_dir="evaluation")

        except Exception as e:
            logger.error(f"交叉验证失败: {str(e)}", exc_info=True)
            # 降级处理：使用简单的训练集评估
            y_true = self.model.history["y"]
            y_pred = self.model.predict(self.model.history)["yhat"]
            metrics_summary = {
                "mae": mean_absolute_error(y_true, y_pred),
                "rmse": np.sqrt(mean_squared_error(y_true, y_pred)),
                "r2": r2_score(y_true, y_pred)
            }
            logger.warning(f"降级为训练集评估: {metrics_summary}")

    def predict_prophet(self, features_df: pd.DataFrame) -> pd.DataFrame:
        """预测未来销量（支持多期预测）"""
        try:
            if not self.model:
                raise ValueError("模型未训练，无法预测")

            # 1. 创建未来时间序列
            future = self.model.make_future_dataframe(
                periods=1,#self.model_params["forecast_periods"],
                freq='D',#self.model_params["forecast_freq"],
                include_history=False
            )
            # 2. 补充未来期的回归因子（简化版：用历史均值填充）
            for reg in self.model.extra_regressors:
                if reg in features_df.columns:
                    future[reg] = features_df[reg].mean()

            # 3. 预测
            forecast = self.model.predict(future)
            # 4. 可视化预测结果
            self._plot_forecast(forecast)

            logger.info(f"完成{1}天销量预测")
            return forecast[["ds", "yhat", "yhat_lower", "yhat_upper"]]

        except Exception as e:
            logger.error(f"预测失败: {str(e)}", exc_info=True)
            raise

    def _plot_forecast(self, forecast: pd.DataFrame) -> None:
        """可视化预测结果"""
        # 预测趋势图
        fig1 = plot_plotly(self.model, forecast)
        fig1.write_html("output/visualization/forecast_trend.html")
        # 组件分解图（趋势/周度/节假日）
        fig2 = plot_components_plotly(self.model, forecast)
        fig2.write_html("output/visualization/forecast_components.html")
        logger.info("预测可视化结果已保存到 output/visualization/")