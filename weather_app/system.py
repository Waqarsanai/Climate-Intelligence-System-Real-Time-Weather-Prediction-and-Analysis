from pathlib import Path

from .config import CONFIG
from .logging_utils import logger
from .visualizer import WeatherVisualizer
from .fetcher import RealTimeWeatherDataFetcher
from .advanced_predictor import AdvancedKarachiPredictor


class InteractiveWeatherSystem:
    """Top-level orchestrator for fetching, predicting, and visualizing weather."""

    def __init__(self):
        self.visualizer = WeatherVisualizer(CONFIG['viz_dir'])
        self.fetcher = RealTimeWeatherDataFetcher(CONFIG['city'])
        self.predictor = AdvancedKarachiPredictor()
        Path(CONFIG['viz_dir']).mkdir(parents=True, exist_ok=True)

    def run_all(self):
        current = self.fetcher.fetch()
        hourly = self.predictor.predict_temperature(current, hours=24)
        extended = self.predictor.predict_extended_forecast(current, days=3)

        paths = {
            'realtime': self.visualizer.plot_realtime_weather(current),
            'hourly': self.visualizer.plot_24hour_forecast(hourly, current),
            'extended': self.visualizer.plot_extended_forecast(extended, current),
        }
        logger.info(f"Generated visualizations: {paths}")
        return paths

    def predict_temperature(self, return_json: bool = False, hours: int = 24):
        """Produce predictions using current realtime weather.

        When `return_json` is True, returns list of dicts suitable for API.
        """
        current = self.fetcher.fetch()
        preds = self.predictor.predict_temperature(current, hours=hours)
        if return_json:
            return preds
        return preds
