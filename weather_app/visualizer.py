import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from pathlib import Path

from .config import CONFIG
from .logging_utils import logger


class WeatherVisualizer:
    def __init__(self, output_dir=CONFIG['viz_dir']):
        self.output_dir = Path(output_dir)
        self.colors = {
            'primary': '#1F77B4',
            'secondary': '#A23B72',
            'success': '#06A77D',
            'warning': '#F18F01',
            'danger': '#C73E1D',
            'info': '#4ECDC4',
            'background': '#F7F9FB'
        }

    def save_plot(self, fig, filename):
        """Save plot with high quality"""
        filepath = self.output_dir / filename
        fig.savefig(filepath, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close(fig)
        logger.info(f"✅ Visualization saved: {filepath.absolute()}")
        return filepath.absolute()

    def plot_realtime_weather(self, weather_data):
        """Visualize real-time weather conditions"""
        fig = plt.figure(figsize=(16, 10))
        gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

        # Temperature gauge
        ax1 = fig.add_subplot(gs[0, :2])
        temp = weather_data.get('temperature', 0)
        ax1.barh([0], [temp], height=0.5, color=self.colors['danger'] if temp > 35 else self.colors['primary'])
        ax1.set_xlim(0, 50)
        ax1.set_yticks([])
        ax1.set_xlabel('Temperature (°C)', fontsize=14, fontweight='bold')
        ax1.set_title(f'Current Temperature: {temp:.1f}°C', fontsize=16, fontweight='bold', pad=20)
        ax1.text(temp, 0.6, f'{temp:.1f}°C', ha='center', fontsize=20, fontweight='bold')

        # Humidity gauge
        ax2 = fig.add_subplot(gs[0, 2])
        humidity = weather_data.get('humidity', 0)
        ax2.pie([humidity, 100-humidity], colors=[self.colors['info'], '#E8E8E8'], 
                startangle=90, counterclock=False, wedgeprops={'width': 0.3})
        ax2.text(0, 0, f'{humidity:.0f}%', ha='center', va='center', fontsize=24, fontweight='bold')
        ax2.set_title('Humidity', fontsize=14, fontweight='bold')

        # Wind speed
        ax3 = fig.add_subplot(gs[1, 0])
        wind = weather_data.get('wind_speed', 0)
        ax3.bar(['Wind'], [wind], color=self.colors['success'], width=0.5)
        ax3.set_ylabel('Speed (m/s)', fontsize=12)
        ax3.set_title(f'Wind: {wind:.1f} m/s', fontsize=14, fontweight='bold')
        ax3.set_ylim(0, 25)

        # Pressure
        ax4 = fig.add_subplot(gs[1, 1])
        pressure = weather_data.get('pressure', 1013)
        ax4.plot([0, 1], [pressure, pressure], linewidth=8, color=self.colors['secondary'], marker='o', markersize=12)
        ax4.set_xlim(-0.1, 1.1)
        ax4.set_ylim(1000, 1025)
        ax4.set_xticks([])
        ax4.set_ylabel('Pressure (mb)', fontsize=12)
        ax4.set_title(f'Pressure: {pressure:.1f} mb', fontsize=14, fontweight='bold')

        # Precipitation
        ax5 = fig.add_subplot(gs[1, 2])
        precip = weather_data.get('precipitation', 0)
        ax5.bar(['Rain'], [precip], color=self.colors['info'], width=0.5)
        ax5.set_ylabel('Precipitation (mm)', fontsize=12)
        ax5.set_title(f'Rainfall: {precip:.2f} mm', fontsize=14, fontweight='bold')

        # Summary
        ax6 = fig.add_subplot(gs[2, :])
        ax6.axis('off')
        summary_text = f"""Location: {CONFIG['city']} | Time: {datetime.now().strftime('%Y-%m-%d %H:%M')}
Source: {weather_data.get('source', 'Unknown')} | Reliability: {weather_data.get('reliability', 0)}%
Feels Like: {weather_data.get('feels_like', temp):.1f}°C | Cloud Cover: {weather_data.get('cloud_cover', 0):.0f}%
Condition: {weather_data.get('description', 'N/A')}"""

        ax6.text(0.5, 0.5, summary_text, ha='center', va='center', fontsize=11,
                bbox=dict(boxstyle='round', facecolor=self.colors['background'], alpha=0.8, pad=1),
                family='monospace')

        fig.suptitle('🌤️ REAL-TIME WEATHER - KARACHI', fontsize=20, fontweight='bold', y=0.98)
        return self.save_plot(fig, 'realtime_weather.jpg')

    def plot_24hour_forecast(self, predictions, current_weather):
        """Visualize 24-hour forecast"""
        fig, axes = plt.subplots(2, 2, figsize=(18, 12))

        times = [p['time'] for p in predictions]
        temps = [p['temp'] for p in predictions]
        hours = list(range(len(predictions)))

        # Line chart
        ax1 = axes[0, 0]
        ax1.plot(hours, temps, marker='o', linewidth=3, markersize=8, color=self.colors['danger'])
        ax1.fill_between(hours, temps, alpha=0.3, color=self.colors['danger'])
        ax1.set_xlabel('Hours Ahead', fontsize=12, fontweight='bold')
        ax1.set_ylabel('Temperature (°C)', fontsize=12, fontweight='bold')
        ax1.set_title('24-Hour Forecast', fontsize=14, fontweight='bold')
        ax1.grid(True, alpha=0.3)

        # Distribution
        ax2 = axes[0, 1]
        if temps:
            ax2.hist(temps, bins=15, color=self.colors['primary'], alpha=0.7, edgecolor='black')
            ax2.axvline(np.mean(temps), color='red', linestyle='--', linewidth=2, label=f'Mean: {np.mean(temps):.1f}°C')
            ax2.legend()
        ax2.set_xlabel('Temperature (°C)', fontsize=12, fontweight='bold')
        ax2.set_ylabel('Frequency', fontsize=12, fontweight='bold')
        ax2.set_title('Distribution', fontsize=14, fontweight='bold')

        # Bar chart
        ax3 = axes[1, 0]
        colors_bar = [self.colors['danger'] if t > 35 else self.colors['warning'] if t > 30 else self.colors['success'] for t in temps]
        ax3.bar(hours, temps, color=colors_bar, alpha=0.8, edgecolor='black')
        ax3.set_xlabel('Hours Ahead', fontsize=12, fontweight='bold')
        ax3.set_ylabel('Temperature (°C)', fontsize=12, fontweight='bold')
        ax3.set_title('Hourly Breakdown', fontsize=14, fontweight='bold')

        # Statistics
        ax4 = axes[1, 1]
        ax4.axis('off')
        if not temps:
            stats_text = "FORECAST STATISTICS\n\nNo data available"
        else:
            stats_text = f"""FORECAST STATISTICS

Current: {current_weather.get('temperature', 0):.1f}°C
Min: {min(temps):.1f}°C
Max: {max(temps):.1f}°C
Average: {np.mean(temps):.1f}°C
Range: {max(temps) - min(temps):.1f}°C

Hottest: {times[temps.index(max(temps))]}
Coolest: {times[temps.index(min(temps))]}

Trend: {'📈 Rising' if temps[-1] > temps[0] else '📉 Falling'}
"""

        ax4.text(0.5, 0.5, stats_text, ha='center', va='center', fontsize=12,
                bbox=dict(boxstyle='round', facecolor=self.colors['background'], alpha=0.9, pad=1.5),
                family='monospace', fontweight='bold')

        fig.suptitle('🎯 24-HOUR FORECAST', fontsize=18, fontweight='bold', y=0.98)
        plt.tight_layout()
        return self.save_plot(fig, '24hour_forecast.jpg')

    def plot_model_performance(self, metrics, y_test, y_pred):
        """Visualize model performance"""
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))

        # Actual vs Predicted
        ax1 = axes[0, 0]
        ax1.scatter(y_test, y_pred, alpha=0.5, s=50, c=self.colors['primary'], edgecolors='black')
        ax1.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=3)
        ax1.set_xlabel('Actual (°C)', fontsize=12, fontweight='bold')
        ax1.set_ylabel('Predicted (°C)', fontsize=12, fontweight='bold')
        ax1.set_title(f'Actual vs Predicted (R² = {metrics["R2"]:.4f})', fontsize=14, fontweight='bold')
        ax1.grid(True, alpha=0.3)

        # Residuals
        ax2 = axes[0, 1]
        residuals = y_test - y_pred
        ax2.scatter(y_pred, residuals, alpha=0.5, s=50, c=self.colors['secondary'], edgecolors='black')
        ax2.axhline(0, color='red', linestyle='--', linewidth=2)
        ax2.set_xlabel('Predicted (°C)', fontsize=12, fontweight='bold')
        ax2.set_ylabel('Residuals (°C)', fontsize=12, fontweight='bold')
        ax2.set_title('Residual Plot', fontsize=14, fontweight='bold')
        ax2.grid(True, alpha=0.3)

        # Error distribution
        ax3 = axes[1, 0]
        ax3.hist(residuals, bins=30, color=self.colors['info'], alpha=0.7, edgecolor='black')
        ax3.axvline(0, color='red', linestyle='--', linewidth=2)
        ax3.set_xlabel('Error (°C)', fontsize=12, fontweight='bold')
        ax3.set_ylabel('Frequency', fontsize=12, fontweight='bold')
        ax3.set_title('Error Distribution', fontsize=14, fontweight='bold')

        # Metrics
        ax4 = axes[1, 1]
        ax4.axis('off')
        accuracy = metrics['R2'] * 100
        status = '✅ EXCELLENT' if accuracy >= 95 else '⚠️ GOOD' if accuracy >= 90 else '❌ NEEDS IMPROVEMENT'

        metrics_text = f"""MODEL PERFORMANCE

Status: {status}

R² Score: {metrics['R2']:.4f} ({accuracy:.2f}%)
RMSE: {metrics['RMSE']:.4f}°C
MAE: {metrics['MAE']:.4f}°C
Within 0.5°C: {metrics.get('within_half', 0):.1f}%

Mean Error: {np.mean(residuals):.4f}°C
Std Error: {np.std(residuals):.4f}°C
Max Error: {np.max(np.abs(residuals)):.2f}°C
"""

        ax4.text(0.5, 0.5, metrics_text, ha='center', va='center', fontsize=11,
                bbox=dict(boxstyle='round', facecolor=self.colors['success'] if accuracy >= 95 else self.colors['warning'], 
                         alpha=0.2, pad=1.5),
                family='monospace', fontweight='bold')

        fig.suptitle('📈 MODEL PERFORMANCE', fontsize=18, fontweight='bold', y=0.98)
        plt.tight_layout()
        return self.save_plot(fig, 'model_performance.jpg')

    def plot_areawise_weather(self, area_predictions):
        """Visualize area-wise weather map"""

        fig, ax = plt.subplots(figsize=(16, 12))

        areas = list(area_predictions.keys())
        temps = [area_predictions[area]['temp'] for area in areas]
        lats = [area_predictions[area]['lat'] for area in areas]
        lons = [area_predictions[area]['lon'] for area in areas]

        scatter = ax.scatter(lons, lats, c=temps, s=500, cmap='RdYlBu_r', 
                           alpha=0.7, edgecolors='black', linewidth=2)

        for i, area in enumerate(areas):
            ax.annotate(f'{area}\n{temps[i]:.1f}°C', (lons[i], lats[i]), 
                       fontsize=8, ha='center', va='center', fontweight='bold',
                       bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))

        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label('Temperature (°C)', fontsize=12, fontweight='bold')

        ax.set_xlabel('Longitude', fontsize=12, fontweight='bold')
        ax.set_ylabel('Latitude', fontsize=12, fontweight='bold')
        ax.set_title('🗺️ AREA-WISE TEMPERATURE MAP', fontsize=16, fontweight='bold', pad=20)
        ax.grid(True, alpha=0.3)

        return self.save_plot(fig, 'areawise_weather.jpg')

    def plot_comparison_chart(self, our_predictions, google_data):
        """Compare predictions"""
        fig, axes = plt.subplots(2, 1, figsize=(16, 10))

        hours = list(range(len(our_predictions)))
        our_temps = [p['temp'] for p in our_predictions]
        google_temps = google_data if google_data and len(google_data) == len(our_temps) else our_temps

        # Line comparison
        ax1 = axes[0]
        ax1.plot(hours, our_temps, marker='o', linewidth=3, markersize=8, 
                color=self.colors['primary'], label='Our Model')
        ax1.plot(hours, google_temps, marker='s', linewidth=3, markersize=8, 
                color=self.colors['danger'], label='External Source', linestyle='--')
        ax1.set_xlabel('Hours Ahead', fontsize=12, fontweight='bold')
        ax1.set_ylabel('Temperature (°C)', fontsize=12, fontweight='bold')
        ax1.set_title('Comparison: Our Model vs External Source', fontsize=14, fontweight='bold')
        ax1.legend(fontsize=12)
        ax1.grid(True, alpha=0.3)

        # Difference
        ax2 = axes[1]
        differences = [our - google for our, google in zip(our_temps, google_temps)]
        colors_diff = [self.colors['success'] if abs(d) < 0.5 else self.colors['warning'] if abs(d) < 1 else self.colors['danger'] for d in differences]
        ax2.bar(hours, differences, color=colors_diff, alpha=0.7, edgecolor='black')
        ax2.axhline(0, color='black', linestyle='-', linewidth=2)
        ax2.set_xlabel('Hours Ahead', fontsize=12, fontweight='bold')
        ax2.set_ylabel('Difference (°C)', fontsize=12, fontweight='bold')
        ax2.set_title('Prediction Difference', fontsize=14, fontweight='bold')
        ax2.grid(True, alpha=0.3, axis='y')

        mae_diff = np.mean(np.abs(differences)) if differences else 0.0
        max_diff = max(np.abs(differences)) if differences else 0.0
        stats_text = f'MAE: {mae_diff:.3f}°C | Max Diff: {max_diff:.3f}°C'
        ax2.text(0.5, 0.95, stats_text, transform=ax2.transAxes, ha='center', va='top',
                fontsize=12, fontweight='bold',
                bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.5))

        fig.suptitle('🔄 PREDICTION COMPARISON', fontsize=18, fontweight='bold', y=0.98)
        plt.tight_layout()
        return self.save_plot(fig, 'comparison_chart.jpg')

    def plot_extended_forecast(self, predictions, current_weather):
        """Visualize extended forecast (multiple days)"""
        if not predictions:
            logger.warning("No predictions provided for extended forecast")
            return None

        fig, axes = plt.subplots(2, 2, figsize=(20, 14))
        fig.patch.set_facecolor('white')

        # Extract data
        times = [p.get('time', '') for p in predictions]
        temps = [p.get('temp', 0) for p in predictions]
        conditions = [p.get('conditions', '') for p in predictions]

        # Temperature trend
        ax1 = axes[0, 0]
        ax1.plot(range(len(temps)), temps, 'o-', linewidth=3, markersize=6, 
                color=self.colors['primary'], markerfacecolor='white', markeredgewidth=2)
        ax1.set_xlabel('Hours Ahead', fontsize=12, fontweight='bold')
        ax1.set_ylabel('Temperature (°C)', fontsize=12, fontweight='bold')
        ax1.set_title('Extended Temperature Forecast', fontsize=14, fontweight='bold')
        ax1.grid(True, alpha=0.3)

        # Add current temperature line
        if current_weather:
            current_temp = current_weather.get('temperature', 0)
            ax1.axhline(y=current_temp, color=self.colors['warning'], linestyle='--', 
                       linewidth=2, label=f'Current: {current_temp:.1f}°C')
            ax1.legend()

        # Daily temperature distribution
        ax2 = axes[0, 1]
        if temps:
            ax2.hist(temps, bins=20, alpha=0.7, color=self.colors['info'], 
                    edgecolor='black', linewidth=1)
            ax2.axvline(np.mean(temps), color=self.colors['danger'], linestyle='--', 
                      linewidth=2, label=f'Mean: {np.mean(temps):.1f}°C')
            ax2.legend()
        ax2.set_xlabel('Temperature (°C)', fontsize=12, fontweight='bold')
        ax2.set_ylabel('Frequency', fontsize=12, fontweight='bold')
        ax2.set_title('Temperature Distribution', fontsize=14, fontweight='bold')
        ax2.grid(True, alpha=0.3)

        # Hourly breakdown (first 24 hours)
        ax3 = axes[1, 0]
        hours_24 = min(24, len(temps))
        if hours_24 > 0:
            ax3.bar(range(hours_24), temps[:hours_24], color=self.colors['success'], 
                   alpha=0.7, edgecolor='black', linewidth=1)
        ax3.set_xlabel('Hour', fontsize=12, fontweight='bold')
        ax3.set_ylabel('Temperature (°C)', fontsize=12, fontweight='bold')
        ax3.set_title('First 24 Hours', fontsize=14, fontweight='bold')
        ax3.grid(True, alpha=0.3, axis='y')

        # Statistics
        ax4 = axes[1, 1]
        ax4.axis('off')
        if temps:
            stats_text = f"""EXTENDED FORECAST STATISTICS

Total Hours: {len(temps)}
Current: {current_weather.get('temperature', 0):.1f}°C
Min: {min(temps):.1f}°C
Max: {max(temps):.1f}°C
Average: {np.mean(temps):.1f}°C
Range: {max(temps) - min(temps):.1f}°C

Trend: {'📈 Rising' if len(temps) > 1 and temps[-1] > temps[0] else '📉 Falling'}
"""
        else:
            stats_text = "EXTENDED FORECAST STATISTICS\n\nNo data available"

        ax4.text(0.5, 0.5, stats_text, ha='center', va='center', fontsize=12,
                bbox=dict(boxstyle='round', facecolor=self.colors['background'], alpha=0.9, pad=1.5),
                family='monospace', fontweight='bold')

        fig.suptitle('📈 EXTENDED WEATHER FORECAST', fontsize=18, fontweight='bold', y=0.98)
        plt.tight_layout()
        return self.save_plot(fig, 'extended_forecast.jpg')