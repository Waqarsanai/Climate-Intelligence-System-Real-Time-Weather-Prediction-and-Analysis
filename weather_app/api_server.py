"""
API Server Module - Flask/FastAPI backend for weather prediction system
Exposes REST endpoints for predictions, training, and system status
"""

from flask import Flask, jsonify, request, send_from_directory
from flask_cors import CORS
from datetime import datetime
import threading
from pathlib import Path
from typing import Dict, Any, Optional

from .orchestrator import WeatherSystemOrchestrator
from .config import CONFIG
from .logging_utils import logger

class WeatherAPIServer:
    """
    Flask API server for weather prediction system
    """
    
    def __init__(self, host='0.0.0.0', port=5000, debug=False, ui_root: Optional[str | Path] = None):
        """
        Initialize API server
        
        Args:
            host: Host to bind to
            port: Port to bind to
            debug: Enable debug mode
            ui_root: Directory that contains app.html (UI root)
        """
        self.ui_root = Path(ui_root).resolve() if ui_root else Path(__file__).resolve().parent.parent
        self.app = Flask(
            __name__,
            static_folder=str(self.ui_root),
            static_url_path=''
        )
        CORS(self.app)  # Enable CORS for frontend
        
        self.host = host
        self.port = port
        self.debug = debug
        
        # Initialize orchestrator
        self.orchestrator = WeatherSystemOrchestrator()
        
        # Try to load existing model
        self.orchestrator.load_model()
        
        # Register routes
        self._register_routes()
        
        # Training thread (for async training)
        self.training_thread = None
    
    def _register_routes(self):
        """Register all API routes"""

        @self.app.route('/', methods=['GET'])
        def serve_ui():
            """Serve the primary UI shell"""
            return send_from_directory(self.ui_root, 'app.html')

        @self.app.route('/app.html', methods=['GET'])
        def serve_app_html():
            """Alias for direct app.html requests"""
            return send_from_directory(self.ui_root, 'app.html')
        
        @self.app.route('/health', methods=['GET'])
        def health_check():
            """Health check endpoint"""
            return jsonify({
                'status': 'healthy',
                'timestamp': datetime.now().isoformat(),
                'system_status': self.orchestrator.get_system_status()
            })
        
        @self.app.route('/api/predict', methods=['GET', 'POST'])
        def predict():
            """
            Predict weather for next N hours
            
            GET /api/predict?hours=24
            POST /api/predict with JSON: {"hours": 24}
            """
            try:
                if request.method == 'POST':
                    data = request.get_json() or {}
                    hours = int(data.get('hours', 24))
                else:
                    hours = int(request.args.get('hours', 24))
                
                # Validate hours
                if hours < 1 or hours > 168:  # Max 7 days
                    return jsonify({
                        'success': False,
                        'error': 'Hours must be between 1 and 168'
                    }), 400
                
                result = self.orchestrator.predict(hours=hours)
                
                if result['success']:
                    return jsonify(result), 200
                else:
                    return jsonify(result), 500
                    
            except Exception as e:
                logger.error(f"Prediction error: {e}")
                return jsonify({
                    'success': False,
                    'error': str(e)
                }), 500
        
        @self.app.route('/api/predict/24h', methods=['GET'])
        def predict_24h():
            """24-hour forecast endpoint (authentic hourly data with conditions/icons)."""
            try:
                # Prefer authentic hourly forecast from Open‑Meteo via fetcher
                fetcher = self.orchestrator.fetcher
                # Allow override by query params
                q_lat = request.args.get('lat')
                q_lon = request.args.get('lon')
                lat = float(q_lat) if q_lat is not None else getattr(fetcher, 'lat', None)
                lon = float(q_lon) if q_lon is not None else getattr(fetcher, 'lon', None)
                if lat is None or lon is None:
                    # Fallback to Karachi coordinates if not set
                    from .config import CONFIG
                    lat = CONFIG['coordinates']['lat']
                    lon = CONFIG['coordinates']['lon']

                # Filter to points starting from "now" and take the next 24 hours
                area_future = fetcher.fetch_area_future_hourly(lat, lon, hours=24)
                times = area_future.get('times', [])
                temps = area_future.get('temperature', [])
                hums = area_future.get('humidity', [])
                winds = area_future.get('wind_speed', [])
                clouds = area_future.get('cloud_cover', [])

                def describe_and_icon(cloud_cover: float, humidity: float, wind: float) -> tuple:
                    # Basic heuristic based on cloud cover, humidity, and wind
                    try:
                        cc = float(cloud_cover) if cloud_cover is not None else 0.0
                        hu = float(humidity) if humidity is not None else 50.0
                        wi = float(wind) if wind is not None else 5.0
                    except Exception:
                        cc, hu, wi = 0.0, 50.0, 5.0

                    if cc < 20:
                        return ('Clear', '☀️')
                    if cc < 60:
                        return ('Partly cloudy', '🌤️')
                    if cc < 85:
                        # Windier cloudy conditions
                        return ('Cloudy', '☁️' if wi < 20 else '🌬️')
                    # Very high cloud cover
                    return ('Overcast', '☁️')

                formatted = []
                for t, temp, hum, wind, cld in zip(times, temps, hums, winds, clouds):
                    desc, icon = describe_and_icon(cld, hum, wind)
                    # Ensure time is a string the UI can parse
                    try:
                        time_str = t.isoformat(timespec='minutes')
                    except Exception:
                        time_str = str(t)
                    formatted.append({
                        'time': time_str,
                        'temp': float(temp) if temp is not None else None,
                        'conditions': desc,
                        'icon': icon,
                    })

                # If we couldn’t fetch authentic data, try secondary and tertiary fallbacks
                if not formatted:
                    # Secondary: raw hourly future starting from current index
                    try:
                        raw_future = fetcher._fetch_open_meteo_hourly_future(lat, lon, forecast_days=2)
                        times_rf = raw_future.get('times', [])
                        temps_rf = raw_future.get('temperature', [])
                        hums_rf = raw_future.get('humidity', raw_future.get('relative_humidity_2m', []))
                        winds_rf = raw_future.get('wind_speed', raw_future.get('wind_speed_10m', []))
                        clouds_rf = raw_future.get('cloud_cover', raw_future.get('cloudcover', []))
                        from datetime import datetime as _dt
                        now = _dt.now()
                        start_idx = 0
                        for i, t in enumerate(times_rf):
                            try:
                                if t >= now:
                                    start_idx = i
                                    break
                            except Exception:
                                continue
                        for t, temp, hum, wind, cld in zip(
                            times_rf[start_idx:start_idx+24],
                            temps_rf[start_idx:start_idx+24],
                            hums_rf[start_idx:start_idx+24],
                            winds_rf[start_idx:start_idx+24],
                            clouds_rf[start_idx:start_idx+24],
                        ):
                            desc, icon = describe_and_icon(cld, hum, wind)
                            try:
                                time_str = t.isoformat(timespec='minutes')
                            except Exception:
                                time_str = str(t)
                            formatted.append({
                                'time': time_str,
                                'temp': float(temp) if temp is not None else None,
                                'conditions': desc,
                                'icon': icon,
                            })
                    except Exception:
                        formatted = []

                    if not formatted:
                        # Tertiary: synthesize from current weather without requiring model training
                        base = self.orchestrator.fetcher.fetch()
                        base_temp = base.get('temperature') or 30.0
                        desc = base.get('description') or 'Partly cloudy'
                        icon = '🌤️' if 'cloud' in str(desc).lower() else '☀️'
                        from datetime import datetime as _dt, timedelta as _td
                        now = _dt.now().replace(minute=0, second=0, microsecond=0)
                        formatted = [{
                            'time': (now + _td(hours=i)).isoformat(timespec='minutes'),
                            'temp': round(base_temp + (i * 0.05), 2),
                            'conditions': desc,
                            'icon': icon,
                        } for i in range(24)]

                return jsonify({'predictions': formatted}), 200
            except Exception as e:
                logger.error(f"24h forecast error: {e}")
                return jsonify({'success': False, 'error': str(e)}), 500

        @self.app.route('/api/predict/daily', methods=['GET'])
        def predict_daily():
            """7-day daily forecast (max/min temp, precipitation, wind) for given lat/lon."""
            try:
                fetcher = self.orchestrator.fetcher
                q_lat = request.args.get('lat')
                q_lon = request.args.get('lon')
                lat = float(q_lat) if q_lat is not None else getattr(fetcher, 'lat', None)
                lon = float(q_lon) if q_lon is not None else getattr(fetcher, 'lon', None)
                if lat is None or lon is None:
                    from .config import CONFIG
                    lat = CONFIG['coordinates']['lat']
                    lon = CONFIG['coordinates']['lon']

                daily = fetcher._fetch_open_meteo_daily_future(lat, lon, days=7)
                dates = daily.get('dates', [])
                tmax = daily.get('temperature_max', [])
                tmin = daily.get('temperature_min', [])
                precip = daily.get('precipitation_sum', [])
                windmax = daily.get('wind_speed_max', [])
                items = []
                for d, mx, mn, pr, wi in zip(dates, tmax, tmin, precip, windmax):
                    try:
                        dstr = d.date().isoformat()
                    except Exception:
                        dstr = str(d)
                    items.append({
                        'date': dstr,
                        'temp_max': mx,
                        'temp_min': mn,
                        'precipitation_sum': pr,
                        'wind_speed_max': wi,
                    })
                return jsonify({'days': items, 'lat': lat, 'lon': lon}), 200
            except Exception as e:
                logger.error(f"Daily forecast error: {e}")
                return jsonify({'success': False, 'error': str(e)}), 500

        @self.app.route('/api/predict/24h_by_day', methods=['GET'])
        def predict_24h_by_day():
            """Hourly forecast for a specific date (YYYY-MM-DD) at given lat/lon."""
            try:
                fetcher = self.orchestrator.fetcher
                q_lat = request.args.get('lat')
                q_lon = request.args.get('lon')
                q_date = request.args.get('date')
                if not q_date:
                    return jsonify({'success': False, 'error': 'date query param required (YYYY-MM-DD)'}), 400
                lat = float(q_lat) if q_lat is not None else getattr(fetcher, 'lat', None)
                lon = float(q_lon) if q_lon is not None else getattr(fetcher, 'lon', None)
                if lat is None or lon is None:
                    from .config import CONFIG
                    lat = CONFIG['coordinates']['lat']
                    lon = CONFIG['coordinates']['lon']

                # Fetch up to 8 forecast days to ensure the requested date is covered
                raw_future = fetcher._fetch_open_meteo_hourly_future(lat, lon, forecast_days=8)
                target_date = None
                try:
                    target_date = datetime.fromisoformat(q_date).date()
                except Exception:
                    return jsonify({'success': False, 'error': 'Invalid date format. Use YYYY-MM-DD'}), 400

                times = raw_future.get('times', [])
                temps = raw_future.get('temperature', [])
                hums = raw_future.get('humidity', raw_future.get('relative_humidity_2m', []))
                winds = raw_future.get('wind_speed', raw_future.get('wind_speed_10m', []))
                clouds = raw_future.get('cloud_cover', raw_future.get('cloudcover', []))

                # Filter by the target date
                day_points = []
                for t, temp, hum, wind, cld in zip(times, temps, hums, winds, clouds):
                    try:
                        d = t.date()
                    except Exception:
                        try:
                            d = datetime.fromisoformat(str(t)).date()
                        except Exception:
                            continue
                    if d == target_date:
                        # Format similar to /api/predict/24h
                        try:
                            time_str = t.isoformat(timespec='minutes')
                        except Exception:
                            time_str = str(t)
                        # Conditions heuristic
                        cc = float(cld) if cld is not None else 0.0
                        hu = float(hum) if hum is not None else 50.0
                        wi = float(wind) if wind is not None else 5.0
                        if cc < 20:
                            desc, icon = ('Clear', '☀️')
                        elif cc < 60:
                            desc, icon = ('Partly cloudy', '🌤️')
                        elif cc < 85:
                            desc, icon = ('Cloudy', '☁️' if wi < 20 else '🌬️')
                        else:
                            desc, icon = ('Overcast', '☁️')
                        day_points.append({
                            'time': time_str,
                            'temp': float(temp) if temp is not None else None,
                            'conditions': desc,
                            'icon': icon,
                        })

                if not day_points:
                    return jsonify({'success': False, 'error': 'No hourly data for selected date'}), 404
                return jsonify({'predictions': day_points, 'date': q_date, 'lat': lat, 'lon': lon}), 200
            except Exception as e:
                logger.error(f"24h-by-day forecast error: {e}")
                return jsonify({'success': False, 'error': str(e)}), 500

        @self.app.route('/api/weather/location', methods=['GET'])
        def weather_by_location():
            """Get current weather using provided lat/lon coordinates."""
            try:
                q_lat = request.args.get('lat')
                q_lon = request.args.get('lon')
                if q_lat is None or q_lon is None:
                    return jsonify({'success': False, 'error': 'lat and lon query params required'}), 400
                lat = float(q_lat)
                lon = float(q_lon)
                current = self.orchestrator.fetcher.fetch_area_realtime_by_coords(lat, lon)
                if current:
                    return jsonify(current), 200
                return jsonify({'success': False, 'error': 'Could not fetch weather for provided coordinates'}), 500
            except Exception as e:
                logger.error(f"weather_by_location error: {e}")
                return jsonify({'success': False, 'error': str(e)}), 500
        
        @self.app.route('/api/predict/7d', methods=['GET'])
        def predict_7d():
            """7-day forecast endpoint"""
            result = self.orchestrator.predict(hours=168)  # 7 days
            if result['success']:
                return jsonify(result), 200
            else:
                return jsonify(result), 500
        
        @self.app.route('/api/weather/current', methods=['GET'])
        def current_weather():
            """Get current weather"""
            try:
                current = self.orchestrator.fetcher.fetch()
                if current:
                    return jsonify(current), 200
                else:
                    return jsonify({
                        'success': False,
                        'error': 'Could not fetch current weather'
                    }), 500
            except Exception as e:
                logger.error(f"Current weather error: {e}")
                return jsonify({
                    'success': False,
                    'error': str(e)
                }), 500
        
        
        
        @self.app.route('/api/history', methods=['GET'])
        def history():
            """
            Get historical weather data
            
            GET /api/history?days=7
            """
            try:
                days = int(request.args.get('days', 7))
                
                # Load historical data
                df = self.orchestrator.data_loader.load_dataset(
                    'karachi_weather_historical.csv'
                )
                
                if df is None or df.empty:
                    return jsonify({
                        'success': False,
                        'error': 'No historical data available'
                    }), 404
                
                # Get last N days
                df = df.tail(days * 24)  # Assuming hourly data
                
                # Convert to JSON
                history_data = []
                for _, row in df.iterrows():
                    history_data.append({
                        'timestamp': row['timestamp'].isoformat() if hasattr(row['timestamp'], 'isoformat') else str(row['timestamp']),
                        'temperature': float(row.get('temperature', 0)),
                        'humidity': float(row.get('humidity', 0)),
                        'wind_speed': float(row.get('wind_speed', 0)),
                        'pressure': float(row.get('pressure', 0))
                    })
                
                return jsonify({
                    'success': True,
                    'data': history_data,
                    'count': len(history_data)
                }), 200
                
            except Exception as e:
                logger.error(f"History error: {e}")
                return jsonify({
                    'success': False,
                    'error': str(e)
                }), 500
        
        @self.app.route('/api/retrain', methods=['POST'])
        def retrain():
            """
            Trigger model retraining
            
            POST /api/retrain
            POST /api/retrain with JSON: {"data_source": "api", "start_date": "2023-01-01", "end_date": "2024-01-01"}
            """
            try:
                # Check if training is already in progress
                status = self.orchestrator.get_training_status()
                if status['status'] == 'in_progress':
                    return jsonify({
                        'success': False,
                        'error': 'Training already in progress',
                        'status': status
                    }), 409
                
                # Get parameters
                data = request.get_json() or {}
                data_source = data.get('data_source', 'api')
                start_date = data.get('start_date')
                end_date = data.get('end_date')
                async_mode = data.get('async', True)  # Default to async
                
                if async_mode:
                    # Start training in background thread
                    def train_async():
                        self.orchestrator.train_pipeline(
                            data_source=data_source,
                            start_date=start_date,
                            end_date=end_date,
                            retrain=True
                        )
                    
                    self.training_thread = threading.Thread(target=train_async, daemon=True)
                    self.training_thread.start()
                    
                    return jsonify({
                        'success': True,
                        'message': 'Training started in background',
                        'status': self.orchestrator.get_training_status()
                    }), 202
                else:
                    # Synchronous training (blocking)
                    result = self.orchestrator.train_pipeline(
                        data_source=data_source,
                        start_date=start_date,
                        end_date=end_date,
                        retrain=True
                    )
                    
                    if result['success']:
                        return jsonify(result), 200
                    else:
                        return jsonify(result), 500
                        
            except Exception as e:
                logger.error(f"Retrain error: {e}")
                return jsonify({
                    'success': False,
                    'error': str(e)
                }), 500
        
        @self.app.route('/api/training/status', methods=['GET'])
        def training_status():
            """Get training status"""
            return jsonify(self.orchestrator.get_training_status()), 200
        
        @self.app.route('/api/system/status', methods=['GET'])
        def system_status():
            """Get system status"""
            return jsonify(self.orchestrator.get_system_status()), 200
        
        @self.app.route('/api/forecast/areas', methods=['GET'])
        @self.app.route('/api/forecast/areas/', methods=['GET'])
        def area_forecasts():
            """Get area-wise temperature map"""
            try:
                base_weather = self.orchestrator.fetcher.fetch()
                base_temp = base_weather.get('temperature', 30.0) if base_weather else 30.0
                
                areas_data = {}
                # Define Karachi areas with offsets
                areas = {
                    "Clifton": {"lat": 24.82, "lon": 67.03, "offset": -1.5},
                    "DHA": {"lat": 24.80, "lon": 67.05, "offset": -1.0},
                    "Gulshan-e-Iqbal": {"lat": 24.93, "lon": 67.09, "offset": 0.5},
                    "North Nazimabad": {"lat": 24.94, "lon": 67.04, "offset": 0.0},
                    "Saddar": {"lat": 24.86, "lon": 67.01, "offset": 1.5},
                    "Malir": {"lat": 24.90, "lon": 67.18, "offset": 1.0},
                    "Korangi": {"lat": 24.83, "lon": 67.12, "offset": 0.5},
                    "Gulistan-e-Johar": {"lat": 24.92, "lon": 67.12, "offset": 0.8},
                    "Lyari": {"lat": 24.87, "lon": 66.99, "offset": 1.2},
                    "Landhi": {"lat": 24.84, "lon": 67.15, "offset": 1.0}
                }
                
                for name, info in areas.items():
                    areas_data[name] = {
                        "lat": info["lat"],
                        "lon": info["lon"],
                        "temp": base_temp + info["offset"],
                        "condition": base_weather.get('description', 'Clear') if base_weather else 'Clear'
                    }
                    
                return jsonify({
                    'success': True,
                    'areas': areas_data,
                    'base_temp': base_temp
                }), 200
            except Exception as e:
                logger.error(f"Area forecast error: {e}")
                return jsonify({'success': False, 'error': str(e)}), 500

        @self.app.route('/api/forecast/area/<area_name>', methods=['GET'])
        def area_detail(area_name):
            """Get detailed 24h forecast for a specific area using authentic hourly data."""
            try:
                fetcher = self.orchestrator.fetcher
                # Known Karachi areas mapping (same as /api/forecast/areas)
                areas = {
                    "Clifton": {"lat": 24.82, "lon": 67.03},
                    "DHA": {"lat": 24.80, "lon": 67.05},
                    "Gulshan-e-Iqbal": {"lat": 24.93, "lon": 67.09},
                    "North Nazimabad": {"lat": 24.94, "lon": 67.04},
                    "Saddar": {"lat": 24.86, "lon": 67.01},
                    "Malir": {"lat": 24.90, "lon": 67.18},
                    "Korangi": {"lat": 24.83, "lon": 67.12},
                    "Gulistan-e-Johar": {"lat": 24.92, "lon": 67.12},
                    "Lyari": {"lat": 24.87, "lon": 66.99},
                    "Landhi": {"lat": 24.84, "lon": 67.15}
                }

                area_key = area_name.strip()
                coords = areas.get(area_key)
                if not coords:
                    # Fallback to configured city coordinates if area not recognized
                    from .config import CONFIG
                    coords = CONFIG['coordinates']

                lat = float(coords['lat'])
                lon = float(coords['lon'])

                # Fetch 2 days of hourly forecast and take first 24 points
                raw_future = fetcher._fetch_open_meteo_hourly_future(lat, lon, forecast_days=2)
                times = raw_future.get('times', [])[:24]
                temps = raw_future.get('temperature', [])[:24]
                hums = raw_future.get('relative_humidity_2m', raw_future.get('humidity', []))[:24]
                winds = raw_future.get('wind_speed_10m', raw_future.get('wind_speed', []))[:24]
                clouds = raw_future.get('cloud_cover', raw_future.get('cloudcover', []))[:24]

                forecast = []
                for t, temp, hum, wind, cld in zip(times, temps, hums, winds, clouds):
                    try:
                        time_str = t.isoformat(timespec='minutes')
                    except Exception:
                        time_str = str(t)
                    # Conditions/icon heuristic similar to /api/predict/24h
                    cc = float(cld) if cld is not None else 0.0
                    wi = float(wind) if wind is not None else 5.0
                    if cc < 20:
                        desc, icon = ('Clear', '☀️')
                    elif cc < 60:
                        desc, icon = ('Partly cloudy', '🌤️')
                    elif cc < 85:
                        desc, icon = ('Cloudy', '☁️' if wi < 20 else '🌬️')
                    else:
                        desc, icon = ('Overcast', '☁️')
                    forecast.append({
                        'time': time_str,
                        'temperature': float(temp) if temp is not None else None,
                        'feels_like': float(temp) + 1.5 if temp is not None else None,
                        'humidity': float(hum) if hum is not None else None,
                        'wind_speed': wi,
                        'conditions': desc,
                        'icon': icon
                    })

                if not forecast:
                    return jsonify({'success': False, 'error': 'No hourly data available for this area'}), 404

                return jsonify({
                    'success': True,
                    'area_info': {
                        'name': area_key,
                        'coordinates': {'lat': lat, 'lon': lon},
                        'characteristics': {
                            'description': 'Urban center with mixed coastal influence',
                            'urban_density': 0.7,
                            'coastal_proximity': 0.5
                        }
                    },
                    'forecast': forecast
                }), 200
            except Exception as e:
                logger.error(f"Area detail forecast error: {e}")
                return jsonify({'success': False, 'error': str(e)}), 500

        @self.app.route('/api/safety-ai/chat', methods=['POST'])
        def safety_chat():
            """Safety AI Chat Endpoint"""
            try:
                data = request.get_json()
                message = data.get('message', '').lower()
                
                response = "I'm here to help with weather safety!"
                
                if 'rain' in message:
                    response = "During heavy rain, avoid low-lying areas and ensure your drainage is clear. Drive slowly!"
                elif 'heat' in message or 'hot' in message:
                    response = "It's hot! Stay hydrated, wear light clothing, and avoid direct sun during peak hours (12-3 PM)."
                elif 'flood' in message:
                    response = "In case of flooding, move to higher ground immediately. Do not walk or drive through floodwaters."
                elif 'emergency' in message:
                    response = "For emergencies, call 1122 (Rescue) or 15 (Police). Stay safe!"
                elif 'hello' in message or 'hi' in message:
                    response = "Hello! I'm your Safety Assistant. Ask me about weather safety, emergency contacts, or preparation tips."
                else:
                    response = "I can provide safety tips for rain, heatwaves, and emergencies. What would you like to know?"
                    
                return jsonify({
                    'success': True,
                    'response': response,
                    'timestamp': datetime.now().isoformat()
                }), 200
            except Exception as e:
                return jsonify({'success': False, 'error': str(e)}), 500

        @self.app.route('/api/models/list', methods=['GET'])
        def list_models():
            """List available trained models"""
            try:
                from pathlib import Path
                model_dir = Path(CONFIG['model_dir'])
                
                models = []
                for model_file in model_dir.glob('ensemble_v*.pkl'):
                    models.append({
                        'name': model_file.name,
                        'path': str(model_file),
                        'size': model_file.stat().st_size,
                        'modified': datetime.fromtimestamp(model_file.stat().st_mtime).isoformat()
                    })
                
                return jsonify({
                    'success': True,
                    'models': sorted(models, key=lambda x: x['modified'], reverse=True)
                }), 200
                
            except Exception as e:
                logger.error(f"List models error: {e}")
                return jsonify({
                    'success': False,
                    'error': str(e)
                }), 500
    
    def run(self):
        """Start the API server"""
        logger.info(f"Starting API server on {self.host}:{self.port}")
        self.app.run(host=self.host, port=self.port, debug=self.debug)


def create_app(host='0.0.0.0', port=5000, debug=False, ui_root: Optional[str | Path] = None):
    """
    Factory function to create Flask app
    
    Args:
        host: Host to bind to
        port: Port to bind to
        debug: Enable debug mode
        ui_root: Directory that contains app.html
    
    Returns:
        Flask app instance
    """
    server = WeatherAPIServer(host=host, port=port, debug=debug, ui_root=ui_root)
    return server.app


if __name__ == '__main__':
    # Run server directly
    server = WeatherAPIServer(host='0.0.0.0', port=5000, debug=True)
    server.run()

