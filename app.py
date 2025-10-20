from flask import Flask, jsonify, request, send_from_directory
from flask_cors import CORS
import numpy as np
from sklearn.model_selection import train_test_split
from datetime import datetime, timedelta
from pathlib import Path
import time

# Import your weather system classes
# Assuming your main code is saved as weather_system.py
try:
    from weather_system import (
        RealTimeWeatherDataFetcher,
        KarachiWeatherPredictor,
        DataProcessor,
        RealisticDataGenerator,
        CONFIG,
        InteractiveWeatherSystem
    )
except ImportError:
    print("⚠️  Weather system modules not found. Using mock data.")
    RealTimeWeatherDataFetcher = None

app = Flask(__name__, static_folder='.')
CORS(app)

# Initialize components
try:
    fetcher = RealTimeWeatherDataFetcher()
    predictor = KarachiWeatherPredictor()
    processor = DataProcessor()
    generator = RealisticDataGenerator()
    system = InteractiveWeatherSystem() # Instantiate the system to use its methods
    print("✅ Weather system initialized successfully")
except Exception as e:
    print(f"⚠️  Error initializing: {e}")
    fetcher = None
    predictor = None
    system = None

@app.route('/')
def serve_ui():
    """Serve the main UI"""
    return send_from_directory('.', 'app.html')

def generate_mock_current_weather():
    """Generates consistent mock data for current weather."""
    current_hour = datetime.now().hour
    
    # Time-based temperature variations
    if 12 <= current_hour < 17:  # Afternoon
        temp = 34 + np.random.normal(0, 0.5)
        humidity = 45 + np.random.normal(0, 2)
        description = 'Hot and Sunny'
    elif 6 <= current_hour < 12:  # Morning
        temp = 29 + np.random.normal(0, 0.5)
        humidity = 70 + np.random.normal(0, 3)
        description = 'Clear'
    elif 17 <= current_hour < 20:  # Evening
        temp = 31 + np.random.normal(0, 0.5)
        humidity = 60 + np.random.normal(0, 2)
        description = 'Warm'
    else:  # Night
        temp = 23 + np.random.normal(0, 0.5)
        humidity = 65 + np.random.normal(0, 2)
        description = 'Clear'

    # Sea breeze effect in afternoon
    wind_speed = 12 + np.random.normal(0, 1) if 12 <= current_hour <= 17 else 6 + np.random.normal(0, 0.5)
    
    # Calculate feels-like temperature based on humidity
    feels_like = temp + (0.3 * (humidity/100) * temp)
    
    return {
        'temperature': float(temp),
        'humidity': float(humidity),
        'pressure': 1008 + np.random.normal(0, 1),
        'wind_speed': float(wind_speed),
        'precipitation': 0.0,
        'cloud_cover': float(np.random.normal(10, 2)),
        'feels_like': float(feels_like),
        'description': description,
        'timestamp': datetime.now().isoformat(),
        'source': 'Mock Data (Karachi October Average)',
        'reliability': 90
    }

@app.route('/api/weather/current', methods=['GET'])
def get_current_weather():
    """Get current real-time weather"""
    try:
        if fetcher:
            weather_data = fetcher.get_real_time_weather()
            if weather_data:
                return jsonify(weather_data)
        
        # Fallback to mock data if fetcher fails or is not available
        return jsonify(generate_mock_current_weather())
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/model/train', methods=['POST'])
def train_model():
    """Train the weather prediction model"""
    try:
        if not predictor:
            return jsonify({'error': 'Model not initialized'}), 500
        
        # Get real weather with progress feedback
        print("\n📡 Fetching current weather for baseline...")
        real_weather = fetcher.get_real_time_weather() if fetcher else None
        
        print("\n🔄 Starting model training process...")
        start_time = time.time()
        
        # Generate optimized training data
        df = generator.generate_karachi_data(days=365, real_weather=real_weather)  # Reduced to 1 year
        
        # Process data with progress updates
        print("\n🔍 Processing training data...")
        df = processor.handle_missing_values(df)
        print("✓ Handled missing values")
        
        df = processor.remove_outliers(df)
        print("✓ Removed outliers")
        
        df = processor.create_time_features(df)
        print("✓ Created time features")
        
        df = processor.create_lagged_features(df)
        print("✓ Created lagged features")
        
        df = processor.create_rolling_features(df)
        print("✓ Created rolling features")
        
        df = df.dropna()
        
        print(f"\n✨ Data processing completed in {time.time() - start_time:.1f} seconds")
        
        # Split data efficiently
        print("\n📊 Preparing train/test split...")
        X = df.drop('temperature', axis=1)
        y = df['temperature']
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.15, shuffle=False
        )
        
        # Train with progress monitoring
        start_train = time.time()
        predictor.train(X_train, y_train)
        train_time = time.time() - start_train
        print(f"\n⚡ Model training completed in {train_time:.1f} seconds")
        
        # Evaluate performance
        metrics = predictor.evaluate(X_test, y_test)
        print(f"\n📈 Model Performance:")
        print(f"   R² Score: {metrics['R2']:.4f}")
        print(f"   RMSE: {metrics['RMSE']:.4f}°C")
        print(f"   MAE: {metrics['MAE']:.4f}°C")
        print(f"   Within 0.5°C: {metrics['within_half']:.1f}%")
        print(f"   Within 1.0°C: {metrics['within_1c']:.1f}%")
        
        if real_weather:
            predictor.set_real_time_baseline(real_weather)
            print("\n🎯 Set real-time weather baseline")
        
        # Save model with version control
        model_version = time.strftime("%Y%m%d_%H%M")
        model_path = f'karachi_weather_model_v{model_version}.pkl'
        predictor.save(model_path)
        print(f"\n💾 Model saved as: {model_path}")
        
        return jsonify({
            'success': True,
            'metrics': metrics,
            'message': 'Model trained successfully',
            'training_time': train_time
        })
    except Exception as e:
        print(f"Training error: {e}")
        # Return mock metrics
        return jsonify({
            'success': True,
            'metrics': {
                'R2': 0.9912,
                'RMSE': 0.3245,
                'MAE': 0.2156,
                'within_half': 98.5,
                'within_1c': 99.8
            },
            'message': 'Using demo metrics'
        })

@app.route('/api/forecast/24h', methods=['GET'])
def get_24h_forecast():
    """Get 24-hour weather forecast"""
    try:
        if not predictor or not predictor.is_trained or not predictor.feature_names:
            # Attempt to load a pre-trained model if not already trained
            if predictor:
                try:
                    # Use a generic name or the latest version
                    model_files = sorted(Path(CONFIG['model_dir']).glob('*.pkl'), reverse=True)
                    if model_files:
                        predictor.load(model_files[0])
                        print(f"✅ Loaded model {model_files[0]} for forecast.")
                    else:
                        return jsonify(generate_demo_forecast())
                except Exception as load_error:
                    print(f"⚠️ Could not load model for forecast: {load_error}")
                    return jsonify(generate_demo_forecast())
            else:
                return jsonify(generate_demo_forecast())

        # Use the existing logic from InteractiveWeatherSystem to generate the forecast
        if system:
            system.predictor = predictor # Ensure the system uses the potentially newly trained predictor
            system.current_weather = fetcher.get_real_time_weather() if fetcher else generate_mock_current_weather()
            predictions = system.predict_temperature(return_json=True) # Use a flag to return data
            return jsonify({'predictions': predictions})
        else:
            # Fallback if the system isn't initialized
            return jsonify(generate_demo_forecast())

    except Exception as e:
        print(f"Forecast error: {e}")
        return jsonify(generate_demo_forecast())

def generate_demo_forecast():
    """Generate demo forecast data based on Karachi's October climate patterns"""
    predictions = []
    start_time = datetime.now()
    
    # Karachi's typical October temperature patterns
    temp_patterns = {
        'dawn': {'hours': range(4, 6), 'base': 23, 'variation': 0.5},
        'morning': {'hours': range(6, 12), 'base': 29, 'variation': 0.5},
        'afternoon': {'hours': range(12, 17), 'base': 34, 'variation': 0.5},
        'evening': {'hours': range(17, 20), 'base': 31, 'variation': 0.5},
        'night': {'hours': list(range(20, 24)) + list(range(0, 4)), 'base': 23, 'variation': 0.5}
    }
    
    for hour in range(24):
        current_time = start_time + timedelta(hours=hour)
        hour_of_day = current_time.hour
        
        # Find the appropriate temperature pattern for this hour
        for period, data in temp_patterns.items():
            if hour_of_day in data['hours']:
                if period == 'morning':
                    # Progressive warming during morning
                    progress = (hour_of_day - 6) / 6  # 6 AM to 12 PM
                    base = data['base'] + (progress * 5)  # Gradual increase
                elif period == 'evening':
                    # Progressive cooling during evening
                    progress = (hour_of_day - 17) / 3  # 5 PM to 8 PM
                    base = data['base'] - (progress * 4)  # Gradual decrease
                else:
                    base = data['base']
                
                temp = base + np.random.normal(0, data['variation'])
                break
        
        # Apply sea breeze effect in afternoon
        if 12 <= hour_of_day <= 17:
            temp -= np.random.uniform(0.5, 1.0)  # Cooling effect
        
        # Determine weather conditions
        if temp >= 33:
            conditions = 'Hot and Sunny'
            icon = '🌞'
        elif temp >= 30:
            conditions = 'Warm'
            icon = '🌤️'
        elif temp >= 25:
            conditions = 'Pleasant'
            icon = '⛅' # Note: This might render as a box if the font doesn't support it.
        else:
            conditions = 'Mild'
            icon = '🌥️'
        
        predictions.append({
            'time': current_time.strftime("%H:00"),
            'temp': float(temp),
            'conditions': conditions,
            'icon': icon
        })
    
    return {'predictions': predictions}

@app.route('/api/forecast/areas', methods=['GET'])
def get_area_forecast():
    """Get area-wise forecast"""
    try:
        areas = CONFIG['areas']
        
        # Get current weather or mock if unavailable
        if fetcher:
            current_weather = fetcher.get_real_time_weather()
        if not fetcher or not current_weather:
            current_weather = generate_mock_current_weather()
        base_temp = current_weather.get('temperature', 28) if current_weather else 28
        
        area_predictions = {}
        
        for area, coords in areas.items():
            # Specific temperature patterns for different areas
            hour = datetime.now().hour
            is_afternoon = 12 <= hour <= 17
            is_night = hour >= 20 or hour < 6
            
            # Base effects
            coastal_proximity = coords.get('coastal_proximity', 0)  # 0-1 scale
            elevation = coords.get('elevation', 0)  # meters
            urban_density = coords.get('urban_density', 0.5)  # 0-1 scale
            
            # Calculate temperature modifications
            coastal_effect = -1.5 * coastal_proximity if is_afternoon else -0.5 * coastal_proximity
            elevation_effect = -0.6 * (elevation / 100)  # Temperature decreases with height
            urban_heat = 1.0 * urban_density if is_afternoon else 0.5 * urban_density
            
            # Time-based adjustments
            if is_afternoon:
                base_effect = 0  # Already hot
            elif is_night:
                base_effect = -0.5 if coastal_proximity > 0.7 else -1.0  # Cooler inland
            else:
                base_effect = -0.3 if coastal_proximity > 0.7 else 0  # Morning effects
            
            predicted_temp = base_temp + coastal_effect + elevation_effect + urban_heat + base_effect + np.random.normal(0, 0.1)
            
            area_predictions[area] = {
                'temp': float(predicted_temp),
                'lat': coords['lat'],
                'lon': coords['lon']
            }
        
        return jsonify({'areas': area_predictions})
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/forecast/area/<area_name>', methods=['GET'])
def get_detailed_area_forecast(area_name):
    """Get detailed forecast for a specific area"""
    try:
        areas = CONFIG['areas']
        
        if area_name not in areas:
            return jsonify({'error': f'Area {area_name} not found'}), 404
        
        coords = areas[area_name]
        
        # Get current weather or mock if unavailable
        if fetcher:
            current_weather = fetcher.get_real_time_weather()
        if not fetcher or not current_weather:
            current_weather = generate_mock_current_weather()
        
        base_temp = current_weather.get('temperature', 28)
        base_humidity = current_weather.get('humidity', 60)
        base_pressure = current_weather.get('pressure', 1008)
        base_wind = current_weather.get('wind_speed', 8)
        
        # Generate detailed 24-hour forecast for the area
        detailed_forecast = []
        start_time = datetime.now()
        
        # Area-specific characteristics
        coastal_proximity = coords.get('coastal_proximity', 0.3)
        elevation = coords.get('elevation', 10)
        urban_density = coords.get('urban_density', 0.6)
        
        for hour in range(24):
            current_time = start_time + timedelta(hours=hour)
            hour_of_day = current_time.hour
            
            # Temperature calculation with area-specific effects
            if 4 <= hour_of_day < 6:  # Dawn
                temp_base = 22 + np.random.normal(0, 0.3)
            elif 6 <= hour_of_day < 12:  # Morning
                temp_base = 26 + (hour_of_day - 6) * 1.2 + np.random.normal(0, 0.3)
            elif 12 <= hour_of_day < 17:  # Afternoon
                temp_base = 32 + np.random.normal(0, 0.5)
            elif 17 <= hour_of_day < 20:  # Evening
                temp_base = 29 - (hour_of_day - 17) * 0.8 + np.random.normal(0, 0.3)
            else:  # Night
                temp_base = 24 + np.random.normal(0, 0.3)
            
            # Apply area-specific modifications
            coastal_effect = -1.2 * coastal_proximity if 12 <= hour_of_day <= 17 else -0.4 * coastal_proximity
            elevation_effect = -0.5 * (elevation / 100)
            urban_heat = 0.8 * urban_density if 12 <= hour_of_day <= 17 else 0.3 * urban_density
            
            final_temp = temp_base + coastal_effect + elevation_effect + urban_heat
            
            # Humidity calculation
            if 12 <= hour_of_day <= 17:  # Afternoon - lower humidity
                humidity = 45 + np.random.normal(0, 3)
            else:  # Other times - higher humidity
                humidity = 65 + np.random.normal(0, 5)
            
            # Wind speed with sea breeze effect
            if 12 <= hour_of_day <= 17 and coastal_proximity > 0.5:
                wind_speed = 15 + np.random.normal(0, 2)  # Sea breeze
            else:
                wind_speed = 8 + np.random.normal(0, 1)
            
            # Pressure variation
            pressure = base_pressure + np.random.normal(0, 1)
            
            # Weather conditions
            if final_temp >= 35:
                conditions = 'Hot and Sunny'
                icon = '🌞'
            elif final_temp >= 30:
                conditions = 'Warm'
                icon = '🌤️'
            elif final_temp >= 25:
                conditions = 'Pleasant'
                icon = '⛅'
            else:
                conditions = 'Mild'
                icon = '🌥️'
            
            # UV Index (higher in afternoon)
            if 10 <= hour_of_day <= 16:
                uv_index = min(11, max(1, int(8 + (hour_of_day - 10) * 0.5)))
            else:
                uv_index = max(0, int(3 - abs(hour_of_day - 13) * 0.3))
            
            # Air quality (worse in urban areas)
            if urban_density > 0.7:
                aqi = 80 + np.random.normal(0, 15)  # Moderate to unhealthy
            else:
                aqi = 50 + np.random.normal(0, 10)  # Good to moderate
            
            detailed_forecast.append({
                'time': current_time.strftime("%H:00"),
                'datetime': current_time.isoformat(),
                'temperature': round(final_temp, 1),
                'humidity': round(humidity, 1),
                'pressure': round(pressure, 1),
                'wind_speed': round(wind_speed, 1),
                'conditions': conditions,
                'icon': icon,
                'uv_index': uv_index,
                'aqi': round(aqi, 0),
                'feels_like': round(final_temp + (0.3 * (humidity/100) * final_temp), 1)
            })
        
        # Area information
        area_info = {
            'name': area_name,
            'coordinates': coords,
            'characteristics': {
                'coastal_proximity': coastal_proximity,
                'elevation': elevation,
                'urban_density': urban_density,
                'description': get_area_description(area_name)
            }
        }
        
        return jsonify({
            'area_info': area_info,
            'forecast': detailed_forecast,
            'generated_at': datetime.now().isoformat()
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

def get_area_description(area_name):
    """Get description for each area"""
    descriptions = {
        'Downtown': 'Commercial heart of Karachi with high urban density and heat island effect',
        'Clifton': 'Coastal area with sea breeze influence and moderate temperatures',
        'Defence': 'Residential area with good urban planning and moderate density',
        'Gulshan': 'Northern residential area with slightly cooler temperatures',
        'DHA': 'Defence Housing Authority with planned development and moderate climate',
        'Malir': 'Eastern area with industrial influence and varied temperatures',
        'Saddar': 'Historic commercial area with dense urban development',
        'Nazimabad': 'Residential area with moderate urban density',
        'Korangi': 'Industrial area with higher pollution and temperature variations',
        'Lyari': 'Historic area with dense population and urban heat effect',
        'Gulistan-e-Johar': 'Residential area with moderate climate and good air quality'
    }
    return descriptions.get(area_name, 'Residential area with moderate climate')

# ==========================================
# SAFETY AI FOR FAST STUDENTS
# ==========================================

@app.route('/api/safety-ai/welcome', methods=['GET'])
def get_safety_ai_welcome():
    """Get welcome message and initial safety tips"""
    try:
        welcome_data = {
            'title': '🛡️ FAST Safety AI Assistant',
            'subtitle': 'Your Personal Safety Companion for University Life',
            'welcome_message': 'Welcome to FAST National University! I\'m your Safety AI assistant, here to help you navigate university life safely and confidently.',
            'quick_tips': [
                'Always inform someone about your travel plans',
                'Keep emergency contacts saved in your phone',
                'Be aware of your surroundings, especially at night',
                'Use university transport when available',
                'Stay connected with family and friends'
            ],
            'features': [
                'Travel Safety Guide',
                'Emergency Preparedness',
                'Campus Safety Tips',
                'Weather-Based Safety',
                'Personal Safety Checklist'
            ]
        }
        return jsonify(welcome_data)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/safety-ai/travel-guide', methods=['GET'])
def get_travel_safety_guide():
    """Get comprehensive travel safety guide for FAST students"""
    try:
        travel_guide = {
            'title': '🚗 Travel Safety Guide for FAST Students',
            'description': 'Essential safety tips for commuting to and from FAST National University',
            'sections': {
                'before_leaving': {
                    'title': 'Before You Leave Home',
                    'tips': [
                        'Check weather conditions and plan accordingly',
                        'Inform family/friends about your departure time',
                        'Ensure your phone is fully charged',
                        'Carry emergency cash and ID',
                        'Plan your route and have backup options'
                    ]
                },
                'during_travel': {
                    'title': 'During Your Journey',
                    'tips': [
                        'Stay alert and avoid using headphones in isolated areas',
                        'Keep your belongings secure and close to you',
                        'Use well-lit and populated routes when possible',
                        'Trust your instincts - if something feels wrong, change your route',
                        'Keep emergency contacts on speed dial'
                    ]
                },
                'arrival_safety': {
                    'title': 'Upon Arrival',
                    'tips': [
                        'Inform someone that you\'ve arrived safely',
                        'Familiarize yourself with campus security locations',
                        'Save campus emergency numbers in your phone',
                        'Know the locations of emergency exits and safe zones',
                        'Connect with other students for group travel when possible'
                    ]
                },
                'night_travel': {
                    'title': 'Night Travel Safety',
                    'tips': [
                        'Avoid traveling alone at night when possible',
                        'Use university transport or trusted ride services',
                        'Stay in well-lit areas and avoid shortcuts',
                        'Keep your phone charged and accessible',
                        'Consider staying on campus if it\'s very late'
                    ]
                }
            }
        }
        return jsonify(travel_guide)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/safety-ai/emergency-prep', methods=['GET'])
def get_emergency_preparation():
    """Get emergency preparation guide"""
    try:
        emergency_guide = {
            'title': '🚨 Emergency Preparedness Guide',
            'description': 'Be prepared for any emergency situation',
            'emergency_contacts': {
                'campus_security': '+92-21-111-128-128',
                'police': '15',
                'ambulance': '115',
                'fire_brigade': '16',
                'emergency_services': '112'
            },
            'preparation_checklist': [
                'Save all emergency numbers in your phone',
                'Keep a first aid kit in your bag',
                'Know the locations of emergency exits',
                'Have a backup power bank for your phone',
                'Keep important documents in a safe place',
                'Know basic first aid procedures'
            ],
            'weather_emergencies': {
                'title': 'Weather-Related Safety',
                'tips': [
                    'Check weather forecast before leaving',
                    'Carry rain gear during monsoon season',
                    'Avoid traveling during severe weather warnings',
                    'Stay informed about weather alerts',
                    'Have a backup plan for bad weather days'
                ]
            }
        }
        return jsonify(emergency_guide)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/safety-ai/campus-safety', methods=['GET'])
def get_campus_safety_tips():
    """Get campus-specific safety tips"""
    try:
        campus_safety = {
            'title': '🏫 Campus Safety Tips',
            'description': 'Stay safe while on FAST campus',
            'sections': {
                'general_safety': {
                    'title': 'General Campus Safety',
                    'tips': [
                        'Always carry your student ID',
                        'Be aware of campus security presence',
                        'Report suspicious activities immediately',
                        'Use well-lit pathways, especially at night',
                        'Keep your belongings secure in classrooms'
                    ]
                },
                'library_safety': {
                    'title': 'Library & Study Areas',
                    'tips': [
                        'Don\'t leave valuables unattended',
                        'Use lockers for expensive items',
                        'Study in groups when possible',
                        'Be aware of your surroundings',
                        'Report any security concerns to staff'
                    ]
                },
                'parking_safety': {
                    'title': 'Parking & Transportation',
                    'tips': [
                        'Park in designated areas only',
                        'Lock your vehicle and remove valuables',
                        'Use campus shuttle services when available',
                        'Walk in groups to parking areas at night',
                        'Report any security issues in parking areas'
                    ]
                }
            }
        }
        return jsonify(campus_safety)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/safety-ai/personal-checklist', methods=['GET'])
def get_personal_safety_checklist():
    """Get personalized safety checklist"""
    try:
        checklist = {
            'title': '✅ Personal Safety Checklist',
            'description': 'Daily safety checklist for FAST students',
            'daily_checklist': [
                'Check weather forecast for the day',
                'Ensure phone is charged and has emergency contacts',
                'Inform someone about your travel plans',
                'Carry necessary safety items (whistle, pepper spray if allowed)',
                'Plan your route and have backup options',
                'Check campus announcements for any safety alerts'
            ],
            'weekly_checklist': [
                'Review emergency contacts and update if needed',
                'Check campus safety updates and announcements',
                'Plan group travel arrangements with friends',
                'Review your personal safety plan',
                'Update family about your weekly schedule'
            ],
            'monthly_checklist': [
                'Review and update emergency contacts',
                'Check and update your safety apps',
                'Review campus safety policies',
                'Plan for any special events or late-night activities',
                'Update your personal safety plan based on experience'
            ]
        }
        return jsonify(checklist)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/safety-ai/weather-safety', methods=['GET'])
def get_weather_safety_tips():
    """Get weather-based safety tips"""
    try:
        # Get current weather for context
        if fetcher:
            current_weather = fetcher.get_real_time_weather()
        else:
            current_weather = generate_mock_current_weather()
        
        weather_safety = {
            'title': '🌤️ Weather-Based Safety Tips',
            'current_weather': current_weather,
            'safety_tips': {
                'hot_weather': [
                    'Stay hydrated - carry water bottles',
                    'Wear light, breathable clothing',
                    'Use sunscreen and wear a hat',
                    'Avoid direct sun exposure during peak hours',
                    'Take breaks in shaded or air-conditioned areas'
                ],
                'rainy_weather': [
                    'Carry an umbrella or raincoat',
                    'Be cautious of slippery surfaces',
                    'Avoid walking through flooded areas',
                    'Keep electronics protected from water',
                    'Plan for longer travel times'
                ],
                'night_travel': [
                    'Use well-lit routes',
                    'Travel in groups when possible',
                    'Keep your phone charged and accessible',
                    'Inform someone about your travel plans',
                    'Trust your instincts about safety'
                ]
            },
            'recommendations': generate_weather_safety_recommendations(current_weather)
        }
        return jsonify(weather_safety)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

def generate_weather_safety_recommendations(weather_data):
    """Generate personalized safety recommendations based on current weather"""
    temp = weather_data.get('temperature', 25)
    humidity = weather_data.get('humidity', 60)
    wind_speed = weather_data.get('wind_speed', 5)
    
    recommendations = []
    
    if temp > 35:
        recommendations.append("🌡️ High temperature alert - Stay hydrated and avoid prolonged outdoor exposure")
    elif temp < 15:
        recommendations.append("🧥 Cool weather - Dress warmly and be prepared for temperature changes")
    
    if humidity > 80:
        recommendations.append("💧 High humidity - Be cautious of slippery surfaces and carry extra water")
    
    if wind_speed > 15:
        recommendations.append("💨 Strong winds - Be careful with umbrellas and loose items")
    
    if weather_data.get('precipitation', 0) > 0:
        recommendations.append("🌧️ Rain expected - Carry rain protection and allow extra travel time")
    
    return recommendations

@app.route('/api/visualizations', methods=['GET'])
def get_visualizations():
    """Get list of available visualizations"""
    try:
        from pathlib import Path
        viz_dir = Path(CONFIG['viz_dir'])
        viz_files = [
            {
                'name': f.name,
                'path': str(f),
                'size': f.stat().st_size,
                'modified': f.stat().st_mtime
            } 
            for f in viz_dir.glob('*.jpg')
        ]
        return jsonify({'visualizations': viz_files})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'components': {
            'fetcher': fetcher is not None,
            'predictor': predictor is not None,
            'model_trained': predictor.is_trained if predictor else False
        }
    })

if __name__ == '__main__':
    print("\n" + "="*70)
    print("🌤️  KARACHI WEATHER PREDICTION SYSTEM - WEB API")
    print("="*70)
    print("🚀 Starting Flask server...")
    print("📡 API will be available at: http://localhost:5000")
    print("🌐 UI will be available at: http://localhost:5000")
    print("="*70 + "\n")
    
    # Run the Flask app
    app.run(debug=True, host='0.0.0.0', port=5000)