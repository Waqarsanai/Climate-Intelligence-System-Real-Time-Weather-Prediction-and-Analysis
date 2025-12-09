# Complete Usage Guide - Karachi Weather Prediction System

## 🚀 Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Train the Model

**Option A: Via CLI**
```bash
python main_controller.py train
```

**Option B: Via Python**
```python
from weather_app.orchestrator import WeatherSystemOrchestrator
orc = WeatherSystemOrchestrator()
res = orc.train_pipeline(data_source='api', retrain=True)
print(res.get('metrics', {}))
```

**Option C: Via API (after starting server)**
```bash
curl -X POST http://localhost:5000/api/retrain
```

### 3. Start the Server

```bash
python main_controller.py server
```

> `main_controller.py` or `python -m weather_app.api_server` can host the API and UI.

### 4. Access the System

- **UI**: http://localhost:5000
- **API**: http://localhost:5000/api/
- **Health Check**: http://localhost:5000/health

## 📋 Detailed Usage

### Training the Model

#### Via CLI
```bash
# Basic training (uses API data source)
python main_controller.py train

# Training with custom dates
python main_controller.py train --start-date 2023-01-01 --end-date 2024-01-01

# Training from file
python main_controller.py train --data-source file
```

#### Via API
```bash
# Start training in background
curl -X POST http://localhost:5000/api/retrain

# Check training status
curl http://localhost:5000/api/training/status

# Training with parameters
curl -X POST http://localhost:5000/api/retrain \
  -H "Content-Type: application/json" \
  -d '{"data_source": "api", "start_date": "2023-01-01", "end_date": "2024-01-01"}'
```

#### Via Python
```python
from weather_app.orchestrator import WeatherSystemOrchestrator

orchestrator = WeatherSystemOrchestrator()
result = orchestrator.train_pipeline(
    data_source='api',
    start_date='2023-01-01',
    end_date='2024-01-01',
    retrain=True
)

if result['success']:
    print(f"MAE: {result['metrics']['mae']:.3f}°C")
    print(f"RMSE: {result['metrics']['rmse']:.3f}°C")
    print(f"R²: {result['metrics']['r2']:.3f}")
```

### Making Predictions

#### Via API
```bash
# 24-hour forecast
curl http://localhost:5000/api/predict?hours=24

# 7-day forecast
curl http://localhost:5000/api/predict/7d

# Custom hours
curl http://localhost:5000/api/predict?hours=48
```

#### Via Python
```python
from weather_app.orchestrator import WeatherSystemOrchestrator

orchestrator = WeatherSystemOrchestrator()
orchestrator.load_model()  # Load trained model

result = orchestrator.predict(hours=24)

if result['success']:
    for pred in result['predictions']:
        print(f"{pred['time']}: {pred['temp']:.1f}°C")
```

### Getting Current Weather

```bash
curl http://localhost:5000/api/weather/current
```

### Getting Historical Data

```bash
# Last 7 days
curl http://localhost:5000/api/history?days=7

# Last 30 days
curl http://localhost:5000/api/history?days=30
```

### Checking System Status

```bash
# Health check
curl http://localhost:5000/health

# System status
curl http://localhost:5000/api/system/status

# Training status
curl http://localhost:5000/api/training/status

# List available models
curl http://localhost:5000/api/models/list
```

## 🖥️ UI Integration

### Updating app.html to Use New API

The existing `app.html` can be updated to use the new API endpoints. Here's an example JavaScript function:

```javascript
// Fetch 24-hour forecast
async function fetch24HourForecast() {
    try {
        const response = await fetch('/api/predict/24h');
        const data = await response.json();
        
        if (data.predictions) {
            // Update UI with predictions
            updateForecastUI(data.predictions);
        }
    } catch (error) {
        console.error('Error fetching forecast:', error);
    }
}

// Trigger retraining
async function triggerRetrain() {
    try {
        const response = await fetch('/api/retrain', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            }
        });
        
        const data = await response.json();
        
        if (data.success) {
            alert('Training started! Check status at /api/training/status');
            // Poll for training status
            pollTrainingStatus();
        }
    } catch (error) {
        console.error('Error starting training:', error);
    }
}

// Poll training status
async function pollTrainingStatus() {
    const interval = setInterval(async () => {
        const response = await fetch('/api/training/status');
        const status = await response.json();
        
        updateTrainingProgress(status);
        
        if (status.status === 'completed' || status.status === 'failed') {
            clearInterval(interval);
        }
    }, 2000); // Poll every 2 seconds
}
```

## 🔧 Configuration

### Environment Variables

Create a `.env` file (optional):

```env
FLASK_HOST=0.0.0.0
FLASK_PORT=5000
FLASK_DEBUG=False
MODEL_DIR=models
DATA_DIR=data
CACHE_DIR=cache
```

### Config File

Edit `weather_app/config.py` to customize:
- City coordinates
- Area definitions
- Directory paths
- Cache settings

## 📦 Model Management

### Saving Models

Models are automatically saved during training to `models/` directory:
- `ensemble_v{timestamp}.pkl` - Ensemble model
- `{model_name}_v{timestamp}.pkl` - Individual models
- `training_metadata_v{timestamp}.json` - Training metrics

### Loading Models

```python
from weather_app.orchestrator import WeatherSystemOrchestrator

orchestrator = WeatherSystemOrchestrator()

# Load latest model
orchestrator.load_model()

# Load specific model
orchestrator.load_model('models/ensemble_v20241116_120000.pkl')
```

### Listing Models

```bash
# Via API
curl http://localhost:5000/api/models/list

# Via Python
from pathlib import Path
from weather_app.config import CONFIG

model_dir = Path(CONFIG['model_dir'])
models = list(model_dir.glob('ensemble_v*.pkl'))
for model in models:
    print(model.name)
```

## 🔄 Retraining Workflow

### Manual Retraining

1. **Stop the server** (if running)
2. **Run training**:
   ```bash
   python main_controller.py train
   ```
3. **Restart server**:
   ```bash
   python main_controller.py server
   ```

### Automatic Retraining via API

1. **Start server**:
   ```bash
   python main_controller.py server
   ```

2. **Trigger retraining**:
   ```bash
   curl -X POST http://localhost:5000/api/retrain
   ```

3. **Monitor status**:
   ```bash
   curl http://localhost:5000/api/training/status
   ```

4. **Server automatically reloads model** when training completes

### Scheduled Retraining

Create a cron job (Linux/Mac) or scheduled task (Windows):

```bash
# Run daily at 2 AM
0 2 * * * cd /path/to/project && python main_controller.py train
```

## 🐛 Troubleshooting

### Issue: "Model not trained"

**Solution**: Train the model first
```bash
python main_controller.py train
```

### Issue: "No data available"

**Solution**: Check internet connection or provide local data file
```bash
# Check if data file exists
ls data/karachi_weather_historical.csv

# Or fetch new data
python -c "from weather_app.data_loader import WeatherDataLoader; \
           loader = WeatherDataLoader(); \
           df = loader.load_from_open_meteo(24.8607, 67.0011, '2023-01-01', '2024-01-01"); \
           loader.save_dataset(df, 'karachi_weather_historical.csv')"
```

### Issue: "Port already in use"

**Solution**: Use a different port
```bash
python main_controller.py server --port 8080
```

### Issue: "Training takes too long"

**Solution**: 
- Reduce data size
- Use fewer models
- Disable deep learning models

### Issue: "Out of memory"

**Solution**:
- Reduce batch size
- Process data in chunks
- Use fewer models
- Increase system RAM

## 📊 Monitoring

### Check Logs

```bash
# View logs in real-time
tail -f weather_prediction.log

# Search for errors
grep ERROR weather_prediction.log

# Search for warnings
grep WARNING weather_prediction.log
```

### Check Training Progress

```bash
# Via API
curl http://localhost:5000/api/training/status

# Response:
# {
#   "status": "in_progress",
#   "progress": 45,
#   "message": "Training models...",
#   "error": null
# }
```

### Check System Health

```bash
curl http://localhost:5000/health

# Response:
# {
#   "status": "healthy",
#   "timestamp": "2024-11-16T12:00:00",
#   "system_status": {
#     "is_trained": true,
#     "model_loaded": true
#   }
# }
```

## 🚀 Deployment

### Local Deployment

1. **Install dependencies**
2. **Train model**
3. **Start server**
4. **Access via browser**

### Cloud Deployment

See `DEPLOYMENT_GUIDE.md` for detailed instructions.

### Docker Deployment (Future)

```dockerfile
FROM python:3.9
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
CMD ["python", "main_controller.py", "server", "--host", "0.0.0.0", "--port", "5000"]
```

## 📝 Example Workflows

### Complete Training and Prediction Workflow

```bash
# 1. Train model
python main_controller.py train

# 2. Start server
python main_controller.py server &

# 3. Make prediction
curl http://localhost:5000/api/predict?hours=24

# 4. Check status
curl http://localhost:5000/health
```

### Development Workflow

```bash
# 1. Make code changes
# 2. Test locally
python main_controller.py train
python main_controller.py predict --hours 24

# 3. Start server in debug mode
python main_controller.py server --debug

# 4. Test API endpoints
curl http://localhost:5000/api/predict?hours=24
```

### Production Workflow

```bash
# 1. Train model
python main_controller.py train

# 2. Verify model
python main_controller.py predict --hours 24

# 3. Start server (no debug)
python main_controller.py server

# 4. Monitor logs
tail -f weather_prediction.log

# 5. Set up scheduled retraining
# (cron job or scheduled task)
```

## 🎯 Best Practices

1. **Always train before first use**
2. **Monitor training status** when retraining via API
3. **Check logs** for errors
4. **Backup models** before retraining
5. **Test predictions** after training
6. **Monitor system health** regularly
7. **Schedule regular retraining** (weekly/monthly)
8. **Keep data updated** for better accuracy

## 📚 Additional Resources

- `SYSTEM_ARCHITECTURE.md` - System architecture details
- `DEPLOYMENT_GUIDE.md` - Deployment instructions
- `README_ADVANCED.md` - Complete documentation
- `COMPLETE_PLAN.md` - Implementation details

# User Guide

This guide explains how to use the Weather App UI, including the Daily Forecast view and location permissions.

## Getting Started
- Open `http://localhost:5000/` in your browser.
- On first load, the app shows weather for a default location (Karachi) unless you grant location access.
- Click "Use My Location" in the header to explicitly trigger the browser’s location permission prompt.

## Location Permission
- When you click "Use My Location", your browser will ask to allow location.
- If you allow, the app fetches weather for your current location.
- If you deny or your environment blocks geolocation, the app continues with the default location and shows a notification.

Tips if the prompt doesn’t appear:
- Check site permissions for `http://localhost:5000/` and allow location.
- Clear blocked permissions and reload the page.
- Some managed/work environments may disable geolocation policies.

## Daily Forecast and Hourly Views
- Use the "Daily Forecast" button to view the 7-day outlook inline.
- Switch between daily and hourly views where available; both are styled to match the app’s theme.
- Charts and stats are displayed in the main content area for quick scanning.

## Notes and Known Behavior
- If the background image is blocked due to ORB policies, it won’t affect core functionality. You can remove/replace the background image in `app.html` if desired.
- Location is requested only when you click "Use My Location" to keep control in your hands and avoid repeated prompts.

## Troubleshooting
- Permission blocked: Reset site permissions and retry the button.
- Still no prompt: Try a different browser or ensure `localhost` is not restricted by policy.
- API issues: Check the terminal logs for errors starting the Flask server.
