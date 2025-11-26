# System Architecture & Control Flow Documentation

## 📁 Project Structure

```
Project/
├── weather_app/                    # Main application package
│   ├── __init__.py
│   ├── config.py                  # Configuration settings
│   ├── data_loader.py             # Loads raw weather data
│   ├── cleaner.py                 # Cleans and preprocesses data
│   ├── features.py                # Feature engineering
│   ├── model_trainer.py           # Trains ML/DL models
│   ├── predictor.py               # Makes predictions (basic)
│   ├── advanced_predictor.py      # Advanced predictions with ensemble
│   ├── orchestrator.py            # Main controller/orchestrator
│   ├── api_server.py              # Flask API server
│   ├── fetcher.py                 # Fetches real-time weather
│   ├── ensemble.py                # Ensemble model methods
│   ├── hyperparameter_tuner.py    # Hyperparameter optimization
│   └── visualizer.py              # Visualization utilities
│
├── main_controller.py             # Unified CLI + API host
├── train_advanced_model.py        # Standalone training script
├── app.html                       # Frontend UI
│
├── models/                        # Trained models storage
├── data/                          # Data storage
├── cache/                         # Cache files
└── weather_visualizations/        # Generated visualizations
```

## 🔄 Control Flow

### 1. Training Pipeline Flow

```
┌─────────────────┐
│  User/CLI       │
│  Triggers Train │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ orchestrator.py │ ◄─── Main Controller
│ train_pipeline()│
└────────┬────────┘
         │
         ├──► Step 1: data_loader.py
         │    └─── load_from_open_meteo() or load_from_file()
         │         Returns: Raw DataFrame
         │
         ├──► Step 2: cleaner.py
         │    └─── clean_dataframe()
         │         Returns: Cleaned DataFrame
         │
         ├──► Step 3: features.py
         │    └─── create_features()
         │         Returns: DataFrame with 100+ features
         │
         ├──► Step 4: data_loader.py
         │    └─── create_training_dataset()
         │         Returns: train_df, val_df, test_df
         │
         ├──► Step 5: model_trainer.py
         │    └─── train_all_models()
         │         Trains: RF, XGBoost, LightGBM, CatBoost, LSTM, etc.
         │         Returns: Model results and metrics
         │
         ├──► Step 6: ensemble.py
         │    └─── fit_weighted_average()
         │         Creates ensemble from all models
         │
         ├──► Step 7: Evaluate on test set
         │    └─── Calculate metrics (MAE, RMSE, R²)
         │
         └──► Step 8: Save models
              └─── Save to models/ directory
                   - Individual models (.pkl or .h5)
                   - Ensemble model (.pkl)
                   - Metadata (.json)
```

### 2. Prediction Flow

```
┌─────────────────┐
│  API Request    │
│  /api/predict   │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  api_server.py  │
│  predict()      │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ orchestrator.py │
│ predict()       │
└────────┬────────┘
         │
         ├──► fetcher.py
         │    └─── fetch() - Get current weather
         │
         ├──► advanced_predictor.py
         │    └─── predict_temperature()
         │         ├─── Prepare features
         │         ├─── Get predictions from all models
         │         └─── Ensemble predictions
         │
         └──► Return JSON response
              └─── {predictions: [...], current_weather: {...}}
```

### 3. API Request Flow

```
┌──────────────┐
│  Frontend UI │
│  (app.html)  │
└──────┬───────┘
       │
       │ HTTP Request
       │ (GET/POST)
       ▼
┌─────────────────┐
│  api_server.py  │
│  Flask Routes   │
└──────┬──────────┘
       │
       ├──► /api/predict → orchestrator.predict()
       ├──► /api/retrain → orchestrator.train_pipeline()
       ├──► /api/weather/current → fetcher.fetch()
       ├──► /api/history → data_loader.load_dataset()
       └──► /api/training/status → orchestrator.get_training_status()
       │
       ▼
┌─────────────────┐
│ orchestrator.py │
│ (Orchestrates)  │
└─────────────────┘
```

## 🧩 Module Responsibilities

### 1. `data_loader.py`
**Purpose**: Load raw weather data from various sources

**Key Functions**:
- `load_from_open_meteo()` - Fetch from Open-Meteo API
- `load_from_file()` - Load from CSV/text files
- `create_training_dataset()` - Split into train/val/test
- `prepare_features_target()` - Prepare X, y arrays

**Input**: API parameters or file path
**Output**: Raw DataFrame or train/val/test splits

### 2. `cleaner.py` (wraps `data_cleaner.py`)
**Purpose**: Clean and preprocess raw data

**Key Functions**:
- `clean_dataframe()` - Main cleaning pipeline
  - Fix timestamps
  - Remove duplicates
  - Convert units (F to C)
  - Handle missing values
  - Remove outliers
  - Validate ranges

**Input**: Raw DataFrame
**Output**: Cleaned DataFrame

### 3. `features.py` (wraps `feature_engineer.py`)
**Purpose**: Create comprehensive features

**Key Functions**:
- `create_features()` - Main feature engineering
  - Temporal features (hour, day, month, cyclical)
  - Lag features (1h, 3h, 6h, 12h, 24h)
  - Rolling statistics (mean, std, min, max)
  - Meteorological features (heat index, dew point)
  - Interaction features
  - Trend features

**Input**: Cleaned DataFrame
**Output**: DataFrame with 100+ features

### 4. `model_trainer.py`
**Purpose**: Train multiple ML/DL models

**Key Functions**:
- `train_all_models()` - Train all available models
  - Random Forest
  - Gradient Boosting
  - XGBoost
  - LightGBM
  - CatBoost
  - LSTM
  - BiLSTM
  - GRU
  - CNN-LSTM

**Input**: X_train, y_train, X_val, y_val
**Output**: Trained models and metrics

### 5. `predictor.py` / `advanced_predictor.py`
**Purpose**: Make weather predictions

**Key Functions**:
- `predict_temperature()` - Generate predictions
  - Prepare features for future timestamps
  - Get predictions from all models
  - Ensemble predictions
  - Apply smoothing

**Input**: Current weather, hours to predict
**Output**: List of predictions

### 6. `orchestrator.py`
**Purpose**: Main controller that coordinates all modules

**Key Functions**:
- `train_pipeline()` - Complete training pipeline
- `predict()` - Generate predictions
- `load_model()` - Load trained models
- `get_training_status()` - Get training progress
- `get_system_status()` - Get system health

**Responsibilities**:
- Coordinate data flow between modules
- Handle errors and exceptions
- Manage training status
- Save/load models

### 7. `api_server.py`
**Purpose**: Flask API server exposing REST endpoints

**Key Endpoints**:
- `GET /health` - Health check
- `GET/POST /api/predict` - Get predictions
- `GET /api/predict/24h` - 24-hour forecast
- `GET /api/predict/7d` - 7-day forecast
- `GET /api/weather/current` - Current weather
- `GET /api/history` - Historical data
- `POST /api/retrain` - Trigger retraining
- `GET /api/training/status` - Training status
- `GET /api/system/status` - System status
- `GET /api/models/list` - List available models

**Responsibilities**:
- Handle HTTP requests
- Validate input
- Call orchestrator methods
- Return JSON responses
- Handle errors gracefully

## 🔌 Integration Points

### Training Integration
```python
# orchestrator.py coordinates:
data_loader → cleaner → features → model_trainer → ensemble → save
```

### Prediction Integration
```python
# api_server.py → orchestrator.py → predictor.py
API Request → orchestrator.predict() → fetcher.fetch() → predictor.predict_temperature()
```

### UI Integration
```html
<!-- app.html makes AJAX calls to API -->
fetch('/api/predict?hours=24')
  .then(response => response.json())
  .then(data => updateUI(data))
```

## 🚀 Execution Paths

### Path 1: Training via CLI
```bash
python main_controller.py train
```
1. `main_controller.py` → `train_model()`
2. Creates `WeatherSystemOrchestrator()`
3. Calls `orchestrator.train_pipeline()`
4. Executes full training pipeline
5. Saves models

### Path 2: Training via API
```bash
POST /api/retrain
```
1. `api_server.py` receives request
2. Checks training status
3. Starts background thread
4. Calls `orchestrator.train_pipeline()`
5. Returns status immediately
6. Training continues in background

### Path 3: Prediction via API
```bash
GET /api/predict?hours=24
```
1. `api_server.py` receives request
2. Validates parameters
3. Calls `orchestrator.predict()`
4. `orchestrator` loads model (if needed)
5. Fetches current weather
6. Generates predictions
7. Returns JSON response

### Path 4: Starting Server
```bash
python main_controller.py server
```
1. Creates `WeatherAPIServer()`
2. Initializes `WeatherSystemOrchestrator()`
3. Loads existing model
4. Registers API routes
5. Starts Flask server
6. Serves UI at `/`
7. Serves API at `/api/*`

## 🔒 Error Handling

### Data Loading Errors
- **Missing data**: Returns empty DataFrame, logs error
- **API failure**: Falls back to file, logs warning
- **Invalid format**: Raises exception, returns error response

### Training Errors
- **Insufficient data**: Validates before training, returns error
- **Model failure**: Continues with other models, logs warning
- **Memory error**: Catches exception, suggests reducing data

### Prediction Errors
- **Model not loaded**: Attempts to load, returns error if fails
- **API failure**: Returns cached data or error message
- **Invalid input**: Validates parameters, returns 400 error

### API Errors
- **500 errors**: Logged, returns JSON error response
- **404 errors**: Returns appropriate error message
- **400 errors**: Returns validation error details

## 📊 Data Flow Diagram

```
┌─────────────┐
│  Data Source│
│ (API/File)  │
└──────┬──────┘
       │
       ▼
┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│ data_loader │────►│   cleaner   │────►│   features  │
└─────────────┘     └─────────────┘     └──────┬──────┘
                                                │
                                                ▼
                                       ┌─────────────┐
                                       │train/val/   │
                                       │test split   │
                                       └──────┬──────┘
                                              │
                                              ▼
                                       ┌─────────────┐
                                       │model_trainer│
                                       └──────┬──────┘
                                              │
                                              ▼
                                       ┌─────────────┐
                                       │  ensemble   │
                                       └──────┬──────┘
                                              │
                                              ▼
                                       ┌─────────────┐
                                       │ Save Models │
                                       └─────────────┘

┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│   fetcher   │────►│ orchestrator│────►│  predictor  │
│(current wx) │     │             │     │             │
└─────────────┘     └──────┬──────┘     └──────┬──────┘
                           │                    │
                           │                    ▼
                           │            ┌─────────────┐
                           │            │  Ensemble   │
                           │            │ Predictions │
                           │            └──────┬──────┘
                           │                   │
                           └───────────────────┘
                                   │
                                   ▼
                            ┌─────────────┐
                            │ JSON Response│
                            └─────────────┘
```

## 🔄 State Management

### Training State
- `status`: 'not_started' | 'in_progress' | 'completed' | 'failed'
- `progress`: 0-100 (percentage)
- `message`: Current status message
- `error`: Error message if failed

### System State
- `is_trained`: Boolean (model loaded and ready)
- `model_loaded`: Boolean (model file loaded)
- `training_status`: Current training state

## 🎯 Best Practices Implemented

1. **Separation of Concerns**: Each module has single responsibility
2. **Error Handling**: Comprehensive try-catch blocks
3. **Logging**: All operations logged
4. **Modularity**: Modules can be used independently
5. **Extensibility**: Easy to add new models/features
6. **Production Ready**: Error handling, logging, status tracking
7. **API Design**: RESTful endpoints, JSON responses
8. **Async Training**: Non-blocking training via threads

## 🔐 Security Considerations

1. **Input Validation**: All API inputs validated
2. **Error Messages**: Don't expose internal details
3. **Rate Limiting**: Can be added to API endpoints
4. **Authentication**: Can be added for production
5. **CORS**: Enabled for frontend access
6. **File Paths**: Validated to prevent directory traversal

## 📈 Scalability

### Current Architecture Supports:
- Multiple concurrent API requests
- Background training
- Model versioning
- Multiple data sources

### Future Enhancements:
- Database for historical data
- Redis for caching
- Message queue for training jobs
- Load balancing for API
- Microservices architecture

## 🧪 Testing Flow

1. **Unit Tests**: Test each module independently
2. **Integration Tests**: Test module interactions
3. **API Tests**: Test endpoints with requests
4. **End-to-End Tests**: Test full pipeline

## 📝 Summary

The system follows a clean, modular architecture:

- **Data Layer**: `data_loader.py`, `cleaner.py`
- **Feature Layer**: `features.py`
- **Model Layer**: `model_trainer.py`, `ensemble.py`
- **Prediction Layer**: `predictor.py`, `advanced_predictor.py`
- **Control Layer**: `orchestrator.py`
- **API Layer**: `api_server.py`
- **UI Layer**: `app.html`

All layers communicate through well-defined interfaces, making the system maintainable, testable, and extensible.
