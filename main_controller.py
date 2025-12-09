import argparse
import sys
from pathlib import Path

from weather_app.api_server import WeatherAPIServer
from weather_app.orchestrator import WeatherSystemOrchestrator
from weather_app.logging_utils import logger

# Force UTF-8 encoding for stdout/stderr to support emojis on Windows
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')
    sys.stderr.reconfigure(encoding='utf-8')


def train_model(data_source='api', start_date=None, end_date=None):
    """Train the model via CLI"""
    print("\n" + "="*70)
    print("🌤️  KARACHI WEATHER PREDICTION SYSTEM - TRAINING")
    print("="*70 + "\n")
    
    orchestrator = WeatherSystemOrchestrator()
    result = orchestrator.train_pipeline(
        data_source=data_source,
        start_date=start_date,
        end_date=end_date,
        retrain=True
    )
    
    if result['success']:
        print("\n✅ Training completed successfully!")
        print(f"   MAE: {result['metrics']['mae']:.3f}°C")
        print(f"   RMSE: {result['metrics']['rmse']:.3f}°C")
        print(f"   R²: {result['metrics']['r2']:.3f}")
        return 0
    else:
        print(f"\n❌ Training failed: {result.get('error', 'Unknown error')}")
        return 1


def start_server(host='0.0.0.0', port=5000, debug=False):
    """Start the API server"""
    print("\n" + "="*70)
    print("🌤️  KARACHI WEATHER PREDICTION SYSTEM - API SERVER")
    print("="*70)
    print(f"🚀 Starting server on http://{host}:{port}")
    print(f"📡 API endpoints available at http://{host}:{port}/api/")
    print(f"🌐 UI available at http://{host}:{port}/")
    print("="*70 + "\n")
    
    project_root = Path(__file__).resolve().parent
    server = WeatherAPIServer(host=host, port=port, debug=debug, ui_root=project_root)
    server.run()


def predict(hours=24):
    """Make a prediction via CLI"""
    print(f"\n🔮 Predicting weather for next {hours} hours...\n")
    
    orchestrator = WeatherSystemOrchestrator()
    orchestrator.load_model()
    
    result = orchestrator.predict(hours=hours)
    
    if result['success']:
        print("Predictions:")
        print("-" * 70)
        for pred in result['predictions']:
            print(f"  {pred['time']}: {pred['temp']:.1f}°C")
        return 0
    else:
        print(f"❌ Prediction failed: {result.get('error', 'Unknown error')}")
        return 1


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description='Karachi Weather Prediction System',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Start API server
  python main_controller.py server
  
  # Train model
  python main_controller.py train
  
  # Make prediction
  python main_controller.py predict --hours 24
  
  # Start server on custom port
  python main_controller.py server --port 8080
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Command to execute')
    
    # Server command
    server_parser = subparsers.add_parser('server', help='Start API server')
    server_parser.add_argument('--host', default='0.0.0.0', help='Host to bind to')
    server_parser.add_argument('--port', type=int, default=5000, help='Port to bind to')
    server_parser.add_argument('--debug', action='store_true', help='Enable debug mode')
    
    # Train command
    train_parser = subparsers.add_parser('train', help='Train the model')
    train_parser.add_argument('--data-source', choices=['api', 'file'], default='api',
                             help='Data source (api or file)')
    train_parser.add_argument('--start-date', help='Start date (YYYY-MM-DD)')
    train_parser.add_argument('--end-date', help='End date (YYYY-MM-DD)')
    
    # Predict command
    predict_parser = subparsers.add_parser('predict', help='Make a prediction')
    predict_parser.add_argument('--hours', type=int, default=24, help='Hours to predict')
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return 1
    
    try:
        if args.command == 'server':
            start_server(host=args.host, port=args.port, debug=args.debug)
        elif args.command == 'train':
            return train_model(
                data_source=args.data_source,
                start_date=args.start_date,
                end_date=args.end_date
            )
        elif args.command == 'predict':
            return predict(hours=args.hours)
        else:
            parser.print_help()
            return 1
    except KeyboardInterrupt:
        print("\n\n⚠️  Interrupted by user")
        return 1
    except Exception as e:
        logger.error(f"Error: {e}")
        print(f"\n❌ Error: {e}")
        return 1


if __name__ == '__main__':
    sys.exit(main())

