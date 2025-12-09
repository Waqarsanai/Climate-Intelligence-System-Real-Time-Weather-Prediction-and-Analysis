from pathlib import Path

CONFIG = {
    'city': 'Karachi',
    'coordinates': {'lat': 24.8607, 'lon': 67.0011},
    'areas': {
        'Downtown': {'lat': 24.7711, 'lon': 67.0141, 'coastal_proximity': 0.2, 'elevation': 8, 'urban_density': 0.9},
        'Clifton': {'lat': 24.7898, 'lon': 67.0859, 'coastal_proximity': 0.9, 'elevation': 5, 'urban_density': 0.7},
        'Defence': {'lat': 24.7786, 'lon': 67.0584, 'coastal_proximity': 0.6, 'elevation': 12, 'urban_density': 0.6},
        'Gulshan': {'lat': 24.9750, 'lon': 67.0808, 'coastal_proximity': 0.1, 'elevation': 25, 'urban_density': 0.5},
        'DHA': {'lat': 24.8081, 'lon': 67.1258, 'coastal_proximity': 0.8, 'elevation': 15, 'urban_density': 0.4},
        'Malir': {'lat': 24.9639, 'lon': 67.1639, 'coastal_proximity': 0.3, 'elevation': 18, 'urban_density': 0.6},
        'Saddar': {'lat': 24.7778, 'lon': 67.0275, 'coastal_proximity': 0.3, 'elevation': 10, 'urban_density': 0.8},
        'Nazimabad': {'lat': 24.9283, 'lon': 67.0567, 'coastal_proximity': 0.2, 'elevation': 20, 'urban_density': 0.7},
        'Korangi': {'lat': 24.8689, 'lon': 67.1861, 'coastal_proximity': 0.4, 'elevation': 12, 'urban_density': 0.8},
        'Lyari': {'lat': 24.8308, 'lon': 67.0133, 'coastal_proximity': 0.4, 'elevation': 8, 'urban_density': 0.9},
        'Gulistan-e-Johar': {'lat': 24.9129, 'lon': 67.1364, 'coastal_proximity': 0.1, 'elevation': 22, 'urban_density': 0.5},
    },
    'cache_timeout': 1800,
    'model_dir': 'models',
    'data_dir': 'data',
    'cache_dir': 'cache',
    'viz_dir': 'weather_visualizations',
    'allow_synthetic_training': True,
}

# Ensure folders exist
for dir_name in [CONFIG['model_dir'], CONFIG['data_dir'], CONFIG['cache_dir'], CONFIG['viz_dir']]:
    Path(dir_name).mkdir(exist_ok=True)
