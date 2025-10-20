// Weather API integrations
const KARACHI_LAT = 24.8607;
const KARACHI_LON = 67.0011;

// OpenMeteo API
async function fetchOpenMeteoData() {
    const response = await fetch(
        `https://api.open-meteo.com/v1/forecast?latitude=${KARACHI_LAT}&longitude=${KARACHI_LON}&current=temperature_2m,relative_humidity_2m,precipitation,weather_code,cloud_cover,pressure_msl,wind_speed_10m&timezone=Asia/Karachi`
    );
    const data = await response.json();
    return {
        source: 'OpenMeteo',
        temperature: data.current.temperature_2m,
        humidity: data.current.relative_humidity_2m,
        pressure: data.current.pressure_msl,
        wind_speed: data.current.wind_speed_10m,
        precipitation: data.current.precipitation,
        cloud_cover: data.current.cloud_cover,
        weather_code: data.current.weather_code
    };
}

// OpenWeatherMap API
async function fetchOpenWeatherData() {
    const API_KEY = 'YOUR_API_KEY'; // Replace with your API key
    const response = await fetch(
        `https://api.openweathermap.org/data/2.5/weather?lat=${KARACHI_LAT}&lon=${KARACHI_LON}&appid=${API_KEY}&units=metric`
    );
    const data = await response.json();
    return {
        source: 'OpenWeather',
        temperature: data.main.temp,
        humidity: data.main.humidity,
        pressure: data.main.pressure,
        wind_speed: data.wind.speed,
        precipitation: data.rain ? data.rain['1h'] || 0 : 0,
        cloud_cover: data.clouds.all,
        weather_code: data.weather[0].id
    };
}

// WeatherAPI
async function fetchWeatherAPIData() {
    const API_KEY = 'YOUR_API_KEY'; // Replace with your API key
    const response = await fetch(
        `https://api.weatherapi.com/v1/current.json?key=${API_KEY}&q=${KARACHI_LAT},${KARACHI_LON}`
    );
    const data = await response.json();
    return {
        source: 'WeatherAPI',
        temperature: data.current.temp_c,
        humidity: data.current.humidity,
        pressure: data.current.pressure_mb,
        wind_speed: data.current.wind_kph / 3.6, // Convert to m/s
        precipitation: data.current.precip_mm,
        cloud_cover: data.current.cloud,
        weather_code: data.current.condition.code
    };
}

// Intelligent data aggregation with outlier detection
function aggregateWeatherData(dataSources) {
    const temperatures = dataSources.map(d => d.temperature);
    const humidities = dataSources.map(d => d.humidity);
    const pressures = dataSources.map(d => d.pressure);
    const windSpeeds = dataSources.map(d => d.wind_speed);
    const precipitations = dataSources.map(d => d.precipitation);
    const cloudCovers = dataSources.map(d => d.cloud_cover);

    // Calculate median for each metric (more robust than mean)
    const getMedian = arr => {
        const sorted = [...arr].sort((a, b) => a - b);
        const mid = Math.floor(sorted.length / 2);
        return sorted.length % 2 ? sorted[mid] : (sorted[mid - 1] + sorted[mid]) / 2;
    };

    // Detect and remove outliers using IQR method
    const filterOutliers = arr => {
        const q1 = getMedian(arr.slice(0, Math.floor(arr.length / 2)));
        const q3 = getMedian(arr.slice(Math.ceil(arr.length / 2)));
        const iqr = q3 - q1;
        const bounds = [q1 - 1.5 * iqr, q3 + 1.5 * iqr];
        return arr.filter(x => x >= bounds[0] && x <= bounds[1]);
    };

    // Calculate weighted average based on data reliability
    const calculateWeightedAverage = arr => {
        if (arr.length === 0) return 0;
        if (arr.length === 1) return arr[0];
        
        const filtered = filterOutliers(arr);
        if (filtered.length === 0) return getMedian(arr);
        
        return filtered.reduce((a, b) => a + b) / filtered.length;
    };

    return {
        temperature: calculateWeightedAverage(temperatures),
        humidity: calculateWeightedAverage(humidities),
        pressure: calculateWeightedAverage(pressures),
        wind_speed: calculateWeightedAverage(windSpeeds),
        precipitation: calculateWeightedAverage(precipitations),
        cloud_cover: calculateWeightedAverage(cloudCovers),
        confidence_score: calculateConfidenceScore(dataSources)
    };
}

// Calculate confidence score based on data agreement
function calculateConfidenceScore(dataSources) {
    if (dataSources.length === 1) return 0.7; // Single source
    
    const maxDeviation = {
        temperature: 2,    // °C
        humidity: 10,      // %
        pressure: 5,       // hPa
        wind_speed: 2,     // m/s
        precipitation: 1,  // mm
        cloud_cover: 20    // %
    };

    let scores = [];
    for (const metric in maxDeviation) {
        const values = dataSources.map(d => d[metric]);
        const range = Math.max(...values) - Math.min(...values);
        const normalizedRange = range / maxDeviation[metric];
        scores.push(Math.max(0, 1 - normalizedRange));
    }

    return scores.reduce((a, b) => a + b) / scores.length;
}

// Export functions
export { 
    fetchOpenMeteoData, 
    fetchOpenWeatherData, 
    fetchWeatherAPIData, 
    aggregateWeatherData 
};