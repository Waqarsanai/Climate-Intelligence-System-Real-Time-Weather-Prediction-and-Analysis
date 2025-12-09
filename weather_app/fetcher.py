import requests
from datetime import datetime
from typing import List, Tuple, Dict, Any
import time

from .logging_utils import logger
from .config import CONFIG


class RealTimeWeatherDataFetcher:
    """Fetch real-time weather data from external APIs or local sources."""

    def __init__(self, city=CONFIG['city']):
        self.city = city
        self.lat = CONFIG['coordinates']['lat']
        self.lon = CONFIG['coordinates']['lon']

    def fetch(self):
        """Return a normalized dict of current weather. Placeholder implementation."""
        # Prefer Open-Meteo realtime if available; fallback to local-simulated
        try:
            data = self._fetch_open_meteo_realtime(self.lat, self.lon)
            logger.info(f"Fetched realtime weather(Open-Meteo) for {self.city}: {data['temperature']}°C")
            return data
        except Exception as e:
            logger.warning(f"Falling back to local-simulated realtime due to: {e}")
            data = {
                'city': self.city,
                'temperature': 32.5,
                'feels_like': 35.0,
                'humidity': 62,
                'wind_speed': 4.2,
                'pressure': 1012.5,
                'precipitation': 0.0,
                'cloud_cover': 35,
                'description': 'Partly cloudy',
                'source': 'local-simulated',
                'reliability': 75,
                'timestamp': datetime.now().isoformat(timespec='seconds'),
            }
            return data

    # Backward-compatible alias used by legacy app routes
    def get_real_time_weather(self):
        """Alias to `fetch()` for compatibility with existing endpoints."""
        return self.fetch()

    def fetch_training_series(self, hours: int = 72) -> Dict[str, Any]:
        """Fetch authentic hourly temperature series for past N hours using Open‑Meteo.

        Returns dict with keys: 'times' (List[datetime]), 'temps' (List[float]).
        Raises on failure.
        """
        times, temps = self._fetch_open_meteo_hourly(self.lat, self.lon, past_days=max(1, hours // 24))
        # Trim to requested hours from the end
        if len(times) > hours:
            times = times[-hours:]
            temps = temps[-hours:]
        return {'times': times, 'temps': temps}

    def fetch_area_realtime_by_coords(self, lat: float, lon: float) -> Dict[str, Any]:
        """
        Fetch current weather for a specific location by coordinates using Open‑Meteo.
        Returns a normalized dict with keys: temperature, humidity, wind_speed, description, timestamp, lat, lon.
        """
        try:
            data = self._fetch_open_meteo_realtime(lat, lon)
            data["lat"] = lat
            data["lon"] = lon
            # Try to label area/city name via reverse geocoding
            try:
                area = self._reverse_geocode(lat, lon)
                if area:
                    data["city"] = area
            except Exception as ge:
                logger.debug(f"Reverse geocoding failed: {ge}")
            return data
        except Exception as e:
            logger.warning(f"Area realtime fetch failed ({lat},{lon}); using fallback: {e}")
            fallback = {
                'city': self.city,
                'temperature': 32.0,
                'feels_like': 33.0,
                'humidity': 60,
                'wind_speed': 4.0,
                'pressure': 1012.0,
                'precipitation': 0.0,
                'cloud_cover': 30,
                'description': 'Partly cloudy',
                'source': 'local-simulated',
                'reliability': 70,
                'timestamp': datetime.now().isoformat(timespec='seconds'),
                'lat': lat,
                'lon': lon,
            }
            return fallback

    def fetch_areas_current(self) -> Dict[str, Dict[str, Any]]:
        """
        Fetch realtime weather for all configured areas.
        Returns a mapping: area_name -> { temperature, lat, lon, description, timestamp }
        """
        areas_data: Dict[str, Dict[str, Any]] = {}
        for area_name, info in CONFIG.get("areas", {}).items():
            lat = info.get("lat")
            lon = info.get("lon")
            if lat is None or lon is None:
                continue
            datum = self.fetch_area_realtime_by_coords(lat, lon)
            areas_data[area_name] = {
                "temperature": datum.get("temperature"),
                "lat": datum.get("lat", lat),
                "lon": datum.get("lon", lon),
                "description": datum.get("description"),
                "timestamp": datum.get("timestamp"),
            }
        return areas_data

    # --------------------------
    # Open‑Meteo provider helpers
    # --------------------------
    def _fetch_open_meteo_realtime(self, lat: float, lon: float) -> Dict[str, Any]:
        url = (
            f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}"
            f"&current_weather=true"
            f"&hourly=temperature_2m,relative_humidity_2m,wind_speed_10m,precipitation,cloudcover,pressure_msl"
            f"&past_days=1&timezone=auto&windspeed_unit=kmh"
        )
        j = self._get_json(url)
        cw = j.get('current_weather') or {}
        # Map WMO weathercode to description
        code = cw.get('weathercode')
        desc = self._wmo_to_description(code)
        data = {
            'city': self.city,
            'temperature': float(cw.get('temperature')) if cw.get('temperature') is not None else None,
            'feels_like': float(cw.get('temperature')) if cw.get('temperature') is not None else None,
            'humidity': None,  # enriched from hourly below
            'wind_speed': float(cw.get('windspeed')) if cw.get('windspeed') is not None else None,  # km/h ensured
            'pressure': None,  # enriched from hourly below (hPa)
            'precipitation': None,
            'cloud_cover': None,
            'description': desc,
            'source': 'open-meteo',
            'reliability': 95,
            'timestamp': cw.get('time') or datetime.now().isoformat(timespec='seconds'),
        }
        # Try to enrich from hourly arrays at current time index
        hourly = j.get('hourly') or {}
        times = hourly.get('time') or []
        if times:
            # Find the closest time match (current time or most recent past time)
            current_time = cw.get('time')
            idx = -1
            if current_time and current_time in times:
                idx = times.index(current_time)
            elif times:
                # If exact match not found, use the last available time (most recent)
                idx = len(times) - 1
            
            if idx >= 0:
                try:
                    hums = hourly.get('relative_humidity_2m') or []
                    precs = hourly.get('precipitation') or []
                    clouds = hourly.get('cloudcover') or []
                    press = hourly.get('pressure_msl') or []
                    data['humidity'] = float(hums[idx]) if idx < len(hums) and hums[idx] is not None else data['humidity']
                    data['precipitation'] = float(precs[idx]) if idx < len(precs) and precs[idx] is not None else data['precipitation']
                    data['cloud_cover'] = float(clouds[idx]) if idx < len(clouds) and clouds[idx] is not None else data['cloud_cover']
                    data['pressure'] = float(press[idx]) if idx < len(press) and press[idx] is not None else data['pressure']
                except Exception:
                    pass
        return data


    def _fetch_open_meteo_hourly(self, lat: float, lon: float, past_days: int = 2) -> Tuple[List[datetime], List[float]]:
        url = (
            f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}"
            f"&hourly=temperature_2m&past_days={past_days}&timezone=auto"
        )
        j = self._get_json(url)
        hourly = j.get('hourly') or {}
        times_str = hourly.get('time') or []
        temps = hourly.get('temperature_2m') or []
        times = []
        for t in times_str:
            try:
                times.append(datetime.fromisoformat(t))
            except Exception:
                # some endpoints may include timezone suffix; be lenient
                try:
                    times.append(datetime.fromisoformat(t.replace('Z', '')))
                except Exception:
                    pass
        temps = [float(x) for x in temps if x is not None]
        # Align lengths
        n = min(len(times), len(temps))
        return times[:n], temps[:n]

    def _reverse_geocode(self, lat: float, lon: float) -> str:
        """Resolve a human-readable area name from coordinates using Nominatim.
        Returns suburb/neighbourhood/city when available.
        """
        try:
            url = (
                f"https://nominatim.openstreetmap.org/reverse?lat={lat}&lon={lon}&format=json&zoom=14"
            )
            headers = {"User-Agent": "WeatherApp/1.0 (https://localhost)"}
            j = self._get_json(url, headers=headers, timeout=8)
            addr = j.get('address', {})
            # Prefer detailed locality fields
            for key in ['suburb', 'neighbourhood', 'quarter', 'village', 'town', 'city_district']:
                if addr.get(key):
                    return addr.get(key)
            return addr.get('city') or addr.get('county') or j.get('display_name')
        except Exception as e:
            logger.debug(f"Reverse geocode error for ({lat},{lon}): {e}")
            return ''

    def _get_json(self, url: str, headers: Dict[str, str] | None = None, timeout: int = 10) -> Dict[str, Any]:
        """HTTP GET with small retry/backoff; raises last error if all retries fail."""
        last_err = None
        for attempt in range(3):
            try:
                r = requests.get(url, headers=headers, timeout=timeout)
                r.raise_for_status()
                return r.json()
            except Exception as e:
                last_err = e
                # jittered backoff: 0.5s, 1.0s
                if attempt < 2:
                    time.sleep(0.5 * (attempt + 1))
        raise last_err

    def _fetch_open_meteo_hourly_future(
        self, lat: float, lon: float, forecast_days: int = 2
    ) -> Dict[str, List[Any]]:
        """
        Fetch future hourly forecast for temperature, humidity, wind, pressure, cloud cover.
        Returns aligned lists: times, temperature, humidity, wind_speed, pressure, cloud_cover.
        """
        url = (
            f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}"
            f"&hourly=temperature_2m,relative_humidity_2m,wind_speed_10m,pressure_msl,cloudcover"
            f"&forecast_days={forecast_days}&timezone=auto&windspeed_unit=kmh"
        )
        j = self._get_json(url)
        hourly = j.get('hourly') or {}
        times_str = hourly.get('time') or []
        temps = hourly.get('temperature_2m') or []
        hums = hourly.get('relative_humidity_2m') or []
        winds = hourly.get('wind_speed_10m') or []
        press = hourly.get('pressure_msl') or []
        clouds = hourly.get('cloudcover') or []

        times: List[datetime] = []
        for t in times_str:
            try:
                times.append(datetime.fromisoformat(t))
            except Exception:
                try:
                    times.append(datetime.fromisoformat(t.replace('Z', '')))
                except Exception:
                    pass

        # Align lengths by minimum
        n = min(len(times), len(temps), len(hums), len(winds), len(press), len(clouds))
        return {
            'times': times[:n],
            'temperature': [float(x) if x is not None else None for x in temps[:n]],
            'humidity': [float(x) if x is not None else None for x in hums[:n]],
            'wind_speed': [float(x) if x is not None else None for x in winds[:n]],
            'pressure': [float(x) if x is not None else None for x in press[:n]],
            'cloud_cover': [float(x) if x is not None else None for x in clouds[:n]],
        }

    def _fetch_open_meteo_daily_future(self, lat: float, lon: float, days: int = 7) -> Dict[str, List[Any]]:
        """
        Fetch future daily forecast for next N days (default 7).
        Returns lists: dates, temperature_max, temperature_min, precipitation_sum, wind_speed_max.
        """
        url = (
            f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}"
            f"&daily=temperature_2m_max,temperature_2m_min,precipitation_sum,wind_speed_10m_max"
            f"&forecast_days={days}&timezone=auto&windspeed_unit=kmh"
        )
        j = self._get_json(url)
        daily = j.get('daily') or {}
        dates_str = daily.get('time') or []
        tmax = daily.get('temperature_2m_max') or []
        tmin = daily.get('temperature_2m_min') or []
        precip = daily.get('precipitation_sum') or []
        windmax = daily.get('wind_speed_10m_max') or []

        dates: List[datetime] = []
        for d in dates_str:
            try:
                dates.append(datetime.fromisoformat(d))
            except Exception:
                try:
                    dates.append(datetime.fromisoformat(d.replace('Z', '')))
                except Exception:
                    pass

        n = min(len(dates), len(tmax), len(tmin), len(precip), len(windmax))
        return {
            'dates': dates[:n],
            'temperature_max': [float(x) if x is not None else None for x in tmax[:n]],
            'temperature_min': [float(x) if x is not None else None for x in tmin[:n]],
            'precipitation_sum': [float(x) if x is not None else None for x in precip[:n]],
            'wind_speed_max': [float(x) if x is not None else None for x in windmax[:n]],
            'lat': lat,
            'lon': lon,
        }

    def fetch_area_future_hourly(self, lat: float, lon: float, hours: int = 24) -> Dict[str, List[Any]]:
        """
        Fetch next N hours of hourly forecast for a specific area.
        Filters to times >= now and truncates to requested hours.
        """
        forecast_days = max(1, (hours + 23) // 24)
        data = self._fetch_open_meteo_hourly_future(lat, lon, forecast_days=forecast_days)
        now = datetime.now()
        filtered = [
            (t, temp, hum, wind, prs, cld)
            for t, temp, hum, wind, prs, cld in zip(
                data['times'],
                data['temperature'],
                data['humidity'],
                data['wind_speed'],
                data['pressure'],
                data['cloud_cover'],
            )
            if t >= now
        ]
        # Take the first `hours` future points
        filtered = filtered[:hours]
        times, temps, hums, winds, press, clouds = (
            [x[0] for x in filtered],
            [x[1] for x in filtered],
            [x[2] for x in filtered],
            [x[3] for x in filtered],
            [x[4] for x in filtered],
            [x[5] for x in filtered],
        )
        return {
            'times': times,
            'temperature': temps,
            'humidity': hums,
            'wind_speed': winds,
            'pressure': press,
            'cloud_cover': clouds,
            'lat': lat,
            'lon': lon,
        }

    def _wmo_to_description(self, code: Any) -> str:
        try:
            code = int(code)
        except Exception:
            return 'Unknown'
        # Minimal mapping
        mapping = {
            0: 'Clear sky',
            1: 'Mainly clear',
            2: 'Partly cloudy',
            3: 'Overcast',
            45: 'Fog',
            48: 'Depositing rime fog',
            51: 'Light drizzle',
            53: 'Moderate drizzle',
            55: 'Dense drizzle',
            61: 'Slight rain',
            63: 'Moderate rain',
            65: 'Heavy rain',
            71: 'Slight snow',
            73: 'Moderate snow',
            75: 'Heavy snow',
            80: 'Rain showers',
            81: 'Rain showers',
            82: 'Violent rain showers',
            95: 'Thunderstorm',
        }
        return mapping.get(code, 'Unknown')
