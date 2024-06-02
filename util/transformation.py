import math

# Constants for WGS84
a = 6378137.0  # semi-major axis
f = 1 / 298.257223563  # flattening
e2 = 2 * f - f ** 2  # square of eccentricity

def wgs84_to_ecef(lat, lon, alt):
    """
    Convert WGS84 coordinates to ECEF.
    
    Parameters:
    lat -- Latitude in degrees
    lon -- Longitude in degrees
    alt -- Altitude in meters
    
    Returns:
    x, y, z -- ECEF coordinates in meters
    """
    # Convert latitude and longitude from degrees to radians
    lat_rad = math.radians(lat)
    lon_rad = math.radians(lon)
    
    # Calculate N
    N = a / math.sqrt(1 - e2 * math.sin(lat_rad) ** 2)
    
    # Calculate ECEF coordinates
    x = (N + alt) * math.cos(lat_rad) * math.cos(lon_rad)
    y = (N + alt) * math.cos(lat_rad) * math.sin(lon_rad)
    z = (N * (1 - e2) + alt) * math.sin(lat_rad)
    
    return x, y, z

def ecef_to_wgs84(x, y, z):
    """
    Convert ECEF coordinates to WGS84 (latitude, longitude, altitude).
    
    Parameters:
    x, y, z - ECEF coordinates in meters
    
    Returns:
    lat - Latitude in degrees
    lon - Longitude in degrees
    alt - Altitude in meters
    """
    # Calculate longitude
    lon = math.atan2(y, x)

    # Calculate the hypotenuse from the z-axis
    hyp = math.sqrt(x**2 + y**2)
    
    # Iteratively calculate latitude and altitude
    lat = math.atan2(z, hyp * (1 - e2))
    N = a / math.sqrt(1 - e2 * math.sin(lat)**2)
    alt = hyp / math.cos(lat) - N
    
    lat_prev = 0
    while abs(lat - lat_prev) > 1e-12:
        lat_prev = lat
        N = a / math.sqrt(1 - e2 * math.sin(lat)**2)
        alt = hyp / math.cos(lat) - N
        lat = math.atan2(z, hyp * (1 - e2 * (N / (N + alt))))        
    
    return lat, lon, alt