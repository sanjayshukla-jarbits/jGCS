"""
Spatial Operations - Geographic and geometric calculations for geofencing

This module provides:
- Distance calculations (2D and 3D)
- Polygon operations and validation
- Coordinate transformations
- Spatial indexing and querying
- Buffer zone calculations
"""

import math
import numpy as np
from typing import Dict, List, Any, Tuple, Optional, Union
from geopy.distance import geodesic, distance
from shapely.geometry import Point, Polygon, LineString, MultiPolygon
from shapely.ops import transform, unary_union
from shapely.validation import make_valid
import pyproj
from functools import partial

class SpatialOperations:
    """Spatial operations for geofencing and navigation"""
    
    def __init__(self):
        # Earth radius in meters
        self.earth_radius = 6371000
        
        # Coordinate reference systems
        self.wgs84 = pyproj.CRS('EPSG:4326')  # WGS84 (lat/lon)
        self.web_mercator = pyproj.CRS('EPSG:3857')  # Web Mercator (for area calculations)
        
    def calculate_distance(self, lat1: float, lon1: float, 
                         lat2: float, lon2: float) -> float:
        """Calculate geodesic distance between two points in meters"""
        
        return geodesic((lat1, lon1), (lat2, lon2)).meters
    
    def calculate_distance_3d(self, point1: Tuple[float, float, float],
                            point2: Tuple[float, float, float]) -> float:
        """Calculate 3D distance between two points (lat, lon, alt)"""
        
        # Horizontal distance
        horizontal_dist = self.calculate_distance(point1[0], point1[1], point2[0], point2[1])
        
        # Vertical distance
        vertical_dist = abs(point1[2] - point2[2])
        
        # 3D distance using Pythagorean theorem
        return math.sqrt(horizontal_dist**2 + vertical_dist**2)
    
    def calculate_bearing(self, lat1: float, lon1: float, 
                        lat2: float, lon2: float) -> float:
        """Calculate initial bearing from point 1 to point 2 in degrees"""
        
        # Convert to radians
        lat1_rad = math.radians(lat1)
        lat2_rad = math.radians(lat2)
        delta_lon_rad = math.radians(lon2 - lon1)
        
        # Calculate bearing
        y = math.sin(delta_lon_rad) * math.cos(lat2_rad)
        x = (math.cos(lat1_rad) * math.sin(lat2_rad) - 
             math.sin(lat1_rad) * math.cos(lat2_rad) * math.cos(delta_lon_rad))
        
        bearing_rad = math.atan2(y, x)
        bearing_deg = math.degrees(bearing_rad)
        
        # Normalize to 0-360 degrees
        return (bearing_deg + 360) % 360
    
    def calculate_destination_point(self, lat: float, lon: float, 
                                  bearing: float, distance: float) -> Tuple[float, float]:
        """Calculate destination point given start point, bearing, and distance"""
        
        # Convert to radians
        lat_rad = math.radians(lat)
        lon_rad = math.radians(lon)
        bearing_rad = math.radians(bearing)
        
        # Angular distance
        angular_dist = distance / self.earth_radius
        
        # Calculate destination
        dest_lat_rad = math.asin(
            math.sin(lat_rad) * math.cos(angular_dist) +
            math.cos(lat_rad) * math.sin(angular_dist) * math.cos(bearing_rad)
        )
        
        dest_lon_rad = lon_rad + math.atan2(
            math.sin(bearing_rad) * math.sin(angular_dist) * math.cos(lat_rad),
            math.cos(angular_dist) - math.sin(lat_rad) * math.sin(dest_lat_rad)
        )
        
        return math.degrees(dest_lat_rad), math.degrees(dest_lon_rad)
    
    def point_in_polygon(self, point: Tuple[float, float], 
                        polygon_coords: List[Tuple[float, float]]) -> bool:
        """Check if point is inside polygon using Shapely"""
        
        try:
            polygon = Polygon(polygon_coords)
            point_geom = Point(point)
            return polygon.contains(point_geom)
        except Exception:
            # Fallback to ray casting algorithm
            return self._point_in_polygon_ray_cast(point, polygon_coords)
    
    def _point_in_polygon_ray_cast(self, point: Tuple[float, float], 
                                  polygon_coords: List[Tuple[float, float]]) -> bool:
        """Ray casting algorithm for point-in-polygon test"""
        
        x, y = point
        n = len(polygon_coords)
        inside = False
        
        p1x, p1y = polygon_coords[0]
        for i in range(1, n + 1):
            p2x, p2y = polygon_coords[i % n]
            
            if y > min(p1y, p2y):
                if y <= max(p1y, p2y):
                    if x <= max(p1x, p2x):
                        if p1y != p2y:
                            xinters = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                        if p1x == p2x or x <= xinters:
                            inside = not inside
            p1x, p1y = p2x, p2y
        
        return inside
    
    def point_to_polygon_distance(self, point: Tuple[float, float], 
                                polygon_coords: List[Tuple[float, float]]) -> float:
        """Calculate minimum distance from point to polygon edge"""
        
        try:
            polygon = Polygon(polygon_coords)
            point_geom = Point(point)
            
            # Distance in degrees
            distance_deg = polygon.distance(point_geom)
            
            # Convert to meters (approximate)
            distance_meters = distance_deg * 111320
            
            return distance_meters
            
        except Exception:
            # Fallback: calculate distance to each edge and return minimum
            return self._point_to_polygon_distance_manual(point, polygon_coords)
    
    def _point_to_polygon_distance_manual(self, point: Tuple[float, float], 
                                        polygon_coords: List[Tuple[float, float]]) -> float:
        """Manual calculation of point to polygon distance"""
        
        min_distance = float('inf')
        
        for i in range(len(polygon_coords)):
            p1 = polygon_coords[i]
            p2 = polygon_coords[(i + 1) % len(polygon_coords)]
            
            # Distance from point to line segment
            segment_distance = self._point_to_line_distance(point, p1, p2)
            min_distance = min(min_distance, segment_distance)
        
        return min_distance
    
    def _point_to_line_distance(self, point: Tuple[float, float], 
                              line_start: Tuple[float, float], 
                              line_end: Tuple[float, float]) -> float:
        """Calculate distance from point to line segment"""
        
        # Use geodesic calculations for accuracy
        line = LineString([line_start, line_end])
        point_geom = Point(point)
        
        # Get closest point on line
        closest_point = line.interpolate(line.project(point_geom))
        
        # Calculate geodesic distance
        return geodesic(point, (closest_point.y, closest_point.x)).meters
    
    def calculate_polygon_area(self, polygon_coords: List[Tuple[float, float]]) -> float:
        """Calculate polygon area in square meters"""
        
        try:
            # Create polygon
            polygon = Polygon(polygon_coords)
            
            # Transform to projected coordinate system for accurate area calculation
            transformer = pyproj.Transformer.from_crs(self.wgs84, self.web_mercator, always_xy=True)
            polygon_projected = transform(transformer.transform, polygon)
            
            # Area in square meters
            return polygon_projected.area
            
        except Exception:
            # Fallback to shoelace formula (approximate)
            return self._calculate_polygon_area_shoelace(polygon_coords)
    
    def _calculate_polygon_area_shoelace(self, coords: List[Tuple[float, float]]) -> float:
        """Calculate polygon area using shoelace formula (approximate)"""
        
        n = len(coords)
        if n < 3:
            return 0
        
        area = 0
        for i in range(n):
            j = (i + 1) % n
            area += coords[i][0] * coords[j][1]
            area -= coords[j][0] * coords[i][1]
        
        area = abs(area) / 2.0
        
        # Convert from degrees squared to meters squared (very approximate)
        # 1 degree ≈ 111,320 meters
        area_m2 = area * (111320 ** 2)
        
        return area_m2
    
    def create_buffer_zone(self, polygon_coords: List[Tuple[float, float]], 
                          buffer_distance_meters: float) -> List[Tuple[float, float]]:
        """Create buffer zone around polygon"""
        
        try:
            # Create polygon
            polygon = Polygon(polygon_coords)
            
            # Convert buffer distance to degrees (approximate)
            # This is rough - better to use projected coordinates
            avg_lat = sum(coord[0] for coord in polygon_coords) / len(polygon_coords)
            buffer_degrees = buffer_distance_meters / (111320 * math.cos(math.radians(avg_lat)))
            
            # Create buffer
            buffered_polygon = polygon.buffer(buffer_degrees)
            
            # Handle MultiPolygon result
            if isinstance(buffered_polygon, MultiPolygon):
                # Return the largest polygon
                largest_polygon = max(buffered_polygon.geoms, key=lambda p: p.area)
                return list(largest_polygon.exterior.coords)
            else:
                return list(buffered_polygon.exterior.coords)
                
        except Exception:
            # Fallback: return original polygon
            return polygon_coords
    
    def create_circular_polygon(self, center: Tuple[float, float], 
                              radius_meters: float, num_points: int = 32) -> List[Tuple[float, float]]:
        """Create circular polygon approximation"""
        
        center_lat, center_lon = center
        polygon_coords = []
        
        for i in range(num_points):
            angle = (2 * math.pi * i) / num_points
            bearing = math.degrees(angle)
            
            point_lat, point_lon = self.calculate_destination_point(
                center_lat, center_lon, bearing, radius_meters
            )
            
            polygon_coords.append((point_lat, point_lon))
        
        # Close the polygon
        polygon_coords.append(polygon_coords[0])
        
        return polygon_coords
    
    def simplify_polygon(self, polygon_coords: List[Tuple[float, float]], 
                        tolerance_meters: float = 10) -> List[Tuple[float, float]]:
        """Simplify polygon by removing redundant points"""
        
        try:
            polygon = Polygon(polygon_coords)
            
            # Convert tolerance to degrees (approximate)
            tolerance_degrees = tolerance_meters / 111320
            
            # Simplify
            simplified_polygon = polygon.simplify(tolerance_degrees, preserve_topology=True)
            
            return list(simplified_polygon.exterior.coords)
            
        except Exception:
            # Return original if simplification fails
            return polygon_coords
    
    def validate_polygon(self, polygon_coords: List[Tuple[float, float]]) -> Dict[str, Any]:
        """Validate polygon and return validation results"""
        
        validation_result = {
            "valid": True,
            "errors": [],
            "warnings": [],
            "corrected_polygon": None
        }
        
        # Check minimum number of points
        if len(polygon_coords) < 3:
            validation_result["valid"] = False
            validation_result["errors"].append("Polygon must have at least 3 points")
            return validation_result
        
        try:
            polygon = Polygon(polygon_coords)
            
            # Check if polygon is valid
            if not polygon.is_valid:
                validation_result["valid"] = False
                validation_result["errors"].append(f"Invalid polygon: {polygon.is_valid_reason}")
                
                # Try to fix the polygon
                try:
                    fixed_polygon = make_valid(polygon)
                    if fixed_polygon.is_valid:
                        validation_result["corrected_polygon"] = list(fixed_polygon.exterior.coords)
                        validation_result["warnings"].append("Polygon was corrected automatically")
                except Exception:
                    pass
            
            # Check for self-intersections
            if polygon.is_valid:
                # Additional checks
                area = polygon.area
                if area == 0:
                    validation_result["warnings"].append("Polygon has zero area")
                
                # Check for very small polygons
                area_m2 = self.calculate_polygon_area(polygon_coords)
                if area_m2 < 1:  # Less than 1 square meter
                    validation_result["warnings"].append("Very small polygon area")
                
                # Check for very large polygons
                if area_m2 > 1000000000:  # 1000 square kilometers
                    validation_result["warnings"].append("Very large polygon area")
        
        except Exception as e:
            validation_result["valid"] = False
            validation_result["errors"].append(f"Polygon validation failed: {str(e)}")
        
        return validation_result
    
    def calculate_polygon_centroid(self, polygon_coords: List[Tuple[float, float]]) -> Tuple[float, float]:
        """Calculate polygon centroid"""
        
        try:
            polygon = Polygon(polygon_coords)
            centroid = polygon.centroid
            return centroid.y, centroid.x  # lat, lon
            
        except Exception:
            # Fallback: arithmetic mean of coordinates
            if not polygon_coords:
                return 0, 0
            
            avg_lat = sum(coord[0] for coord in polygon_coords) / len(polygon_coords)
            avg_lon = sum(coord[1] for coord in polygon_coords) / len(polygon_coords)
            
            return avg_lat, avg_lon
    
    def calculate_polygon_bounds(self, polygon_coords: List[Tuple[float, float]]) -> Dict[str, float]:
        """Calculate polygon bounding box"""
        
        if not polygon_coords:
            return {"min_lat": 0, "max_lat": 0, "min_lon": 0, "max_lon": 0}
        
        lats = [coord[0] for coord in polygon_coords]
        lons = [coord[1] for coord in polygon_coords]
        
        return {
            "min_lat": min(lats),
            "max_lat": max(lats),
            "min_lon": min(lons),
            "max_lon": max(lons)
        }
    
    def polygons_intersect(self, poly1_coords: List[Tuple[float, float]], 
                          poly2_coords: List[Tuple[float, float]]) -> bool:
        """Check if two polygons intersect"""
        
        try:
            poly1 = Polygon(poly1_coords)
            poly2 = Polygon(poly2_coords)
            
            return poly1.intersects(poly2)
            
        except Exception:
            # Fallback: check if any vertex of one polygon is inside the other
            for point in poly1_coords:
                if self.point_in_polygon(point, poly2_coords):
                    return True
            
            for point in poly2_coords:
                if self.point_in_polygon(point, poly1_coords):
                    return True
            
            return False
    
    def calculate_intersection_area(self, poly1_coords: List[Tuple[float, float]], 
                                  poly2_coords: List[Tuple[float, float]]) -> float:
        """Calculate intersection area between two polygons"""
        
        try:
            poly1 = Polygon(poly1_coords)
            poly2 = Polygon(poly2_coords)
            
            intersection = poly1.intersection(poly2)
            
            if intersection.is_empty:
                return 0
            
            # Transform to projected coordinates for accurate area calculation
            transformer = pyproj.Transformer.from_crs(self.wgs84, self.web_mercator, always_xy=True)
            intersection_projected = transform(transformer.transform, intersection)
            
            return intersection_projected.area
            
        except Exception:
            return 0
    
    def create_corridor_polygon(self, path_coords: List[Tuple[float, float]], 
                              width_meters: float) -> List[Tuple[float, float]]:
        """Create corridor polygon from path and width"""
        
        try:
            # Create LineString from path
            line = LineString(path_coords)
            
            # Convert width to degrees (approximate)
            avg_lat = sum(coord[0] for coord in path_coords) / len(path_coords)
            width_degrees = width_meters / (111320 * math.cos(math.radians(avg_lat)))
            
            # Create buffer around line
            corridor_polygon = line.buffer(width_degrees / 2)  # Half width on each side
            
            return list(corridor_polygon.exterior.coords)
            
        except Exception:
            # Fallback: create simple rectangle around path
            return self._create_simple_corridor(path_coords, width_meters)
    
    def _create_simple_corridor(self, path_coords: List[Tuple[float, float]], 
                               width_meters: float) -> List[Tuple[float, float]]:
        """Create simple rectangular corridor around path"""
        
        if len(path_coords) < 2:
            return path_coords
        
        corridor_coords = []
        half_width = width_meters / 2
        
        # Create perpendicular offsets for each segment
        for i in range(len(path_coords) - 1):
            p1 = path_coords[i]
            p2 = path_coords[i + 1]
            
            # Calculate bearing
            bearing = self.calculate_bearing(p1[0], p1[1], p2[0], p2[1])
            
            # Perpendicular bearings
            left_bearing = (bearing - 90) % 360
            right_bearing = (bearing + 90) % 360
            
            # Offset points
            left_p1 = self.calculate_destination_point(p1[0], p1[1], left_bearing, half_width)
            right_p1 = self.calculate_destination_point(p1[0], p1[1], right_bearing, half_width)
            
            if i == 0:
                corridor_coords.append(left_p1)
            
            if i == len(path_coords) - 2:  # Last segment
                left_p2 = self.calculate_destination_point(p2[0], p2[1], left_bearing, half_width)
                right_p2 = self.calculate_destination_point(p2[0], p2[1], right_bearing, half_width)
                
                corridor_coords.append(left_p2)
                corridor_coords.append(right_p2)
        
        # Add right side points in reverse order
        for i in range(len(path_coords) - 1, -1, -1):
            p = path_coords[i]
            
            if i == 0:
                bearing = self.calculate_bearing(p[0], p[1], path_coords[1][0], path_coords[1][1])
            else:
                bearing = self.calculate_bearing(path_coords[i-1][0], path_coords[i-1][1], p[0], p[1])
            
            right_bearing = (bearing + 90) % 360
            right_p = self.calculate_destination_point(p[0], p[1], right_bearing, half_width)
            corridor_coords.append(right_p)
        
        # Close the polygon
        if corridor_coords:
            corridor_coords.append(corridor_coords[0])
        
        return corridor_coords
    
    def get_spatial_statistics(self, polygon_coords: List[Tuple[float, float]]) -> Dict[str, Any]:
        """Get comprehensive spatial statistics for polygon"""
        
        if not polygon_coords:
            return {}
        
        try:
            # Basic measurements
            area_m2 = self.calculate_polygon_area(polygon_coords)
            centroid = self.calculate_polygon_centroid(polygon_coords)
            bounds = self.calculate_polygon_bounds(polygon_coords)
            
            # Calculate perimeter
            perimeter = 0
            for i in range(len(polygon_coords)):
                p1 = polygon_coords[i]
                p2 = polygon_coords[(i + 1) % len(polygon_coords)]
                perimeter += self.calculate_distance(p1[0], p1[1], p2[0], p2[1])
            
            # Calculate dimensions
            width = self.calculate_distance(bounds["min_lat"], bounds["min_lon"], 
                                          bounds["min_lat"], bounds["max_lon"])
            height = self.calculate_distance(bounds["min_lat"], bounds["min_lon"], 
                                           bounds["max_lat"], bounds["min_lon"])
            
            return {
                "area_m2": area_m2,
                "area_km2": area_m2 / 1000000,
                "perimeter_m": perimeter,
                "perimeter_km": perimeter / 1000,
                "centroid": {"latitude": centroid[0], "longitude": centroid[1]},
                "bounds": bounds,
                "width_m": width,
                "height_m": height,
                "vertex_count": len(polygon_coords)
            }
            
        except Exception as e:
            return {"error": str(e)}