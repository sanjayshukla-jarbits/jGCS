"""
Survey Pattern Generation - Generate flight patterns for different mission types

This module generates:
- Grid survey patterns with optimal coverage
- Corridor survey patterns for linear features
- Structure scan patterns for 3D inspection
- Optimized waypoint sequences
"""

import numpy as np
import math
from shapely.geometry import Polygon, Point, LineString, MultiPoint
from shapely.ops import transform
from typing import List, Dict, Any, Tuple, Optional
from geopy import distance
from geopy.distance import distance as geopy_distance

from src.models.mission import Waypoint

class PatternGenerator:
    """Generates flight patterns for various mission types"""
    
    def __init__(self):
        self.earth_radius_m = 6371000  # Earth radius in meters
    
    def generate_survey_pattern(self, aoi: Dict[str, Any], altitude: float, 
                               spacing: float, overlap: float = 75, 
                               sidelap: float = 65) -> List[Waypoint]:
        """Generate grid survey pattern for area coverage"""
        
        # Extract coordinates from AOI
        coordinates = aoi.get("coordinates", [[]])[0]  # Assume first ring for polygon
        if len(coordinates) < 3:
            raise ValueError("AOI must have at least 3 coordinates")
        
        # Create polygon
        polygon = Polygon(coordinates)
        if not polygon.is_valid:
            raise ValueError("Invalid AOI polygon")
        
        # Get bounding box
        minx, miny, maxx, maxy = polygon.bounds
        
        # Calculate line spacing based on overlap
        line_spacing = self._calculate_line_spacing(spacing, sidelap)
        
        # Generate grid lines
        waypoints = []
        
        # Calculate number of lines needed
        width = geopy_distance((miny, minx), (miny, maxx)).meters
        num_lines = max(1, int(width / line_spacing) + 1)
        
        # Generate parallel lines
        direction_alternating = True
        for i in range(num_lines):
            # Calculate longitude for this line
            progress = i / max(1, num_lines - 1)
            line_lon = minx + progress * (maxx - minx)
            
            # Find intersections with polygon
            line = LineString([(line_lon, miny - 0.001), (line_lon, maxy + 0.001)])
            intersections = polygon.intersection(line)
            
            if intersections.is_empty:
                continue
            
            # Extract intersection points
            if hasattr(intersections, 'geoms'):
                points = []
                for geom in intersections.geoms:
                    if hasattr(geom, 'coords'):
                        points.extend(list(geom.coords))
            else:
                if hasattr(intersections, 'coords'):
                    points = list(intersections.coords)
                else:
                    continue
            
            if len(points) < 2:
                continue
            
            # Sort points by latitude
            points = sorted(points, key=lambda p: p[1])
            
            # Take first and last points (entry and exit of polygon)
            start_point = points[0]
            end_point = points[-1]
            
            # Alternate direction for lawn-mower pattern
            if direction_alternating and i % 2 == 1:
                start_point, end_point = end_point, start_point
            
            # Create waypoints for this line
            waypoints.append(Waypoint(
                latitude=start_point[1],
                longitude=start_point[0],
                altitude=altitude,
                waypoint_type="survey"
            ))
            
            waypoints.append(Waypoint(
                latitude=end_point[1],
                longitude=end_point[0], 
                altitude=altitude,
                waypoint_type="survey"
            ))
        
        return waypoints
    
    def generate_corridor_pattern(self, path: List[Dict[str, float]], width: float, 
                                 altitude: float) -> List[Waypoint]:
        """Generate corridor survey pattern along a path"""
        
        if len(path) < 2:
            raise ValueError("Corridor path must have at least 2 points")
        
        waypoints = []
        
        # Convert path to LineString
        path_coords = [(p["longitude"], p["latitude"]) for p in path]
        path_line = LineString(path_coords)
        
        # Calculate waypoints along the corridor
        # For simplicity, create waypoints along the centerline
        # In practice, you might want to create multiple parallel lines
        
        total_length = self._calculate_path_length(path)
        segment_distance = 50  # 50m between waypoints
        num_segments = max(1, int(total_length / segment_distance))
        
        for i in range(num_segments + 1):
            progress = i / max(1, num_segments)
            
            # Interpolate along path
            point_on_path = path_line.interpolate(progress, normalized=True)
            
            waypoint = Waypoint(
                latitude=point_on_path.y,
                longitude=point_on_path.x,
                altitude=altitude,
                waypoint_type="corridor"
            )
            waypoints.append(waypoint)
        
        return waypoints
    
    def generate_structure_scan_pattern(self, structure_bounds: Dict[str, Any], 
                                      min_altitude: float, max_altitude: float,
                                      layers: int, radius: float) -> List[Waypoint]:
        """Generate 3D structure scanning pattern"""
        
        # Get structure center point
        if "center" in structure_bounds:
            center_lat = structure_bounds["center"]["latitude"]
            center_lon = structure_bounds["center"]["longitude"]
        else:
            # Calculate center from bounds
            bounds = structure_bounds.get("bounds", {})
            center_lat = (bounds.get("north", 0) + bounds.get("south", 0)) / 2
            center_lon = (bounds.get("east", 0) + bounds.get("west", 0)) / 2
        
        waypoints = []
        
        # Generate circular patterns at different altitudes
        for layer in range(layers):
            # Calculate altitude for this layer
            if layers > 1:
                altitude_progress = layer / (layers - 1)
            else:
                altitude_progress = 0
            
            layer_altitude = min_altitude + altitude_progress * (max_altitude - min_altitude)
            
            # Generate circular waypoints around structure
            num_points = 8  # 8 points per circle
            for i in range(num_points):
                angle = (2 * math.pi * i) / num_points
                
                # Calculate offset position
                lat_offset = (radius * math.cos(angle)) / 111320  # Rough conversion to degrees
                lon_offset = (radius * math.sin(angle)) / (111320 * math.cos(math.radians(center_lat)))
                
                waypoint = Waypoint(
                    latitude=center_lat + lat_offset,
                    longitude=center_lon + lon_offset,
                    altitude=layer_altitude,
                    waypoint_type="inspection",
                    action="gimbal_point_to_structure",
                    action_params={
                        "target_lat": center_lat,
                        "target_lon": center_lon
                    }
                )
                waypoints.append(waypoint)
        
        return waypoints
    
    def optimize_waypoint_sequence(self, waypoints: List[Waypoint], 
                                  start_position: Tuple[float, float] = None) -> List[Waypoint]:
        """Optimize waypoint sequence to minimize flight time"""
        
        if len(waypoints) <= 2:
            return waypoints
        
        # Simple nearest neighbor optimization
        # For more complex scenarios, consider using TSP algorithms
        
        if start_position is None:
            start_position = (waypoints[0].latitude, waypoints[0].longitude)
        
        optimized = []
        remaining = waypoints.copy()
        current_pos = start_position
        
        while remaining:
            # Find nearest waypoint
            nearest_wp = None
            min_distance = float('inf')
            
            for wp in remaining:
                wp_pos = (wp.latitude, wp.longitude)
                dist = geopy_distance(current_pos, wp_pos).meters
                
                if dist < min_distance:
                    min_distance = dist
                    nearest_wp = wp
            
            if nearest_wp:
                optimized.append(nearest_wp)
                remaining.remove(nearest_wp)
                current_pos = (nearest_wp.latitude, nearest_wp.longitude)
        
        return optimized
    
    def _calculate_line_spacing(self, camera_spacing: float, sidelap: float) -> float:
        """Calculate spacing between survey lines based on sidelap"""
        
        # Line spacing = camera spacing * (1 - sidelap/100)
        return camera_spacing * (1 - sidelap / 100)
    
    def _calculate_path_length(self, path: List[Dict[str, float]]) -> float:
        """Calculate total length of a path in meters"""
        
        total_length = 0
        for i in range(len(path) - 1):
            point1 = (path[i]["latitude"], path[i]["longitude"])
            point2 = (path[i + 1]["latitude"], path[i + 1]["longitude"])
            
            segment_length = geopy_distance(point1, point2).meters
            total_length += segment_length
        
        return total_length
    
    def _meters_to_degrees_lat(self, meters: float) -> float:
        """Convert meters to degrees latitude"""
        return meters / 111320  # Rough conversion
    
    def _meters_to_degrees_lon(self, meters: float, latitude: float) -> float:
        """Convert meters to degrees longitude at given latitude"""
        return meters / (111320 * math.cos(math.radians(latitude)))
    
    def generate_search_pattern(self, search_area: Dict[str, Any], 
                              pattern_type: str = "expanding_square",
                              altitude: float = 100) -> List[Waypoint]:
        """Generate search patterns for search and rescue missions"""
        
        # Get center point of search area
        coordinates = search_area.get("coordinates", [[]])[0]
        polygon = Polygon(coordinates)
        centroid = polygon.centroid
        
        center_lat, center_lon = centroid.y, centroid.x
        
        waypoints = []
        
        if pattern_type == "expanding_square":
            # Generate expanding square search pattern
            square_size = 100  # Start with 100m squares
            num_squares = 5    # 5 expanding squares
            
            for square in range(num_squares):
                side_length = square_size * (square + 1)
                
                # Generate square corners
                half_side = side_length / 2
                
                corners = [
                    (center_lat + self._meters_to_degrees_lat(half_side),
                     center_lon + self._meters_to_degrees_lon(half_side, center_lat)),
                    (center_lat + self._meters_to_degrees_lat(half_side),
                     center_lon - self._meters_to_degrees_lon(half_side, center_lat)),
                    (center_lat - self._meters_to_degrees_lat(half_side),
                     center_lon - self._meters_to_degrees_lon(half_side, center_lat)),
                    (center_lat - self._meters_to_degrees_lat(half_side),
                     center_lon + self._meters_to_degrees_lon(half_side, center_lat)),
                    # Return to first corner to close square
                    (center_lat + self._meters_to_degrees_lat(half_side),
                     center_lon + self._meters_to_degrees_lon(half_side, center_lat))
                ]
                
                for lat, lon in corners:
                    waypoint = Waypoint(
                        latitude=lat,
                        longitude=lon,
                        altitude=altitude,
                        waypoint_type="search"
                    )
                    waypoints.append(waypoint)
        
        elif pattern_type == "spiral":
            # Generate spiral search pattern
            spiral_spacing = 50  # 50m between spiral arms
            max_radius = 500     # Maximum spiral radius
            
            angle = 0
            radius = 0
            angle_step = math.pi / 8  # 22.5 degree steps
            
            while radius < max_radius:
                # Calculate position
                lat_offset = (radius * math.cos(angle)) / 111320
                lon_offset = (radius * math.sin(angle)) / (111320 * math.cos(math.radians(center_lat)))
                
                waypoint = Waypoint(
                    latitude=center_lat + lat_offset,
                    longitude=center_lon + lon_offset,
                    altitude=altitude,
                    waypoint_type="search"
                )
                waypoints.append(waypoint)
                
                # Update for next point
                angle += angle_step
                radius = (spiral_spacing * angle) / (2 * math.pi)
        
        return waypoints
    
    def generate_perimeter_pattern(self, area: Dict[str, Any], 
                                  altitude: float = 50,
                                  offset_distance: float = 20) -> List[Waypoint]:
        """Generate perimeter patrol pattern around an area"""
        
        coordinates = area.get("coordinates", [[]])[0]
        polygon = Polygon(coordinates)
        
        # Create buffer around polygon for offset patrol
        # Convert offset distance to degrees (approximate)
        offset_degrees = offset_distance / 111320
        buffered_polygon = polygon.buffer(offset_degrees)
        
        waypoints = []
        
        # Get exterior coordinates of buffered polygon
        if hasattr(buffered_polygon, 'exterior'):
            exterior_coords = list(buffered_polygon.exterior.coords)
        else:
            exterior_coords = coordinates
        
        # Create waypoints along perimeter
        for coord in exterior_coords:
            waypoint = Waypoint(
                latitude=coord[1],
                longitude=coord[0],
                altitude=altitude,
                waypoint_type="patrol"
            )
            waypoints.append(waypoint)
        
        return waypoints