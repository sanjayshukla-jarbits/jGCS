"""
Collision Detector - Detect potential collisions between vehicles

This module provides:
- Path-based collision detection
- Time-based conflict analysis
- 3D spatial conflict detection
- Safety distance validation
- Collision avoidance recommendations
"""

import numpy as np
import math
from typing import Dict, List, Any, Tuple, Optional
from datetime import datetime, timedelta
from dataclasses import dataclass
from geopy.distance import geodesic
from shapely.geometry import Point, LineString
from shapely.ops import nearest_points

@dataclass
class CollisionThreat:
    """Collision threat information"""
    vehicle_a: str
    vehicle_b: str
    threat_level: str  # "low", "medium", "high", "critical"
    min_distance: float  # meters
    time_to_collision: float  # seconds
    collision_point: Tuple[float, float, float]  # lat, lon, alt
    relative_velocity: float  # m/s
    recommended_action: str

@dataclass
class TrajectoryPoint:
    """Point along vehicle trajectory"""
    latitude: float
    longitude: float
    altitude: float
    timestamp: float
    speed: float
    heading: float

class CollisionDetector:
    """Detect potential collisions between multiple vehicles"""
    
    def __init__(self):
        self.safety_margins = {
            "critical": 10,   # meters - immediate danger
            "high": 25,      # meters - high risk
            "medium": 50,    # meters - moderate risk  
            "low": 100       # meters - caution
        }
        
        self.time_horizons = {
            "immediate": 30,   # seconds
            "short_term": 120, # seconds
            "medium_term": 300 # seconds
        }
    
    def detect_path_conflicts(self, path_a: List[Tuple[float, float, float]], 
                            path_b: List[Tuple[float, float, float]], 
                            safety_distance: float) -> List[Dict[str, Any]]:
        """Detect conflicts between two flight paths"""
        
        conflicts = []
        
        if not path_a or not path_b:
            return conflicts
        
        # Create LineString objects for path analysis
        try:
            # Convert to 2D for shapely (lat, lon only)
            path_a_2d = [(point[1], point[0]) for point in path_a]  # lon, lat for shapely
            path_b_2d = [(point[1], point[0]) for point in path_b]
            
            line_a = LineString(path_a_2d)
            line_b = LineString(path_b_2d)
            
            # Find closest approach points
            nearest_a, nearest_b = nearest_points(line_a, line_b)
            
            # Calculate horizontal distance
            horizontal_distance = geodesic(
                (nearest_a.y, nearest_a.x),  # lat, lon
                (nearest_b.y, nearest_b.x)   # lat, lon
            ).meters
            
            # Find corresponding altitudes (simplified - would interpolate in real implementation)
            alt_a = self._interpolate_altitude_at_point(path_a, (nearest_a.y, nearest_a.x))
            alt_b = self._interpolate_altitude_at_point(path_b, (nearest_b.y, nearest_b.x))
            
            # Calculate 3D distance
            altitude_diff = abs(alt_a - alt_b)
            distance_3d = math.sqrt(horizontal_distance**2 + altitude_diff**2)
            
            if distance_3d < safety_distance:
                # Estimate time to conflict (simplified)
                estimated_time = self._estimate_time_to_conflict(path_a, path_b, nearest_a, nearest_b)
                
                conflicts.append({
                    "point": (nearest_a.y, nearest_a.x, (alt_a + alt_b) / 2),
                    "distance": distance_3d,
                    "horizontal_distance": horizontal_distance,
                    "altitude_separation": altitude_diff,
                    "time": estimated_time,
                    "severity": self._assess_conflict_severity(distance_3d, safety_distance)
                })
        
        except Exception as e:
            # Handle degenerate cases (single point paths, etc.)
            pass
        
        return conflicts
    
    def detect_real_time_conflicts(self, vehicle_states: Dict[str, Dict[str, Any]], 
                                 prediction_horizon: float = 300) -> List[CollisionThreat]:
        """Detect real-time collision threats based on current vehicle states"""
        
        threats = []
        vehicle_ids = list(vehicle_states.keys())
        
        # Generate predicted trajectories for each vehicle
        predicted_trajectories = {}
        for vehicle_id, state in vehicle_states.items():
            trajectory = self._predict_trajectory(state, prediction_horizon)
            predicted_trajectories[vehicle_id] = trajectory
        
        # Check all vehicle pairs for conflicts
        for i, vehicle_a in enumerate(vehicle_ids):
            for j, vehicle_b in enumerate(vehicle_ids[i+1:], i+1):
                
                threat = self._analyze_trajectory_conflict(
                    vehicle_a, predicted_trajectories[vehicle_a],
                    vehicle_b, predicted_trajectories[vehicle_b]
                )
                
                if threat:
                    threats.append(threat)
        
        # Sort by threat level and time to collision
        threats.sort(key=lambda t: (
            ["low", "medium", "high", "critical"].index(t.threat_level),
            t.time_to_collision
        ))
        
        return threats
    
    def _predict_trajectory(self, vehicle_state: Dict[str, Any], 
                          horizon: float) -> List[TrajectoryPoint]:
        """Predict vehicle trajectory based on current state"""
        
        trajectory = []
        
        # Current state
        current_lat = vehicle_state.get("latitude", 0)
        current_lon = vehicle_state.get("longitude", 0)
        current_alt = vehicle_state.get("altitude", 0)
        current_speed = vehicle_state.get("ground_speed", 10)  # m/s
        current_heading = vehicle_state.get("heading", 0)  # degrees
        
        # Time step for prediction
        time_step = 5.0  # seconds
        num_steps = int(horizon / time_step)
        
        for step in range(num_steps + 1):
            time = step * time_step
            
            # Simple linear prediction (constant velocity)
            distance_traveled = current_speed * time
            
            # Convert heading to radians
            heading_rad = math.radians(current_heading)
            
            # Calculate new position
            # Approximate conversion: 1 degree latitude = 111,320 meters
            lat_change = (distance_traveled * math.cos(heading_rad)) / 111320
            lon_change = (distance_traveled * math.sin(heading_rad)) / (111320 * math.cos(math.radians(current_lat)))
            
            trajectory_point = TrajectoryPoint(
                latitude=current_lat + lat_change,
                longitude=current_lon + lon_change,
                altitude=current_alt,  # Assume constant altitude for now
                timestamp=time,
                speed=current_speed,
                heading=current_heading
            )
            
            trajectory.append(trajectory_point)
        
        return trajectory
    
    def _analyze_trajectory_conflict(self, vehicle_a: str, trajectory_a: List[TrajectoryPoint],
                                   vehicle_b: str, trajectory_b: List[TrajectoryPoint]) -> Optional[CollisionThreat]:
        """Analyze conflict between two predicted trajectories"""
        
        if not trajectory_a or not trajectory_b:
            return None
        
        min_distance = float('inf')
        min_distance_time = 0
        collision_point = None
        
        # Check distance at each time step
        for i, point_a in enumerate(trajectory_a):
            if i < len(trajectory_b):
                point_b = trajectory_b[i]
                
                # Calculate 3D distance
                horizontal_distance = geodesic(
                    (point_a.latitude, point_a.longitude),
                    (point_b.latitude, point_b.longitude)
                ).meters
                
                altitude_diff = abs(point_a.altitude - point_b.altitude)
                distance_3d = math.sqrt(horizontal_distance**2 + altitude_diff**2)
                
                if distance_3d < min_distance:
                    min_distance = distance_3d
                    min_distance_time = point_a.timestamp
                    collision_point = (
                        (point_a.latitude + point_b.latitude) / 2,
                        (point_a.longitude + point_b.longitude) / 2,
                        (point_a.altitude + point_b.altitude) / 2
                    )
        
        # Determine threat level
        threat_level = self._determine_threat_level(min_distance)
        
        if threat_level == "low":
            return None  # Not a significant threat
        
        # Calculate relative velocity
        rel_velocity = self._calculate_relative_velocity(trajectory_a[0], trajectory_b[0])
        
        # Generate recommendation
        recommendation = self._generate_avoidance_recommendation(
            vehicle_a, vehicle_b, min_distance, threat_level
        )
        
        return CollisionThreat(
            vehicle_a=vehicle_a,
            vehicle_b=vehicle_b,
            threat_level=threat_level,
            min_distance=min_distance,
            time_to_collision=min_distance_time,
            collision_point=collision_point,
            relative_velocity=rel_velocity,
            recommended_action=recommendation
        )
    
    def detect_formation_conflicts(self, formation_config: Dict[str, Any], 
                                 vehicle_states: Dict[str, Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Detect conflicts within formation flight"""
        
        conflicts = []
        formation_vehicles = formation_config.get("vehicles", [])
        required_separation = formation_config.get("separation_distance", 50)
        
        for i, vehicle_a in enumerate(formation_vehicles):
            for j, vehicle_b in enumerate(formation_vehicles[i+1:], i+1):
                
                if vehicle_a not in vehicle_states or vehicle_b not in vehicle_states:
                    continue
                
                state_a = vehicle_states[vehicle_a]
                state_b = vehicle_states[vehicle_b]
                
                # Calculate current separation
                current_separation = geodesic(
                    (state_a.get("latitude", 0), state_a.get("longitude", 0)),
                    (state_b.get("latitude", 0), state_b.get("longitude", 0))
                ).meters
                
                altitude_sep = abs(state_a.get("altitude", 0) - state_b.get("altitude", 0))
                separation_3d = math.sqrt(current_separation**2 + altitude_sep**2)
                
                # Check if separation is too close or too far
                tolerance = required_separation * 0.2  # 20% tolerance
                
                if separation_3d < required_separation - tolerance:
                    conflicts.append({
                        "type": "formation_too_close",
                        "vehicle_a": vehicle_a,
                        "vehicle_b": vehicle_b,
                        "current_separation": separation_3d,
                        "required_separation": required_separation,
                        "severity": "high" if separation_3d < required_separation * 0.5 else "medium"
                    })
                
                elif separation_3d > required_separation + tolerance:
                    conflicts.append({
                        "type": "formation_too_far",
                        "vehicle_a": vehicle_a,
                        "vehicle_b": vehicle_b,
                        "current_separation": separation_3d,
                        "required_separation": required_separation,
                        "severity": "low"
                    })
        
        return conflicts
    
    def check_airspace_conflicts(self, vehicle_positions: Dict[str, Dict[str, Any]], 
                               airspace_constraints: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Check for conflicts with airspace constraints"""
        
        conflicts = []
        
        for constraint in airspace_constraints:
            constraint_type = constraint.get("type", "polygon")
            constraint_name = constraint.get("name", "Airspace Constraint")
            
            for vehicle_id, position in vehicle_positions.items():
                lat = position.get("latitude", 0)
                lon = position.get("longitude", 0)
                alt = position.get("altitude", 0)
                
                if constraint_type == "polygon":
                    # Check if vehicle is in restricted polygon
                    coordinates = constraint.get("coordinates", [])
                    if self._point_in_polygon(lat, lon, coordinates):
                        conflicts.append({
                            "type": "airspace_violation",
                            "vehicle_id": vehicle_id,
                            "constraint_name": constraint_name,
                            "position": [lat, lon, alt],
                            "severity": constraint.get("severity", "medium")
                        })
                
                elif constraint_type == "cylinder":
                    # Check if vehicle is in restricted cylinder
                    center_lat = constraint.get("center_latitude", 0)
                    center_lon = constraint.get("center_longitude", 0)
                    radius = constraint.get("radius_meters", 1000)
                    min_alt = constraint.get("min_altitude", 0)
                    max_alt = constraint.get("max_altitude", 1000)
                    
                    distance = geodesic((lat, lon), (center_lat, center_lon)).meters
                    
                    if distance < radius and min_alt <= alt <= max_alt:
                        conflicts.append({
                            "type": "airspace_violation",
                            "vehicle_id": vehicle_id,
                            "constraint_name": constraint_name,
                            "position": [lat, lon, alt],
                            "distance_to_center": distance,
                            "severity": constraint.get("severity", "medium")
                        })
        
        return conflicts
    
    def _interpolate_altitude_at_point(self, path: List[Tuple[float, float, float]], 
                                     target_point: Tuple[float, float]) -> float:
        """Interpolate altitude at a specific point along path"""
        
        if not path:
            return 0
        
        # Find closest point in path (simplified)
        min_distance = float('inf')
        closest_altitude = path[0][2]
        
        for lat, lon, alt in path:
            distance = geodesic(target_point, (lat, lon)).meters
            if distance < min_distance:
                min_distance = distance
                closest_altitude = alt
        
        return closest_altitude
    
    def _estimate_time_to_conflict(self, path_a: List[Tuple[float, float, float]], 
                                 path_b: List[Tuple[float, float, float]],
                                 point_a, point_b) -> float:
        """Estimate time until vehicles reach conflict point"""
        
        # Simplified estimation - assume constant speed
        default_speed = 15  # m/s
        
        # Distance from start of path A to conflict point
        if path_a:
            dist_a = geodesic((path_a[0][0], path_a[0][1]), (point_a.y, point_a.x)).meters
            time_a = dist_a / default_speed
        else:
            time_a = 0
        
        # Distance from start of path B to conflict point
        if path_b:
            dist_b = geodesic((path_b[0][0], path_b[0][1]), (point_b.y, point_b.x)).meters
            time_b = dist_b / default_speed
        else:
            time_b = 0
        
        # Return the maximum time (when both vehicles will be at conflict point)
        return max(time_a, time_b)
    
    def _assess_conflict_severity(self, distance: float, safety_distance: float) -> str:
        """Assess severity of conflict based on distance"""
        
        ratio = distance / safety_distance
        
        if ratio < 0.2:
            return "critical"
        elif ratio < 0.4:
            return "high"
        elif ratio < 0.7:
            return "medium"
        else:
            return "low"
    
    def _determine_threat_level(self, distance: float) -> str:
        """Determine threat level based on distance"""
        
        if distance < self.safety_margins["critical"]:
            return "critical"
        elif distance < self.safety_margins["high"]:
            return "high"
        elif distance < self.safety_margins["medium"]:
            return "medium"
        elif distance < self.safety_margins["low"]:
            return "low"
        else:
            return "minimal"
    
    def _calculate_relative_velocity(self, point_a: TrajectoryPoint, 
                                   point_b: TrajectoryPoint) -> float:
        """Calculate relative velocity between two trajectory points"""
        
        # Convert headings to velocity components
        heading_a_rad = math.radians(point_a.heading)
        heading_b_rad = math.radians(point_b.heading)
        
        # Velocity components for vehicle A
        vx_a = point_a.speed * math.sin(heading_a_rad)
        vy_a = point_a.speed * math.cos(heading_a_rad)
        
        # Velocity components for vehicle B
        vx_b = point_b.speed * math.sin(heading_b_rad)
        vy_b = point_b.speed * math.cos(heading_b_rad)
        
        # Relative velocity components
        rel_vx = vx_a - vx_b
        rel_vy = vy_a - vy_b
        
        # Relative velocity magnitude
        rel_velocity = math.sqrt(rel_vx**2 + rel_vy**2)
        
        return rel_velocity
    
    def _generate_avoidance_recommendation(self, vehicle_a: str, vehicle_b: str, 
                                         distance: float, threat_level: str) -> str:
        """Generate collision avoidance recommendation"""
        
        if threat_level == "critical":
            return f"IMMEDIATE ACTION: {vehicle_a} climb 50m, {vehicle_b} descend 50m"
        elif threat_level == "high":
            return f"URGENT: {vehicle_a} adjust altitude +30m, {vehicle_b} hold current altitude"
        elif threat_level == "medium":
            return f"CAUTION: {vehicle_a} reduce speed, maintain separation"
        else:
            return "Monitor situation, maintain awareness"
    
    def _point_in_polygon(self, lat: float, lon: float, 
                         polygon_coords: List[List[float]]) -> bool:
        """Check if point is inside polygon using ray casting algorithm"""
        
        if not polygon_coords or len(polygon_coords[0]) < 3:
            return False
        
        coords = polygon_coords[0]  # Assume first ring
        n = len(coords)
        inside = False
        
        p1x, p1y = coords[0]
        for i in range(1, n + 1):
            p2x, p2y = coords[i % n]
            
            if lat > min(p1y, p2y):
                if lat <= max(p1y, p2y):
                    if lon <= max(p1x, p2x):
                        if p1y != p2y:
                            xinters = (lat - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                        if p1x == p2x or lon <= xinters:
                            inside = not inside
            p1x, p1y = p2x, p2y
        
        return inside
    
    def get_safety_statistics(self, detection_results: List[CollisionThreat]) -> Dict[str, Any]:
        """Get safety statistics from detection results"""
        
        total_threats = len(detection_results)
        
        threat_counts = {
            "critical": 0,
            "high": 0,
            "medium": 0,
            "low": 0
        }
        
        min_separation = float('inf')
        avg_time_to_collision = 0
        
        for threat in detection_results:
            threat_counts[threat.threat_level] += 1
            min_separation = min(min_separation, threat.min_distance)
            avg_time_to_collision += threat.time_to_collision
        
        if total_threats > 0:
            avg_time_to_collision /= total_threats
        
        return {
            "total_threats": total_threats,
            "threat_breakdown": threat_counts,
            "minimum_separation": min_separation if min_separation != float('inf') else 0,
            "average_time_to_collision": avg_time_to_collision,
            "overall_safety_level": self._calculate_overall_safety_level(threat_counts)
        }
    
    def _calculate_overall_safety_level(self, threat_counts: Dict[str, int]) -> str:
        """Calculate overall safety level"""
        
        if threat_counts["critical"] > 0:
            return "critical"
        elif threat_counts["high"] > 2:
            return "high"
        elif threat_counts["medium"] > 5:
            return "medium"
        elif threat_counts["low"] > 0:
            return "low"
        else:
            return "safe"