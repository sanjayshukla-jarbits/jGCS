"""
Zone Validator - Validate zones and spatial constraints

This module provides:
- Zone geometry validation
- Spatial constraint checking
- Multi-zone conflict detection
- Zone compliance verification
"""

from typing import Dict, List, Any, Tuple, Optional
from datetime import datetime
from dataclasses import dataclass
from shapely.geometry import Polygon, Point, MultiPolygon
from shapely.validation import make_valid
from shapely.ops import unary_union
import numpy as np

from src.modules.geofence_manager.spatial_operations import SpatialOperations

@dataclass
class ValidationResult:
    """Zone validation result"""
    valid: bool
    errors: List[str]
    warnings: List[str]
    corrected_geometry: Optional[Dict[str, Any]] = None

@dataclass
class ZoneConflict:
    """Conflict between zones"""
    zone_1: str
    zone_2: str
    conflict_type: str  # "overlap", "contains", "adjacent"
    conflict_area: float  # square meters
    severity: str  # "low", "medium", "high", "critical"

class ZoneValidator:
    """Validate zones and spatial constraints"""
    
    def __init__(self):
        self.spatial_ops = SpatialOperations()
        
        # Validation parameters
        self.min_zone_area = 100  # square meters
        self.max_zone_area = 10000000  # 10 square kilometers
        self.min_vertices = 3
        self.max_vertices = 1000
        self.min_buffer_distance = 5  # meters
        
        # Validation rules
        self.zone_type_rules = {
            "no_fly_zone": {
                "requires_buffer": True,
                "min_area": 1000,
                "max_vertices": 100,
                "allow_holes": False
            },
            "restricted_area": {
                "requires_buffer": True,
                "min_area": 500,
                "max_vertices": 200,
                "allow_holes": True
            },
            "operational_area": {
                "requires_buffer": False,
                "min_area": 10000,  # Larger minimum for operational areas
                "max_vertices": 500,
                "allow_holes": True
            },
            "landing_zone": {
                "requires_buffer": False,
                "min_area": 100,
                "max_vertices": 50,
                "allow_holes": False,
                "shape_requirements": "convex"  # Landing zones should be convex
            },
            "surveillance_zone": {
                "requires_buffer": False,
                "min_area": 5000,
                "max_vertices": 300,
                "allow_holes": True
            }
        }
    
    def validate_zone(self, zone_data: Dict[str, Any]) -> ValidationResult:
        """Validate individual zone geometry and properties"""
        
        result = ValidationResult(valid=True, errors=[], warnings=[])
        
        # Basic structure validation
        if not self._validate_zone_structure(zone_data, result):
            return result
        
        # Geometry validation
        if not self._validate_zone_geometry(zone_data, result):
            return result
        
        # Type-specific validation
        zone_type = zone_data.get("type", "operational_area")
        if not self._validate_zone_type_rules(zone_data, zone_type, result):
            return result
        
        # Altitude validation
        self._validate_altitude_constraints(zone_data, result)
        
        # Temporal validation
        self._validate_temporal_constraints(zone_data, result)
        
        return result
    
    def _validate_zone_structure(self, zone_data: Dict[str, Any], result: ValidationResult) -> bool:
        """Validate basic zone data structure"""
        
        required_fields = ["coordinates", "type"]
        for field in required_fields:
            if field not in zone_data:
                result.errors.append(f"Missing required field: {field}")
                result.valid = False
        
        # Validate coordinates structure
        coordinates = zone_data.get("coordinates", [])
        if not coordinates or not isinstance(coordinates, list):
            result.errors.append("Coordinates must be a non-empty list")
            result.valid = False
            return False
        
        # Check if it's a nested coordinate structure
        if len(coordinates) > 0 and isinstance(coordinates[0], list):
            # Polygon with potential holes
            for i, ring in enumerate(coordinates):
                if not isinstance(ring, list) or len(ring) < 3:
                    result.errors.append(f"Ring {i} must have at least 3 coordinate pairs")
                    result.valid = False
                
                for j, coord in enumerate(ring):
                    if not isinstance(coord, (list, tuple)) or len(coord) < 2:
                        result.errors.append(f"Coordinate {j} in ring {i} must have at least 2 values (lat, lon)")
                        result.valid = False
        else:
            # Simple coordinate list
            if len(coordinates) < 3:
                result.errors.append("Zone must have at least 3 coordinate pairs")
                result.valid = False
            
            for i, coord in enumerate(coordinates):
                if not isinstance(coord, (list, tuple)) or len(coord) < 2:
                    result.errors.append(f"Coordinate {i} must have at least 2 values (lat, lon)")
                    result.valid = False
        
        return result.valid
    
    def _validate_zone_geometry(self, zone_data: Dict[str, Any], result: ValidationResult) -> bool:
        """Validate zone geometry using Shapely"""
        
        try:
            coordinates = zone_data["coordinates"]
            
            # Handle different coordinate structures
            if isinstance(coordinates[0], list) and len(coordinates[0]) > 0 and isinstance(coordinates[0][0], (list, tuple)):
                # Nested structure (polygon with holes)
                polygon_coords = coordinates
            else:
                # Simple coordinate list
                polygon_coords = [coordinates]
            
            # Create Shapely polygon
            try:
                polygon = Polygon(polygon_coords[0], holes=polygon_coords[1:] if len(polygon_coords) > 1 else None)
            except Exception as e:
                result.errors.append(f"Invalid polygon geometry: {str(e)}")
                result.valid = False
                return False
            
            # Check if polygon is valid
            if not polygon.is_valid:
                result.errors.append(f"Invalid polygon: {polygon.is_valid_reason}")
                
                # Try to fix the polygon
                try:
                    fixed_polygon = make_valid(polygon)
                    if fixed_polygon.is_valid:
                        result.warnings.append("Polygon was automatically corrected")
                        result.corrected_geometry = {
                            "coordinates": [list(fixed_polygon.exterior.coords)]
                        }
                        polygon = fixed_polygon
                    else:
                        result.valid = False
                        return False
                except:
                    result.valid = False
                    return False
            
            # Geometric validation checks
            area = self.spatial_ops.calculate_polygon_area(polygon_coords[0])
            
            # Area validation
            if area < self.min_zone_area:
                result.warnings.append(f"Zone area ({area:.1f}m²) is very small (minimum recommended: {self.min_zone_area}m²)")
            elif area > self.max_zone_area:
                result.errors.append(f"Zone area ({area:.1f}m²) exceeds maximum allowed ({self.max_zone_area}m²)")
                result.valid = False
            
            # Vertex count validation
            vertex_count = len(polygon_coords[0])
            if vertex_count < self.min_vertices:
                result.errors.append(f"Zone must have at least {self.min_vertices} vertices")
                result.valid = False
            elif vertex_count > self.max_vertices:
                result.warnings.append(f"Zone has many vertices ({vertex_count}), consider simplifying")
            
            # Self-intersection check (already covered by polygon.is_valid)
            if polygon.is_valid:
                # Check for very small segments
                coords = list(polygon.exterior.coords)
                for i in range(len(coords) - 1):
                    segment_length = self.spatial_ops.calculate_distance(
                        coords[i][1], coords[i][0], coords[i+1][1], coords[i+1][0]
                    )
                    if segment_length < 1:  # Less than 1 meter
                        result.warnings.append(f"Very short segment detected ({segment_length:.2f}m)")
                
                # Check for sharp angles that might cause navigation issues
                self._check_sharp_angles(coords, result)
            
        except Exception as e:
            result.errors.append(f"Geometry validation failed: {str(e)}")
            result.valid = False
        
        return result.valid
    
    def _validate_zone_type_rules(self, zone_data: Dict[str, Any], zone_type: str, result: ValidationResult) -> bool:
        """Validate zone against type-specific rules"""
        
        if zone_type not in self.zone_type_rules:
            result.warnings.append(f"Unknown zone type: {zone_type}")
            return True  # Continue validation with default rules
        
        rules = self.zone_type_rules[zone_type]
        coordinates = zone_data["coordinates"]
        
        # Handle coordinate structure
        if isinstance(coordinates[0], list) and isinstance(coordinates[0][0], (list, tuple)):
            main_coords = coordinates[0]
        else:
            main_coords = coordinates
        
        # Area validation
        area = self.spatial_ops.calculate_polygon_area(main_coords)
        min_area = rules.get("min_area", self.min_zone_area)
        
        if area < min_area:
            result.errors.append(f"{zone_type} requires minimum area of {min_area}m² (current: {area:.1f}m²)")
            result.valid = False
        
        # Vertex count validation
        max_vertices = rules.get("max_vertices", self.max_vertices)
        if len(main_coords) > max_vertices:
            result.errors.append(f"{zone_type} allows maximum {max_vertices} vertices (current: {len(main_coords)})")
            result.valid = False
        
        # Holes validation
        has_holes = len(coordinates) > 1 if isinstance(coordinates[0], list) and isinstance(coordinates[0][0], (list, tuple)) else False
        if has_holes and not rules.get("allow_holes", True):
            result.errors.append(f"{zone_type} does not allow holes in geometry")
            result.valid = False
        
        # Shape requirements
        if "shape_requirements" in rules:
            shape_req = rules["shape_requirements"]
            if shape_req == "convex":
                if not self._is_convex(main_coords):
                    result.errors.append(f"{zone_type} requires convex geometry")
                    result.valid = False
        
        # Buffer requirements
        if rules.get("requires_buffer", False):
            if "buffer_distance" not in zone_data or zone_data["buffer_distance"] < self.min_buffer_distance:
                result.warnings.append(f"{zone_type} should have buffer distance (recommended: ≥{self.min_buffer_distance}m)")
        
        return result.valid
    
    def _validate_altitude_constraints(self, zone_data: Dict[str, Any], result: ValidationResult) -> None:
        """Validate altitude constraints"""
        
        min_altitude = zone_data.get("min_altitude")
        max_altitude = zone_data.get("max_altitude")
        
        if min_altitude is not None and max_altitude is not None:
            if min_altitude >= max_altitude:
                result.errors.append("Minimum altitude must be less than maximum altitude")
                result.valid = False
            
            if max_altitude - min_altitude < 10:
                result.warnings.append("Very small altitude range (less than 10m)")
        
        if min_altitude is not None and min_altitude < 0:
            result.warnings.append("Negative minimum altitude specified")
        
        if max_altitude is not None and max_altitude > 1500:
            result.warnings.append("Very high maximum altitude (above 1500m)")
    
    def _validate_temporal_constraints(self, zone_data: Dict[str, Any], result: ValidationResult) -> None:
        """Validate temporal constraints"""
        
        start_time = zone_data.get("start_time")
        end_time = zone_data.get("end_time")
        
        if start_time and end_time:
            try:
                if isinstance(start_time, str):
                    start_dt = datetime.fromisoformat(start_time.replace('Z', '+00:00'))
                else:
                    start_dt = start_time
                
                if isinstance(end_time, str):
                    end_dt = datetime.fromisoformat(end_time.replace('Z', '+00:00'))
                else:
                    end_dt = end_time
                
                if start_dt >= end_dt:
                    result.errors.append("Start time must be before end time")
                    result.valid = False
                
                # Check if the zone is already expired
                if end_dt < datetime.now():
                    result.warnings.append("Zone end time is in the past")
                
            except Exception as e:
                result.errors.append(f"Invalid time format: {str(e)}")
                result.valid = False
    
    def _check_sharp_angles(self, coordinates: List[Tuple[float, float]], result: ValidationResult) -> None:
        """Check for sharp angles in polygon"""
        
        min_angle_degrees = 15  # Minimum angle threshold
        
        for i in range(len(coordinates) - 1):
            prev_idx = (i - 1) % (len(coordinates) - 1)
            curr_idx = i
            next_idx = (i + 1) % (len(coordinates) - 1)
            
            # Calculate angle at current vertex
            angle = self._calculate_angle(
                coordinates[prev_idx], coordinates[curr_idx], coordinates[next_idx]
            )
            
            if angle < min_angle_degrees:
                result.warnings.append(f"Sharp angle ({angle:.1f}°) detected at vertex {i}")
    
    def _calculate_angle(self, p1: Tuple[float, float], p2: Tuple[float, float], p3: Tuple[float, float]) -> float:
        """Calculate angle at p2 formed by p1-p2-p3"""
        
        # Convert to vectors
        v1 = (p1[0] - p2[0], p1[1] - p2[1])
        v2 = (p3[0] - p2[0], p3[1] - p2[1])
        
        # Calculate angle using dot product
        dot_product = v1[0] * v2[0] + v1[1] * v2[1]
        mag1 = (v1[0]**2 + v1[1]**2)**0.5
        mag2 = (v2[0]**2 + v2[1]**2)**0.5
        
        if mag1 == 0 or mag2 == 0:
            return 180  # Degenerate case
        
        cos_angle = dot_product / (mag1 * mag2)
        cos_angle = max(-1, min(1, cos_angle))  # Clamp to valid range
        
        angle_rad = np.arccos(cos_angle)
        return np.degrees(angle_rad)
    
    def _is_convex(self, coordinates: List[Tuple[float, float]]) -> bool:
        """Check if polygon is convex"""
        
        if len(coordinates) < 4:  # Less than 3 unique points
            return True
        
        coords = coordinates[:-1] if coordinates[0] == coordinates[-1] else coordinates
        
        if len(coords) < 3:
            return True
        
        sign = None
        
        for i in range(len(coords)):
            p1 = coords[i]
            p2 = coords[(i + 1) % len(coords)]
            p3 = coords[(i + 2) % len(coords)]
            
            # Calculate cross product to determine turn direction
            cross = (p2[0] - p1[0]) * (p3[1] - p2[1]) - (p2[1] - p1[1]) * (p3[0] - p2[0])
            
            if abs(cross) > 1e-10:  # Not collinear
                if sign is None:
                    sign = cross > 0
                elif (cross > 0) != sign:
                    return False  # Found turn in opposite direction
        
        return True
    
    def validate_multiple_zones(self, zones: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Validate multiple zones and check for conflicts"""
        
        validation_results = {}
        zone_conflicts = []
        
        # Validate each zone individually
        for i, zone in enumerate(zones):
            zone_id = zone.get("id", f"zone_{i}")
            validation_results[zone_id] = self.validate_zone(zone)
        
        # Check for conflicts between zones
        valid_zones = {
            zone_id: zone for i, (zone_id, zone) in enumerate(zip(validation_results.keys(), zones))
            if validation_results[zone_id].valid
        }
        
        zone_conflicts = self._detect_zone_conflicts(valid_zones, zones)
        
        # Summary
        total_zones = len(zones)
        valid_zones_count = sum(1 for result in validation_results.values() if result.valid)
        
        return {
            "total_zones": total_zones,
            "valid_zones": valid_zones_count,
            "invalid_zones": total_zones - valid_zones_count,
            "zone_results": validation_results,
            "conflicts": zone_conflicts,
            "validation_summary": {
                "all_valid": valid_zones_count == total_zones,
                "has_conflicts": len(zone_conflicts) > 0,
                "total_errors": sum(len(result.errors) for result in validation_results.values()),
                "total_warnings": sum(len(result.warnings) for result in validation_results.values())
            }
        }
    
    def _detect_zone_conflicts(self, valid_zone_ids: Dict[str, Any], all_zones: List[Dict[str, Any]]) -> List[ZoneConflict]:
        """Detect conflicts between valid zones"""
        
        conflicts = []
        zone_id_to_data = {}
        
        # Map zone IDs to data
        for i, zone in enumerate(all_zones):
            zone_id = zone.get("id", f"zone_{i}")
            if zone_id in valid_zone_ids:
                zone_id_to_data[zone_id] = zone
        
        zone_ids = list(zone_id_to_data.keys())
        
        # Check all pairs of zones
        for i, zone_id_1 in enumerate(zone_ids):
            for zone_id_2 in zone_ids[i+1:]:
                
                zone_1 = zone_id_to_data[zone_id_1]
                zone_2 = zone_id_to_data[zone_id_2]
                
                conflict = self._check_zone_pair_conflict(zone_id_1, zone_1, zone_id_2, zone_2)
                if conflict:
                    conflicts.append(conflict)
        
        return conflicts
    
    def _check_zone_pair_conflict(self, zone_id_1: str, zone_1: Dict[str, Any], 
                                zone_id_2: str, zone_2: Dict[str, Any]) -> Optional[ZoneConflict]:
        """Check for conflict between two zones"""
        
        try:
            # Get coordinates
            coords_1 = self._normalize_coordinates(zone_1["coordinates"])
            coords_2 = self._normalize_coordinates(zone_2["coordinates"])
            
            # Create polygons
            poly_1 = Polygon(coords_1)
            poly_2 = Polygon(coords_2)
            
            # Check for spatial relationships
            if poly_1.intersects(poly_2):
                intersection_area = poly_1.intersection(poly_2).area
                intersection_area_m2 = intersection_area * (111320 ** 2)  # Rough conversion
                
                # Determine conflict type
                if poly_1.contains(poly_2) or poly_2.contains(poly_1):
                    conflict_type = "contains"
                elif intersection_area_m2 > 100:  # Significant overlap
                    conflict_type = "overlap"
                else:
                    conflict_type = "adjacent"
                
                # Determine severity
                severity = self._assess_conflict_severity(
                    zone_1, zone_2, conflict_type, intersection_area_m2
                )
                
                return ZoneConflict(
                    zone_1=zone_id_1,
                    zone_2=zone_id_2,
                    conflict_type=conflict_type,
                    conflict_area=intersection_area_m2,
                    severity=severity
                )
            
        except Exception as e:
            # If we can't check the conflict, assume there might be one
            return ZoneConflict(
                zone_1=zone_id_1,
                zone_2=zone_id_2,
                conflict_type="validation_error",
                conflict_area=0,
                severity="medium"
            )
        
        return None
    
    def _normalize_coordinates(self, coordinates: List[Any]) -> List[Tuple[float, float]]:
        """Normalize coordinates to simple list format"""
        
        if isinstance(coordinates[0], list) and isinstance(coordinates[0][0], (list, tuple)):
            # Nested structure - take the outer ring
            return [(coord[1], coord[0]) for coord in coordinates[0]]  # lon, lat for Shapely
        else:
            # Simple structure
            return [(coord[1], coord[0]) for coord in coordinates]  # lon, lat for Shapely
    
    def _assess_conflict_severity(self, zone_1: Dict[str, Any], zone_2: Dict[str, Any], 
                                conflict_type: str, conflict_area: float) -> str:
        """Assess severity of zone conflict"""
        
        type_1 = zone_1.get("type", "operational_area")
        type_2 = zone_2.get("type", "operational_area")
        
        # Critical conflicts
        critical_combinations = [
            ("no_fly_zone", "operational_area"),
            ("no_fly_zone", "landing_zone"),
            ("restricted_area", "landing_zone")
        ]
        
        if (type_1, type_2) in critical_combinations or (type_2, type_1) in critical_combinations:
            return "critical"
        
        # High severity conflicts
        if conflict_type == "contains":
            return "high"
        
        if conflict_type == "overlap" and conflict_area > 10000:  # > 1 hectare
            return "high"
        
        # Medium severity
        if conflict_type == "overlap" and conflict_area > 1000:  # > 0.1 hectare
            return "medium"
        
        # Low severity
        return "low"
    
    def suggest_zone_fixes(self, zone_data: Dict[str, Any], validation_result: ValidationResult) -> List[Dict[str, Any]]:
        """Suggest fixes for zone validation issues"""
        
        suggestions = []
        
        for error in validation_result.errors:
            if "area" in error.lower() and "small" in error.lower():
                suggestions.append({
                    "type": "expand_zone",
                    "description": "Expand zone boundaries to meet minimum area requirements",
                    "action": "buffer",
                    "parameters": {"buffer_distance": 50}
                })
            
            elif "vertices" in error.lower() and "many" in error.lower():
                suggestions.append({
                    "type": "simplify_geometry",
                    "description": "Simplify zone geometry by reducing number of vertices",
                    "action": "simplify",
                    "parameters": {"tolerance": 10}
                })
            
            elif "invalid polygon" in error.lower():
                suggestions.append({
                    "type": "fix_geometry",
                    "description": "Automatically correct polygon geometry issues",
                    "action": "make_valid",
                    "parameters": {}
                })
        
        for warning in validation_result.warnings:
            if "sharp angle" in warning.lower():
                suggestions.append({
                    "type": "smooth_corners",
                    "description": "Smooth sharp corners to improve navigation",
                    "action": "smooth",
                    "parameters": {"smoothing_factor": 0.1}
                })
        
        return suggestions