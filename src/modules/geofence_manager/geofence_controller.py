"""
Geofence Manager Module - Spatial validation and no-fly zone management

This module handles:
- Polygon geofence validation
- Circular geofence validation
- No-fly zone constraint checking
- Multi-vehicle geofence coordination
"""

from shapely.geometry import Polygon, Point, MultiPolygon
from shapely.ops import unary_union
import geopandas as gpd
from typing import Dict, List, Any, Tuple, Optional
from datetime import datetime

from src.modules.base_module import BaseModule
from src.models.process import ProcessStep
from src.orchestrator.execution_context import ExecutionContext
from src.modules.geofence_manager.spatial_operations import SpatialOperations
from src.modules.geofence_manager.zone_validator import ZoneValidator

class GeofenceManagerModule(BaseModule):
    """Module for geofence validation and spatial constraint checking"""
    
    def __init__(self, module_name: str, implementation: str):
        super().__init__(module_name, implementation)
        self.spatial_ops = SpatialOperations()
        self.zone_validator = ZoneValidator()
        self.active_geofences: Dict[str, Dict[str, Any]] = {}
    
    async def _execute_implementation(self, step_config: ProcessStep, context: ExecutionContext) -> Dict[str, Any]:
        """Execute geofence management implementation"""
        
        implementation_map = {
            "polygon_validator": self._validate_polygon_geofences,
            "circle_validator": self._validate_circular_geofences,
            "no_fly_zone_validator": self._validate_no_fly_zones
        }
        
        if self.implementation not in implementation_map:
            raise ValueError(f"Unknown geofence implementation: {self.implementation}")
        
        return await implementation_map[self.implementation](step_config, context)
    
    async def _validate_parameters(self, parameters: Dict[str, Any]) -> None:
        """Validate geofence parameters"""
        
        if self.implementation in ["polygon_validator", "no_fly_zone_validator"]:
            if "restricted_zones" not in parameters and "inclusion_zones" not in parameters:
                raise ValueError("Either restricted_zones or inclusion_zones must be provided")
        elif self.implementation == "circle_validator":
            if "circular_zones" not in parameters:
                raise ValueError("circular_zones parameter required for circle_validator")
        
        # Validate mission data availability
        if "validate_all_vehicles" in parameters and parameters["validate_all_vehicles"]:
            # Will need access to all vehicle missions
            pass
    
    async def _validate_polygon_geofences(self, step_config: ProcessStep, context: ExecutionContext) -> Dict[str, Any]:
        """Validate mission against polygon geofences"""
        
        params = step_config.parameters
        restricted_zones = params.get("restricted_zones", [])
        inclusion_zones = params.get("inclusion_zones", [])
        safety_buffer_meters = params.get("safety_buffer_meters", 50)
        validate_all_vehicles = params.get("validate_all_vehicles", False)
        
        # Get mission data
        if validate_all_vehicles:
            # Validate all managed vehicles
            managed_vehicles = context.get_shared_data("managed_vehicles", {})
            mission_data = {}
            for vehicle_id in managed_vehicles.keys():
                vehicle_mission = context.get_shared_data(f"mission_data_{vehicle_id}")
                if vehicle_mission:
                    mission_data[vehicle_id] = vehicle_mission
        else:
            # Validate current mission
            mission_data = {"current": context.get_shared_data("mission_data", {})}
        
        if not mission_data:
            raise ValueError("No mission data available for geofence validation")
        
        validation_results = {}
        overall_violations = []
        overall_warnings = []
        
        for vehicle_id, vehicle_mission in mission_data.items():
            waypoints = vehicle_mission.get("waypoints", [])
            if not waypoints:
                continue
            
            violations = []
            warnings = []
            
            # Process restricted zones (exclusion zones)
            for zone in restricted_zones:
                zone_violations = self._check_zone_violations(
                    waypoints, zone, "restricted", safety_buffer_meters
                )
                violations.extend(zone_violations)
            
            # Process inclusion zones (must stay within)
            for zone in inclusion_zones:
                zone_violations = self._check_zone_violations(
                    waypoints, zone, "inclusion", safety_buffer_meters
                )
                violations.extend(zone_violations)
            
            # Check altitude restrictions
            altitude_violations = self._check_altitude_restrictions(waypoints, params)
            violations.extend(altitude_violations)
            
            # Generate warnings for close approaches
            zone_warnings = self._check_zone_warnings(waypoints, restricted_zones, safety_buffer_meters * 2)
            warnings.extend(zone_warnings)
            
            validation_results[vehicle_id] = {
                "violations": violations,
                "warnings": warnings,
                "compliance_status": "compliant" if len(violations) == 0 else "non_compliant",
                "waypoints_checked": len(waypoints)
            }
            
            overall_violations.extend(violations)
            overall_warnings.extend(warnings)
        
        # Store geofence validation results in context
        context.set_shared_data("geofence_validation", validation_results)
        
        return {
            "validation_type": "polygon_geofences",
            "vehicle_results": validation_results,
            "overall_summary": {
                "total_violations": len(overall_violations),
                "total_warnings": len(overall_warnings),
                "vehicles_compliant": len([r for r in validation_results.values() if r["compliance_status"] == "compliant"]),
                "vehicles_non_compliant": len([r for r in validation_results.values() if r["compliance_status"] == "non_compliant"])
            },
            "safety_buffer_meters": safety_buffer_meters,
            "restricted_zones_count": len(restricted_zones),
            "inclusion_zones_count": len(inclusion_zones),
            "validation_timestamp": datetime.now().isoformat()
        }
    
    async def _validate_circular_geofences(self, step_config: ProcessStep, context: ExecutionContext) -> Dict[str, Any]:
        """Validate mission against circular geofences"""
        
        params = step_config.parameters
        circular_zones = params["circular_zones"]
        safety_buffer_meters = params.get("safety_buffer_meters", 50)
        
        # Get mission data
        mission_data = context.get_shared_data("mission_data", {})
        waypoints = mission_data.get("waypoints", [])
        
        if not waypoints:
            raise ValueError("No waypoints found in mission data")
        
        violations = []
        warnings = []
        
        for zone in circular_zones:
            center_lat = zone.get("center_latitude")
            center_lon = zone.get("center_longitude")
            radius_meters = zone.get("radius_meters")
            zone_type = zone.get("type", "restricted")  # restricted or required
            
            if not all([center_lat, center_lon, radius_meters]):
                continue
            
            center_point = Point(center_lon, center_lat)
            
            # Check each waypoint
            for i, waypoint in enumerate(waypoints):
                wp_point = Point(waypoint.get("longitude", 0), waypoint.get("latitude", 0))
                
                # Calculate distance (approximate)
                distance_meters = self.spatial_ops.calculate_distance(
                    center_lat, center_lon, 
                    waypoint.get("latitude", 0), waypoint.get("longitude", 0)
                )
                
                if zone_type == "restricted":
                    # Should not enter restricted circular zone
                    if distance_meters < radius_meters:
                        violations.append({
                            "type": "circular_zone_violation",
                            "waypoint_index": i,
                            "zone_name": zone.get("name", f"Circular Zone {radius_meters}m"),
                            "distance_to_center": distance_meters,
                            "zone_radius": radius_meters,
                            "violation_depth": radius_meters - distance_meters
                        })
                    elif distance_meters < radius_meters + safety_buffer_meters:
                        warnings.append({
                            "type": "circular_zone_warning",
                            "waypoint_index": i,
                            "zone_name": zone.get("name", f"Circular Zone {radius_meters}m"),
                            "distance_to_center": distance_meters,
                            "buffer_penetration": (radius_meters + safety_buffer_meters) - distance_meters
                        })
                
                elif zone_type == "required":
                    # Must stay within required circular zone
                    if distance_meters > radius_meters:
                        violations.append({
                            "type": "required_zone_exit",
                            "waypoint_index": i,
                            "zone_name": zone.get("name", f"Required Zone {radius_meters}m"),
                            "distance_to_center": distance_meters,
                            "zone_radius": radius_meters,
                            "exit_distance": distance_meters - radius_meters
                        })
        
        return {
            "validation_type": "circular_geofences",
            "violations": violations,
            "warnings": warnings,
            "zones_checked": len(circular_zones),
            "waypoints_checked": len(waypoints),
            "compliance_status": "compliant" if len(violations) == 0 else "non_compliant",
            "validation_timestamp": datetime.now().isoformat()
        }
    
    async def _validate_no_fly_zones(self, step_config: ProcessStep, context: ExecutionContext) -> Dict[str, Any]:
        """Validate mission against no-fly zones"""
        
        params = step_config.parameters
        restricted_zones = params.get("restricted_zones", [])
        safety_buffer_meters = params.get("safety_buffer_meters", 100)
        altitude_restrictions = params.get("altitude_restrictions", [])
        
        # Get mission data
        mission_data = context.get_shared_data("mission_data", {})
        waypoints = mission_data.get("waypoints", [])
        
        if not waypoints:
            raise ValueError("No waypoints found in mission data")
        
        violations = []
        warnings = []
        critical_violations = []
        
        # Check each waypoint against all no-fly zones
        for i, waypoint in enumerate(waypoints):
            wp_lat = waypoint.get("latitude", 0)
            wp_lon = waypoint.get("longitude", 0)
            wp_alt = waypoint.get("altitude", 0)
            
            for zone in restricted_zones:
                zone_type = zone.get("type", "polygon")
                zone_name = zone.get("name", f"No-Fly Zone {i}")
                zone_reason = zone.get("reason", "Restricted airspace")
                
                if zone_type == "polygon":
                    # Check polygon no-fly zone
                    coordinates = zone.get("coordinates", [])
                    if len(coordinates) >= 3:
                        zone_polygon = Polygon([(c[0], c[1]) for c in coordinates])
                        wp_point = Point(wp_lon, wp_lat)
                        
                        if zone_polygon.contains(wp_point):
                            violation = {
                                "type": "no_fly_zone_violation",
                                "severity": "critical",
                                "waypoint_index": i,
                                "zone_name": zone_name,
                                "zone_reason": zone_reason,
                                "coordinates": [wp_lat, wp_lon, wp_alt]
                            }
                            violations.append(violation)
                            critical_violations.append(violation)
                        
                        # Check safety buffer
                        elif zone_polygon.distance(wp_point) * 111320 < safety_buffer_meters:
                            warnings.append({
                                "type": "no_fly_zone_proximity",
                                "waypoint_index": i,
                                "zone_name": zone_name,
                                "distance_to_zone": zone_polygon.distance(wp_point) * 111320,
                                "safety_buffer": safety_buffer_meters
                            })
                
                elif zone_type == "circle":
                    # Check circular no-fly zone
                    center_lat = zone.get("center_latitude")
                    center_lon = zone.get("center_longitude")
                    radius = zone.get("radius_meters")
                    
                    if all([center_lat, center_lon, radius]):
                        distance = self.spatial_ops.calculate_distance(
                            center_lat, center_lon, wp_lat, wp_lon
                        )
                        
                        if distance < radius:
                            violation = {
                                "type": "no_fly_zone_violation",
                                "severity": "critical",
                                "waypoint_index": i,
                                "zone_name": zone_name,
                                "zone_reason": zone_reason,
                                "distance_to_center": distance,
                                "zone_radius": radius
                            }
                            violations.append(violation)
                            critical_violations.append(violation)
                
                # Check altitude restrictions
                min_alt = zone.get("min_altitude")
                max_alt = zone.get("max_altitude")
                
                if min_alt is not None and wp_alt < min_alt:
                    violations.append({
                        "type": "altitude_restriction_violation",
                        "severity": "high",
                        "waypoint_index": i,
                        "current_altitude": wp_alt,
                        "minimum_altitude": min_alt,
                        "zone_name": zone_name
                    })
                
                if max_alt is not None and wp_alt > max_alt:
                    violations.append({
                        "type": "altitude_restriction_violation",
                        "severity": "high",
                        "waypoint_index": i,
                        "current_altitude": wp_alt,
                        "maximum_altitude": max_alt,
                        "zone_name": zone_name
                    })
        
        # Generate remediation suggestions
        remediation_suggestions = self._generate_remediation_suggestions(violations, waypoints)
        
        return {
            "validation_type": "no_fly_zones",
            "violations": violations,
            "warnings": warnings,
            "critical_violations": critical_violations,
            "total_zones_checked": len(restricted_zones),
            "waypoints_checked": len(waypoints),
            "compliance_status": "compliant" if len(violations) == 0 else "non_compliant",
            "safety_assessment": self._assess_safety_risk(violations, critical_violations),
            "remediation_suggestions": remediation_suggestions,
            "validation_timestamp": datetime.now().isoformat()
        }
    
    def _check_zone_violations(self, waypoints: List[Dict], zone: Dict[str, Any], 
                             zone_type: str, safety_buffer: float) -> List[Dict[str, Any]]:
        """Check for zone violations"""
        
        violations = []
        coordinates = zone.get("coordinates", [])
        
        if len(coordinates) < 3:
            return violations
        
        zone_polygon = Polygon(coordinates)
        zone_name = zone.get("name", f"{zone_type}_zone")
        
        for i, waypoint in enumerate(waypoints):
            wp_point = Point(waypoint.get("longitude", 0), waypoint.get("latitude", 0))
            
            if zone_type == "restricted":
                # Should not be inside restricted zone
                if zone_polygon.contains(wp_point):
                    violations.append({
                        "type": f"{zone_type}_zone_violation",
                        "waypoint_index": i,
                        "zone_name": zone_name,
                        "reason": zone.get("reason", "Restricted airspace"),
                        "coordinates": [waypoint.get("latitude", 0), waypoint.get("longitude", 0)]
                    })
            
            elif zone_type == "inclusion":
                # Should be inside inclusion zone
                if not zone_polygon.contains(wp_point):
                    violations.append({
                        "type": f"{zone_type}_zone_violation",
                        "waypoint_index": i,
                        "zone_name": zone_name,
                        "reason": "Outside required operational area",
                        "coordinates": [waypoint.get("latitude", 0), waypoint.get("longitude", 0)]
                    })
        
        return violations
    
    def _check_altitude_restrictions(self, waypoints: List[Dict], params: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Check altitude-based restrictions"""
        
        violations = []
        max_altitude = params.get("max_altitude")
        min_altitude = params.get("min_altitude")
        
        for i, waypoint in enumerate(waypoints):
            altitude = waypoint.get("altitude", 0)
            
            if max_altitude and altitude > max_altitude:
                violations.append({
                    "type": "altitude_limit_violation",
                    "waypoint_index": i,
                    "current_altitude": altitude,
                    "max_allowed": max_altitude,
                    "excess": altitude - max_altitude
                })
            
            if min_altitude and altitude < min_altitude:
                violations.append({
                    "type": "altitude_limit_violation",
                    "waypoint_index": i,
                    "current_altitude": altitude,
                    "min_required": min_altitude,
                    "deficit": min_altitude - altitude
                })
        
        return violations
    
    def _check_zone_warnings(self, waypoints: List[Dict], zones: List[Dict], 
                           warning_distance: float) -> List[Dict[str, Any]]:
        """Check for zone proximity warnings"""
        
        warnings = []
        
        for zone in zones:
            coordinates = zone.get("coordinates", [])
            if len(coordinates) < 3:
                continue
            
            zone_polygon = Polygon(coordinates)
            zone_name = zone.get("name", "Restricted Zone")
            
            for i, waypoint in enumerate(waypoints):
                wp_point = Point(waypoint.get("longitude", 0), waypoint.get("latitude", 0))
                
                # Calculate distance to zone boundary
                distance_to_zone = zone_polygon.distance(wp_point) * 111320  # Convert to meters
                
                if 0 < distance_to_zone < warning_distance:
                    warnings.append({
                        "type": "zone_proximity_warning",
                        "waypoint_index": i,
                        "zone_name": zone_name,
                        "distance_to_zone": distance_to_zone,
                        "warning_threshold": warning_distance
                    })
        
        return warnings
    
    def _assess_safety_risk(self, violations: List[Dict], critical_violations: List[Dict]) -> str:
        """Assess overall safety risk level"""
        
        if len(critical_violations) > 0:
            return "critical"
        elif len(violations) > 5:
            return "high"
        elif len(violations) > 2:
            return "medium"
        elif len(violations) > 0:
            return "low"
        else:
            return "minimal"
    
    def _generate_remediation_suggestions(self, violations: List[Dict], 
                                        waypoints: List[Dict]) -> List[str]:
        """Generate suggestions to fix violations"""
        
        suggestions = []
        
        if not violations:
            suggestions.append("Mission is compliant with all geofence restrictions")
            return suggestions
        
        # Count violation types
        zone_violations = len([v for v in violations if "zone_violation" in v.get("type", "")])
        altitude_violations = len([v for v in violations if "altitude" in v.get("type", "")])
        
        if zone_violations > 0:
            suggestions.append(f"Modify flight path to avoid {zone_violations} restricted zone violations")
            suggestions.append("Consider alternative routing around no-fly zones")
        
        if altitude_violations > 0:
            suggestions.append(f"Adjust altitude for {altitude_violations} waypoints to meet restrictions")
        
        if len(violations) > len(waypoints) * 0.5:
            suggestions.append("Consider complete mission replanning due to extensive violations")
        
        return suggestions
    
    def add_geofence(self, geofence_id: str, geofence_data: Dict[str, Any]) -> None:
        """Add a new active geofence"""
        
        self.active_geofences[geofence_id] = {
            **geofence_data,
            "created_at": datetime.now(),
            "status": "active"
        }
    
    def remove_geofence(self, geofence_id: str) -> bool:
        """Remove an active geofence"""
        
        if geofence_id in self.active_geofences:
            del self.active_geofences[geofence_id]
            return True
        return False
    
    def get_active_geofences(self) -> Dict[str, Dict[str, Any]]:
        """Get all active geofences"""
        return self.active_geofences.copy()