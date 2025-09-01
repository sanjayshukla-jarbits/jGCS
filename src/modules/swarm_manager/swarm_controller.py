"""
Swarm Manager Module - Multi-vehicle coordination and conflict resolution

This module handles:
- Collision detection and avoidance
- Formation flight planning
- Swarm analytics and optimization
- Multi-vehicle mission coordination
"""

import numpy as np
from shapely.geometry import Point, LineString
from typing import Dict, List, Any, Tuple, Optional
from datetime import datetime, timedelta
from dataclasses import dataclass

from src.modules.base_module import BaseModule
from src.models.process import ProcessStep
from src.orchestrator.execution_context import ExecutionContext
from src.modules.swarm_manager.collision_detector import CollisionDetector
from src.modules.swarm_manager.analytics_engine import AnalyticsEngine

@dataclass
class VehicleState:
    """Current state of a vehicle in the swarm"""
    vehicle_id: str
    position: Tuple[float, float, float]  # lat, lon, alt
    velocity: Tuple[float, float, float]  # vx, vy, vz
    heading: float
    status: str
    last_update: datetime

@dataclass
class FormationConfig:
    """Formation flight configuration"""
    formation_type: str  # "line", "V", "diamond", "circle"
    leader_vehicle: str
    follower_vehicles: List[str]
    separation_distance: float
    altitude_offset: float

class SwarmManagerModule(BaseModule):
    """Module for swarm coordination and multi-vehicle operations"""
    
    def __init__(self, module_name: str, implementation: str):
        super().__init__(module_name, implementation)
        self.collision_detector = CollisionDetector()
        self.analytics_engine = AnalyticsEngine()
        self.vehicle_states: Dict[str, VehicleState] = {}
        self.formation_configs: Dict[str, FormationConfig] = {}
    
    async def _execute_implementation(self, step_config: ProcessStep, context: ExecutionContext) -> Dict[str, Any]:
        """Execute swarm management implementation"""
        
        implementation_map = {
            "collision_detector": self._detect_collisions,
            "formation_planner": self._plan_formation_flight,
            "analytics_processor": self._process_swarm_analytics
        }
        
        if self.implementation not in implementation_map:
            raise ValueError(f"Unknown swarm implementation: {self.implementation}")
        
        return await implementation_map[self.implementation](step_config, context)
    
    async def _validate_parameters(self, parameters: Dict[str, Any]) -> None:
        """Validate swarm management parameters"""
        
        if self.implementation == "collision_detector":
            required = ["vehicle_missions", "safety_distance"]
        elif self.implementation == "formation_planner":
            required = ["formation_type", "leader_vehicle", "follower_vehicles"]
        elif self.implementation == "analytics_processor":
            required = ["analysis_type", "time_window"]
        else:
            required = []
        
        self._validate_required_parameters(parameters, required)
        
        # Additional validation for formation planning
        if self.implementation == "formation_planner":
            formation_type = parameters["formation_type"]
            if formation_type not in ["line", "V", "diamond", "circle", "custom"]:
                raise ValueError(f"Unsupported formation type: {formation_type}")
            
            follower_vehicles = parameters["follower_vehicles"]
            if not isinstance(follower_vehicles, list) or len(follower_vehicles) == 0:
                raise ValueError("At least one follower vehicle must be specified")
    
    async def _detect_collisions(self, step_config: ProcessStep, context: ExecutionContext) -> Dict[str, Any]:
        """Detect potential collisions between vehicles"""
        
        params = step_config.parameters
        vehicle_missions = params["vehicle_missions"]
        safety_distance = params["safety_distance"]
        time_horizon = params.get("time_horizon", 300)  # 5 minutes default
        
        collision_results = []
        
        # Extract vehicle paths from missions
        vehicle_paths = {}
        for mission_data in vehicle_missions:
            vehicle_id = mission_data["vehicle_id"]
            waypoints = mission_data.get("mission_data", {}).get("waypoints", [])
            
            if waypoints:
                path_points = [(wp.get("latitude", 0), wp.get("longitude", 0), wp.get("altitude", 0)) 
                              for wp in waypoints]
                vehicle_paths[vehicle_id] = path_points
        
        # Detect collisions between all vehicle pairs
        vehicle_ids = list(vehicle_paths.keys())
        
        for i, vehicle_a in enumerate(vehicle_ids):
            for j, vehicle_b in enumerate(vehicle_ids[i+1:], i+1):
                
                path_a = vehicle_paths[vehicle_a]
                path_b = vehicle_paths[vehicle_b]
                
                # Check for potential collisions
                collision_points = self.collision_detector.detect_path_conflicts(
                    path_a, path_b, safety_distance
                )
                
                if collision_points:
                    for collision in collision_points:
                        collision_results.append({
                            "vehicle_a": vehicle_a,
                            "vehicle_b": vehicle_b,
                            "collision_point": collision["point"],
                            "min_distance": collision["distance"],
                            "estimated_time": collision["time"],
                            "severity": self._assess_collision_severity(collision["distance"], safety_distance),
                            "resolution_suggestions": self._generate_collision_resolution(
                                vehicle_a, vehicle_b, collision
                            )
                        })
        
        # Calculate overall collision risk
        total_conflicts = len(collision_results)
        critical_conflicts = len([c for c in collision_results if c["severity"] == "critical"])
        
        # Store collision data in context
        context.set_shared_data("collision_analysis", {
            "conflicts": collision_results,
            "total_conflicts": total_conflicts,
            "critical_conflicts": critical_conflicts
        })
        
        return {
            "collision_analysis": {
                "total_conflicts": total_conflicts,
                "critical_conflicts": critical_conflicts,
                "conflicts": collision_results,
                "safety_distance": safety_distance,
                "time_horizon": time_horizon,
                "analysis_timestamp": datetime.now().isoformat()
            },
            "recommendations": self._generate_swarm_recommendations(collision_results),
            "overall_risk_level": self._calculate_overall_risk(collision_results)
        }
    
    async def _plan_formation_flight(self, step_config: ProcessStep, context: ExecutionContext) -> Dict[str, Any]:
        """Plan formation flight for multiple vehicles"""
        
        params = step_config.parameters
        formation_type = params["formation_type"]
        leader_vehicle = params["leader_vehicle"]
        follower_vehicles = params["follower_vehicles"]
        separation_distance = params.get("separation_distance", 50)  # meters
        altitude_offset = params.get("altitude_offset", 10)  # meters
        
        # Get leader mission from context
        leader_mission = context.get_shared_data(f"mission_data_{leader_vehicle}")
        if not leader_mission:
            raise ValueError(f"No mission data found for leader vehicle {leader_vehicle}")
        
        leader_waypoints = leader_mission.get("waypoints", [])
        if not leader_waypoints:
            raise ValueError("Leader vehicle has no waypoints")
        
        # Generate formation waypoints for each follower
        formation_missions = {}
        formation_config = FormationConfig(
            formation_type=formation_type,
            leader_vehicle=leader_vehicle,
            follower_vehicles=follower_vehicles,
            separation_distance=separation_distance,
            altitude_offset=altitude_offset
        )
        
        for i, follower_id in enumerate(follower_vehicles):
            formation_waypoints = self._generate_formation_waypoints(
                leader_waypoints, formation_type, i, separation_distance, altitude_offset
            )
            
            formation_missions[follower_id] = {
                "mission_id": f"formation_{follower_id}_{datetime.now().timestamp()}",
                "mission_type": "formation_flight",
                "formation_role": "follower",
                "formation_position": i,
                "leader_vehicle": leader_vehicle,
                "waypoints": formation_waypoints,
                "formation_config": {
                    "type": formation_type,
                    "separation_distance": separation_distance,
                    "altitude_offset": altitude_offset
                },
                "generated_at": datetime.now().isoformat()
            }
            
            # Store individual formation mission in context
            context.set_shared_data(f"formation_mission_{follower_id}", formation_missions[follower_id])
        
        # Store formation configuration
        self.formation_configs[f"{leader_vehicle}_formation"] = formation_config
        
        # Calculate formation statistics
        formation_stats = self._calculate_formation_statistics(
            leader_waypoints, formation_missions, separation_distance
        )
        
        return {
            "formation_plan": {
                "formation_type": formation_type,
                "leader_vehicle": leader_vehicle,
                "follower_vehicles": follower_vehicles,
                "formation_missions": formation_missions,
                "formation_statistics": formation_stats,
                "safety_analysis": self._analyze_formation_safety(formation_missions, separation_distance)
            },
            "formation_vehicles": len(follower_vehicles) + 1,  # +1 for leader
            "estimated_formation_time": formation_stats.get("total_flight_time", 0),
            "generated_at": datetime.now().isoformat()
        }
    
    async def _process_swarm_analytics(self, step_config: ProcessStep, context: ExecutionContext) -> Dict[str, Any]:
        """Process swarm analytics and optimization"""
        
        params = step_config.parameters
        analysis_type = params["analysis_type"]
        time_window = params.get("time_window", 3600)  # 1 hour default
        
        # Get mission data from context
        managed_vehicles = context.get_shared_data("managed_vehicles", {})
        collision_analysis = context.get_shared_data("collision_analysis", {})
        
        analytics_results = {}
        
        if analysis_type == "efficiency":
            analytics_results = await self._analyze_swarm_efficiency(
                managed_vehicles, time_window
            )
        
        elif analysis_type == "coverage":
            analytics_results = await self._analyze_swarm_coverage(
                managed_vehicles, time_window
            )
        
        elif analysis_type == "coordination":
            analytics_results = await self._analyze_swarm_coordination(
                managed_vehicles, collision_analysis, time_window
            )
        
        elif analysis_type == "resource_utilization":
            analytics_results = await self._analyze_resource_utilization(
                managed_vehicles, time_window
            )
        
        else:
            # Comprehensive analysis
            analytics_results = {
                "efficiency": await self._analyze_swarm_efficiency(managed_vehicles, time_window),
                "coverage": await self._analyze_swarm_coverage(managed_vehicles, time_window),
                "coordination": await self._analyze_swarm_coordination(managed_vehicles, collision_analysis, time_window),
                "resource_utilization": await self._analyze_resource_utilization(managed_vehicles, time_window)
            }
        
        # Generate optimization recommendations
        optimization_recommendations = self._generate_optimization_recommendations(analytics_results)
        
        return {
            "analytics_type": analysis_type,
            "time_window_seconds": time_window,
            "analytics_results": analytics_results,
            "optimization_recommendations": optimization_recommendations,
            "analysis_timestamp": datetime.now().isoformat(),
            "vehicles_analyzed": len(managed_vehicles)
        }
    
    def _generate_formation_waypoints(self, leader_waypoints: List[Dict], formation_type: str, 
                                    follower_index: int, separation_distance: float, 
                                    altitude_offset: float) -> List[Dict]:
        """Generate formation waypoints for a follower vehicle"""
        
        formation_waypoints = []
        
        for waypoint in leader_waypoints:
            leader_lat = waypoint.get("latitude", 0)
            leader_lon = waypoint.get("longitude", 0)
            leader_alt = waypoint.get("altitude", 50)
            
            # Calculate offset based on formation type
            if formation_type == "line":
                # Line formation: vehicles in a straight line
                lat_offset, lon_offset = self._calculate_line_formation_offset(
                    follower_index, separation_distance, 90  # perpendicular to flight path
                )
            
            elif formation_type == "V":
                # V formation: classic V-shape
                lat_offset, lon_offset = self._calculate_v_formation_offset(
                    follower_index, separation_distance
                )
            
            elif formation_type == "diamond":
                # Diamond formation: diamond shape
                lat_offset, lon_offset = self._calculate_diamond_formation_offset(
                    follower_index, separation_distance
                )
            
            elif formation_type == "circle":
                # Circular formation: vehicles in circle around leader
                lat_offset, lon_offset = self._calculate_circle_formation_offset(
                    follower_index, len(leader_waypoints), separation_distance
                )
            
            else:
                # Default to line formation
                lat_offset, lon_offset = self._calculate_line_formation_offset(
                    follower_index, separation_distance, 90
                )
            
            # Apply offsets
            formation_waypoint = waypoint.copy()
            formation_waypoint["latitude"] = leader_lat + lat_offset
            formation_waypoint["longitude"] = leader_lon + lon_offset
            formation_waypoint["altitude"] = leader_alt + (altitude_offset * (follower_index + 1))
            formation_waypoint["formation_position"] = follower_index
            
            formation_waypoints.append(formation_waypoint)
        
        return formation_waypoints
    
    def _calculate_line_formation_offset(self, index: int, separation: float, 
                                       bearing: float) -> Tuple[float, float]:
        """Calculate offset for line formation"""
        
        # Convert separation to degrees (approximate)
        separation_deg = separation / 111320  # meters to degrees
        
        # Calculate position along the line
        position_offset = (index + 1) * separation_deg
        
        # Apply bearing
        bearing_rad = np.radians(bearing)
        lat_offset = position_offset * np.cos(bearing_rad)
        lon_offset = position_offset * np.sin(bearing_rad)
        
        return lat_offset, lon_offset
    
    def _calculate_v_formation_offset(self, index: int, separation: float) -> Tuple[float, float]:
        """Calculate offset for V formation"""
        
        separation_deg = separation / 111320
        
        # V formation: alternating left and right
        side = 1 if index % 2 == 0 else -1  # Alternate sides
        row = (index // 2) + 1  # Row in formation
        
        lat_offset = -row * separation_deg * 0.5  # Behind leader
        lon_offset = side * row * separation_deg * 0.866  # 60-degree angle
        
        return lat_offset, lon_offset
    
    def _calculate_diamond_formation_offset(self, index: int, separation: float) -> Tuple[float, float]:
        """Calculate offset for diamond formation"""
        
        separation_deg = separation / 111320
        
        # Diamond positions: right, left, rear
        if index == 0:  # Right
            lat_offset = 0
            lon_offset = separation_deg
        elif index == 1:  # Left  
            lat_offset = 0
            lon_offset = -separation_deg
        else:  # Rear
            lat_offset = -separation_deg
            lon_offset = 0
        
        return lat_offset, lon_offset
    
    def _calculate_circle_formation_offset(self, index: int, total_followers: int, 
                                         radius: float) -> Tuple[float, float]:
        """Calculate offset for circular formation"""
        
        radius_deg = radius / 111320
        
        # Distribute evenly around circle
        angle = (2 * np.pi * index) / total_followers
        
        lat_offset = radius_deg * np.cos(angle)
        lon_offset = radius_deg * np.sin(angle)
        
        return lat_offset, lon_offset
    
    def _assess_collision_severity(self, min_distance: float, safety_distance: float) -> str:
        """Assess severity of potential collision"""
        
        ratio = min_distance / safety_distance
        
        if ratio < 0.3:
            return "critical"
        elif ratio < 0.6:
            return "high"
        elif ratio < 0.8:
            return "medium"
        else:
            return "low"
    
    def _generate_collision_resolution(self, vehicle_a: str, vehicle_b: str, 
                                     collision: Dict[str, Any]) -> List[str]:
        """Generate collision resolution suggestions"""
        
        suggestions = []
        
        # Altitude separation
        suggestions.append(f"Adjust {vehicle_a} altitude by +20m")
        suggestions.append(f"Adjust {vehicle_b} altitude by -20m")
        
        # Temporal separation
        suggestions.append(f"Delay {vehicle_b} mission by 2 minutes")
        
        # Path modification
        suggestions.append(f"Modify {vehicle_a} path to avoid conflict zone")
        
        return suggestions
    
    def _generate_swarm_recommendations(self, collision_results: List[Dict]) -> List[str]:
        """Generate overall swarm coordination recommendations"""
        
        recommendations = []
        
        if len(collision_results) == 0:
            recommendations.append("No conflicts detected - swarm coordination optimal")
        
        elif len(collision_results) < 3:
            recommendations.append("Minor conflicts detected - consider altitude or temporal separation")
        
        else:
            recommendations.append("Multiple conflicts detected - recommend mission replanning")
            recommendations.append("Consider implementing dynamic collision avoidance")
        
        # Check for critical conflicts
        critical_conflicts = [c for c in collision_results if c["severity"] == "critical"]
        if critical_conflicts:
            recommendations.append("CRITICAL: Immediate intervention required for safety")
        
        return recommendations
    
    def _calculate_overall_risk(self, collision_results: List[Dict]) -> str:
        """Calculate overall swarm risk level"""
        
        if not collision_results:
            return "low"
        
        critical_count = len([c for c in collision_results if c["severity"] == "critical"])
        high_count = len([c for c in collision_results if c["severity"] == "high"])
        
        if critical_count > 0:
            return "critical"
        elif high_count > 2:
            return "high"
        elif len(collision_results) > 5:
            return "medium"
        else:
            return "low"
    
    def _calculate_formation_statistics(self, leader_waypoints: List[Dict], 
                                      formation_missions: Dict[str, Dict], 
                                      separation_distance: float) -> Dict[str, Any]:
        """Calculate formation flight statistics"""
        
        # Calculate total formation distance
        total_distance = 0
        if len(leader_waypoints) > 1:
            for i in range(len(leader_waypoints) - 1):
                wp1 = leader_waypoints[i]
                wp2 = leader_waypoints[i + 1]
                
                # Simple distance calculation (would use proper geo calculation in production)
                lat_diff = wp2["latitude"] - wp1["latitude"]
                lon_diff = wp2["longitude"] - wp1["longitude"]
                alt_diff = wp2.get("altitude", 0) - wp1.get("altitude", 0)
                
                distance = np.sqrt(lat_diff**2 + lon_diff**2 + (alt_diff/111320)**2) * 111320
                total_distance += distance
        
        # Estimate flight time (assuming average speed)
        avg_speed = 15  # m/s default
        total_flight_time = total_distance / avg_speed if avg_speed > 0 else 0
        
        return {
            "total_distance_meters": total_distance,
            "total_flight_time_seconds": total_flight_time,
            "formation_area_coverage": self._calculate_formation_coverage_area(separation_distance, len(formation_missions)),
            "coordination_complexity": len(formation_missions) * 2,  # Complexity factor
            "safety_margin": separation_distance
        }
    
    def _calculate_formation_coverage_area(self, separation_distance: float, num_vehicles: int) -> float:
        """Calculate area covered by formation"""
        
        # Approximate coverage area based on separation and number of vehicles
        return (separation_distance ** 2) * num_vehicles * 0.8  # 80% efficiency factor
    
    def _analyze_formation_safety(self, formation_missions: Dict[str, Dict], 
                                separation_distance: float) -> Dict[str, Any]:
        """Analyze formation safety metrics"""
        
        safety_analysis = {
            "minimum_separation": separation_distance,
            "collision_risk": "low" if separation_distance >= 50 else "medium",
            "formation_stability": "stable" if len(formation_missions) <= 4 else "complex",
            "coordination_load": len(formation_missions),
            "safety_recommendations": []
        }
        
        if separation_distance < 30:
            safety_analysis["safety_recommendations"].append("Increase separation distance to minimum 30m")
        
        if len(formation_missions) > 6:
            safety_analysis["safety_recommendations"].append("Consider splitting into smaller formations")
        
        return safety_analysis
    
    async def _analyze_swarm_efficiency(self, managed_vehicles: Dict, time_window: int) -> Dict[str, Any]:
        """Analyze swarm operational efficiency"""
        
        return {
            "total_vehicles": len(managed_vehicles),
            "active_vehicles": len([v for v in managed_vehicles.values() if v.get("status") == "active"]),
            "efficiency_score": 0.85,  # Placeholder - would calculate based on real metrics
            "resource_utilization": 0.78,
            "coordination_effectiveness": 0.82
        }
    
    async def _analyze_swarm_coverage(self, managed_vehicles: Dict, time_window: int) -> Dict[str, Any]:
        """Analyze swarm area coverage"""
        
        return {
            "total_coverage_area": 15000,  # Placeholder - square meters
            "coverage_efficiency": 0.73,
            "overlap_percentage": 12.5,
            "gap_percentage": 3.2
        }
    
    async def _analyze_swarm_coordination(self, managed_vehicles: Dict, collision_analysis: Dict, 
                                       time_window: int) -> Dict[str, Any]:
        """Analyze swarm coordination metrics"""
        
        return {
            "coordination_score": 0.88,
            "conflict_resolution_time": 2.3,  # seconds average
            "communication_latency": 0.15,    # seconds
            "formation_maintenance": 0.91     # formation accuracy
        }
    
    async def _analyze_resource_utilization(self, managed_vehicles: Dict, time_window: int) -> Dict[str, Any]:
        """Analyze resource utilization across swarm"""
        
        return {
            "average_battery_usage": 0.65,
            "flight_time_efficiency": 0.82,
            "payload_utilization": 0.71,
            "communication_bandwidth_usage": 0.45
        }
    
    def _generate_optimization_recommendations(self, analytics_results: Dict) -> List[str]:
        """Generate optimization recommendations based on analytics"""
        
        recommendations = []
        
        # Check efficiency
        if isinstance(analytics_results.get("efficiency"), dict):
            eff_score = analytics_results["efficiency"].get("efficiency_score", 0)
            if eff_score < 0.7:
                recommendations.append("Consider optimizing flight paths for better efficiency")
        
        # Check coordination
        if isinstance(analytics_results.get("coordination"), dict):
            coord_score = analytics_results["coordination"].get("coordination_score", 0)
            if coord_score < 0.8:
                recommendations.append("Improve inter-vehicle communication protocols")
        
        # Default recommendation
        if not recommendations:
            recommendations.append("Swarm performance is optimal")
        
        return recommendations