"""
Mission Planner Module - Main mission planning implementation

This module handles:
- Survey mission planning (grid patterns)
- Corridor mission planning (linear surveys) 
- Structure scan planning (3D inspection)
- Custom mission planning (user-defined waypoints)
"""

import numpy as np
from shapely.geometry import Polygon, Point, LineString
from typing import Dict, List, Any, Tuple
from datetime import datetime
import uuid

from src.modules.base_module import BaseModule
from src.models.mission import MissionDefinition, Waypoint, MissionStatistics
from src.models.process import ProcessStep
from src.orchestrator.execution_context import ExecutionContext
from src.modules.mission_planner.pattern_generator import PatternGenerator
from src.modules.mission_planner.qgc_converter import QGCConverter
from src.modules.mission_planner.flight_calculator import FlightCalculator

class MissionPlannerModule(BaseModule):
    """Module for mission planning and generation"""
    
    def __init__(self, module_name: str, implementation: str):
        super().__init__(module_name, implementation)
        self.pattern_generator = PatternGenerator()
        self.qgc_converter = QGCConverter()
        self.flight_calculator = FlightCalculator()
    
    async def _execute_implementation(self, step_config: ProcessStep, context: ExecutionContext) -> Dict[str, Any]:
        """Execute mission planning implementation"""
        
        implementation_map = {
            "survey_planner": self._plan_survey_mission,
            "corridor_planner": self._plan_corridor_mission,
            "structure_scan_planner": self._plan_structure_scan_mission,
            "custom_planner": self._plan_custom_mission
        }
        
        if self.implementation not in implementation_map:
            raise ValueError(f"Unknown mission planning implementation: {self.implementation}")
        
        return await implementation_map[self.implementation](step_config, context)
    
    async def _validate_parameters(self, parameters: Dict[str, Any]) -> None:
        """Validate mission planning parameters"""
        
        common_required = ["vehicle_id", "mission_type"]
        
        for param in common_required:
            if param not in parameters:
                raise ValueError(f"Required parameter '{param}' missing")
        
        if self.implementation == "survey_planner":
            survey_required = ["aoi", "flight_params"]
            self._validate_required_parameters(parameters, survey_required)
            
            # Validate Area of Interest (AOI)
            aoi = parameters["aoi"]
            if not isinstance(aoi, dict) or "coordinates" not in aoi:
                raise ValueError("AOI must contain coordinates")
            
            # Validate flight parameters
            flight_params = parameters["flight_params"]
            required_flight_params = ["altitude", "speed", "grid_spacing"]
            for param in required_flight_params:
                if param not in flight_params:
                    raise ValueError(f"Flight parameter '{param}' missing")
        
        elif self.implementation == "corridor_planner":
            corridor_required = ["corridor_path", "corridor_width", "flight_params"]
            self._validate_required_parameters(parameters, corridor_required)
        
        elif self.implementation == "structure_scan_planner":
            structure_required = ["structure_bounds", "scan_params"]
            self._validate_required_parameters(parameters, structure_required)
        
        elif self.implementation == "custom_planner":
            custom_required = ["waypoints"]
            self._validate_required_parameters(parameters, custom_required)
    
    async def _plan_survey_mission(self, step_config: ProcessStep, context: ExecutionContext) -> Dict[str, Any]:
        """Plan a survey mission with grid pattern"""
        
        params = step_config.parameters
        vehicle_id = params["vehicle_id"]
        aoi = params["aoi"]
        flight_params = params["flight_params"]
        
        # Extract flight parameters
        altitude = flight_params["altitude"]
        speed = flight_params["speed"]
        grid_spacing = flight_params["grid_spacing"]
        overlap = flight_params.get("overlap", 75)
        sidelap = flight_params.get("sidelap", 65)
        gimbal_pitch = flight_params.get("gimbal_pitch", -90)
        
        # Generate survey pattern
        survey_waypoints = self.pattern_generator.generate_survey_pattern(
            aoi=aoi,
            altitude=altitude,
            spacing=grid_spacing,
            overlap=overlap,
            sidelap=sidelap
        )
        
        if not survey_waypoints:
            raise ValueError("Failed to generate survey waypoints")
        
        # Add takeoff and landing waypoints
        first_waypoint = survey_waypoints[0]
        last_waypoint = survey_waypoints[-1]
        
        # Takeoff waypoint (above first survey point)
        takeoff_wp = Waypoint(
            latitude=first_waypoint.latitude,
            longitude=first_waypoint.longitude,
            altitude=altitude + 10,  # 10m above survey altitude
            speed=speed,
            waypoint_type="takeoff"
        )
        
        # Landing waypoint (same as last survey point but on ground)
        landing_wp = Waypoint(
            latitude=last_waypoint.latitude,
            longitude=last_waypoint.longitude,
            altitude=0,
            speed=3,  # Slow landing speed
            waypoint_type="landing"
        )
        
        # Combine all waypoints
        all_waypoints = [takeoff_wp] + survey_waypoints + [landing_wp]
        
        # Calculate mission statistics
        mission_stats = self.flight_calculator.calculate_mission_statistics(
            waypoints=all_waypoints,
            speed=speed,
            vehicle_id=vehicle_id
        )
        
        # Generate QGC format
        qgc_mission = self.qgc_converter.convert_to_qgc(
            waypoints=all_waypoints,
            mission_name=f"Survey_{vehicle_id}_{datetime.now().strftime('%Y%m%d_%H%M')}"
        )
        
        # Store mission data in context for other steps
        mission_data = {
            "waypoints": [wp.to_dict() for wp in all_waypoints],
            "statistics": mission_stats,
            "qgc_format": qgc_mission
        }
        context.set_shared_data(f"mission_data_{vehicle_id}", mission_data)
        context.set_shared_data("mission_data", mission_data)  # Generic key
        
        return {
            "mission_id": str(uuid.uuid4()),
            "vehicle_id": vehicle_id,
            "mission_type": "survey",
            "waypoints": [wp.to_dict() for wp in all_waypoints],
            "statistics": mission_stats,
            "qgc_mission": qgc_mission,
            "flight_parameters": flight_params,
            "area_of_interest": aoi,
            "generated_at": datetime.now().isoformat()
        }
    
    async def _plan_corridor_mission(self, step_config: ProcessStep, context: ExecutionContext) -> Dict[str, Any]:
        """Plan a corridor survey mission"""
        
        params = step_config.parameters
        vehicle_id = params["vehicle_id"]
        corridor_path = params["corridor_path"]
        corridor_width = params["corridor_width"]
        flight_params = params["flight_params"]
        
        altitude = flight_params["altitude"]
        speed = flight_params["speed"]
        
        # Generate corridor pattern
        corridor_waypoints = self.pattern_generator.generate_corridor_pattern(
            path=corridor_path,
            width=corridor_width,
            altitude=altitude
        )
        
        if not corridor_waypoints:
            raise ValueError("Failed to generate corridor waypoints")
        
        # Calculate mission statistics
        mission_stats = self.flight_calculator.calculate_mission_statistics(
            waypoints=corridor_waypoints,
            speed=speed,
            vehicle_id=vehicle_id
        )
        
        # Generate QGC format
        qgc_mission = self.qgc_converter.convert_to_qgc(
            waypoints=corridor_waypoints,
            mission_name=f"Corridor_{vehicle_id}_{datetime.now().strftime('%Y%m%d_%H%M')}"
        )
        
        # Store mission data
        mission_data = {
            "waypoints": [wp.to_dict() for wp in corridor_waypoints],
            "statistics": mission_stats,
            "qgc_format": qgc_mission
        }
        context.set_shared_data(f"mission_data_{vehicle_id}", mission_data)
        
        return {
            "mission_id": str(uuid.uuid4()),
            "vehicle_id": vehicle_id,
            "mission_type": "corridor",
            "waypoints": [wp.to_dict() for wp in corridor_waypoints],
            "statistics": mission_stats,
            "qgc_mission": qgc_mission,
            "corridor_parameters": {
                "path": corridor_path,
                "width": corridor_width,
                "flight_params": flight_params
            },
            "generated_at": datetime.now().isoformat()
        }
    
    async def _plan_structure_scan_mission(self, step_config: ProcessStep, context: ExecutionContext) -> Dict[str, Any]:
        """Plan a 3D structure scanning mission"""
        
        params = step_config.parameters
        vehicle_id = params["vehicle_id"]
        structure_bounds = params["structure_bounds"]
        scan_params = params["scan_params"]
        
        # Extract scan parameters
        min_altitude = scan_params.get("min_altitude", 20)
        max_altitude = scan_params.get("max_altitude", 100)
        altitude_layers = scan_params.get("altitude_layers", 3)
        scan_radius = scan_params.get("scan_radius", 50)
        speed = scan_params.get("speed", 5)
        
        # Generate 3D structure scan pattern
        structure_waypoints = self.pattern_generator.generate_structure_scan_pattern(
            structure_bounds=structure_bounds,
            min_altitude=min_altitude,
            max_altitude=max_altitude,
            layers=altitude_layers,
            radius=scan_radius
        )
        
        if not structure_waypoints:
            raise ValueError("Failed to generate structure scan waypoints")
        
        # Calculate mission statistics
        mission_stats = self.flight_calculator.calculate_mission_statistics(
            waypoints=structure_waypoints,
            speed=speed,
            vehicle_id=vehicle_id
        )
        
        # Generate QGC format
        qgc_mission = self.qgc_converter.convert_to_qgc(
            waypoints=structure_waypoints,
            mission_name=f"Structure_{vehicle_id}_{datetime.now().strftime('%Y%m%d_%H%M')}"
        )
        
        # Store mission data
        mission_data = {
            "waypoints": [wp.to_dict() for wp in structure_waypoints],
            "statistics": mission_stats,
            "qgc_format": qgc_mission
        }
        context.set_shared_data(f"mission_data_{vehicle_id}", mission_data)
        
        return {
            "mission_id": str(uuid.uuid4()),
            "vehicle_id": vehicle_id,
            "mission_type": "structure_scan",
            "waypoints": [wp.to_dict() for wp in structure_waypoints],
            "statistics": mission_stats,
            "qgc_mission": qgc_mission,
            "scan_parameters": scan_params,
            "structure_bounds": structure_bounds,
            "generated_at": datetime.now().isoformat()
        }
    
    async def _plan_custom_mission(self, step_config: ProcessStep, context: ExecutionContext) -> Dict[str, Any]:
        """Plan a custom mission from user-defined waypoints"""
        
        params = step_config.parameters
        vehicle_id = params["vehicle_id"]
        user_waypoints = params["waypoints"]
        flight_params = params.get("flight_params", {"speed": 10})
        
        speed = flight_params.get("speed", 10)
        
        # Convert user waypoints to Waypoint objects
        waypoints = []
        for i, wp_data in enumerate(user_waypoints):
            waypoint = Waypoint(
                latitude=wp_data["latitude"],
                longitude=wp_data["longitude"],
                altitude=wp_data.get("altitude", 50),
                speed=wp_data.get("speed", speed),
                waypoint_type=wp_data.get("type", "waypoint"),
                action=wp_data.get("action"),
                action_params=wp_data.get("action_params", {})
            )
            waypoints.append(waypoint)
        
        if not waypoints:
            raise ValueError("No valid waypoints provided")
        
        # Calculate mission statistics
        mission_stats = self.flight_calculator.calculate_mission_statistics(
            waypoints=waypoints,
            speed=speed,
            vehicle_id=vehicle_id
        )
        
        # Generate QGC format
        qgc_mission = self.qgc_converter.convert_to_qgc(
            waypoints=waypoints,
            mission_name=f"Custom_{vehicle_id}_{datetime.now().strftime('%Y%m%d_%H%M')}"
        )
        
        # Store mission data
        mission_data = {
            "waypoints": [wp.to_dict() for wp in waypoints],
            "statistics": mission_stats,
            "qgc_format": qgc_mission
        }
        context.set_shared_data(f"mission_data_{vehicle_id}", mission_data)
        
        return {
            "mission_id": str(uuid.uuid4()),
            "vehicle_id": vehicle_id,
            "mission_type": "custom",
            "waypoints": [wp.to_dict() for wp in waypoints],
            "statistics": mission_stats,
            "qgc_mission": qgc_mission,
            "flight_parameters": flight_params,
            "generated_at": datetime.now().isoformat()
        }
    
    def _validate_flight_parameters(self, flight_params: Dict[str, Any]) -> None:
        """Validate flight parameters are within reasonable bounds"""
        
        altitude = flight_params.get("altitude", 0)
        speed = flight_params.get("speed", 0)
        
        if altitude < 5 or altitude > 500:
            raise ValueError(f"Altitude {altitude}m is outside valid range (5-500m)")
        
        if speed < 1 or speed > 30:
            raise ValueError(f"Speed {speed}m/s is outside valid range (1-30m/s)")
        
        if "overlap" in flight_params:
            overlap = flight_params["overlap"]
            if overlap < 50 or overlap > 95:
                raise ValueError(f"Overlap {overlap}% is outside valid range (50-95%)")
        
        if "sidelap" in flight_params:
            sidelap = flight_params["sidelap"]
            if sidelap < 30 or sidelap > 90:
                raise ValueError(f"Sidelap {sidelap}% is outside valid range (30-90%)")