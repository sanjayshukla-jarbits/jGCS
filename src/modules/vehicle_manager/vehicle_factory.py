"""
Vehicle Factory - Create different types of vehicles with appropriate configurations

This module provides:
- Vehicle type factory pattern implementation
- Multi-rotor vehicle creation
- Fixed-wing vehicle creation  
- VTOL vehicle creation
- Generic vehicle creation
"""

from typing import Dict, Any, Optional
from dataclasses import dataclass
from datetime import datetime
from abc import ABC, abstractmethod

@dataclass
class VehicleCapabilities:
    """Vehicle capability definition"""
    can_hover: bool
    requires_runway: bool
    min_speed: float
    max_speed: float
    max_altitude: float
    max_payload: float
    flight_modes: list
    has_gimbal: bool = False
    has_vtol_capability: bool = False

class BaseVehicle(ABC):
    """Abstract base vehicle class"""
    
    def __init__(self, vehicle_id: str, vehicle_type: str):
        self.vehicle_id = vehicle_id
        self.vehicle_type = vehicle_type
        self.capabilities: Optional[VehicleCapabilities] = None
        self.configuration: Dict[str, Any] = {}
        self.created_at = datetime.now()
        self.status = "initialized"
    
    @abstractmethod
    def configure(self, config: Dict[str, Any]) -> None:
        """Configure vehicle with specific parameters"""
        pass
    
    @abstractmethod
    def validate_mission_compatibility(self, mission_data: Dict[str, Any]) -> Dict[str, Any]:
        """Validate if mission is compatible with vehicle capabilities"""
        pass
    
    def get_capabilities(self) -> Dict[str, Any]:
        """Get vehicle capabilities"""
        if self.capabilities:
            return {
                "can_hover": self.capabilities.can_hover,
                "requires_runway": self.capabilities.requires_runway,
                "min_speed": self.capabilities.min_speed,
                "max_speed": self.capabilities.max_speed,
                "max_altitude": self.capabilities.max_altitude,
                "max_payload": self.capabilities.max_payload,
                "flight_modes": self.capabilities.flight_modes,
                "has_gimbal": self.capabilities.has_gimbal,
                "has_vtol_capability": self.capabilities.has_vtol_capability
            }
        return {}
    
    def get_status(self) -> Dict[str, Any]:
        """Get vehicle status information"""
        return {
            "vehicle_id": self.vehicle_id,
            "vehicle_type": self.vehicle_type,
            "status": self.status,
            "created_at": self.created_at.isoformat(),
            "configured": len(self.configuration) > 0
        }

class MultiRotorVehicle(BaseVehicle):
    """Multi-rotor vehicle implementation"""
    
    def __init__(self, vehicle_id: str):
        super().__init__(vehicle_id, "multi_rotor")
        self.capabilities = VehicleCapabilities(
            can_hover=True,
            requires_runway=False,
            min_speed=0,  # Can hover
            max_speed=25,
            max_altitude=500,
            max_payload=5.0,  # kg
            flight_modes=["manual", "stabilize", "altitude_hold", "position_hold", "guided", "auto", "rtl", "land"],
            has_gimbal=True
        )
    
    def configure(self, config: Dict[str, Any]) -> None:
        """Configure multi-rotor specific parameters"""
        
        self.configuration = {
            "rotor_count": config.get("rotor_count", 4),
            "max_climb_rate": config.get("max_climb_rate", 5.0),  # m/s
            "max_descent_rate": config.get("max_descent_rate", 3.0),  # m/s
            "hover_accuracy": config.get("hover_accuracy", 2.0),  # meters
            "battery_type": config.get("battery_type", "LiPo"),
            "battery_capacity": config.get("battery_capacity", 16000),  # mAh
            "motor_count": config.get("motor_count", 4),
            "propeller_size": config.get("propeller_size", "10x4.5"),
            "frame_size": config.get("frame_size", 450),  # mm
            "takeoff_weight": config.get("takeoff_weight", 2.5),  # kg
            "hover_time": config.get("hover_time", 25),  # minutes
        }
        
        # Update capabilities based on configuration
        if "max_speed" in config:
            self.capabilities.max_speed = min(config["max_speed"], 25)  # Cap at 25 m/s
        
        if "max_altitude" in config:
            self.capabilities.max_altitude = config["max_altitude"]
        
        self.status = "configured"
    
    def validate_mission_compatibility(self, mission_data: Dict[str, Any]) -> Dict[str, Any]:
        """Validate mission compatibility for multi-rotor"""
        
        validation_result = {
            "compatible": True,
            "warnings": [],
            "errors": [],
            "recommendations": []
        }
        
        waypoints = mission_data.get("waypoints", [])
        
        for i, waypoint in enumerate(waypoints):
            # Check altitude limits
            altitude = waypoint.get("altitude", 0)
            if altitude > self.capabilities.max_altitude:
                validation_result["warnings"].append(
                    f"Waypoint {i}: Altitude {altitude}m may exceed vehicle limits"
                )
            
            if speed > self.capabilities.max_speed:
                validation_result["warnings"].append(
                    f"Waypoint {i}: Speed {speed}m/s may exceed vehicle limits"
                )
            
            # Check hover capability
            if not self.capabilities.can_hover and speed == 0:
                validation_result["errors"].append(
                    f"Waypoint {i}: Vehicle cannot hover"
                )
                validation_result["compatible"] = False
        
        validation_result["warnings"].append(
            "Generic vehicle type - capabilities assumed, verify with actual vehicle specifications"
        )
        
        return validation_result

class VehicleFactory:
    """Factory for creating different types of vehicles"""
    
    def __init__(self):
        self.vehicle_registry = {}
        self.supported_types = {
            "multi_rotor": MultiRotorVehicle,
            "multirotor": MultiRotorVehicle,  # Alternative spelling
            "quadcopter": MultiRotorVehicle,
            "hexacopter": MultiRotorVehicle,
            "octocopter": MultiRotorVehicle,
            "fixed_wing": FixedWingVehicle,
            "fixedwing": FixedWingVehicle,  # Alternative spelling
            "plane": FixedWingVehicle,
            "vtol": VTOLVehicle,
            "quadplane": VTOLVehicle,
            "tiltrotor": VTOLVehicle,
            "tailsitter": VTOLVehicle,
            "generic": GenericVehicle,
            "unknown": GenericVehicle
        }
    
    def create_vehicle(self, vehicle_type: str, vehicle_id: str, 
                      config: Dict[str, Any] = None) -> BaseVehicle:
        """Create a vehicle instance of the specified type"""
        
        vehicle_type_lower = vehicle_type.lower()
        
        if vehicle_type_lower not in self.supported_types:
            # Default to generic vehicle for unsupported types
            vehicle_type_lower = "generic"
        
        vehicle_class = self.supported_types[vehicle_type_lower]
        vehicle = vehicle_class(vehicle_id)
        
        # Configure vehicle if config provided
        if config:
            vehicle.configure(config)
        
        # Register vehicle
        self.vehicle_registry[vehicle_id] = vehicle
        
        return vehicle
    
    def get_vehicle(self, vehicle_id: str) -> Optional[BaseVehicle]:
        """Get existing vehicle instance"""
        return self.vehicle_registry.get(vehicle_id)
    
    def remove_vehicle(self, vehicle_id: str) -> bool:
        """Remove vehicle from registry"""
        if vehicle_id in self.vehicle_registry:
            del self.vehicle_registry[vehicle_id]
            return True
        return False
    
    def list_vehicles(self) -> Dict[str, Dict[str, Any]]:
        """List all registered vehicles"""
        return {
            vehicle_id: vehicle.get_status() 
            for vehicle_id, vehicle in self.vehicle_registry.items()
        }
    
    def get_supported_types(self) -> List[str]:
        """Get list of supported vehicle types"""
        # Return unique types (remove duplicates)
        unique_types = list(set(self.supported_types.values()))
        return [cls.__name__.replace("Vehicle", "").lower() for cls in unique_types]
    
    def validate_vehicle_for_mission(self, vehicle_id: str, 
                                   mission_data: Dict[str, Any]) -> Dict[str, Any]:
        """Validate if a vehicle is suitable for a mission"""
        
        vehicle = self.get_vehicle(vehicle_id)
        if not vehicle:
            return {
                "compatible": False,
                "errors": [f"Vehicle {vehicle_id} not found"],
                "warnings": [],
                "recommendations": []
            }
        
        return vehicle.validate_mission_compatibility(mission_data)
    
    def recommend_vehicle_type(self, mission_requirements: Dict[str, Any]) -> Dict[str, Any]:
        """Recommend best vehicle type for mission requirements"""
        
        recommendations = []
        
        # Analyze mission requirements
        requires_hover = mission_requirements.get("requires_hover", False)
        requires_speed = mission_requirements.get("max_speed", 15)
        requires_endurance = mission_requirements.get("endurance_minutes", 30)
        requires_payload = mission_requirements.get("payload_kg", 1.0)
        mission_range = mission_requirements.get("range_km", 10)
        
        # Multi-rotor recommendation
        if requires_hover and requires_speed <= 20 and mission_range <= 15:
            recommendations.append({
                "vehicle_type": "multi_rotor",
                "suitability_score": 0.9,
                "reasons": ["Excellent hover capability", "Good for short-range missions", "High maneuverability"]
            })
        
        # Fixed-wing recommendation  
        if not requires_hover and mission_range > 20:
            recommendations.append({
                "vehicle_type": "fixed_wing",
                "suitability_score": 0.85,
                "reasons": ["Excellent endurance", "High speed capability", "Long range efficiency"]
            })
        
        # VTOL recommendation
        if requires_hover and mission_range > 10:
            recommendations.append({
                "vehicle_type": "vtol",
                "suitability_score": 0.8,
                "reasons": ["Combines hover and forward flight", "Good compromise for mixed missions"]
            })
        
        # Sort by suitability score
        recommendations.sort(key=lambda x: x["suitability_score"], reverse=True)
        
        return {
            "mission_requirements": mission_requirements,
            "recommendations": recommendations,
            "top_recommendation": recommendations[0] if recommendations else None
        }
    
    def create_vehicle_from_template(self, template_name: str, vehicle_id: str) -> BaseVehicle:
        """Create vehicle from predefined template"""
        
        templates = {
            "survey_multirotor": {
                "type": "multi_rotor",
                "config": {
                    "rotor_count": 4,
                    "max_speed": 15,
                    "hover_time": 30,
                    "battery_capacity": 16000,
                    "has_gimbal": True,
                    "max_payload": 2.0
                }
            },
            "mapping_fixedwing": {
                "type": "fixed_wing", 
                "config": {
                    "wingspan": 2.0,
                    "cruise_speed": 22,
                    "endurance": 120,
                    "service_ceiling": 1000,
                    "has_gimbal": True
                }
            },
            "cargo_vtol": {
                "type": "vtol",
                "config": {
                    "vtol_type": "quadplane",
                    "max_payload": 10.0,
                    "forward_speed_max": 30,
                    "hover_endurance": 15,
                    "forward_endurance": 90
                }
            },
            "inspection_drone": {
                "type": "multi_rotor",
                "config": {
                    "rotor_count": 6,
                    "max_speed": 12,
                    "hover_accuracy": 1.0,
                    "has_gimbal": True,
                    "battery_capacity": 20000,
                    "hover_time": 45
                }
            }
        }
        
        if template_name not in templates:
            raise ValueError(f"Unknown template: {template_name}. Available: {list(templates.keys())}")
        
        template = templates[template_name]
        return self.create_vehicle(
            vehicle_type=template["type"],
            vehicle_id=vehicle_id,
            config=template["config"]
        )
    
    def get_vehicle_comparison(self, vehicle_ids: List[str]) -> Dict[str, Any]:
        """Compare multiple vehicles"""
        
        comparison = {
            "vehicles": {},
            "comparison_matrix": {}
        }
        
        vehicles = []
        for vehicle_id in vehicle_ids:
            vehicle = self.get_vehicle(vehicle_id)
            if vehicle:
                vehicles.append(vehicle)
                comparison["vehicles"][vehicle_id] = {
                    "type": vehicle.vehicle_type,
                    "capabilities": vehicle.get_capabilities(),
                    "status": vehicle.get_status()
                }
        
        if len(vehicles) < 2:
            return comparison
        
        # Create comparison matrix
        capabilities = ["can_hover", "max_speed", "max_altitude", "max_payload"]
        
        for cap in capabilities:
            comparison["comparison_matrix"][cap] = {}
            for vehicle in vehicles:
                caps = vehicle.get_capabilities()
                comparison["comparison_matrix"][cap][vehicle.vehicle_id] = caps.get(cap, "N/A")
        
        return comparison
    
    def export_vehicle_config(self, vehicle_id: str) -> Dict[str, Any]:
        """Export vehicle configuration"""
        
        vehicle = self.get_vehicle(vehicle_id)
        if not vehicle:
            return {}
        
        return {
            "vehicle_id": vehicle_id,
            "vehicle_type": vehicle.vehicle_type,
            "capabilities": vehicle.get_capabilities(),
            "configuration": vehicle.configuration,
            "status": vehicle.get_status(),
            "exported_at": datetime.now().isoformat()
        }
    
    def import_vehicle_config(self, config_data: Dict[str, Any]) -> BaseVehicle:
        """Import vehicle from configuration"""
        
        vehicle_id = config_data["vehicle_id"]
        vehicle_type = config_data["vehicle_type"]
        configuration = config_data.get("configuration", {})
        
        return self.create_vehicle(vehicle_type, vehicle_id, configuration)
    
    def clear_registry(self) -> None:
        """Clear all vehicles from registry"""
        self.vehicle_registry.clear()
    
    def get_registry_statistics(self) -> Dict[str, Any]:
        """Get statistics about vehicle registry"""
        
        type_counts = {}
        configured_count = 0
        
        for vehicle in self.vehicle_registry.values():
            vehicle_type = vehicle.vehicle_type
            type_counts[vehicle_type] = type_counts.get(vehicle_type, 0) + 1
            
            if vehicle.status == "configured":
                configured_count += 1
        
        return {
            "total_vehicles": len(self.vehicle_registry),
            "configured_vehicles": configured_count,
            "vehicle_types": type_counts,
            "supported_types": self.get_supported_types()
        }result["errors"].append(
                    f"Waypoint {i}: Altitude {altitude}m exceeds max altitude {self.capabilities.max_altitude}m"
                )
                validation_result["compatible"] = False
            
            # Check speed limits
            speed = waypoint.get("speed", 10)
            if speed > self.capabilities.max_speed:
                validation_result["warnings"].append(
                    f"Waypoint {i}: Speed {speed}m/s exceeds recommended max {self.capabilities.max_speed}m/s"
                )
        
        # Check mission duration vs battery capacity
        estimated_flight_time = mission_data.get("statistics", {}).get("total_flight_time_minutes", 0)
        hover_time = self.configuration.get("hover_time", 25)
        
        if estimated_flight_time > hover_time * 0.8:  # 80% of max flight time
            validation_result["warnings"].append(
                f"Mission duration {estimated_flight_time}min may exceed safe battery capacity"
            )
        
        # Recommendations
        if any("survey" in wp.get("waypoint_type", "") for wp in waypoints):
            validation_result["recommendations"].append(
                "Consider using gimbal stabilization for survey missions"
            )
        
        return validation_result

class FixedWingVehicle(BaseVehicle):
    """Fixed-wing vehicle implementation"""
    
    def __init__(self, vehicle_id: str):
        super().__init__(vehicle_id, "fixed_wing")
        self.capabilities = VehicleCapabilities(
            can_hover=False,
            requires_runway=True,
            min_speed=12,  # Stall speed
            max_speed=35,
            max_altitude=1000,
            max_payload=10.0,  # kg
            flight_modes=["manual", "stabilize", "fbwa", "fbwb", "cruise", "auto", "rtl", "circle"],
            has_gimbal=True
        )
    
    def configure(self, config: Dict[str, Any]) -> None:
        """Configure fixed-wing specific parameters"""
        
        self.configuration = {
            "wingspan": config.get("wingspan", 1.5),  # meters
            "wing_area": config.get("wing_area", 0.3),  # m²
            "stall_speed": config.get("stall_speed", 12),  # m/s
            "cruise_speed": config.get("cruise_speed", 20),  # m/s
            "max_speed": config.get("max_speed", 30),  # m/s
            "min_turn_radius": config.get("min_turn_radius", 50),  # meters
            "takeoff_distance": config.get("takeoff_distance", 100),  # meters
            "landing_distance": config.get("landing_distance", 80),  # meters
            "service_ceiling": config.get("service_ceiling", 1000),  # meters
            "endurance": config.get("endurance", 90),  # minutes
            "battery_capacity": config.get("battery_capacity", 20000),  # mAh
        }
        
        # Update capabilities
        self.capabilities.min_speed = self.configuration["stall_speed"]
        self.capabilities.max_speed = self.configuration["max_speed"]
        self.capabilities.max_altitude = self.configuration["service_ceiling"]
        
        self.status = "configured"
    
    def validate_mission_compatibility(self, mission_data: Dict[str, Any]) -> Dict[str, Any]:
        """Validate mission compatibility for fixed-wing"""
        
        validation_result = {
            "compatible": True,
            "warnings": [],
            "errors": [],
            "recommendations": []
        }
        
        waypoints = mission_data.get("waypoints", [])
        
        # Check for hovering waypoints (not compatible)
        for i, waypoint in enumerate(waypoints):
            wp_type = waypoint.get("waypoint_type", "waypoint")
            if wp_type == "loiter" and waypoint.get("hold_time", 0) == 0:
                validation_result["errors"].append(
                    f"Waypoint {i}: Fixed-wing cannot hover at waypoint"
                )
                validation_result["compatible"] = False
            
            # Check minimum speed requirements
            speed = waypoint.get("speed", 15)
            if speed < self.capabilities.min_speed:
                validation_result["errors"].append(
                    f"Waypoint {i}: Speed {speed}m/s below stall speed {self.capabilities.min_speed}m/s"
                )
                validation_result["compatible"] = False
        
        # Check for runway requirements
        has_takeoff = any(wp.get("waypoint_type") == "takeoff" for wp in waypoints)
        has_landing = any(wp.get("waypoint_type") == "land" for wp in waypoints)
        
        if not has_takeoff:
            validation_result["warnings"].append("No explicit takeoff waypoint found")
        
        if not has_landing:
            validation_result["warnings"].append("No explicit landing waypoint found")
        
        # Check turn radius constraints
        for i in range(len(waypoints) - 1):
            current = waypoints[i]
            next_wp = waypoints[i + 1]
            
            # Calculate required turn radius (simplified)
            speed = current.get("speed", 20)
            # This is a simplified check - would need proper geospatial calculation
            
        validation_result["recommendations"].append(
            "Ensure adequate runway length for takeoff and landing"
        )
        validation_result["recommendations"].append(
            "Consider wind conditions for fixed-wing operations"
        )
        
        return validation_result

class VTOLVehicle(BaseVehicle):
    """VTOL vehicle implementation"""
    
    def __init__(self, vehicle_id: str):
        super().__init__(vehicle_id, "vtol")
        self.capabilities = VehicleCapabilities(
            can_hover=True,
            requires_runway=False,
            min_speed=0,  # Can hover
            max_speed=30,
            max_altitude=800,
            max_payload=8.0,  # kg
            flight_modes=["manual", "stabilize", "altitude_hold", "position_hold", "guided", "auto", "rtl", "qhover", "qstabilize"],
            has_gimbal=True,
            has_vtol_capability=True
        )
    
    def configure(self, config: Dict[str, Any]) -> None:
        """Configure VTOL specific parameters"""
        
        self.configuration = {
            "vtol_type": config.get("vtol_type", "quadplane"),  # quadplane, tiltrotor, tailsitter
            "hover_motors": config.get("hover_motors", 4),
            "forward_motor": config.get("forward_motor", 1),
            "transition_speed": config.get("transition_speed", 18),  # m/s
            "hover_speed_max": config.get("hover_speed_max", 15),  # m/s
            "forward_speed_max": config.get("forward_speed_max", 25),  # m/s
            "hover_efficiency": config.get("hover_efficiency", 0.7),  # 0-1
            "forward_efficiency": config.get("forward_efficiency", 0.85),  # 0-1
            "transition_time": config.get("transition_time", 5),  # seconds
            "battery_capacity": config.get("battery_capacity", 22000),  # mAh
            "hover_endurance": config.get("hover_endurance", 20),  # minutes
            "forward_endurance": config.get("forward_endurance", 60),  # minutes
        }
        
        # Update capabilities
        if "forward_speed_max" in config:
            self.capabilities.max_speed = config["forward_speed_max"]
        
        self.status = "configured"
    
    def validate_mission_compatibility(self, mission_data: Dict[str, Any]) -> Dict[str, Any]:
        """Validate mission compatibility for VTOL"""
        
        validation_result = {
            "compatible": True,
            "warnings": [],
            "errors": [],
            "recommendations": []
        }
        
        waypoints = mission_data.get("waypoints", [])
        
        # Analyze flight phases
        hover_time = 0
        forward_time = 0
        transitions = 0
        
        for i, waypoint in enumerate(waypoints):
            wp_type = waypoint.get("waypoint_type", "waypoint")
            speed = waypoint.get("speed", 10)
            
            # Determine if this is hover or forward flight
            if speed < self.configuration["transition_speed"] or wp_type in ["takeoff", "land", "loiter"]:
                hover_time += 1  # Simplified time calculation
            else:
                forward_time += 1
                
            # Count transitions
            if i > 0:
                prev_speed = waypoints[i-1].get("speed", 10)
                transition_threshold = self.configuration["transition_speed"]
                
                if (speed >= transition_threshold and prev_speed < transition_threshold) or \
                   (speed < transition_threshold and prev_speed >= transition_threshold):
                    transitions += 1
        
        # Check endurance limits
        hover_endurance = self.configuration["hover_endurance"]
        forward_endurance = self.configuration["forward_endurance"]
        
        if hover_time > hover_endurance * 0.8:
            validation_result["warnings"].append(
                f"Hover time may exceed efficient hover endurance ({hover_endurance}min)"
            )
        
        # Check transition frequency
        if transitions > 10:
            validation_result["warnings"].append(
                f"High number of transitions ({transitions}) may reduce efficiency"
            )
        
        validation_result["recommendations"].append(
            "Optimize mission to minimize hover time for better efficiency"
        )
        validation_result["recommendations"].append(
            "Group hover operations to reduce transition frequency"
        )
        
        return validation_result

class GenericVehicle(BaseVehicle):
    """Generic vehicle implementation for unknown or custom vehicle types"""
    
    def __init__(self, vehicle_id: str):
        super().__init__(vehicle_id, "generic")
        self.capabilities = VehicleCapabilities(
            can_hover=True,  # Assume most flexible capabilities
            requires_runway=False,
            min_speed=0,
            max_speed=20,
            max_altitude=400,
            max_payload=3.0,
            flight_modes=["manual", "stabilize", "guided", "auto"],
            has_gimbal=False
        )
    
    def configure(self, config: Dict[str, Any]) -> None:
        """Configure generic vehicle parameters"""
        
        self.configuration = {
            "actual_type": config.get("actual_type", "unknown"),
            "max_speed": config.get("max_speed", 20),
            "cruise_speed": config.get("cruise_speed", 15),
            "max_altitude": config.get("max_altitude", 400),
            "battery_capacity": config.get("battery_capacity", 12000),
            "estimated_flight_time": config.get("estimated_flight_time", 30),
            "can_hover": config.get("can_hover", True),
            "requires_runway": config.get("requires_runway", False),
        }
        
        # Update capabilities based on configuration
        self.capabilities.can_hover = self.configuration["can_hover"]
        self.capabilities.requires_runway = self.configuration["requires_runway"]
        self.capabilities.max_speed = self.configuration["max_speed"]
        self.capabilities.max_altitude = self.configuration["max_altitude"]
        
        if not self.capabilities.can_hover:
            self.capabilities.min_speed = self.configuration["cruise_speed"] * 0.5
        
        self.status = "configured"
    
    def validate_mission_compatibility(self, mission_data: Dict[str, Any]) -> Dict[str, Any]:
        """Validate mission compatibility for generic vehicle"""
        
        validation_result = {
            "compatible": True,
            "warnings": [],
            "errors": [],
            "recommendations": []
        }
        
        waypoints = mission_data.get("waypoints", [])
        
        # Basic validation
        for i, waypoint in enumerate(waypoints):
            altitude = waypoint.get("altitude", 0)
            speed = waypoint.get("speed", 10)
            
            if altitude > self.capabilities.max_altitude:
                validation_