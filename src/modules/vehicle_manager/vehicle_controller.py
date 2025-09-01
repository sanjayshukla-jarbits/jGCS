"""
Vehicle Manager Module - Vehicle management and fleet coordination

This module handles:
- Multi-rotor vehicle management
- Fixed-wing vehicle management  
- VTOL vehicle management
- Generic vehicle management
- Vehicle configuration and capabilities
"""

import asyncio
from typing import Dict, List, Any, Optional
from datetime import datetime
from dataclasses import dataclass, asdict

from src.modules.base_module import BaseModule
from src.models.process import ProcessStep
from src.orchestrator.execution_context import ExecutionContext
from src.modules.vehicle_manager.vehicle_factory import VehicleFactory
from src.modules.vehicle_manager.configuration import VehicleConfiguration

@dataclass
class VehicleStatus:
    """Current status of a vehicle"""
    vehicle_id: str
    vehicle_type: str
    status: str  # "idle", "armed", "flying", "landing", "emergency", "offline"
    location: Optional[Dict[str, float]] = None
    battery_level: Optional[float] = None
    mission_progress: Optional[float] = None
    last_update: datetime = None

class VehicleManagerModule(BaseModule):
    """Module for vehicle management and coordination"""
    
    def __init__(self, module_name: str, implementation: str):
        super().__init__(module_name, implementation)
        self.vehicle_factory = VehicleFactory()
        self.vehicle_config = VehicleConfiguration()
        self.managed_vehicles: Dict[str, Dict[str, Any]] = {}
        self.vehicle_status: Dict[str, VehicleStatus] = {}
    
    async def _execute_implementation(self, step_config: ProcessStep, context: ExecutionContext) -> Dict[str, Any]:
        """Execute vehicle management implementation"""
        
        implementation_map = {
            "multi_rotor_manager": self._manage_multi_rotor_vehicles,
            "fixed_wing_manager": self._manage_fixed_wing_vehicles,
            "vtol_manager": self._manage_vtol_vehicles,
            "generic_manager": self._manage_generic_vehicles
        }
        
        if self.implementation not in implementation_map:
            raise ValueError(f"Unknown vehicle manager implementation: {self.implementation}")
        
        return await implementation_map[self.implementation](step_config, context)
    
    async def _validate_parameters(self, parameters: Dict[str, Any]) -> None:
        """Validate vehicle management parameters"""
        
        required_params = ["vehicles"]
        self._validate_required_parameters(parameters, required_params)
        
        vehicles = parameters["vehicles"]
        if not isinstance(vehicles, list) or len(vehicles) == 0:
            raise ValueError("At least one vehicle must be specified")
        
        for vehicle in vehicles:
            if "vehicle_id" not in vehicle:
                raise ValueError("Each vehicle must have a vehicle_id")
            
            if "vehicle_type" not in vehicle:
                raise ValueError("Each vehicle must have a vehicle_type")
    
    async def _manage_multi_rotor_vehicles(self, step_config: ProcessStep, context: ExecutionContext) -> Dict[str, Any]:
        """Manage multi-rotor vehicles"""
        
        params = step_config.parameters
        vehicles = params["vehicles"]
        management_action = params.get("action", "configure")
        
        results = {}
        
        for vehicle_data in vehicles:
            vehicle_id = vehicle_data["vehicle_id"]
            
            # Create multi-rotor vehicle instance
            vehicle = self.vehicle_factory.create_vehicle(
                vehicle_type="multi_rotor",
                vehicle_id=vehicle_id,
                config=vehicle_data
            )
            
            # Store managed vehicle
            self.managed_vehicles[vehicle_id] = {
                "vehicle": vehicle,
                "type": "multi_rotor",
                "config": vehicle_data,
                "managed_since": datetime.now()
            }
            
            # Initialize vehicle status
            self.vehicle_status[vehicle_id] = VehicleStatus(
                vehicle_id=vehicle_id,
                vehicle_type="multi_rotor",
                status="idle",
                last_update=datetime.now()
            )
            
            # Perform management action
            if management_action == "configure":
                result = await self._configure_multi_rotor(vehicle, vehicle_data)
            elif management_action == "validate":
                result = await self._validate_multi_rotor(vehicle, vehicle_data)
            elif management_action == "prepare_mission":
                result = await self._prepare_multi_rotor_mission(vehicle, vehicle_data)
            else:
                result = {"success": True, "message": "Vehicle registered"}
            
            results[vehicle_id] = result
        
        # Store vehicle fleet information in context
        context.set_shared_data("managed_vehicles", self.managed_vehicles)
        context.set_shared_data("vehicle_status", {k: asdict(v) for k, v in self.vehicle_status.items()})
        
        return {
            "management_type": "multi_rotor",
            "action": management_action,
            "vehicle_results": results,
            "total_vehicles": len(vehicles),
            "successful_vehicles": len([r for r in results.values() if r.get("success", False)]),
            "timestamp": datetime.now().isoformat()
        }
    
    async def _manage_fixed_wing_vehicles(self, step_config: ProcessStep, context: ExecutionContext) -> Dict[str, Any]:
        """Manage fixed-wing vehicles"""
        
        params = step_config.parameters
        vehicles = params["vehicles"]
        management_action = params.get("action", "configure")
        
        results = {}
        
        for vehicle_data in vehicles:
            vehicle_id = vehicle_data["vehicle_id"]
            
            # Create fixed-wing vehicle instance
            vehicle = self.vehicle_factory.create_vehicle(
                vehicle_type="fixed_wing",
                vehicle_id=vehicle_id,
                config=vehicle_data
            )
            
            # Store managed vehicle
            self.managed_vehicles[vehicle_id] = {
                "vehicle": vehicle,
                "type": "fixed_wing",
                "config": vehicle_data,
                "managed_since": datetime.now()
            }
            
            # Initialize vehicle status
            self.vehicle_status[vehicle_id] = VehicleStatus(
                vehicle_id=vehicle_id,
                vehicle_type="fixed_wing",
                status="idle",
                last_update=datetime.now()
            )
            
            # Perform management action
            if management_action == "configure":
                result = await self._configure_fixed_wing(vehicle, vehicle_data)
            elif management_action == "validate":
                result = await self._validate_fixed_wing(vehicle, vehicle_data)
            elif management_action == "prepare_mission":
                result = await self._prepare_fixed_wing_mission(vehicle, vehicle_data)
            else:
                result = {"success": True, "message": "Vehicle registered"}
            
            results[vehicle_id] = result
        
        # Store vehicle fleet information in context
        context.set_shared_data("managed_vehicles", self.managed_vehicles)
        context.set_shared_data("vehicle_status", {k: asdict(v) for k, v in self.vehicle_status.items()})
        
        return {
            "management_type": "fixed_wing",
            "action": management_action,
            "vehicle_results": results,
            "total_vehicles": len(vehicles),
            "successful_vehicles": len([r for r in results.values() if r.get("success", False)]),
            "timestamp": datetime.now().isoformat()
        }
    
    async def _manage_vtol_vehicles(self, step_config: ProcessStep, context: ExecutionContext) -> Dict[str, Any]:
        """Manage VTOL vehicles"""
        
        params = step_config.parameters
        vehicles = params["vehicles"]
        management_action = params.get("action", "configure")
        
        results = {}
        
        for vehicle_data in vehicles:
            vehicle_id = vehicle_data["vehicle_id"]
            
            # Create VTOL vehicle instance
            vehicle = self.vehicle_factory.create_vehicle(
                vehicle_type="vtol",
                vehicle_id=vehicle_id,
                config=vehicle_data
            )
            
            # Store managed vehicle
            self.managed_vehicles[vehicle_id] = {
                "vehicle": vehicle,
                "type": "vtol",
                "config": vehicle_data,
                "managed_since": datetime.now()
            }
            
            # Initialize vehicle status
            self.vehicle_status[vehicle_id] = VehicleStatus(
                vehicle_id=vehicle_id,
                vehicle_type="vtol",
                status="idle",
                last_update=datetime.now()
            )
            
            # Perform management action
            if management_action == "configure":
                result = await self._configure_vtol(vehicle, vehicle_data)
            elif management_action == "validate":
                result = await self._validate_vtol(vehicle, vehicle_data)
            elif management_action == "prepare_mission":
                result = await self._prepare_vtol_mission(vehicle, vehicle_data)
            else:
                result = {"success": True, "message": "Vehicle registered"}
            
            results[vehicle_id] = result
        
        # Store vehicle fleet information in context
        context.set_shared_data("managed_vehicles", self.managed_vehicles)
        context.set_shared_data("vehicle_status", {k: asdict(v) for k, v in self.vehicle_status.items()})
        
        return {
            "management_type": "vtol",
            "action": management_action,
            "vehicle_results": results,
            "total_vehicles": len(vehicles),
            "successful_vehicles": len([r for r in results.values() if r.get("success", False)]),
            "timestamp": datetime.now().isoformat()
        }
    
    async def _manage_generic_vehicles(self, step_config: ProcessStep, context: ExecutionContext) -> Dict[str, Any]:
        """Manage generic vehicles"""
        
        params = step_config.parameters
        vehicles = params["vehicles"]
        management_action = params.get("action", "configure")
        
        results = {}
        
        for vehicle_data in vehicles:
            vehicle_id = vehicle_data["vehicle_id"]
            vehicle_type = vehicle_data.get("vehicle_type", "generic")
            
            # Create generic vehicle instance
            vehicle = self.vehicle_factory.create_vehicle(
                vehicle_type="generic",
                vehicle_id=vehicle_id,
                config=vehicle_data
            )
            
            # Store managed vehicle
            self.managed_vehicles[vehicle_id] = {
                "vehicle": vehicle,
                "type": vehicle_type,
                "config": vehicle_data,
                "managed_since": datetime.now()
            }
            
            # Initialize vehicle status
            self.vehicle_status[vehicle_id] = VehicleStatus(
                vehicle_id=vehicle_id,
                vehicle_type=vehicle_type,
                status="idle",
                last_update=datetime.now()
            )
            
            # Perform management action
            if management_action == "configure":
                result = await self._configure_generic_vehicle(vehicle, vehicle_data)
            elif management_action == "validate":
                result = await self._validate_generic_vehicle(vehicle, vehicle_data)
            elif management_action == "prepare_mission":
                result = await self._prepare_generic_mission(vehicle, vehicle_data)
            else:
                result = {"success": True, "message": "Vehicle registered"}
            
            results[vehicle_id] = result
        
        # Store vehicle fleet information in context
        context.set_shared_data("managed_vehicles", self.managed_vehicles)
        context.set_shared_data("vehicle_status", {k: asdict(v) for k, v in self.vehicle_status.items()})
        
        return {
            "management_type": "generic",
            "action": management_action,
            "vehicle_results": results,
            "total_vehicles": len(vehicles),
            "successful_vehicles": len([r for r in results.values() if r.get("success", False)]),
            "timestamp": datetime.now().isoformat()
        }
    
    async def _configure_multi_rotor(self, vehicle: Any, config: Dict[str, Any]) -> Dict[str, Any]:
        """Configure multi-rotor specific parameters"""
        
        try:
            # Multi-rotor specific configuration
            max_climb_rate = config.get("max_climb_rate", 5.0)  # m/s
            max_descent_rate = config.get("max_descent_rate", 3.0)  # m/s
            max_horizontal_speed = config.get("max_horizontal_speed", 20.0)  # m/s
            hover_accuracy = config.get("hover_accuracy", 2.0)  # meters
            
            # Validate multi-rotor constraints
            if max_climb_rate > 10.0:
                return {"success": False, "error": "Max climb rate exceeds multi-rotor limits"}
            
            if max_horizontal_speed > 25.0:
                return {"success": False, "error": "Max speed exceeds multi-rotor limits"}
            
            # Configure vehicle parameters
            vehicle_params = {
                "max_climb_rate": max_climb_rate,
                "max_descent_rate": max_descent_rate,
                "max_horizontal_speed": max_horizontal_speed,
                "hover_accuracy": hover_accuracy,
                "flight_mode_capabilities": ["manual", "stabilize", "altitude_hold", "position_hold", "guided", "auto"],
                "can_hover": True,
                "requires_runway": False,
                "min_speed": 0  # Can hover
            }
            
            vehicle.configure(vehicle_params)
            
            return {
                "success": True,
                "message": "Multi-rotor configured successfully",
                "configuration": vehicle_params
            }
            
        except Exception as e:
            return {"success": False, "error": f"Multi-rotor configuration failed: {str(e)}"}
    
    async def _configure_fixed_wing(self, vehicle: Any, config: Dict[str, Any]) -> Dict[str, Any]:
        """Configure fixed-wing specific parameters"""
        
        try:
            # Fixed-wing specific configuration
            stall_speed = config.get("stall_speed", 12.0)  # m/s
            max_speed = config.get("max_speed", 30.0)  # m/s
            cruise_speed = config.get("cruise_speed", 20.0)  # m/s
            min_turn_radius = config.get("min_turn_radius", 50.0)  # meters
            
            # Validate fixed-wing constraints
            if stall_speed >= cruise_speed:
                return {"success": False, "error": "Stall speed must be less than cruise speed"}
            
            if cruise_speed > max_speed:
                return {"success": False, "error": "Cruise speed cannot exceed max speed"}
            
            # Configure vehicle parameters
            vehicle_params = {
                "stall_speed": stall_speed,
                "max_speed": max_speed,
                "cruise_speed": cruise_speed,
                "min_turn_radius": min_turn_radius,
                "flight_mode_capabilities": ["manual", "stabilize", "cruise", "auto"],
                "can_hover": False,
                "requires_runway": True,
                "min_speed": stall_speed
            }
            
            vehicle.configure(vehicle_params)
            
            return {
                "success": True,
                "message": "Fixed-wing configured successfully",
                "configuration": vehicle_params
            }
            
        except Exception as e:
            return {"success": False, "error": f"Fixed-wing configuration failed: {str(e)}"}
    
    async def _configure_vtol(self, vehicle: Any, config: Dict[str, Any]) -> Dict[str, Any]:
        """Configure VTOL specific parameters"""
        
        try:
            # VTOL specific configuration
            hover_speed_max = config.get("hover_speed_max", 15.0)  # m/s
            forward_speed_max = config.get("forward_speed_max", 25.0)  # m/s
            transition_speed = config.get("transition_speed", 18.0)  # m/s
            hover_efficiency = config.get("hover_efficiency", 0.7)  # 0-1
            
            # Validate VTOL constraints
            if transition_speed > forward_speed_max:
                return {"success": False, "error": "Transition speed cannot exceed max forward speed"}
            
            # Configure vehicle parameters
            vehicle_params = {
                "hover_speed_max": hover_speed_max,
                "forward_speed_max": forward_speed_max,
                "transition_speed": transition_speed,
                "hover_efficiency": hover_efficiency,
                "flight_mode_capabilities": ["manual", "stabilize", "altitude_hold", "position_hold", "guided", "auto", "vtol_transition"],
                "can_hover": True,
                "requires_runway": False,
                "min_speed": 0,  # Can hover
                "has_vtol_capability": True
            }
            
            vehicle.configure(vehicle_params)
            
            return {
                "success": True,
                "message": "VTOL configured successfully",
                "configuration": vehicle_params
            }
            
        except Exception as e:
            return {"success": False, "error": f"VTOL configuration failed: {str(e)}"}
    
    async def _configure_generic_vehicle(self, vehicle: Any, config: Dict[str, Any]) -> Dict[str, Any]:
        """Configure generic vehicle parameters"""
        
        try:
            # Generic vehicle configuration
            max_speed = config.get("max_speed", 20.0)  # m/s
            cruise_speed = config.get("cruise_speed", 15.0)  # m/s
            can_hover = config.get("can_hover", True)
            requires_runway = config.get("requires_runway", False)
            
            # Configure vehicle parameters
            vehicle_params = {
                "max_speed": max_speed,
                "cruise_speed": cruise_speed,
                "can_hover": can_hover,
                "requires_runway": requires_runway,
                "flight_mode_capabilities": ["manual", "stabilize", "guided", "auto"],
                "min_speed": 0 if can_hover else cruise_speed * 0.5
            }
            
            vehicle.configure(vehicle_params)
            
            return {
                "success": True,
                "message": "Generic vehicle configured successfully",
                "configuration": vehicle_params
            }
            
        except Exception as e:
            return {"success": False, "error": f"Generic vehicle configuration failed: {str(e)}"}
    
    async def _validate_multi_rotor(self, vehicle: Any, config: Dict[str, Any]) -> Dict[str, Any]:
        """Validate multi-rotor vehicle configuration"""
        
        validation_results = []
        
        # Check battery configuration
        battery_capacity = config.get("battery_capacity", 0)
        if battery_capacity < 5000:  # mAh
            validation_results.append("Warning: Low battery capacity for multi-rotor")
        
        # Check payload capacity
        max_payload = config.get("max_payload", 0)
        if max_payload < 1.0:  # kg
            validation_results.append("Warning: Low payload capacity")
        
        # Check flight time estimation
        estimated_flight_time = config.get("estimated_flight_time", 0)
        if estimated_flight_time < 20:  # minutes
            validation_results.append("Warning: Short estimated flight time")
        
        return {
            "success": True,
            "validation_results": validation_results,
            "vehicle_type": "multi_rotor"
        }
    
    async def _validate_fixed_wing(self, vehicle: Any, config: Dict[str, Any]) -> Dict[str, Any]:
        """Validate fixed-wing vehicle configuration"""
        
        validation_results = []
        
        # Check runway requirements
        if not config.get("runway_length"):
            validation_results.append("Warning: Runway length not specified")
        
        # Check endurance
        estimated_flight_time = config.get("estimated_flight_time", 0)
        if estimated_flight_time < 60:  # minutes
            validation_results.append("Warning: Short flight time for fixed-wing")
        
        return {
            "success": True,
            "validation_results": validation_results,
            "vehicle_type": "fixed_wing"
        }
    
    async def _validate_vtol(self, vehicle: Any, config: Dict[str, Any]) -> Dict[str, Any]:
        """Validate VTOL vehicle configuration"""
        
        validation_results = []
        
        # Check transition capabilities
        if not config.get("transition_speed"):
            validation_results.append("Warning: Transition speed not specified")
        
        # Check power requirements
        hover_power = config.get("hover_power_consumption", 0)
        forward_power = config.get("forward_power_consumption", 0)
        
        if hover_power > forward_power * 2:
            validation_results.append("Warning: High hover power consumption")
        
        return {
            "success": True,
            "validation_results": validation_results,
            "vehicle_type": "vtol"
        }
    
    async def _validate_generic_vehicle(self, vehicle: Any, config: Dict[str, Any]) -> Dict[str, Any]:
        """Validate generic vehicle configuration"""
        
        validation_results = []
        
        # Basic validation checks
        if not config.get("max_speed"):
            validation_results.append("Warning: Max speed not specified")
        
        if not config.get("estimated_flight_time"):
            validation_results.append("Warning: Flight time not specified")
        
        return {
            "success": True,
            "validation_results": validation_results,
            "vehicle_type": "generic"
        }
    
    def update_vehicle_status(self, vehicle_id: str, status_update: Dict[str, Any]) -> None:
        """Update vehicle status"""
        
        if vehicle_id in self.vehicle_status:
            status = self.vehicle_status[vehicle_id]
            
            if "status" in status_update:
                status.status = status_update["status"]
            
            if "location" in status_update:
                status.location = status_update["location"]
            
            if "battery_level" in status_update:
                status.battery_level = status_update["battery_level"]
            
            if "mission_progress" in status_update:
                status.mission_progress = status_update["mission_progress"]
            
            status.last_update = datetime.now()
    
    def get_vehicle_status(self, vehicle_id: str = None) -> Dict[str, Any]:
        """Get vehicle status"""
        
        if vehicle_id:
            status = self.vehicle_status.get(vehicle_id)
            return asdict(status) if status else None
        else:
            return {vid: asdict(status) for vid, status in self.vehicle_status.items()}
    
    def get_fleet_summary(self) -> Dict[str, Any]:
        """Get fleet summary"""
        
        total_vehicles = len(self.managed_vehicles)
        
        # Count by type
        type_counts = {}
        status_counts = {}
        
        for vehicle_id, vehicle_data in self.managed_vehicles.items():
            vehicle_type = vehicle_data["type"]
            type_counts[vehicle_type] = type_counts.get(vehicle_type, 0) + 1
            
            status = self.vehicle_status.get(vehicle_id)
            if status:
                status_counts[status.status] = status_counts.get(status.status, 0) + 1
        
        return {
            "total_vehicles": total_vehicles,
            "vehicle_types": type_counts,
            "vehicle_statuses": status_counts,
            "last_updated": datetime.now().isoformat()
        }