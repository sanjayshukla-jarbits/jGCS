"""
SkyrouteX Modules Package - Core modules for UAV mission planning and execution

This package contains all the core modules for the SkyrouteX system:
- sensor_manager: External sensor library integration and monitoring
- mission_planner: Mission planning and generation logic
- vehicle_manager: Vehicle configuration and fleet management
- geofence_manager: Spatial validation and geofencing
- swarm_manager: Multi-vehicle coordination and swarm operations
- base_module: Base class for all modules
"""

from src.modules.base_module import BaseModule, ModuleResult

__all__ = [
    "BaseModule",
    "ModuleResult"
]

# Version information
__version__ = "1.0.0"
__author__ = "SkyrouteX Development Team"