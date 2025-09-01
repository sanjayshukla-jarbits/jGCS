"""
Vehicle Manager Module Package - Vehicle configuration and fleet management

This package provides comprehensive vehicle management capabilities:
- Vehicle controller and management logic
- Vehicle type factory (multi-rotor, fixed-wing, VTOL)
- Vehicle configuration management
- Kafka-based fleet coordination
- Vehicle state streaming
- Command dispatch system
"""

from src.modules.vehicle_manager.vehicle_controller import VehicleManagerModule
from src.modules.vehicle_manager.vehicle_factory import VehicleFactory
from src.modules.vehicle_manager.configuration import VehicleConfiguration

__all__ = [
    "VehicleManagerModule",
    "VehicleFactory",
    "VehicleConfiguration"
]