"""
Sensor Manager Module Package - External sensor integration and monitoring

This package provides comprehensive sensor management capabilities:
- External sensor library client integration
- Server-side fact groups (QGC-style)
- Sensor bounds validation
- Kafka-based sensor monitoring
- Stream-based anomaly detection  
- Multi-sensor data fusion
"""

from src.modules.sensor_manager.sensor_client import SensorManagerModule
from src.modules.sensor_manager.fact_groups import FactGroupManager
from src.modules.sensor_manager.bounds_validator import BoundsValidator

__all__ = [
    "SensorManagerModule",
    "FactGroupManager", 
    "BoundsValidator"
]