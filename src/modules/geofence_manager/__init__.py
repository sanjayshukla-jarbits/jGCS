"""
Geofence Manager Module Package - Spatial validation and geofencing

This package provides comprehensive geofencing capabilities:
- Geofencing logic and spatial validation
- Zone validation and constraint checking
- Spatial operations and calculations
- Dynamic geofence updates via streams
- Real-time breach detection
- Geospatial indexing for fast queries
"""

from src.modules.geofence_manager.geofence_controller import GeofenceManagerModule
from src.modules.geofence_manager.zone_validator import ZoneValidator
from src.modules.geofence_manager.spatial_operations import SpatialOperations

__all__ = [
    "GeofenceManagerModule",
    "ZoneValidator", 
    "SpatialOperations"
]