"""
QGroundControl Format Converter - Convert missions to QGC format

This module provides:
- QGroundControl mission format conversion
- MAVLink command mapping
- Mission item serialization
- QGC plan file generation
"""

import json
from typing import Dict, List, Any, Optional
from datetime import datetime
from dataclasses import dataclass

from src.models.mission import Waypoint

@dataclass
class QGCMissionItem:
    """QGroundControl mission item"""
    command: int
    coordinate: List[float]  # [lat, lon, alt]
    autoContinue: bool = True
    frame: int = 3  # MAV_FRAME_GLOBAL_RELATIVE_ALT
    params: List[float] = None
    type: str = "SimpleItem"
    
    def __post_init__(self):
        if self.params is None:
            self.params = [0.0, 0.0, 0.0, float('nan'), 0.0, 0.0, 0.0]

class QGCConverter:
    """Converts waypoint missions to QGroundControl format"""
    
    def __init__(self):
        # MAVLink command mappings
        self.mavlink_commands = {
            "takeoff": 22,      # MAV_CMD_NAV_TAKEOFF
            "waypoint": 16,     # MAV_CMD_NAV_WAYPOINT
            "land": 21,         # MAV_CMD_NAV_LAND
            "rtl": 20,          # MAV_CMD_NAV_RETURN_TO_LAUNCH
            "loiter": 17,       # MAV_CMD_NAV_LOITER_UNLIM
            "survey": 16,       # MAV_CMD_NAV_WAYPOINT (for survey points)
            "corridor": 16,     # MAV_CMD_NAV_WAYPOINT (for corridor points)
            "inspection": 16,   # MAV_CMD_NAV_WAYPOINT (for inspection points)
            "patrol": 16,       # MAV_CMD_NAV_WAYPOINT (for patrol points)
            "search": 16,       # MAV_CMD_NAV_WAYPOINT (for search points)
        }
        
        # Frame types
        self.frame_types = {
            "global": 0,                    # MAV_FRAME_GLOBAL
            "global_relative": 3,           # MAV_FRAME_GLOBAL_RELATIVE_ALT
            "local_ned": 1,                 # MAV_FRAME_LOCAL_NED
            "body_offset": 8                # MAV_FRAME_BODY_OFFSET_NED
        }
    
    def convert_to_qgc(self, waypoints: List[Waypoint], mission_name: str = "Mission",
                       vehicle_type: str = "multirotor") -> Dict[str, Any]:
        """Convert waypoints to QGC mission format"""
        
        if not waypoints:
            raise ValueError("No waypoints provided for conversion")
        
        # Create QGC mission items
        mission_items = []
        
        for i, waypoint in enumerate(waypoints):
            qgc_item = self._convert_waypoint_to_qgc_item(waypoint, i)
            mission_items.append(qgc_item)
        
        # Create QGC plan structure
        qgc_plan = {
            "fileType": "Plan",
            "geoFence": {
                "circles": [],
                "polygons": [],
                "version": 2
            },
            "groundStation": "QGroundControl",
            "mission": {
                "cruiseSpeed": 15,
                "firmwareType": 12,  # ArduPilot
                "globalPlanAltitudeMode": 1,  # Relative altitude
                "hoverSpeed": 5,
                "items": [self._create_home_item(waypoints[0])] + mission_items,
                "plannedHomePosition": [
                    waypoints[0].latitude,
                    waypoints[0].longitude,
                    waypoints[0].altitude
                ],
                "vehicleType": 2 if vehicle_type == "multirotor" else 1,  # 2=MultiRotor, 1=FixedWing
                "version": 2
            },
            "rallyPoints": {
                "points": [],
                "version": 2
            },
            "version": 1
        }
        
        return qgc_plan
    
    def _convert_waypoint_to_qgc_item(self, waypoint: Waypoint, sequence: int) -> Dict[str, Any]:
        """Convert a single waypoint to QGC mission item"""
        
        waypoint_type = getattr(waypoint, 'waypoint_type', 'waypoint')
        command = self.mavlink_commands.get(waypoint_type, 16)  # Default to NAV_WAYPOINT
        
        # Create base QGC item
        qgc_item = {
            "AMSLAltAboveTerrain": None,
            "Altitude": waypoint.altitude,
            "AltitudeMode": 1,  # Relative
            "autoContinue": True,
            "command": command,
            "doJumpId": sequence + 1,
            "frame": 3,  # MAV_FRAME_GLOBAL_RELATIVE_ALT
            "params": [0, 0, 0, float('nan'), waypoint.latitude, waypoint.longitude, waypoint.altitude],
            "type": "SimpleItem"
        }
        
        # Customize parameters based on waypoint type
        if waypoint_type == "takeoff":
            qgc_item["params"] = [
                15,  # Minimum pitch angle
                0, 0, float('nan'),
                waypoint.latitude, waypoint.longitude, waypoint.altitude
            ]
        
        elif waypoint_type == "land":
            qgc_item["params"] = [
                0,   # Abort altitude
                0,   # Precision land mode
                0, float('nan'),
                waypoint.latitude, waypoint.longitude, 0  # Land at ground level
            ]
        
        elif waypoint_type == "loiter":
            qgc_item["params"] = [
                0,   # Loiter radius
                0,   # Loiter direction
                0, float('nan'),
                waypoint.latitude, waypoint.longitude, waypoint.altitude
            ]
        
        elif waypoint_type == "rtl":
            qgc_item["command"] = 20  # MAV_CMD_NAV_RETURN_TO_LAUNCH
            qgc_item["params"] = [0, 0, 0, float('nan'), 0, 0, 0]
        
        else:
            # Standard waypoint
            hold_time = getattr(waypoint, 'hold_time', 0)
            accept_radius = getattr(waypoint, 'accept_radius', 3)
            pass_radius = getattr(waypoint, 'pass_radius', 0)
            
            qgc_item["params"] = [
                hold_time,      # Hold time
                accept_radius,  # Accept radius
                pass_radius,    # Pass radius
                float('nan'),   # Yaw angle
                waypoint.latitude, waypoint.longitude, waypoint.altitude
            ]
        
        # Add actions if present
        if hasattr(waypoint, 'action') and waypoint.action:
            qgc_item = self._add_waypoint_action(qgc_item, waypoint)
        
        return qgc_item
    
    def _create_home_item(self, first_waypoint: Waypoint) -> Dict[str, Any]:
        """Create home position item"""
        
        return {
            "AMSLAltAboveTerrain": None,
            "Altitude": 0,
            "AltitudeMode": 1,
            "autoContinue": True,
            "command": 16,  # MAV_CMD_NAV_WAYPOINT
            "doJumpId": 1,
            "frame": 0,  # MAV_FRAME_GLOBAL
            "params": [0, 0, 0, float('nan'), first_waypoint.latitude, first_waypoint.longitude, 0],
            "type": "SimpleItem"
        }
    
    def _add_waypoint_action(self, qgc_item: Dict[str, Any], waypoint: Waypoint) -> Dict[str, Any]:
        """Add action commands to waypoint"""
        
        action = waypoint.action
        action_params = getattr(waypoint, 'action_params', {})
        
        if action == "take_photo":
            # Add camera trigger command
            qgc_item["command"] = 2000  # MAV_CMD_IMAGE_START_CAPTURE
            qgc_item["params"] = [
                0,  # Camera ID
                0,  # Interval
                1,  # Total images
                0,  # Sequence number
                0, 0, 0
            ]
        
        elif action == "start_video":
            qgc_item["command"] = 2500  # MAV_CMD_VIDEO_START_CAPTURE
            qgc_item["params"] = [0, 0, 0, 0, 0, 0, 0]
        
        elif action == "stop_video":
            qgc_item["command"] = 2501  # MAV_CMD_VIDEO_STOP_CAPTURE
            qgc_item["params"] = [0, 0, 0, 0, 0, 0, 0]
        
        elif action == "gimbal_point_to_structure":
            # Point gimbal at target
            target_lat = action_params.get("target_lat", 0)
            target_lon = action_params.get("target_lon", 0)
            
            qgc_item["command"] = 205  # MAV_CMD_DO_MOUNT_CONTROL
            qgc_item["params"] = [
                target_lat,   # Pitch
                0,           # Roll
                target_lon,  # Yaw
                0, 0, 0,     # Reserved
                2            # Mount mode: MAV_MOUNT_MODE_GPS_POINT
            ]
        
        elif action == "change_speed":
            speed = action_params.get("speed", 10)
            qgc_item["command"] = 178  # MAV_CMD_DO_CHANGE_SPEED
            qgc_item["params"] = [
                1,     # Speed type (1=Ground Speed)
                speed, # Speed value
                -1,    # Throttle (-1=no change)
                0, 0, 0, 0
            ]
        
        return qgc_item
    
    def save_qgc_plan(self, qgc_plan: Dict[str, Any], filepath: str) -> None:
        """Save QGC plan to file"""
        
        try:
            with open(filepath, 'w') as f:
                json.dump(qgc_plan, f, indent=2)
        except Exception as e:
            raise Exception(f"Failed to save QGC plan to {filepath}: {str(e)}")
    
    def load_qgc_plan(self, filepath: str) -> Dict[str, Any]:
        """Load QGC plan from file"""
        
        try:
            with open(filepath, 'r') as f:
                return json.load(f)
        except Exception as e:
            raise Exception(f"Failed to load QGC plan from {filepath}: {str(e)}")
    
    def convert_from_qgc(self, qgc_plan: Dict[str, Any]) -> List[Waypoint]:
        """Convert QGC plan back to waypoints"""
        
        waypoints = []
        
        mission_items = qgc_plan.get("mission", {}).get("items", [])
        
        # Skip home item (first item)
        for item in mission_items[1:]:
            waypoint = self._convert_qgc_item_to_waypoint(item)
            if waypoint:
                waypoints.append(waypoint)
        
        return waypoints
    
    def _convert_qgc_item_to_waypoint(self, qgc_item: Dict[str, Any]) -> Optional[Waypoint]:
        """Convert QGC mission item back to waypoint"""
        
        command = qgc_item.get("command", 16)
        params = qgc_item.get("params", [])
        
        if len(params) < 7:
            return None
        
        # Extract coordinates
        latitude = params[4]
        longitude = params[5] 
        altitude = params[6]
        
        # Determine waypoint type from command
        waypoint_type = "waypoint"  # default
        for wt, cmd in self.mavlink_commands.items():
            if cmd == command:
                waypoint_type = wt
                break
        
        # Create waypoint
        waypoint = Waypoint(
            latitude=latitude,
            longitude=longitude,
            altitude=altitude,
            waypoint_type=waypoint_type
        )
        
        # Extract additional parameters
        if waypoint_type == "waypoint" and len(params) >= 3:
            waypoint.hold_time = params[0]
            waypoint.accept_radius = params[1]
            waypoint.pass_radius = params[2]
        
        return waypoint
    
    def validate_qgc_plan(self, qgc_plan: Dict[str, Any]) -> Dict[str, Any]:
        """Validate QGC plan format"""
        
        validation_results = {
            "valid": True,
            "errors": [],
            "warnings": []
        }
        
        # Check required fields
        if "mission" not in qgc_plan:
            validation_results["valid"] = False
            validation_results["errors"].append("Missing 'mission' section")
        
        mission = qgc_plan.get("mission", {})
        
        if "items" not in mission:
            validation_results["valid"] = False
            validation_results["errors"].append("Missing mission items")
        
        items = mission.get("items", [])
        
        if len(items) < 2:  # At least home + one waypoint
            validation_results["warnings"].append("Mission has very few waypoints")
        
        # Validate each mission item
        for i, item in enumerate(items):
            item_errors = self._validate_mission_item(item, i)
            validation_results["errors"].extend(item_errors)
        
        # Check for logical issues
        if len(items) > 1:
            # Check if mission has takeoff
            has_takeoff = any(item.get("command") == 22 for item in items[1:])  # Skip home
            if not has_takeoff:
                validation_results["warnings"].append("Mission does not include explicit takeoff command")
            
            # Check if mission has landing or RTL
            has_landing = any(item.get("command") in [20, 21] for item in items)  # RTL or LAND
            if not has_landing:
                validation_results["warnings"].append("Mission does not include landing or RTL command")
        
        validation_results["valid"] = len(validation_results["errors"]) == 0
        
        return validation_results
    
    def _validate_mission_item(self, item: Dict[str, Any], index: int) -> List[str]:
        """Validate individual mission item"""
        
        errors = []
        
        # Check required fields
        if "command" not in item:
            errors.append(f"Item {index}: Missing command field")
        
        if "params" not in item:
            errors.append(f"Item {index}: Missing params field")
        else:
            params = item["params"]
            if not isinstance(params, list) or len(params) < 7:
                errors.append(f"Item {index}: Invalid params array (must have 7 elements)")
            else:
                # Validate coordinates
                lat, lon, alt = params[4], params[5], params[6]
                
                if not (-90 <= lat <= 90):
                    errors.append(f"Item {index}: Invalid latitude {lat}")
                
                if not (-180 <= lon <= 180):
                    errors.append(f"Item {index}: Invalid longitude {lon}")
                
                if alt < 0:
                    errors.append(f"Item {index}: Invalid altitude {alt}")
        
        return errors
    
    def get_mission_statistics(self, qgc_plan: Dict[str, Any]) -> Dict[str, Any]:
        """Get statistics about QGC mission plan"""
        
        mission = qgc_plan.get("mission", {})
        items = mission.get("items", [])
        
        # Count different command types
        command_counts = {}
        total_distance = 0
        
        prev_lat, prev_lon = None, None
        
        for item in items:
            command = item.get("command", 0)
            command_counts[command] = command_counts.get(command, 0) + 1
            
            params = item.get("params", [])
            if len(params) >= 6:
                lat, lon = params[4], params[5]
                
                # Calculate distance from previous point
                if prev_lat is not None and prev_lon is not None:
                    # Simple distance calculation (would use proper geo calculation in production)
                    distance = ((lat - prev_lat) ** 2 + (lon - prev_lon) ** 2) ** 0.5 * 111320
                    total_distance += distance
                
                prev_lat, prev_lon = lat, lon
        
        return {
            "total_waypoints": len(items) - 1,  # Exclude home item
            "total_distance_meters": total_distance,
            "command_breakdown": command_counts,
            "estimated_flight_time_minutes": total_distance / 250,  # Rough estimate at 15 m/s
            "file_size_bytes": len(json.dumps(qgc_plan).encode('utf-8')),
            "qgc_version": qgc_plan.get("version", "unknown")
        }