"""
Server-side Fact Groups (QGC-style) - Structured sensor data organization

This module provides QGroundControl-style fact groups for organizing
and accessing sensor data in a structured, hierarchical manner.
"""

from typing import Dict, Any, Optional, List, Union
from datetime import datetime
from dataclasses import dataclass, field
from enum import Enum
import json

class FactType(Enum):
    """Types of facts that can be stored"""
    SENSOR_VALUE = "sensor_value"
    SYSTEM_STATUS = "system_status" 
    VEHICLE_STATE = "vehicle_state"
    MISSION_DATA = "mission_data"
    ENVIRONMENTAL = "environmental"

@dataclass
class Fact:
    """Individual fact with metadata"""
    name: str
    value: Any
    fact_type: FactType
    unit: Optional[str] = None
    timestamp: datetime = field(default_factory=datetime.now)
    source: Optional[str] = None
    quality: float = 1.0  # 0.0 to 1.0, quality indicator
    min_value: Optional[float] = None
    max_value: Optional[float] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert fact to dictionary"""
        return {
            "name": self.name,
            "value": self.value,
            "type": self.fact_type.value,
            "unit": self.unit,
            "timestamp": self.timestamp.isoformat(),
            "source": self.source,
            "quality": self.quality,
            "min_value": self.min_value,
            "max_value": self.max_value
        }

class FactGroup:
    """Group of related facts"""
    
    def __init__(self, group_name: str):
        self.group_name = group_name
        self.facts: Dict[str, Fact] = {}
        self.created_at = datetime.now()
        self.last_updated = datetime.now()
    
    def add_fact(self, fact: Fact) -> None:
        """Add or update a fact in this group"""
        self.facts[fact.name] = fact
        self.last_updated = datetime.now()
    
    def get_fact(self, name: str) -> Optional[Fact]:
        """Get a specific fact by name"""
        return self.facts.get(name)
    
    def get_fact_value(self, name: str, default: Any = None) -> Any:
        """Get just the value of a fact"""
        fact = self.facts.get(name)
        return fact.value if fact else default
    
    def remove_fact(self, name: str) -> bool:
        """Remove a fact from the group"""
        if name in self.facts:
            del self.facts[name]
            self.last_updated = datetime.now()
            return True
        return False
    
    def get_all_facts(self) -> Dict[str, Fact]:
        """Get all facts in this group"""
        return self.facts.copy()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert fact group to dictionary"""
        return {
            "group_name": self.group_name,
            "created_at": self.created_at.isoformat(),
            "last_updated": self.last_updated.isoformat(),
            "facts": {name: fact.to_dict() for name, fact in self.facts.items()}
        }

class FactGroupManager:
    """Manager for organizing fact groups (QGC-style)"""
    
    def __init__(self):
        self.fact_groups: Dict[str, FactGroup] = {}
        self._initialize_standard_groups()
    
    def _initialize_standard_groups(self) -> None:
        """Initialize standard QGC-style fact groups"""
        
        # Vehicle fact group
        self.create_group("vehicle")
        
        # GPS fact group  
        self.create_group("gps")
        
        # Battery fact group
        self.create_group("battery")
        
        # Attitude fact group
        self.create_group("attitude")
        
        # Mission fact group
        self.create_group("mission")
        
        # System fact group
        self.create_group("system")
        
        # Wind fact group
        self.create_group("wind")
        
        # Sensors fact group
        self.create_group("sensors")
    
    def create_group(self, group_name: str) -> FactGroup:
        """Create a new fact group"""
        if group_name in self.fact_groups:
            return self.fact_groups[group_name]
        
        fact_group = FactGroup(group_name)
        self.fact_groups[group_name] = fact_group
        return fact_group
    
    def get_group(self, group_name: str) -> Optional[FactGroup]:
        """Get a fact group by name"""
        return self.fact_groups.get(group_name)
    
    def add_fact(self, group_name: str, fact: Fact) -> None:
        """Add a fact to a specific group"""
        if group_name not in self.fact_groups:
            self.create_group(group_name)
        
        self.fact_groups[group_name].add_fact(fact)
    
    def get_fact(self, group_name: str, fact_name: str) -> Optional[Fact]:
        """Get a specific fact from a group"""
        group = self.fact_groups.get(group_name)
        return group.get_fact(fact_name) if group else None
    
    def get_fact_value(self, group_name: str, fact_name: str, default: Any = None) -> Any:
        """Get just the value of a specific fact"""
        fact = self.get_fact(group_name, fact_name)
        return fact.value if fact else default
    
    def update_vehicle_facts(self, telemetry: Dict[str, Any]) -> None:
        """Update vehicle fact group from telemetry data"""
        
        vehicle_group = self.get_group("vehicle")
        if not vehicle_group:
            return
        
        # Update basic vehicle facts
        if "armed" in telemetry:
            vehicle_group.add_fact(Fact(
                "armed", telemetry["armed"], FactType.VEHICLE_STATE,
                source="telemetry"
            ))
        
        if "flight_mode" in telemetry:
            vehicle_group.add_fact(Fact(
                "flight_mode", telemetry["flight_mode"], FactType.VEHICLE_STATE,
                source="telemetry"
            ))
        
        if "ready_to_fly" in telemetry:
            vehicle_group.add_fact(Fact(
                "ready_to_fly", telemetry["ready_to_fly"], FactType.VEHICLE_STATE,
                source="telemetry"
            ))
    
    def update_gps_facts(self, gps_data: Dict[str, Any]) -> None:
        """Update GPS fact group from GPS data"""
        
        gps_group = self.get_group("gps")
        if not gps_group:
            return
        
        if "latitude" in gps_data:
            gps_group.add_fact(Fact(
                "latitude", gps_data["latitude"], FactType.SENSOR_VALUE,
                unit="degrees", source="gps"
            ))
        
        if "longitude" in gps_data:
            gps_group.add_fact(Fact(
                "longitude", gps_data["longitude"], FactType.SENSOR_VALUE,
                unit="degrees", source="gps"
            ))
        
        if "altitude" in gps_data:
            gps_group.add_fact(Fact(
                "altitude", gps_data["altitude"], FactType.SENSOR_VALUE,
                unit="meters", source="gps"
            ))
        
        if "hdop" in gps_data:
            gps_group.add_fact(Fact(
                "hdop", gps_data["hdop"], FactType.SENSOR_VALUE,
                unit="", source="gps", min_value=0, max_value=50
            ))
        
        if "satellite_count" in gps_data:
            gps_group.add_fact(Fact(
                "satellite_count", gps_data["satellite_count"], FactType.SENSOR_VALUE,
                unit="count", source="gps", min_value=0, max_value=32
            ))
    
    def update_battery_facts(self, battery_data: Dict[str, Any]) -> None:
        """Update battery fact group from battery data"""
        
        battery_group = self.get_group("battery")
        if not battery_group:
            return
        
        if "voltage" in battery_data:
            battery_group.add_fact(Fact(
                "voltage", battery_data["voltage"], FactType.SENSOR_VALUE,
                unit="volts", source="battery", min_value=0, max_value=30
            ))
        
        if "current" in battery_data:
            battery_group.add_fact(Fact(
                "current", battery_data["current"], FactType.SENSOR_VALUE,
                unit="amps", source="battery"
            ))
        
        if "consumed" in battery_data:
            battery_group.add_fact(Fact(
                "consumed", battery_data["consumed"], FactType.SENSOR_VALUE,
                unit="mah", source="battery", min_value=0
            ))
        
        if "percent_remaining" in battery_data:
            battery_group.add_fact(Fact(
                "percent_remaining", battery_data["percent_remaining"], FactType.SENSOR_VALUE,
                unit="percent", source="battery", min_value=0, max_value=100
            ))
    
    def update_attitude_facts(self, attitude_data: Dict[str, Any]) -> None:
        """Update attitude fact group from IMU data"""
        
        attitude_group = self.get_group("attitude")
        if not attitude_group:
            return
        
        if "roll" in attitude_data:
            attitude_group.add_fact(Fact(
                "roll", attitude_data["roll"], FactType.SENSOR_VALUE,
                unit="degrees", source="imu", min_value=-180, max_value=180
            ))
        
        if "pitch" in attitude_data:
            attitude_group.add_fact(Fact(
                "pitch", attitude_data["pitch"], FactType.SENSOR_VALUE,
                unit="degrees", source="imu", min_value=-90, max_value=90
            ))
        
        if "yaw" in attitude_data:
            attitude_group.add_fact(Fact(
                "yaw", attitude_data["yaw"], FactType.SENSOR_VALUE,
                unit="degrees", source="imu", min_value=-180, max_value=180
            ))
    
    def get_summary_dict(self) -> Dict[str, Any]:
        """Get a summary of all fact groups"""
        
        summary = {
            "total_groups": len(self.fact_groups),
            "groups": {}
        }
        
        for group_name, group in self.fact_groups.items():
            summary["groups"][group_name] = {
                "fact_count": len(group.facts),
                "last_updated": group.last_updated.isoformat(),
                "sample_facts": list(group.facts.keys())[:5]  # First 5 fact names
            }
        
        return summary
    
    def export_to_json(self) -> str:
        """Export all fact groups to JSON"""
        
        export_data = {
            "exported_at": datetime.now().isoformat(),
            "fact_groups": {name: group.to_dict() for name, group in self.fact_groups.items()}
        }
        
        return json.dumps(export_data, indent=2)
    
    def clear_old_facts(self, max_age_hours: int = 24) -> int:
        """Clear facts older than specified hours"""
        
        cutoff_time = datetime.now() - timedelta(hours=max_age_hours)
        removed_count = 0
        
        for group in self.fact_groups.values():
            facts_to_remove = []
            
            for fact_name, fact in group.facts.items():
                if fact.timestamp < cutoff_time:
                    facts_to_remove.append(fact_name)
            
            for fact_name in facts_to_remove:
                group.remove_fact(fact_name)
                removed_count += 1
        
        return removed_count