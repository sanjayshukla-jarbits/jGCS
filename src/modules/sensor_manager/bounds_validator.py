"""
Sensor Bounds Validator - Validate sensor readings against operational bounds

This module provides comprehensive validation of sensor data against:
- Operational bounds (min/max values)
- Rate of change limits
- Quality thresholds
- Cross-sensor validation
"""

from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass
from enum import Enum
import numpy as np

class ViolationType(Enum):
    """Types of bounds violations"""
    MIN_VALUE = "min_value"
    MAX_VALUE = "max_value"
    RATE_LIMIT = "rate_limit" 
    QUALITY_LOW = "quality_low"
    STALE_DATA = "stale_data"
    CROSS_VALIDATION = "cross_validation"

class SeverityLevel(Enum):
    """Severity levels for violations"""
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"
    EMERGENCY = "emergency"

@dataclass
class BoundsViolation:
    """Represents a bounds violation"""
    sensor_id: str
    violation_type: ViolationType
    severity: SeverityLevel
    current_value: Any
    expected_range: Optional[Tuple[float, float]]
    message: str
    timestamp: datetime
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "sensor_id": self.sensor_id,
            "violation_type": self.violation_type.value,
            "severity": self.severity.value,
            "current_value": self.current_value,
            "expected_range": self.expected_range,
            "message": self.message,
            "timestamp": self.timestamp.isoformat()
        }

@dataclass
class SensorBounds:
    """Sensor operational bounds configuration"""
    sensor_id: str
    min_value: Optional[float] = None
    max_value: Optional[float] = None
    warning_min: Optional[float] = None
    warning_max: Optional[float] = None
    max_rate_change: Optional[float] = None  # Per second
    min_quality: float = 0.5
    max_age_seconds: int = 60
    cross_validation_sensors: List[str] = None
    
    def __post_init__(self):
        if self.cross_validation_sensors is None:
            self.cross_validation_sensors = []

class BoundsValidator:
    """Validates sensor data against operational bounds"""
    
    def __init__(self):
        self.sensor_bounds: Dict[str, SensorBounds] = {}
        self.sensor_history: Dict[str, List[Dict[str, Any]]] = {}
        self.max_history_size = 100  # Keep last 100 readings per sensor
    
    def configure_sensor_bounds(self, sensor_id: str, bounds: SensorBounds) -> None:
        """Configure bounds for a specific sensor"""
        self.sensor_bounds[sensor_id] = bounds
    
    def configure_from_sensor_data(self, sensors: List[Dict[str, Any]]) -> None:
        """Configure bounds from sensor configuration data"""
        
        for sensor in sensors:
            sensor_id = sensor.get("sensor_id")
            if not sensor_id:
                continue
            
            operational_bounds = sensor.get("operational_bounds", {})
            safety_limits = sensor.get("safety_limits", {})
            
            bounds = SensorBounds(
                sensor_id=sensor_id,
                min_value=operational_bounds.get("min_value"),
                max_value=operational_bounds.get("max_value"),
                warning_min=safety_limits.get("warning_min"),
                warning_max=safety_limits.get("warning_max"),
                max_rate_change=safety_limits.get("max_rate_change"),
                min_quality=safety_limits.get("min_quality", 0.5),
                max_age_seconds=safety_limits.get("max_age_seconds", 60)
            )
            
            self.configure_sensor_bounds(sensor_id, bounds)
    
    def validate_sensor_reading(self, sensor_id: str, value: float, 
                               quality: float = 1.0, timestamp: datetime = None) -> List[BoundsViolation]:
        """Validate a single sensor reading"""
        
        if timestamp is None:
            timestamp = datetime.now()
        
        violations = []
        bounds = self.sensor_bounds.get(sensor_id)
        
        if not bounds:
            # No bounds configured, just store the reading
            self._store_reading(sensor_id, value, quality, timestamp)
            return violations
        
        # Check basic min/max bounds
        if bounds.min_value is not None and value < bounds.min_value:
            violations.append(BoundsViolation(
                sensor_id=sensor_id,
                violation_type=ViolationType.MIN_VALUE,
                severity=SeverityLevel.CRITICAL,
                current_value=value,
                expected_range=(bounds.min_value, bounds.max_value),
                message=f"Value {value} below minimum {bounds.min_value}",
                timestamp=timestamp
            ))
        
        if bounds.max_value is not None and value > bounds.max_value:
            violations.append(BoundsViolation(
                sensor_id=sensor_id,
                violation_type=ViolationType.MAX_VALUE,
                severity=SeverityLevel.CRITICAL,
                current_value=value,
                expected_range=(bounds.min_value, bounds.max_value),
                message=f"Value {value} above maximum {bounds.max_value}",
                timestamp=timestamp
            ))
        
        # Check warning thresholds
        if bounds.warning_min is not None and value < bounds.warning_min:
            violations.append(BoundsViolation(
                sensor_id=sensor_id,
                violation_type=ViolationType.MIN_VALUE,
                severity=SeverityLevel.WARNING,
                current_value=value,
                expected_range=(bounds.warning_min, bounds.warning_max),
                message=f"Value {value} approaching minimum threshold {bounds.warning_min}",
                timestamp=timestamp
            ))
        
        if bounds.warning_max is not None and value > bounds.warning_max:
            violations.append(BoundsViolation(
                sensor_id=sensor_id,
                violation_type=ViolationType.MAX_VALUE,
                severity=SeverityLevel.WARNING,
                current_value=value,
                expected_range=(bounds.warning_min, bounds.warning_max),
                message=f"Value {value} approaching maximum threshold {bounds.warning_max}",
                timestamp=timestamp
            ))
        
        # Check quality
        if quality < bounds.min_quality:
            violations.append(BoundsViolation(
                sensor_id=sensor_id,
                violation_type=ViolationType.QUALITY_LOW,
                severity=SeverityLevel.WARNING,
                current_value=quality,
                expected_range=(bounds.min_quality, 1.0),
                message=f"Sensor quality {quality} below minimum {bounds.min_quality}",
                timestamp=timestamp
            ))
        
        # Check data age
        age_seconds = (datetime.now() - timestamp).total_seconds()
        if age_seconds > bounds.max_age_seconds:
            violations.append(BoundsViolation(
                sensor_id=sensor_id,
                violation_type=ViolationType.STALE_DATA,
                severity=SeverityLevel.WARNING,
                current_value=age_seconds,
                expected_range=(0, bounds.max_age_seconds),
                message=f"Data age {age_seconds}s exceeds maximum {bounds.max_age_seconds}s",
                timestamp=timestamp
            ))
        
        # Check rate of change
        rate_violations = self._check_rate_of_change(sensor_id, value, timestamp, bounds)
        violations.extend(rate_violations)
        
        # Store the reading for future rate calculations
        self._store_reading(sensor_id, value, quality, timestamp)
        
        return violations
    
    def validate_sensor_batch(self, sensor_readings: Dict[str, Dict[str, Any]]) -> Dict[str, List[BoundsViolation]]:
        """Validate a batch of sensor readings"""
        
        all_violations = {}
        
        for sensor_id, reading in sensor_readings.items():
            value = reading.get("value")
            quality = reading.get("quality", 1.0)
            timestamp = reading.get("timestamp")
            
            if timestamp and isinstance(timestamp, str):
                timestamp = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
            elif timestamp is None:
                timestamp = datetime.now()
            
            if value is not None:
                violations = self.validate_sensor_reading(sensor_id, value, quality, timestamp)
                if violations:
                    all_violations[sensor_id] = violations
        
        # Perform cross-sensor validation
        cross_violations = self._perform_cross_validation(sensor_readings)
        
        # Merge cross-validation violations
        for sensor_id, violations in cross_violations.items():
            if sensor_id in all_violations:
                all_violations[sensor_id].extend(violations)
            else:
                all_violations[sensor_id] = violations
        
        return all_violations
    
    def _check_rate_of_change(self, sensor_id: str, current_value: float, 
                             timestamp: datetime, bounds: SensorBounds) -> List[BoundsViolation]:
        """Check if rate of change exceeds limits"""
        
        violations = []
        
        if bounds.max_rate_change is None:
            return violations
        
        history = self.sensor_history.get(sensor_id, [])
        if not history:
            return violations
        
        # Get the most recent reading
        last_reading = history[-1]
        last_value = last_reading["value"]
        last_timestamp = last_reading["timestamp"]
        
        time_diff = (timestamp - last_timestamp).total_seconds()
        if time_diff <= 0:
            return violations
        
        rate = abs(current_value - last_value) / time_diff
        
        if rate > bounds.max_rate_change:
            violations.append(BoundsViolation(
                sensor_id=sensor_id,
                violation_type=ViolationType.RATE_LIMIT,
                severity=SeverityLevel.WARNING,
                current_value=rate,
                expected_range=(0, bounds.max_rate_change),
                message=f"Rate of change {rate:.2f}/s exceeds limit {bounds.max_rate_change}/s",
                timestamp=timestamp
            ))
        
        return violations
    
    def _perform_cross_validation(self, sensor_readings: Dict[str, Dict[str, Any]]) -> Dict[str, List[BoundsViolation]]:
        """Perform cross-sensor validation"""
        
        violations = {}
        
        for sensor_id, bounds in self.sensor_bounds.items():
            if not bounds.cross_validation_sensors:
                continue
            
            primary_reading = sensor_readings.get(sensor_id)
            if not primary_reading:
                continue
            
            primary_value = primary_reading.get("value")
            if primary_value is None:
                continue
            
            # Compare with cross-validation sensors
            for cross_sensor_id in bounds.cross_validation_sensors:
                cross_reading = sensor_readings.get(cross_sensor_id)
                if not cross_reading:
                    continue
                
                cross_value = cross_reading.get("value")
                if cross_value is None:
                    continue
                
                # Check if values are reasonably consistent
                # This is a basic implementation - real cross-validation would be domain-specific
                if abs(primary_value - cross_value) > abs(primary_value * 0.2):  # 20% tolerance
                    violation = BoundsViolation(
                        sensor_id=sensor_id,
                        violation_type=ViolationType.CROSS_VALIDATION,
                        severity=SeverityLevel.WARNING,
                        current_value=primary_value,
                        expected_range=None,
                        message=f"Cross-validation failed with {cross_sensor_id}: {primary_value} vs {cross_value}",
                        timestamp=primary_reading.get("timestamp", datetime.now())
                    )
                    
                    if sensor_id not in violations:
                        violations[sensor_id] = []
                    violations[sensor_id].append(violation)
        
        return violations
    
    def _store_reading(self, sensor_id: str, value: float, quality: float, timestamp: datetime) -> None:
        """Store sensor reading for historical analysis"""
        
        if sensor_id not in self.sensor_history:
            self.sensor_history[sensor_id] = []
        
        reading = {
            "value": value,
            "quality": quality,
            "timestamp": timestamp
        }
        
        self.sensor_history[sensor_id].append(reading)
        
        # Limit history size
        if len(self.sensor_history[sensor_id]) > self.max_history_size:
            self.sensor_history[sensor_id] = self.sensor_history[sensor_id][-self.max_history_size:]
    
    def get_sensor_statistics(self, sensor_id: str) -> Dict[str, Any]:
        """Get statistical information about a sensor"""
        
        history = self.sensor_history.get(sensor_id, [])
        if not history:
            return {"sensor_id": sensor_id, "readings_count": 0}
        
        values = [r["value"] for r in history]
        qualities = [r["quality"] for r in history]
        
        stats = {
            "sensor_id": sensor_id,
            "readings_count": len(history),
            "value_stats": {
                "min": min(values),
                "max": max(values),
                "mean": np.mean(values),
                "std": np.std(values)
            },
            "quality_stats": {
                "min": min(qualities),
                "max": max(qualities),
                "mean": np.mean(qualities)
            },
            "last_reading": history[-1]["timestamp"].isoformat(),
            "first_reading": history[0]["timestamp"].isoformat()
        }
        
        return stats
    
    def clear_sensor_history(self, sensor_id: str = None) -> None:
        """Clear sensor history for one or all sensors"""
        
        if sensor_id:
            self.sensor_history.pop(sensor_id, None)
        else:
            self.sensor_history.clear()