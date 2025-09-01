"""
Multi-sensor Data Fusion Stream - Real-time sensor data fusion and estimation

This module provides:
- Kalman filter-based sensor fusion
- Weighted averaging with quality metrics
- Time-synchronized data fusion
- Uncertainty propagation
"""

import asyncio
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from collections import defaultdict, deque
from filterpy.kalman import KalmanFilter
from filterpy.common import Q_discrete_white_noise
import json

from src.utils.logger import get_logger

@dataclass
class SensorReading:
    """Individual sensor reading with metadata"""
    sensor_id: str
    value: float
    quality: float
    uncertainty: float
    timestamp: datetime
    sensor_type: str
    unit: str = ""
    
@dataclass
class FusedEstimate:
    """Fused sensor estimate result"""
    parameter: str
    fused_value: float
    uncertainty: float
    quality: float
    contributing_sensors: List[str]
    fusion_method: str
    timestamp: datetime
    confidence_interval: Tuple[float, float] = field(default_factory=lambda: (0.0, 0.0))

class KalmanFusionFilter:
    """Kalman filter for sensor fusion"""
    
    def __init__(self, dim_state: int = 2, dim_measurement: int = 1):
        self.kf = KalmanFilter(dim_x=dim_state, dim_z=dim_measurement)
        self._initialize_filter()
        self.last_update = None
        
    def _initialize_filter(self):
        """Initialize Kalman filter matrices"""
        # State: [value, rate_of_change]
        self.kf.x = np.array([0., 0.])  # Initial state
        
        # State transition matrix (constant velocity model)
        self.kf.F = np.array([[1., 1.],
                              [0., 1.]])
        
        # Measurement matrix (we measure the value directly)
        self.kf.H = np.array([[1., 0.]])
        
        # Measurement noise
        self.kf.R *= 0.1  # Measurement uncertainty
        
        # Process noise
        self.kf.Q = Q_discrete_white_noise(2, dt=1.0, var=0.01)
        
        # Initial covariance
        self.kf.P *= 1000
    
    def update(self, measurement: float, measurement_noise: float, dt: float = 1.0):
        """Update filter with new measurement"""
        
        # Update time step
        self.kf.F[0, 1] = dt
        self.kf.Q = Q_discrete_white_noise(2, dt=dt, var=0.01)
        
        # Update measurement noise based on sensor quality
        self.kf.R[0, 0] = measurement_noise
        
        # Predict and update
        self.kf.predict()
        self.kf.update(measurement)
        
        return self.kf.x[0], np.sqrt(self.kf.P[0, 0])  # Value and uncertainty

class WeightedFusion:
    """Weighted averaging fusion based on sensor quality and uncertainty"""
    
    def __init__(self):
        self.sensor_weights: Dict[str, float] = {}
        
    def calculate_weights(self, readings: List[SensorReading]) -> Dict[str, float]:
        """Calculate fusion weights based on quality and uncertainty"""
        
        weights = {}
        total_weight = 0
        
        for reading in readings:
            # Weight based on quality and inverse uncertainty
            uncertainty_weight = 1.0 / (reading.uncertainty + 1e-6)  # Avoid division by zero
            quality_weight = reading.quality
            
            # Combined weight
            weight = uncertainty_weight * quality_weight
            weights[reading.sensor_id] = weight
            total_weight += weight
        
        # Normalize weights
        if total_weight > 0:
            for sensor_id in weights:
                weights[sensor_id] /= total_weight
        
        return weights
    
    def fuse_readings(self, readings: List[SensorReading]) -> FusedEstimate:
        """Fuse multiple sensor readings using weighted average"""
        
        if not readings:
            raise ValueError("No readings provided for fusion")
        
        weights = self.calculate_weights(readings)
        
        # Weighted average
        fused_value = 0
        total_weight = 0
        uncertainty_sum = 0
        quality_sum = 0
        
        for reading in readings:
            weight = weights[reading.sensor_id]
            fused_value += weight * reading.value
            uncertainty_sum += weight * (reading.uncertainty ** 2)
            quality_sum += weight * reading.quality
            total_weight += weight
        
        # Calculate fused uncertainty (weighted RMS)
        fused_uncertainty = np.sqrt(uncertainty_sum)
        
        # Calculate confidence interval (95%)
        confidence_interval = (
            fused_value - 1.96 * fused_uncertainty,
            fused_value + 1.96 * fused_uncertainty
        )
        
        return FusedEstimate(
            parameter=readings[0].sensor_type,
            fused_value=fused_value,
            uncertainty=fused_uncertainty,
            quality=quality_sum,
            contributing_sensors=[r.sensor_id for r in readings],
            fusion_method="weighted_average",
            timestamp=max(r.timestamp for r in readings),
            confidence_interval=confidence_interval
        )

class TimeSynchronizer:
    """Time synchronization for sensor data"""
    
    def __init__(self, sync_window: timedelta = timedelta(seconds=1)):
        self.sync_window = sync_window
        self.reading_buffers: Dict[str, deque] = defaultdict(lambda: deque(maxlen=100))
        
    def add_reading(self, reading: SensorReading) -> None:
        """Add reading to buffer"""
        parameter_key = f"{reading.sensor_type}_{reading.unit}"
        self.reading_buffers[parameter_key].append(reading)
    
    def get_synchronized_readings(self, parameter: str, reference_time: datetime) -> List[SensorReading]:
        """Get readings synchronized around reference time"""
        
        if parameter not in self.reading_buffers:
            return []
        
        synchronized = []
        
        for reading in self.reading_buffers[parameter]:
            time_diff = abs((reading.timestamp - reference_time).total_seconds())
            
            if time_diff <= self.sync_window.total_seconds():
                synchronized.append(reading)
        
        return synchronized
    
    def cleanup_old_readings(self, max_age: timedelta = timedelta(minutes=10)) -> None:
        """Remove old readings from buffers"""
        
        cutoff_time = datetime.now() - max_age
        
        for parameter, buffer in self.reading_buffers.items():
            # Filter out old readings
            valid_readings = deque([r for r in buffer if r.timestamp > cutoff_time], maxlen=100)
            self.reading_buffers[parameter] = valid_readings

class SensorFusionStream:
    """Main sensor fusion stream processor"""
    
    def __init__(self):
        self.logger = get_logger(__name__)
        
        # Fusion components
        self.time_synchronizer = TimeSynchronizer()
        self.weighted_fusion = WeightedFusion()
        self.kalman_filters: Dict[str, KalmanFusionFilter] = {}
        
        # Configuration
        self.fusion_parameters = {
            "position": ["gps", "ins", "vision"],
            "altitude": ["gps", "barometer", "rangefinder"],
            "attitude": ["imu", "magnetometer", "gps"],
            "velocity": ["gps", "ins", "airspeed"],
            "temperature": ["temp_sensor_1", "temp_sensor_2"],
            "battery_voltage": ["battery_monitor_1", "battery_monitor_2"]
        }
        
        # Fusion results
        self.latest_estimates: Dict[str, FusedEstimate] = {}
        self.fusion_history: Dict[str, deque] = defaultdict(lambda: deque(maxlen=1000))
        
        # Quality tracking
        self.sensor_performance: Dict[str, Dict[str, float]] = defaultdict(dict)
        
    def configure_fusion_parameters(self, config: Dict[str, List[str]]) -> None:
        """Configure which sensors contribute to which parameters"""
        self.fusion_parameters = config
        self.logger.info(f"Configured fusion for {len(config)} parameters")
    
    async def process_sensor_reading(self, sensor_id: str, value: float,
                                   quality: float = 1.0, uncertainty: float = 0.1,
                                   sensor_type: str = "unknown", unit: str = "",
                                   timestamp: datetime = None) -> List[FusedEstimate]:
        """Process incoming sensor reading and perform fusion"""
        
        if timestamp is None:
            timestamp = datetime.now()
        
        reading = SensorReading(
            sensor_id=sensor_id,
            value=value,
            quality=quality,
            uncertainty=uncertainty,
            timestamp=timestamp,
            sensor_type=sensor_type,
            unit=unit
        )
        
        # Add to time synchronizer
        self.time_synchronizer.add_reading(reading)
        
        # Update sensor performance tracking
        self._update_sensor_performance(reading)
        
        # Find fusion opportunities
        fusion_results = []
        
        for parameter, contributing_sensors in self.fusion_parameters.items():
            if sensor_id in contributing_sensors or sensor_type in contributing_sensors:
                result = await self._attempt_fusion(parameter, timestamp)
                if result:
                    fusion_results.append(result)
        
        return fusion_results
    
    async def _attempt_fusion(self, parameter: str, reference_time: datetime) -> Optional[FusedEstimate]:
        """Attempt to fuse sensors for a specific parameter"""
        
        # Get synchronized readings
        sync_readings = self.time_synchronizer.get_synchronized_readings(parameter, reference_time)
        
        if len(sync_readings) < 2:  # Need at least 2 sensors for fusion
            return None
        
        try:
            # Choose fusion method based on availability and configuration
            if len(sync_readings) >= 3 and parameter in ["position", "altitude", "attitude"]:
                # Use Kalman filter for critical navigation parameters
                result = await self._kalman_fusion(parameter, sync_readings, reference_time)
            else:
                # Use weighted fusion for other parameters
                result = self.weighted_fusion.fuse_readings(sync_readings)
            
            # Store result
            if result:
                self.latest_estimates[parameter] = result
                self.fusion_history[parameter].append(result)
                
                self.logger.debug(
                    f"Fused {parameter}: {result.fused_value:.3f} ± {result.uncertainty:.3f} "
                    f"from {len(result.contributing_sensors)} sensors"
                )
            
            return result
            
        except Exception as e:
            self.logger.error(f"Fusion failed for {parameter}: {e}")
            return None
    
    async def _kalman_fusion(self, parameter: str, readings: List[SensorReading], 
                           reference_time: datetime) -> Optional[FusedEstimate]:
        """Perform Kalman filter-based fusion"""
        
        # Initialize filter if needed
        if parameter not in self.kalman_filters:
            self.kalman_filters[parameter] = KalmanFusionFilter()
        
        kf = self.kalman_filters[parameter]
        
        # Sort readings by timestamp
        readings.sort(key=lambda r: r.timestamp)
        
        # Process each reading through the filter
        fused_value = None
        fused_uncertainty = None
        
        last_time = kf.last_update or reference_time
        
        for reading in readings:
            dt = (reading.timestamp - last_time).total_seconds()
            dt = max(dt, 0.1)  # Minimum time step
            
            # Update filter
            fused_value, fused_uncertainty = kf.update(
                reading.value, 
                reading.uncertainty,
                dt
            )
            
            last_time = reading.timestamp
        
        kf.last_update = reference_time
        
        if fused_value is not None:
            # Calculate overall quality
            quality = np.mean([r.quality for r in readings])
            
            # Confidence interval
            confidence_interval = (
                fused_value - 1.96 * fused_uncertainty,
                fused_value + 1.96 * fused_uncertainty
            )
            
            return FusedEstimate(
                parameter=parameter,
                fused_value=float(fused_value),
                uncertainty=float(fused_uncertainty),
                quality=float(quality),
                contributing_sensors=[r.sensor_id for r in readings],
                fusion_method="kalman_filter",
                timestamp=reference_time,
                confidence_interval=confidence_interval
            )
        
        return None
    
    def _update_sensor_performance(self, reading: SensorReading) -> None:
        """Update sensor performance metrics"""
        
        sensor_id = reading.sensor_id
        
        if sensor_id not in self.sensor_performance:
            self.sensor_performance[sensor_id] = {
                "avg_quality": reading.quality,
                "avg_uncertainty": reading.uncertainty,
                "reading_count": 1,
                "last_update": reading.timestamp
            }
        else:
            perf = self.sensor_performance[sensor_id]
            count = perf["reading_count"]
            
            # Update running averages
            perf["avg_quality"] = (perf["avg_quality"] * count + reading.quality) / (count + 1)
            perf["avg_uncertainty"] = (perf["avg_uncertainty"] * count + reading.uncertainty) / (count + 1)
            perf["reading_count"] = count + 1
            perf["last_update"] = reading.timestamp
    
    def get_latest_estimate(self, parameter: str) -> Optional[Dict[str, Any]]:
        """Get latest fused estimate for a parameter"""
        
        estimate = self.latest_estimates.get(parameter)
        if estimate:
            return {
                "parameter": estimate.parameter,
                "value": estimate.fused_value,
                "uncertainty": estimate.uncertainty,
                "quality": estimate.quality,
                "contributing_sensors": estimate.contributing_sensors,
                "fusion_method": estimate.fusion_method,
                "timestamp": estimate.timestamp.isoformat(),
                "confidence_interval": estimate.confidence_interval
            }
        
        return None
    
    def get_all_estimates(self) -> Dict[str, Dict[str, Any]]:
        """Get all latest fused estimates"""
        
        return {
            parameter: self.get_latest_estimate(parameter)
            for parameter in self.latest_estimates.keys()
        }
    
    def get_sensor_performance_report(self) -> Dict[str, Any]:
        """Get sensor performance report"""
        
        report = {
            "total_sensors": len(self.sensor_performance),
            "sensors": {}
        }
        
        for sensor_id, perf in self.sensor_performance.items():
            age_minutes = (datetime.now() - perf["last_update"]).total_seconds() / 60
            
            report["sensors"][sensor_id] = {
                "average_quality": round(perf["avg_quality"], 3),
                "average_uncertainty": round(perf["avg_uncertainty"], 4),
                "total_readings": perf["reading_count"],
                "last_update_minutes_ago": round(age_minutes, 1),
                "status": "active" if age_minutes < 5 else "stale"
            }
        
        return report
    
    def get_fusion_history(self, parameter: str, hours: int = 1) -> List[Dict[str, Any]]:
        """Get fusion history for a parameter"""
        
        if parameter not in self.fusion_history:
            return []
        
        cutoff_time = datetime.now() - timedelta(hours=hours)
        
        history = []
        for estimate in self.fusion_history[parameter]:
            if estimate.timestamp > cutoff_time:
                history.append({
                    "timestamp": estimate.timestamp.isoformat(),
                    "value": estimate.fused_value,
                    "uncertainty": estimate.uncertainty,
                    "quality": estimate.quality,
                    "contributing_sensors": estimate.contributing_sensors,
                    "fusion_method": estimate.fusion_method
                })
        
        return sorted(history, key=lambda x: x["timestamp"])
    
    async def cleanup_old_data(self) -> None:
        """Clean up old data to prevent memory leaks"""
        
        # Clean up time synchronizer
        self.time_synchronizer.cleanup_old_readings()
        
        # Clean up old performance data
        cutoff_time = datetime.now() - timedelta(hours=24)
        
        sensors_to_remove = []
        for sensor_id, perf in self.sensor_performance.items():
            if perf["last_update"] < cutoff_time:
                sensors_to_remove.append(sensor_id)
        
        for sensor_id in sensors_to_remove:
            del self.sensor_performance[sensor_id]
        
        self.logger.debug(f"Cleaned up data for {len(sensors_to_remove)} stale sensors")
    
    async def start_periodic_cleanup(self, interval_minutes: int = 30) -> None:
        """Start periodic cleanup task"""
        
        while True:
            await asyncio.sleep(interval_minutes * 60)
            await self.cleanup_old_data()