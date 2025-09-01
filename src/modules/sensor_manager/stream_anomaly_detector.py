"""
Stream-based Anomaly Detection - Real-time anomaly detection for sensor streams

This module provides:
- Statistical anomaly detection (Z-score, IQR)
- Machine learning-based anomaly detection
- Pattern-based anomaly detection
- Multi-sensor correlation analysis
"""

import asyncio
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass
from collections import deque
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
import json

from src.utils.logger import get_logger

@dataclass
class AnomalyDetection:
    """Anomaly detection result"""
    sensor_id: str
    timestamp: datetime
    value: float
    anomaly_score: float
    anomaly_type: str
    confidence: float
    description: str
    severity: str  # "low", "medium", "high"

class StatisticalDetector:
    """Statistical anomaly detection using Z-score and IQR methods"""
    
    def __init__(self, window_size: int = 100, z_threshold: float = 3.0):
        self.window_size = window_size
        self.z_threshold = z_threshold
        self.sensor_windows: Dict[str, deque] = {}
        
    def add_value(self, sensor_id: str, value: float) -> Optional[AnomalyDetection]:
        """Add a new value and check for anomalies"""
        
        if sensor_id not in self.sensor_windows:
            self.sensor_windows[sensor_id] = deque(maxlen=self.window_size)
        
        window = self.sensor_windows[sensor_id]
        window.append(value)
        
        # Need at least 10 values for statistical analysis
        if len(window) < 10:
            return None
        
        values = np.array(window)
        
        # Z-score anomaly detection
        mean = np.mean(values[:-1])  # Exclude current value from baseline
        std = np.std(values[:-1])
        
        if std == 0:  # No variation in data
            return None
        
        z_score = abs((value - mean) / std)
        
        if z_score > self.z_threshold:
            return AnomalyDetection(
                sensor_id=sensor_id,
                timestamp=datetime.now(),
                value=value,
                anomaly_score=z_score,
                anomaly_type="statistical_outlier",
                confidence=min(z_score / self.z_threshold, 1.0),
                description=f"Z-score {z_score:.2f} exceeds threshold {self.z_threshold}",
                severity="high" if z_score > self.z_threshold * 1.5 else "medium"
            )
        
        return None

class PatternDetector:
    """Pattern-based anomaly detection"""
    
    def __init__(self, pattern_length: int = 10):
        self.pattern_length = pattern_length
        self.sensor_patterns: Dict[str, List[List[float]]] = {}
        
    def add_sequence(self, sensor_id: str, sequence: List[float]) -> Optional[AnomalyDetection]:
        """Add a sequence and check for pattern anomalies"""
        
        if len(sequence) < self.pattern_length:
            return None
        
        if sensor_id not in self.sensor_patterns:
            self.sensor_patterns[sensor_id] = []
        
        patterns = self.sensor_patterns[sensor_id]
        current_pattern = sequence[-self.pattern_length:]
        
        if len(patterns) < 5:  # Need some patterns to compare against
            patterns.append(current_pattern)
            return None
        
        # Calculate similarity with existing patterns
        similarities = []
        for pattern in patterns:
            similarity = self._calculate_pattern_similarity(current_pattern, pattern)
            similarities.append(similarity)
        
        avg_similarity = np.mean(similarities)
        
        # If current pattern is very different from all previous patterns
        if avg_similarity < 0.3:  # Threshold for pattern anomaly
            patterns.append(current_pattern)
            # Keep only recent patterns
            if len(patterns) > 20:
                patterns.pop(0)
            
            return AnomalyDetection(
                sensor_id=sensor_id,
                timestamp=datetime.now(),
                value=sequence[-1],
                anomaly_score=1 - avg_similarity,
                anomaly_type="pattern_anomaly",
                confidence=1 - avg_similarity,
                description=f"Pattern similarity {avg_similarity:.2f} below threshold",
                severity="medium" if avg_similarity < 0.2 else "low"
            )
        
        patterns.append(current_pattern)
        if len(patterns) > 20:
            patterns.pop(0)
        
        return None
    
    def _calculate_pattern_similarity(self, pattern1: List[float], pattern2: List[float]) -> float:
        """Calculate similarity between two patterns"""
        
        if len(pattern1) != len(pattern2):
            return 0.0
        
        # Normalize patterns
        p1 = np.array(pattern1)
        p2 = np.array(pattern2)
        
        # Calculate correlation coefficient
        if np.std(p1) == 0 or np.std(p2) == 0:
            return 1.0 if np.allclose(p1, p2) else 0.0
        
        correlation = np.corrcoef(p1, p2)[0, 1]
        return max(0, correlation)  # Return positive correlation only

class MLAnomalyDetector:
    """Machine learning-based anomaly detection using Isolation Forest"""
    
    def __init__(self, contamination: float = 0.1, window_size: int = 200):
        self.contamination = contamination
        self.window_size = window_size
        self.models: Dict[str, IsolationForest] = {}
        self.scalers: Dict[str, StandardScaler] = {}
        self.sensor_data: Dict[str, deque] = {}
        self.training_interval = 50  # Retrain every N samples
        self.sample_count: Dict[str, int] = {}
        
    def add_value(self, sensor_id: str, value: float, features: Dict[str, float] = None) -> Optional[AnomalyDetection]:
        """Add value with optional additional features"""
        
        if sensor_id not in self.sensor_data:
            self.sensor_data[sensor_id] = deque(maxlen=self.window_size)
            self.sample_count[sensor_id] = 0
        
        # Create feature vector
        feature_vector = [value]
        if features:
            feature_vector.extend(features.values())
        
        self.sensor_data[sensor_id].append(feature_vector)
        self.sample_count[sensor_id] += 1
        
        # Need sufficient data for training
        if len(self.sensor_data[sensor_id]) < 50:
            return None
        
        # Retrain model periodically
        if (self.sample_count[sensor_id] % self.training_interval == 0 or 
            sensor_id not in self.models):
            self._train_model(sensor_id)
        
        # Predict anomaly
        if sensor_id in self.models:
            return self._predict_anomaly(sensor_id, feature_vector)
        
        return None
    
    def _train_model(self, sensor_id: str) -> None:
        """Train isolation forest model for sensor"""
        
        data = np.array(list(self.sensor_data[sensor_id]))
        
        # Initialize scaler if needed
        if sensor_id not in self.scalers:
            self.scalers[sensor_id] = StandardScaler()
        
        # Scale features
        scaled_data = self.scalers[sensor_id].fit_transform(data)
        
        # Train isolation forest
        model = IsolationForest(
            contamination=self.contamination,
            random_state=42,
            n_estimators=100
        )
        model.fit(scaled_data)
        
        self.models[sensor_id] = model
    
    def _predict_anomaly(self, sensor_id: str, feature_vector: List[float]) -> Optional[AnomalyDetection]:
        """Predict if current value is anomalous"""
        
        model = self.models[sensor_id]
        scaler = self.scalers[sensor_id]
        
        # Scale features
        scaled_features = scaler.transform([feature_vector])
        
        # Predict
        prediction = model.predict(scaled_features)[0]
        anomaly_score = model.decision_function(scaled_features)[0]
        
        # Convert decision function output to 0-1 scale
        normalized_score = 1 / (1 + np.exp(anomaly_score))  # Sigmoid transformation
        
        if prediction == -1:  # Anomaly detected
            return AnomalyDetection(
                sensor_id=sensor_id,
                timestamp=datetime.now(),
                value=feature_vector[0],
                anomaly_score=normalized_score,
                anomaly_type="ml_anomaly",
                confidence=normalized_score,
                description=f"ML model detected anomaly (score: {normalized_score:.3f})",
                severity="high" if normalized_score > 0.8 else "medium"
            )
        
        return None

class StreamAnomalyDetector:
    """Main stream anomaly detection coordinator"""
    
    def __init__(self):
        self.logger = get_logger(__name__)
        
        # Initialize detectors
        self.statistical_detector = StatisticalDetector()
        self.pattern_detector = PatternDetector()
        self.ml_detector = MLAnomalyDetector()
        
        # Multi-sensor correlation
        self.correlation_groups: Dict[str, List[str]] = {}
        self.correlation_data: Dict[str, deque] = {}
        
        # Anomaly history
        self.anomaly_history: List[AnomalyDetection] = []
        self.max_history = 1000
        
    def configure_correlation_groups(self, groups: Dict[str, List[str]]) -> None:
        """Configure sensor groups for correlation analysis"""
        self.correlation_groups = groups
        
        for group_name, sensor_ids in groups.items():
            if group_name not in self.correlation_data:
                self.correlation_data[group_name] = deque(maxlen=100)
    
    async def process_sensor_value(self, sensor_id: str, value: float, 
                                  timestamp: datetime = None,
                                  additional_features: Dict[str, float] = None) -> List[AnomalyDetection]:
        """Process a sensor value through all anomaly detectors"""
        
        if timestamp is None:
            timestamp = datetime.now()
        
        anomalies = []
        
        try:
            # Statistical detection
            stat_anomaly = self.statistical_detector.add_value(sensor_id, value)
            if stat_anomaly:
                anomalies.append(stat_anomaly)
            
            # ML-based detection
            ml_anomaly = self.ml_detector.add_value(sensor_id, value, additional_features)
            if ml_anomaly:
                anomalies.append(ml_anomaly)
            
            # Pattern detection (needs sequence)
            pattern_anomaly = await self._check_pattern_anomaly(sensor_id, value)
            if pattern_anomaly:
                anomalies.append(pattern_anomaly)
            
            # Correlation analysis
            correlation_anomalies = await self._check_correlation_anomalies(sensor_id, value)
            anomalies.extend(correlation_anomalies)
            
            # Store anomalies in history
            for anomaly in anomalies:
                self.anomaly_history.append(anomaly)
                if len(self.anomaly_history) > self.max_history:
                    self.anomaly_history.pop(0)
            
            if anomalies:
                self.logger.warning(f"Detected {len(anomalies)} anomalies for sensor {sensor_id}")
            
        except Exception as e:
            self.logger.error(f"Error in anomaly detection for {sensor_id}: {e}")
        
        return anomalies
    
    async def _check_pattern_anomaly(self, sensor_id: str, value: float) -> Optional[AnomalyDetection]:
        """Check for pattern anomalies using recent sequence"""
        
        # Get recent values for pattern analysis
        if hasattr(self.statistical_detector, 'sensor_windows'):
            window = self.statistical_detector.sensor_windows.get(sensor_id)
            if window and len(window) >= self.pattern_detector.pattern_length:
                sequence = list(window)
                return self.pattern_detector.add_sequence(sensor_id, sequence)
        
        return None
    
    async def _check_correlation_anomalies(self, sensor_id: str, value: float) -> List[AnomalyDetection]:
        """Check for correlation-based anomalies"""
        
        anomalies = []
        
        # Find correlation groups containing this sensor
        for group_name, sensor_ids in self.correlation_groups.items():
            if sensor_id in sensor_ids:
                anomaly = await self._analyze_sensor_correlation(group_name, sensor_id, value)
                if anomaly:
                    anomalies.append(anomaly)
        
        return anomalies
    
    async def _analyze_sensor_correlation(self, group_name: str, sensor_id: str, value: float) -> Optional[AnomalyDetection]:
        """Analyze correlation within a sensor group"""
        
        # This is a simplified correlation analysis
        # In practice, this would be more sophisticated
        
        if group_name not in self.correlation_data:
            return None
        
        group_data = self.correlation_data[group_name]
        
        # Store current readings for all sensors in group
        current_readings = {sensor_id: value}
        
        # Get recent readings from other sensors (simplified)
        # In a real implementation, this would maintain a proper correlation matrix
        
        if len(group_data) > 10:
            # Simple correlation check: if this sensor deviates significantly
            # from its historical relationship with the group
            recent_values = [reading.get(sensor_id, 0) for reading in list(group_data)[-10:]]
            if recent_values:
                mean_recent = np.mean(recent_values)
                std_recent = np.std(recent_values)
                
                if std_recent > 0:
                    deviation = abs(value - mean_recent) / std_recent
                    
                    if deviation > 2.5:  # Correlation anomaly threshold
                        return AnomalyDetection(
                            sensor_id=sensor_id,
                            timestamp=datetime.now(),
                            value=value,
                            anomaly_score=deviation / 2.5,
                            anomaly_type="correlation_anomaly",
                            confidence=min(deviation / 2.5, 1.0),
                            description=f"Sensor correlation deviation: {deviation:.2f} std devs",
                            severity="medium"
                        )
        
        # Store current reading
        group_data.append(current_readings)
        
        return None
    
    def get_anomaly_summary(self, hours: int = 24) -> Dict[str, Any]:
        """Get summary of anomalies in the specified time window"""
        
        cutoff_time = datetime.now() - timedelta(hours=hours)
        recent_anomalies = [
            a for a in self.anomaly_history 
            if a.timestamp > cutoff_time
        ]
        
        # Group by sensor
        sensor_anomalies = {}
        for anomaly in recent_anomalies:
            if anomaly.sensor_id not in sensor_anomalies:
                sensor_anomalies[anomaly.sensor_id] = []
            sensor_anomalies[anomaly.sensor_id].append(anomaly)
        
        # Calculate statistics
        total_anomalies = len(recent_anomalies)
        anomaly_types = {}
        severity_counts = {"low": 0, "medium": 0, "high": 0}
        
        for anomaly in recent_anomalies:
            # Count by type
            if anomaly.anomaly_type not in anomaly_types:
                anomaly_types[anomaly.anomaly_type] = 0
            anomaly_types[anomaly.anomaly_type] += 1
            
            # Count by severity
            severity_counts[anomaly.severity] += 1
        
        return {
            "time_window_hours": hours,
            "total_anomalies": total_anomalies,
            "affected_sensors": len(sensor_anomalies),
            "anomaly_types": anomaly_types,
            "severity_distribution": severity_counts,
            "sensor_breakdown": {
                sensor_id: len(anomalies) 
                for sensor_id, anomalies in sensor_anomalies.items()
            }
        }
    
    def get_sensor_anomalies(self, sensor_id: str, hours: int = 24) -> List[Dict[str, Any]]:
        """Get anomalies for a specific sensor"""
        
        cutoff_time = datetime.now() - timedelta(hours=hours)
        sensor_anomalies = [
            {
                "timestamp": a.timestamp.isoformat(),
                "value": a.value,
                "anomaly_score": a.anomaly_score,
                "anomaly_type": a.anomaly_type,
                "confidence": a.confidence,
                "description": a.description,
                "severity": a.severity
            }
            for a in self.anomaly_history 
            if a.sensor_id == sensor_id and a.timestamp > cutoff_time
        ]
        
        return sorted(sensor_anomalies, key=lambda x: x["timestamp"], reverse=True)