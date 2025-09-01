"""
Kafka-based Sensor Monitoring - Real-time sensor data streaming and monitoring

This module provides:
- Kafka-based sensor data ingestion
- Real-time sensor health monitoring
- Sensor data stream processing
- Automated alerting for sensor anomalies
"""

import asyncio
import json
from typing import Dict, List, Any, Optional, Callable
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from kafka import KafkaProducer, KafkaConsumer
from kafka.errors import KafkaError
import logging

from src.modules.sensor_manager.bounds_validator import BoundsValidator, BoundsViolation
from src.modules.sensor_manager.fact_groups import FactGroupManager, Fact, FactType
from src.utils.logger import get_logger

@dataclass
class SensorHealthStatus:
    """Sensor health status information"""
    sensor_id: str
    status: str  # "healthy", "warning", "critical", "offline"
    last_reading: Optional[datetime]
    violation_count: int
    quality_score: float
    uptime_percent: float
    
class KafkaSensorMonitor:
    """Kafka-based real-time sensor monitoring system"""
    
    def __init__(self, kafka_config: Dict[str, Any]):
        self.kafka_config = kafka_config
        self.logger = get_logger(__name__)
        self.bounds_validator = BoundsValidator()
        self.fact_manager = FactGroupManager()
        
        # Sensor monitoring state
        self.sensor_health: Dict[str, SensorHealthStatus] = {}
        self.alert_callbacks: List[Callable] = []
        self.running = False
        
        # Kafka components
        self.producer: Optional[KafkaProducer] = None
        self.consumer: Optional[KafkaConsumer] = None
        
        # Topics
        self.sensor_data_topic = "uav-sensor-data"
        self.sensor_alerts_topic = "sensor-alerts"
        self.sensor_health_topic = "sensor-health"
        
    async def start_monitoring(self) -> None:
        """Start the Kafka sensor monitoring system"""
        
        try:
            # Initialize Kafka producer
            self.producer = KafkaProducer(
                bootstrap_servers=self.kafka_config.get("bootstrap_servers", ["localhost:9092"]),
                value_serializer=lambda v: json.dumps(v, default=str).encode('utf-8'),
                key_serializer=lambda k: k.encode('utf-8') if k else None,
                acks='all',
                retries=3,
                retry_backoff_ms=1000
            )
            
            # Initialize Kafka consumer
            self.consumer = KafkaConsumer(
                self.sensor_data_topic,
                bootstrap_servers=self.kafka_config.get("bootstrap_servers", ["localhost:9092"]),
                value_deserializer=lambda m: json.loads(m.decode('utf-8')),
                key_deserializer=lambda k: k.decode('utf-8') if k else None,
                group_id='sensor-monitor-group',
                auto_offset_reset='latest',
                enable_auto_commit=True
            )
            
            self.running = True
            self.logger.info("Kafka sensor monitoring started")
            
            # Start monitoring loop
            await self._monitoring_loop()
            
        except Exception as e:
            self.logger.error(f"Failed to start Kafka sensor monitoring: {e}")
            raise
    
    async def stop_monitoring(self) -> None:
        """Stop the Kafka sensor monitoring system"""
        
        self.running = False
        
        if self.producer:
            self.producer.close()
        
        if self.consumer:
            self.consumer.close()
        
        self.logger.info("Kafka sensor monitoring stopped")
    
    async def _monitoring_loop(self) -> None:
        """Main monitoring loop"""
        
        while self.running:
            try:
                # Poll for new sensor data
                message_batch = self.consumer.poll(timeout_ms=1000)
                
                for topic_partition, messages in message_batch.items():
                    for message in messages:
                        await self._process_sensor_message(message)
                
                # Perform periodic health checks
                await self._perform_health_checks()
                
                # Small delay to prevent overwhelming the system
                await asyncio.sleep(0.1)
                
            except Exception as e:
                self.logger.error(f"Error in monitoring loop: {e}")
                await asyncio.sleep(1)  # Back off on error
    
    async def _process_sensor_message(self, message) -> None:
        """Process incoming sensor data message"""
        
        try:
            sensor_id = message.key
            sensor_data = message.value
            
            if not sensor_id or not sensor_data:
                return
            
            # Extract sensor reading information
            value = sensor_data.get("value")
            quality = sensor_data.get("quality", 1.0)
            timestamp_str = sensor_data.get("timestamp")
            
            if timestamp_str:
                timestamp = datetime.fromisoformat(timestamp_str.replace('Z', '+00:00'))
            else:
                timestamp = datetime.now()
            
            # Update fact groups
            self._update_sensor_facts(sensor_id, sensor_data)
            
            # Validate sensor bounds
            violations = []
            if value is not None:
                violations = self.bounds_validator.validate_sensor_reading(
                    sensor_id, value, quality, timestamp
                )
            
            # Update sensor health status
            self._update_sensor_health(sensor_id, timestamp, violations, quality)
            
            # Send alerts if violations found
            if violations:
                await self._send_sensor_alerts(sensor_id, violations)
            
            # Log sensor activity
            self.logger.debug(f"Processed sensor data: {sensor_id} = {value}")
            
        except Exception as e:
            self.logger.error(f"Error processing sensor message: {e}")
    
    def _update_sensor_facts(self, sensor_id: str, sensor_data: Dict[str, Any]) -> None:
        """Update fact groups with sensor data"""
        
        # Determine the appropriate fact group based on sensor type
        sensor_type = sensor_data.get("sensor_type", "unknown")
        
        if sensor_type in ["gps", "gnss"]:
            self.fact_manager.update_gps_facts(sensor_data)
        elif sensor_type in ["battery", "power"]:
            self.fact_manager.update_battery_facts(sensor_data)
        elif sensor_type in ["imu", "attitude"]:
            self.fact_manager.update_attitude_facts(sensor_data)
        else:
            # Generic sensor fact group
            fact_group = self.fact_manager.get_group("sensors")
            if fact_group:
                fact = Fact(
                    name=f"{sensor_id}_value",
                    value=sensor_data.get("value"),
                    fact_type=FactType.SENSOR_VALUE,
                    unit=sensor_data.get("unit"),
                    source=sensor_id,
                    quality=sensor_data.get("quality", 1.0)
                )
                fact_group.add_fact(fact)
    
    def _update_sensor_health(self, sensor_id: str, timestamp: datetime, 
                             violations: List[BoundsViolation], quality: float) -> None:
        """Update sensor health status"""
        
        if sensor_id not in self.sensor_health:
            self.sensor_health[sensor_id] = SensorHealthStatus(
                sensor_id=sensor_id,
                status="healthy",
                last_reading=timestamp,
                violation_count=0,
                quality_score=quality,
                uptime_percent=100.0
            )
        
        health = self.sensor_health[sensor_id]
        
        # Update basic information
        health.last_reading = timestamp
        health.quality_score = (health.quality_score * 0.9) + (quality * 0.1)  # Exponential moving average
        
        # Update violation count
        if violations:
            health.violation_count += len(violations)
            
            # Determine status based on violation severity
            critical_violations = [v for v in violations if v.severity.value == "critical"]
            if critical_violations:
                health.status = "critical"
            elif health.status != "critical":
                health.status = "warning"
        else:
            # No violations - potentially improve status
            if health.status == "warning" and health.violation_count == 0:
                health.status = "healthy"
        
        # Calculate uptime (simplified)
        if timestamp:
            time_since_start = (datetime.now() - timestamp).total_seconds()
            if time_since_start > 300:  # 5 minutes offline threshold
                health.status = "offline"
                health.uptime_percent = max(0, health.uptime_percent - 1)
            else:
                health.uptime_percent = min(100, health.uptime_percent + 0.1)
    
    async def _send_sensor_alerts(self, sensor_id: str, violations: List[BoundsViolation]) -> None:
        """Send sensor alerts to Kafka"""
        
        try:
            for violation in violations:
                alert_data = {
                    "alert_type": "sensor_violation",
                    "sensor_id": sensor_id,
                    "violation": violation.to_dict(),
                    "timestamp": datetime.now().isoformat(),
                    "alert_id": f"sensor_{sensor_id}_{datetime.now().timestamp()}"
                }
                
                # Send to alerts topic
                future = self.producer.send(
                    self.sensor_alerts_topic,
                    key=sensor_id,
                    value=alert_data
                )
                
                # Don't block - fire and forget
                future.add_callback(lambda metadata: self.logger.debug(f"Alert sent: {metadata}"))
                future.add_errback(lambda e: self.logger.error(f"Failed to send alert: {e}"))
            
            # Call registered alert callbacks
            for callback in self.alert_callbacks:
                try:
                    await callback(sensor_id, violations)
                except Exception as e:
                    self.logger.error(f"Alert callback error: {e}")
                    
        except Exception as e:
            self.logger.error(f"Failed to send sensor alerts: {e}")
    
    async def _perform_health_checks(self) -> None:
        """Perform periodic health checks and publish status"""
        
        # Check for stale sensors
        current_time = datetime.now()
        stale_threshold = timedelta(minutes=5)
        
        for sensor_id, health in self.sensor_health.items():
            if health.last_reading and (current_time - health.last_reading) > stale_threshold:
                if health.status != "offline":
                    health.status = "offline"
                    await self._send_stale_sensor_alert(sensor_id, health)
        
        # Publish health status every 30 seconds
        if hasattr(self, '_last_health_publish'):
            if (current_time - self._last_health_publish).total_seconds() < 30:
                return
        
        await self._publish_health_status()
        self._last_health_publish = current_time
    
    async def _send_stale_sensor_alert(self, sensor_id: str, health: SensorHealthStatus) -> None:
        """Send alert for stale/offline sensor"""
        
        try:
            alert_data = {
                "alert_type": "sensor_offline",
                "sensor_id": sensor_id,
                "last_reading": health.last_reading.isoformat() if health.last_reading else None,
                "timestamp": datetime.now().isoformat(),
                "alert_id": f"offline_{sensor_id}_{datetime.now().timestamp()}"
            }
            
            future = self.producer.send(
                self.sensor_alerts_topic,
                key=sensor_id,
                value=alert_data
            )
            
            future.add_errback(lambda e: self.logger.error(f"Failed to send offline alert: {e}"))
            
        except Exception as e:
            self.logger.error(f"Failed to send stale sensor alert: {e}")
    
    async def _publish_health_status(self) -> None:
        """Publish overall sensor health status"""
        
        try:
            health_summary = {
                "timestamp": datetime.now().isoformat(),
                "total_sensors": len(self.sensor_health),
                "healthy_sensors": len([h for h in self.sensor_health.values() if h.status == "healthy"]),
                "warning_sensors": len([h for h in self.sensor_health.values() if h.status == "warning"]),
                "critical_sensors": len([h for h in self.sensor_health.values() if h.status == "critical"]),
                "offline_sensors": len([h for h in self.sensor_health.values() if h.status == "offline"]),
                "sensor_details": {
                    sensor_id: asdict(health) for sensor_id, health in self.sensor_health.items()
                }
            }
            
            future = self.producer.send(
                self.sensor_health_topic,
                key="health_summary",
                value=health_summary
            )
            
            future.add_errback(lambda e: self.logger.error(f"Failed to publish health status: {e}"))
            
        except Exception as e:
            self.logger.error(f"Failed to publish health status: {e}")
    
    def add_alert_callback(self, callback: Callable) -> None:
        """Add a callback function for sensor alerts"""
        self.alert_callbacks.append(callback)
    
    def remove_alert_callback(self, callback: Callable) -> None:
        """Remove a callback function"""
        if callback in self.alert_callbacks:
            self.alert_callbacks.remove(callback)
    
    def get_sensor_health(self, sensor_id: str = None) -> Dict[str, Any]:
        """Get sensor health information"""
        
        if sensor_id:
            health = self.sensor_health.get(sensor_id)
            return asdict(health) if health else None
        else:
            return {
                sensor_id: asdict(health) 
                for sensor_id, health in self.sensor_health.items()
            }
    
    def configure_sensor_bounds(self, sensor_configs: List[Dict[str, Any]]) -> None:
        """Configure sensor bounds for validation"""
        self.bounds_validator.configure_from_sensor_data(sensor_configs)
        self.logger.info(f"Configured bounds for {len(sensor_configs)} sensors")
    
    async def inject_test_data(self, sensor_id: str, test_data: Dict[str, Any]) -> None:
        """Inject test sensor data for testing purposes"""
        
        try:
            test_message = {
                "sensor_id": sensor_id,
                "value": test_data.get("value"),
                "quality": test_data.get("quality", 1.0),
                "sensor_type": test_data.get("sensor_type", "test"),
                "unit": test_data.get("unit", ""),
                "timestamp": datetime.now().isoformat()
            }
            
            future = self.producer.send(
                self.sensor_data_topic,
                key=sensor_id,
                value=test_message
            )
            
            await asyncio.wrap_future(future)
            self.logger.info(f"Injected test data for sensor {sensor_id}")
            
        except Exception as e:
            self.logger.error(f"Failed to inject test data: {e}")