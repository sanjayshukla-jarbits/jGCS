"""
Vehicle State Stream - Real-time vehicle state streaming and management

This module provides:
- Real-time vehicle state streaming
- State change detection and notifications
- Historical state tracking
- State-based triggers and alerts
"""

import asyncio
import json
from typing import Dict, List, Any, Optional, Callable
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from collections import deque
from kafka import KafkaProducer, KafkaConsumer
import uuid

from src.utils.logger import get_logger

@dataclass
class VehicleState:
    """Vehicle state information"""
    vehicle_id: str
    timestamp: datetime
    position: Optional[Dict[str, float]]  # lat, lon, alt
    velocity: Optional[Dict[str, float]]  # vx, vy, vz, ground_speed
    attitude: Optional[Dict[str, float]]  # roll, pitch, yaw
    battery: Optional[Dict[str, float]]   # voltage, current, percentage
    status: str  # "idle", "armed", "flying", "landing", "emergency", "offline"
    flight_mode: str
    health_indicators: Dict[str, Any]
    mission_info: Optional[Dict[str, Any]]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary with timestamp formatting"""
        result = asdict(self)
        result['timestamp'] = self.timestamp.isoformat()
        return result

@dataclass
class StateChange:
    """Vehicle state change event"""
    change_id: str
    vehicle_id: str
    change_type: str  # "status", "position", "battery", "mission"
    old_value: Any
    new_value: Any
    timestamp: datetime
    significance: str  # "minor", "moderate", "major", "critical"

class VehicleStateStream:
    """Real-time vehicle state streaming system"""
    
    def __init__(self, kafka_config: Dict[str, Any]):
        self.kafka_config = kafka_config
        self.logger = get_logger(__name__)
        
        # Kafka components
        self.producer: Optional[KafkaProducer] = None
        self.consumer: Optional[KafkaConsumer] = None
        
        # Topics
        self.state_updates_topic = "vehicle-state-updates"
        self.state_changes_topic = "vehicle-state-changes"
        self.state_alerts_topic = "vehicle-state-alerts"
        
        # State management
        self.current_states: Dict[str, VehicleState] = {}
        self.state_history: Dict[str, deque] = {}
        self.state_change_handlers: Dict[str, List[Callable]] = {}
        
        # Configuration
        self.history_size = 1000  # Number of historical states per vehicle
        self.change_detection_thresholds = {
            "position_change_meters": 10,
            "battery_change_percent": 5,
            "altitude_change_meters": 20,
            "speed_change_ms": 3
        }
        
        # State monitoring
        self.running = False
        self.stream_task = None
        
    async def start_streaming(self) -> None:
        """Start vehicle state streaming"""
        
        try:
            # Initialize Kafka producer
            self.producer = KafkaProducer(
                bootstrap_servers=self.kafka_config.get("bootstrap_servers", ["localhost:9092"]),
                value_serializer=lambda v: json.dumps(v, default=str).encode('utf-8'),
                key_serializer=lambda k: k.encode('utf-8') if k else None,
                acks='all',
                batch_size=16384,
                linger_ms=10  # Small delay for batching
            )
            
            # Initialize Kafka consumer
            self.consumer = KafkaConsumer(
                self.state_updates_topic,
                bootstrap_servers=self.kafka_config.get("bootstrap_servers", ["localhost:9092"]),
                value_deserializer=lambda m: json.loads(m.decode('utf-8')),
                key_deserializer=lambda k: k.decode('utf-8') if k else None,
                group_id='vehicle-state-processor',
                auto_offset_reset='latest',
                max_poll_records=500
            )
            
            self.running = True
            self.stream_task = asyncio.create_task(self._processing_loop())
            
            self.logger.info("Started vehicle state streaming")
            
        except Exception as e:
            self.logger.error(f"Failed to start vehicle state streaming: {e}")
            raise
    
    async def stop_streaming(self) -> None:
        """Stop vehicle state streaming"""
        
        self.running = False
        
        if self.stream_task:
            self.stream_task.cancel()
            try:
                await self.stream_task
            except asyncio.CancelledError:
                pass
        
        if self.producer:
            self.producer.close()
        
        if self.consumer:
            self.consumer.close()
        
        self.logger.info("Stopped vehicle state streaming")
    
    async def _processing_loop(self) -> None:
        """Main state processing loop"""
        
        while self.running:
            try:
                # Poll for state updates
                message_batch = self.consumer.poll(timeout_ms=1000)
                
                for topic_partition, messages in message_batch.items():
                    for message in messages:
                        await self._process_state_update(message)
                
                # Perform periodic maintenance
                await self._maintenance_tasks()
                
                # Small delay
                await asyncio.sleep(0.1)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Error in state processing loop: {e}")
                await asyncio.sleep(1)
    
    async def _process_state_update(self, message) -> None:
        """Process incoming state update"""
        
        try:
            vehicle_id = message.key
            state_data = message.value
            
            if not vehicle_id:
                return
            
            # Parse timestamp
            timestamp = datetime.now()
            if "timestamp" in state_data:
                try:
                    timestamp = datetime.fromisoformat(state_data["timestamp"].replace('Z', '+00:00'))
                except:
                    pass
            
            # Create vehicle state
            new_state = VehicleState(
                vehicle_id=vehicle_id,
                timestamp=timestamp,
                position=state_data.get("position"),
                velocity=state_data.get("velocity"),
                attitude=state_data.get("attitude"),
                battery=state_data.get("battery"),
                status=state_data.get("status", "unknown"),
                flight_mode=state_data.get("flight_mode", "unknown"),
                health_indicators=state_data.get("health_indicators", {}),
                mission_info=state_data.get("mission_info")
            )
            
            # Detect changes
            changes = await self._detect_state_changes(vehicle_id, new_state)
            
            # Update current state
            old_state = self.current_states.get(vehicle_id)
            self.current_states[vehicle_id] = new_state
            
            # Store in history
            if vehicle_id not in self.state_history:
                self.state_history[vehicle_id] = deque(maxlen=self.history_size)
            self.state_history[vehicle_id].append(new_state)
            
            # Process changes
            for change in changes:
                await self._handle_state_change(change)
            
            # Publish state change events
            if changes:
                await self._publish_state_changes(changes)
            
        except Exception as e:
            self.logger.error(f"Error processing state update: {e}")
    
    async def _detect_state_changes(self, vehicle_id: str, new_state: VehicleState) -> List[StateChange]:
        """Detect significant state changes"""
        
        changes = []
        old_state = self.current_states.get(vehicle_id)
        
        if not old_state:
            return changes  # No previous state to compare
        
        # Status change
        if old_state.status != new_state.status:
            changes.append(StateChange(
                change_id=str(uuid.uuid4()),
                vehicle_id=vehicle_id,
                change_type="status",
                old_value=old_state.status,
                new_value=new_state.status,
                timestamp=new_state.timestamp,
                significance=self._assess_status_change_significance(old_state.status, new_state.status)
            ))
        
        # Flight mode change
        if old_state.flight_mode != new_state.flight_mode:
            changes.append(StateChange(
                change_id=str(uuid.uuid4()),
                vehicle_id=vehicle_id,
                change_type="flight_mode",
                old_value=old_state.flight_mode,
                new_value=new_state.flight_mode,
                timestamp=new_state.timestamp,
                significance="moderate"
            ))
        
        # Position change
        if old_state.position and new_state.position:
            distance_change = self._calculate_position_distance(old_state.position, new_state.position)
            if distance_change > self.change_detection_thresholds["position_change_meters"]:
                changes.append(StateChange(
                    change_id=str(uuid.uuid4()),
                    vehicle_id=vehicle_id,
                    change_type="position",
                    old_value=old_state.position,
                    new_value=new_state.position,
                    timestamp=new_state.timestamp,
                    significance="minor" if distance_change < 50 else "moderate"
                ))
        
        # Battery change
        if (old_state.battery and new_state.battery and 
            "percentage" in old_state.battery and "percentage" in new_state.battery):
            
            battery_change = abs(old_state.battery["percentage"] - new_state.battery["percentage"])
            if battery_change > self.change_detection_thresholds["battery_change_percent"]:
                significance = "minor"
                if new_state.battery["percentage"] < 20:
                    significance = "critical"
                elif new_state.battery["percentage"] < 30:
                    significance = "major"
                
                changes.append(StateChange(
                    change_id=str(uuid.uuid4()),
                    vehicle_id=vehicle_id,
                    change_type="battery",
                    old_value=old_state.battery["percentage"],
                    new_value=new_state.battery["percentage"],
                    timestamp=new_state.timestamp,
                    significance=significance
                ))
        
        # Altitude change
        if (old_state.position and new_state.position and 
            "alt" in old_state.position and "alt" in new_state.position):
            
            altitude_change = abs(old_state.position["alt"] - new_state.position["alt"])
            if altitude_change > self.change_detection_thresholds["altitude_change_meters"]:
                changes.append(StateChange(
                    change_id=str(uuid.uuid4()),
                    vehicle_id=vehicle_id,
                    change_type="altitude",
                    old_value=old_state.position["alt"],
                    new_value=new_state.position["alt"],
                    timestamp=new_state.timestamp,
                    significance="minor" if altitude_change < 100 else "moderate"
                ))
        
        return changes
    
    def _assess_status_change_significance(self, old_status: str, new_status: str) -> str:
        """Assess significance of status change"""
        
        critical_transitions = [
            ("flying", "emergency"),
            ("armed", "emergency"),
            ("idle", "emergency"),
            ("flying", "offline"),
            ("armed", "offline")
        ]
        
        major_transitions = [
            ("idle", "armed"),
            ("armed", "flying"),
            ("flying", "landing"),
            ("landing", "idle")
        ]
        
        if (old_status, new_status) in critical_transitions:
            return "critical"
        elif (old_status, new_status) in major_transitions:
            return "major"
        elif old_status != new_status:
            return "moderate"
        else:
            return "minor"
    
    def _calculate_position_distance(self, pos1: Dict[str, float], pos2: Dict[str, float]) -> float:
        """Calculate distance between two positions in meters"""
        
        if not all(key in pos1 for key in ["lat", "lon"]) or not all(key in pos2 for key in ["lat", "lon"]):
            return 0
        
        # Simple distance calculation (Haversine would be more accurate)
        lat_diff = pos1["lat"] - pos2["lat"]
        lon_diff = pos1["lon"] - pos2["lon"]
        alt_diff = pos1.get("alt", 0) - pos2.get("alt", 0)
        
        # Rough conversion to meters
        horizontal_distance = ((lat_diff * 111320) ** 2 + (lon_diff * 111320) ** 2) ** 0.5
        total_distance = (horizontal_distance ** 2 + alt_diff ** 2) ** 0.5
        
        return total_distance
    
    async def _handle_state_change(self, change: StateChange) -> None:
        """Handle individual state change"""
        
        # Call registered handlers
        handlers = self.state_change_handlers.get(change.change_type, [])
        handlers.extend(self.state_change_handlers.get("all", []))
        
        for handler in handlers:
            try:
                await handler(change)
            except Exception as e:
                self.logger.error(f"Error in state change handler: {e}")
        
        # Generate alerts for significant changes
        if change.significance in ["major", "critical"]:
            await self._generate_state_alert(change)
    
    async def _publish_state_changes(self, changes: List[StateChange]) -> None:
        """Publish state changes to Kafka"""
        
        try:
            for change in changes:
                change_message = {
                    "change_id": change.change_id,
                    "vehicle_id": change.vehicle_id,
                    "change_type": change.change_type,
                    "old_value": change.old_value,
                    "new_value": change.new_value,
                    "timestamp": change.timestamp.isoformat(),
                    "significance": change.significance
                }
                
                future = self.producer.send(
                    self.state_changes_topic,
                    key=change.vehicle_id,
                    value=change_message
                )
                
                future.add_errback(lambda e: self.logger.error(f"Failed to publish state change: {e}"))
                
        except Exception as e:
            self.logger.error(f"Error publishing state changes: {e}")
    
    async def _generate_state_alert(self, change: StateChange) -> None:
        """Generate alert for significant state change"""
        
        alert_data = {
            "alert_id": str(uuid.uuid4()),
            "alert_type": "vehicle_state_change",
            "vehicle_id": change.vehicle_id,
            "change_type": change.change_type,
            "significance": change.significance,
            "old_value": change.old_value,
            "new_value": change.new_value,
            "timestamp": change.timestamp.isoformat(),
            "message": self._generate_alert_message(change)
        }
        
        try:
            future = self.producer.send(
                self.state_alerts_topic,
                key=change.vehicle_id,
                value=alert_data
            )
            
            future.add_errback(lambda e: self.logger.error(f"Failed to publish state alert: {e}"))
            
        except Exception as e:
            self.logger.error(f"Error generating state alert: {e}")
    
    def _generate_alert_message(self, change: StateChange) -> str:
        """Generate human-readable alert message"""
        
        if change.change_type == "status":
            return f"Vehicle {change.vehicle_id} status changed from {change.old_value} to {change.new_value}"
        elif change.change_type == "battery":
            return f"Vehicle {change.vehicle_id} battery level changed from {change.old_value}% to {change.new_value}%"
        elif change.change_type == "position":
            return f"Vehicle {change.vehicle_id} position changed significantly"
        elif change.change_type == "altitude":
            return f"Vehicle {change.vehicle_id} altitude changed from {change.old_value}m to {change.new_value}m"
        else:
            return f"Vehicle {change.vehicle_id} {change.change_type} changed from {change.old_value} to {change.new_value}"
    
    async def _maintenance_tasks(self) -> None:
        """Perform periodic maintenance tasks"""
        
        # Clean up old history periodically
        if not hasattr(self, '_last_cleanup') or \
           (datetime.now() - self._last_cleanup).total_seconds() > 3600:  # Every hour
            
            await self._cleanup_old_data()
            self._last_cleanup = datetime.now()
    
    async def _cleanup_old_data(self) -> None:
        """Clean up old state data"""
        
        cutoff_time = datetime.now() - timedelta(hours=24)
        
        # Clean up offline vehicles
        offline_vehicles = []
        for vehicle_id, state in self.current_states.items():
            if (datetime.now() - state.timestamp).total_seconds() > 7200:  # 2 hours offline
                offline_vehicles.append(vehicle_id)
        
        for vehicle_id in offline_vehicles:
            del self.current_states[vehicle_id]
            if vehicle_id in self.state_history:
                del self.state_history[vehicle_id]
        
        if offline_vehicles:
            self.logger.info(f"Cleaned up {len(offline_vehicles)} offline vehicles")
    
    def add_state_change_handler(self, change_type: str, handler: Callable) -> None:
        """Add handler for state changes"""
        
        if change_type not in self.state_change_handlers:
            self.state_change_handlers[change_type] = []
        
        self.state_change_handlers[change_type].append(handler)
    
    def get_current_state(self, vehicle_id: str) -> Optional[Dict[str, Any]]:
        """Get current state of vehicle"""
        
        state = self.current_states.get(vehicle_id)
        return state.to_dict() if state else None
    
    def get_all_states(self) -> Dict[str, Dict[str, Any]]:
        """Get current states of all vehicles"""
        
        return {
            vehicle_id: state.to_dict()
            for vehicle_id, state in self.current_states.items()
        }
    
    def get_state_history(self, vehicle_id: str, hours: int = 24) -> List[Dict[str, Any]]:
        """Get state history for vehicle"""
        
        if vehicle_id not in self.state_history:
            return []
        
        cutoff_time = datetime.now() - timedelta(hours=hours)
        
        history = []
        for state in self.state_history[vehicle_id]:
            if state.timestamp > cutoff_time:
                history.append(state.to_dict())
        
        return sorted(history, key=lambda s: s["timestamp"])
    
    def get_vehicles_by_status(self, status: str) -> List[str]:
        """Get list of vehicles with specific status"""
        
        return [
            vehicle_id for vehicle_id, state in self.current_states.items()
            if state.status == status
        ]
    
    def get_fleet_summary(self) -> Dict[str, Any]:
        """Get fleet state summary"""
        
        total_vehicles = len(self.current_states)
        status_counts = {}
        
        for state in self.current_states.values():
            status = state.status
            status_counts[status] = status_counts.get(status, 0) + 1
        
        # Calculate average battery level
        battery_levels = []
        for state in self.current_states.values():
            if state.battery and "percentage" in state.battery:
                battery_levels.append(state.battery["percentage"])
        
        avg_battery = sum(battery_levels) / len(battery_levels) if battery_levels else 0
        
        return {
            "total_vehicles": total_vehicles,
            "status_breakdown": status_counts,
            "average_battery_level": round(avg_battery, 1),
            "vehicles_online": len([s for s in self.current_states.values() 
                                  if (datetime.now() - s.timestamp).total_seconds() < 300]),
            "last_updated": datetime.now().isoformat()
        }