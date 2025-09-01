"""
Kafka-triggered Mission Replanning - Real-time mission replanning based on events

This module provides:
- Event-driven mission replanning
- Real-time mission updates via Kafka
- Conflict resolution and avoidance
- Dynamic route optimization
"""

import asyncio
import json
from typing import Dict, List, Any, Optional, Callable
from datetime import datetime, timedelta
from dataclasses import dataclass
from kafka import KafkaProducer, KafkaConsumer
from kafka.errors import KafkaError

from src.modules.mission_planner.mission_controller import MissionPlannerModule
from src.modules.mission_planner.pattern_generator import PatternGenerator
from src.utils.logger import get_logger

@dataclass
class ReplanningEvent:
    """Event that triggers mission replanning"""
    event_id: str
    event_type: str  # "weather", "conflict", "obstacle", "emergency", "optimization"
    priority: str    # "low", "medium", "high", "critical"
    affected_vehicles: List[str]
    event_data: Dict[str, Any]
    timestamp: datetime
    requires_immediate_action: bool = False

@dataclass
class ReplanningResult:
    """Result of mission replanning operation"""
    replan_id: str
    trigger_event: str
    affected_vehicles: List[str]
    new_missions: Dict[str, Dict[str, Any]]
    changes_summary: Dict[str, Any]
    replanning_time_ms: float
    success: bool
    warnings: List[str]

class KafkaReplanner:
    """Kafka-based real-time mission replanning system"""
    
    def __init__(self, kafka_config: Dict[str, Any]):
        self.kafka_config = kafka_config
        self.logger = get_logger(__name__)
        
        # Mission planner components
        self.mission_planner = MissionPlannerModule("mission_planner", "survey_planner")
        self.pattern_generator = PatternGenerator()
        
        # Kafka components
        self.producer: Optional[KafkaProducer] = None
        self.consumer: Optional[KafkaConsumer] = None
        
        # Topics
        self.replanning_events_topic = "mission-replanning-events"
        self.mission_updates_topic = "mission-updates"
        self.vehicle_commands_topic = "uav-commands"
        
        # State management
        self.active_missions: Dict[str, Dict[str, Any]] = {}
        self.replanning_callbacks: List[Callable] = []
        self.running = False
        
        # Replanning strategies
        self.replanning_strategies = {
            "weather": self._handle_weather_replanning,
            "conflict": self._handle_conflict_replanning,
            "obstacle": self._handle_obstacle_replanning,
            "emergency": self._handle_emergency_replanning,
            "optimization": self._handle_optimization_replanning
        }
    
    async def start_replanner(self) -> None:
        """Start the Kafka replanning system"""
        
        try:
            # Initialize Kafka producer
            self.producer = KafkaProducer(
                bootstrap_servers=self.kafka_config.get("bootstrap_servers", ["localhost:9092"]),
                value_serializer=lambda v: json.dumps(v, default=str).encode('utf-8'),
                key_serializer=lambda k: k.encode('utf-8') if k else None,
                acks='all',
                retries=3
            )
            
            # Initialize Kafka consumer
            self.consumer = KafkaConsumer(
                self.replanning_events_topic,
                bootstrap_servers=self.kafka_config.get("bootstrap_servers", ["localhost:9092"]),
                value_deserializer=lambda m: json.loads(m.decode('utf-8')),
                key_deserializer=lambda k: k.decode('utf-8') if k else None,
                group_id='mission-replanner-group',
                auto_offset_reset='latest'
            )
            
            self.running = True
            self.logger.info("Kafka mission replanner started")
            
            # Start replanning loop
            await self._replanning_loop()
            
        except Exception as e:
            self.logger.error(f"Failed to start Kafka replanner: {e}")
            raise
    
    async def stop_replanner(self) -> None:
        """Stop the Kafka replanning system"""
        
        self.running = False
        
        if self.producer:
            self.producer.close()
        
        if self.consumer:
            self.consumer.close()
        
        self.logger.info("Kafka mission replanner stopped")
    
    async def _replanning_loop(self) -> None:
        """Main replanning event processing loop"""
        
        while self.running:
            try:
                # Poll for replanning events
                message_batch = self.consumer.poll(timeout_ms=1000)
                
                for topic_partition, messages in message_batch.items():
                    for message in messages:
                        await self._process_replanning_event(message)
                
                # Small delay to prevent overwhelming the system
                await asyncio.sleep(0.1)
                
            except Exception as e:
                self.logger.error(f"Error in replanning loop: {e}")
                await asyncio.sleep(1)
    
    async def _process_replanning_event(self, message) -> None:
        """Process incoming replanning event"""
        
        try:
            event_data = message.value
            
            # Create replanning event object
            event = ReplanningEvent(
                event_id=event_data.get("event_id"),
                event_type=event_data.get("event_type"),
                priority=event_data.get("priority", "medium"),
                affected_vehicles=event_data.get("affected_vehicles", []),
                event_data=event_data.get("event_data", {}),
                timestamp=datetime.fromisoformat(event_data.get("timestamp", datetime.now().isoformat())),
                requires_immediate_action=event_data.get("requires_immediate_action", False)
            )
            
            self.logger.info(f"Processing replanning event: {event.event_type} for vehicles {event.affected_vehicles}")
            
            # Execute replanning based on event type
            if event.event_type in self.replanning_strategies:
                result = await self.replanning_strategies[event.event_type](event)
                
                if result.success:
                    await self._publish_mission_updates(result)
                    await self._send_vehicle_commands(result)
                    
                    # Notify callbacks
                    for callback in self.replanning_callbacks:
                        try:
                            await callback(event, result)
                        except Exception as e:
                            self.logger.error(f"Replanning callback error: {e}")
                
                self.logger.info(f"Replanning completed in {result.replanning_time_ms:.1f}ms")
            else:
                self.logger.warning(f"No replanning strategy for event type: {event.event_type}")
                
        except Exception as e:
            self.logger.error(f"Error processing replanning event: {e}")
    
    async def _handle_weather_replanning(self, event: ReplanningEvent) -> ReplanningResult:
        """Handle weather-related replanning"""
        
        start_time = datetime.now()
        
        weather_data = event.event_data
        affected_vehicles = event.affected_vehicles
        
        new_missions = {}
        changes_summary = {}
        warnings = []
        
        for vehicle_id in affected_vehicles:
            current_mission = self.active_missions.get(vehicle_id)
            if not current_mission:
                continue
            
            # Get weather constraints
            wind_speed = weather_data.get("wind_speed", 0)
            visibility = weather_data.get("visibility", 10000)
            precipitation = weather_data.get("precipitation", 0)
            
            # Determine if mission needs modification
            if wind_speed > 15:  # High wind threshold
                # Reduce altitude and speed
                modified_mission = await self._modify_mission_for_weather(
                    current_mission, {"reduce_altitude": 20, "reduce_speed": 0.5}
                )
                new_missions[vehicle_id] = modified_mission
                changes_summary[vehicle_id] = "Reduced altitude and speed due to high winds"
                
            elif visibility < 1000:  # Low visibility
                # Return to home or hold position
                rtl_mission = await self._generate_return_to_launch_mission(current_mission)
                new_missions[vehicle_id] = rtl_mission
                changes_summary[vehicle_id] = "Return to launch due to low visibility"
                warnings.append(f"Vehicle {vehicle_id} returning due to weather")
            
            elif precipitation > 5:  # Heavy precipitation
                # Land immediately at nearest safe location
                land_mission = await self._generate_emergency_land_mission(current_mission)
                new_missions[vehicle_id] = land_mission
                changes_summary[vehicle_id] = "Emergency landing due to precipitation"
                warnings.append(f"Vehicle {vehicle_id} landing due to weather")
        
        processing_time = (datetime.now() - start_time).total_seconds() * 1000
        
        return ReplanningResult(
            replan_id=f"weather_{event.event_id}",
            trigger_event=event.event_id,
            affected_vehicles=affected_vehicles,
            new_missions=new_missions,
            changes_summary=changes_summary,
            replanning_time_ms=processing_time,
            success=len(new_missions) > 0,
            warnings=warnings
        )
    
    async def _handle_conflict_replanning(self, event: ReplanningEvent) -> ReplanningResult:
        """Handle conflict avoidance replanning"""
        
        start_time = datetime.now()
        
        conflict_data = event.event_data
        affected_vehicles = event.affected_vehicles
        
        new_missions = {}
        changes_summary = {}
        warnings = []
        
        # Get conflict details
        conflict_zone = conflict_data.get("conflict_zone")
        conflict_time = conflict_data.get("conflict_time")
        conflicting_vehicles = conflict_data.get("conflicting_vehicles", [])
        
        for vehicle_id in affected_vehicles:
            current_mission = self.active_missions.get(vehicle_id)
            if not current_mission:
                continue
            
            if vehicle_id in conflicting_vehicles:
                # Apply conflict resolution strategy
                if len(conflicting_vehicles) == 2:
                    # Two-vehicle conflict - use altitude separation
                    altitude_offset = 20 if conflicting_vehicles.index(vehicle_id) == 0 else -20
                    modified_mission = await self._modify_mission_altitude(
                        current_mission, altitude_offset
                    )
                    new_missions[vehicle_id] = modified_mission
                    changes_summary[vehicle_id] = f"Altitude adjusted by {altitude_offset}m for conflict avoidance"
                
                else:
                    # Multi-vehicle conflict - use temporal separation
                    delay_seconds = conflicting_vehicles.index(vehicle_id) * 60  # 1 minute delays
                    modified_mission = await self._add_mission_delay(
                        current_mission, delay_seconds
                    )
                    new_missions[vehicle_id] = modified_mission
                    changes_summary[vehicle_id] = f"Mission delayed by {delay_seconds}s for conflict avoidance"
        
        processing_time = (datetime.now() - start_time).total_seconds() * 1000
        
        return ReplanningResult(
            replan_id=f"conflict_{event.event_id}",
            trigger_event=event.event_id,
            affected_vehicles=affected_vehicles,
            new_missions=new_missions,
            changes_summary=changes_summary,
            replanning_time_ms=processing_time,
            success=len(new_missions) > 0,
            warnings=warnings
        )
    
    async def _handle_obstacle_replanning(self, event: ReplanningEvent) -> ReplanningResult:
        """Handle obstacle avoidance replanning"""
        
        start_time = datetime.now()
        
        obstacle_data = event.event_data
        affected_vehicles = event.affected_vehicles
        
        new_missions = {}
        changes_summary = {}
        warnings = []
        
        obstacle_location = obstacle_data.get("location")
        obstacle_radius = obstacle_data.get("radius", 100)  # Default 100m radius
        
        for vehicle_id in affected_vehicles:
            current_mission = self.active_missions.get(vehicle_id)
            if not current_mission:
                continue
            
            # Generate new path avoiding obstacle
            modified_mission = await self._generate_obstacle_avoiding_path(
                current_mission, obstacle_location, obstacle_radius
            )
            
            if modified_mission:
                new_missions[vehicle_id] = modified_mission
                changes_summary[vehicle_id] = "Path modified to avoid detected obstacle"
            else:
                warnings.append(f"Could not generate obstacle-avoiding path for {vehicle_id}")
        
        processing_time = (datetime.now() - start_time).total_seconds() * 1000
        
        return ReplanningResult(
            replan_id=f"obstacle_{event.event_id}",
            trigger_event=event.event_id,
            affected_vehicles=affected_vehicles,
            new_missions=new_missions,
            changes_summary=changes_summary,
            replanning_time_ms=processing_time,
            success=len(new_missions) > 0,
            warnings=warnings
        )
    
    async def _handle_emergency_replanning(self, event: ReplanningEvent) -> ReplanningResult:
        """Handle emergency replanning"""
        
        start_time = datetime.now()
        
        emergency_data = event.event_data
        emergency_type = emergency_data.get("emergency_type")
        affected_vehicles = event.affected_vehicles
        
        new_missions = {}
        changes_summary = {}
        warnings = []
        
        for vehicle_id in affected_vehicles:
            current_mission = self.active_missions.get(vehicle_id)
            if not current_mission:
                continue
            
            if emergency_type == "low_battery":
                # Generate return to launch mission
                rtl_mission = await self._generate_return_to_launch_mission(current_mission)
                new_missions[vehicle_id] = rtl_mission
                changes_summary[vehicle_id] = "Return to launch due to low battery"
                
            elif emergency_type == "system_failure":
                # Generate emergency landing mission
                land_mission = await self._generate_emergency_land_mission(current_mission)
                new_missions[vehicle_id] = land_mission
                changes_summary[vehicle_id] = "Emergency landing due to system failure"
                
            elif emergency_type == "communication_loss":
                # Activate failsafe mission
                failsafe_mission = await self._generate_failsafe_mission(current_mission)
                new_missions[vehicle_id] = failsafe_mission
                changes_summary[vehicle_id] = "Failsafe mission activated"
            
            warnings.append(f"Emergency replanning for {vehicle_id}: {emergency_type}")
        
        processing_time = (datetime.now() - start_time).total_seconds() * 1000
        
        return ReplanningResult(
            replan_id=f"emergency_{event.event_id}",
            trigger_event=event.event_id,
            affected_vehicles=affected_vehicles,
            new_missions=new_missions,
            changes_summary=changes_summary,
            replanning_time_ms=processing_time,
            success=len(new_missions) > 0,
            warnings=warnings
        )
    
    async def _handle_optimization_replanning(self, event: ReplanningEvent) -> ReplanningResult:
        """Handle optimization-based replanning"""
        
        start_time = datetime.now()
        
        optimization_data = event.event_data
        affected_vehicles = event.affected_vehicles
        
        new_missions = {}
        changes_summary = {}
        warnings = []
        
        optimization_type = optimization_data.get("type", "efficiency")
        
        if optimization_type == "efficiency":
            # Optimize for flight efficiency
            for vehicle_id in affected_vehicles:
                current_mission = self.active_missions.get(vehicle_id)
                if not current_mission:
                    continue
                
                optimized_mission = await self._optimize_mission_efficiency(current_mission)
                if optimized_mission:
                    new_missions[vehicle_id] = optimized_mission
                    changes_summary[vehicle_id] = "Mission optimized for efficiency"
        
        elif optimization_type == "coverage":
            # Optimize for area coverage
            for vehicle_id in affected_vehicles:
                current_mission = self.active_missions.get(vehicle_id)
                if not current_mission:
                    continue
                
                optimized_mission = await self._optimize_mission_coverage(current_mission)
                if optimized_mission:
                    new_missions[vehicle_id] = optimized_mission
                    changes_summary[vehicle_id] = "Mission optimized for coverage"
        
        processing_time = (datetime.now() - start_time).total_seconds() * 1000
        
        return ReplanningResult(
            replan_id=f"optimization_{event.event_id}",
            trigger_event=event.event_id,
            affected_vehicles=affected_vehicles,
            new_missions=new_missions,
            changes_summary=changes_summary,
            replanning_time_ms=processing_time,
            success=len(new_missions) > 0,
            warnings=warnings
        )
    
    async def _modify_mission_for_weather(self, mission: Dict[str, Any], 
                                        modifications: Dict[str, Any]) -> Dict[str, Any]:
        """Modify mission parameters for weather conditions"""
        
        modified_mission = mission.copy()
        waypoints = modified_mission.get("waypoints", [])
        
        altitude_reduction = modifications.get("reduce_altitude", 0)
        speed_factor = modifications.get("reduce_speed", 1.0)
        
        for wp in waypoints:
            if altitude_reduction > 0:
                wp["altitude"] = max(10, wp.get("altitude", 50) - altitude_reduction)
            
            if "speed" in wp:
                wp["speed"] *= speed_factor
        
        modified_mission["waypoints"] = waypoints
        modified_mission["modified_at"] = datetime.now().isoformat()
        modified_mission["modification_reason"] = "weather_adaptation"
        
        return modified_mission
    
    async def _generate_return_to_launch_mission(self, current_mission: Dict[str, Any]) -> Dict[str, Any]:
        """Generate return to launch mission"""
        
        # Get current position (simplified - would get from telemetry in real system)
        waypoints = current_mission.get("waypoints", [])
        if not waypoints:
            return current_mission
        
        # Assume first waypoint is takeoff location
        home_position = waypoints[0]
        
        rtl_mission = {
            "mission_id": f"RTL_{current_mission.get('mission_id', 'unknown')}",
            "mission_type": "return_to_launch",
            "waypoints": [
                {
                    "latitude": home_position.get("latitude"),
                    "longitude": home_position.get("longitude"), 
                    "altitude": home_position.get("altitude", 50),
                    "speed": 10,
                    "waypoint_type": "rtl"
                }
            ],
            "generated_at": datetime.now().isoformat(),
            "parent_mission": current_mission.get("mission_id")
        }
        
        return rtl_mission
    
    async def _generate_emergency_land_mission(self, current_mission: Dict[str, Any]) -> Dict[str, Any]:
        """Generate emergency landing mission"""
        
        # Get current position (simplified)
        waypoints = current_mission.get("waypoints", [])
        if not waypoints:
            return current_mission
        
        # Use current position for emergency landing
        current_position = waypoints[-1] if waypoints else waypoints[0]
        
        land_mission = {
            "mission_id": f"LAND_{current_mission.get('mission_id', 'unknown')}",
            "mission_type": "emergency_land",
            "waypoints": [
                {
                    "latitude": current_position.get("latitude"),
                    "longitude": current_position.get("longitude"),
                    "altitude": 0,
                    "speed": 3,
                    "waypoint_type": "land"
                }
            ],
            "generated_at": datetime.now().isoformat(),
            "parent_mission": current_mission.get("mission_id")
        }
        
        return land_mission
    
    async def _publish_mission_updates(self, result: ReplanningResult) -> None:
        """Publish mission updates to Kafka"""
        
        try:
            update_message = {
                "replan_id": result.replan_id,
                "trigger_event": result.trigger_event,
                "timestamp": datetime.now().isoformat(),
                "affected_vehicles": result.affected_vehicles,
                "mission_updates": result.new_missions,
                "changes_summary": result.changes_summary,
                "success": result.success,
                "warnings": result.warnings
            }
            
            future = self.producer.send(
                self.mission_updates_topic,
                key="mission_update",
                value=update_message
            )
            
            future.add_errback(lambda e: self.logger.error(f"Failed to publish mission update: {e}"))
            
        except Exception as e:
            self.logger.error(f"Failed to publish mission updates: {e}")
    
    async def _send_vehicle_commands(self, result: ReplanningResult) -> None:
        """Send updated mission commands to vehicles"""
        
        try:
            for vehicle_id, mission_data in result.new_missions.items():
                command_message = {
                    "command_type": "mission_update",
                    "vehicle_id": vehicle_id,
                    "mission_data": mission_data,
                    "replan_id": result.replan_id,
                    "timestamp": datetime.now().isoformat(),
                    "priority": "high"
                }
                
                future = self.producer.send(
                    self.vehicle_commands_topic,
                    key=vehicle_id,
                    value=command_message
                )
                
                future.add_errback(lambda e: self.logger.error(f"Failed to send vehicle command: {e}"))
                
        except Exception as e:
            self.logger.error(f"Failed to send vehicle commands: {e}")
    
    def add_replanning_callback(self, callback: Callable) -> None:
        """Add callback for replanning events"""
        self.replanning_callbacks.append(callback)
    
    def update_active_mission(self, vehicle_id: str, mission_data: Dict[str, Any]) -> None:
        """Update active mission for a vehicle"""
        self.active_missions[vehicle_id] = mission_data
    
    def get_active_missions(self) -> Dict[str, Dict[str, Any]]:
        """Get all active missions"""
        return self.active_missions.copy()
    
    async def trigger_replanning_event(self, event_type: str, affected_vehicles: List[str],
                                     event_data: Dict[str, Any], priority: str = "medium") -> str:
        """Manually trigger a replanning event"""
        
        event_id = f"{event_type}_{datetime.now().timestamp()}"
        
        event_message = {
            "event_id": event_id,
            "event_type": event_type,
            "priority": priority,
            "affected_vehicles": affected_vehicles,
            "event_data": event_data,
            "timestamp": datetime.now().isoformat(),
            "requires_immediate_action": priority in ["high", "critical"]
        }
        
        try:
            future = self.producer.send(
                self.replanning_events_topic,
                key=event_id,
                value=event_message
            )
            
            await asyncio.wrap_future(future)
            self.logger.info(f"Triggered replanning event: {event_id}")
            
            return event_id
            
        except Exception as e:
            self.logger.error(f"Failed to trigger replanning event: {e}")
            raise