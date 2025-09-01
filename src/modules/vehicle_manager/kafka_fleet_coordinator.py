"""
Kafka Fleet Coordinator - Kafka-based fleet coordination and management

This module provides:
- Real-time fleet coordination via Kafka
- Fleet-wide messaging and commands
- Resource allocation and optimization
- Fleet health monitoring
"""

import asyncio
import json
from typing import Dict, List, Any, Optional, Callable
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from kafka import KafkaProducer, KafkaConsumer
from kafka.errors import KafkaError
import uuid

from src.utils.logger import get_logger

@dataclass
class FleetCommand:
    """Fleet-wide command"""
    command_id: str
    command_type: str  # "formation", "mission", "emergency", "maintenance"
    target_vehicles: List[str]
    command_data: Dict[str, Any]
    priority: int  # 1-10, 10 being highest
    issued_at: datetime
    expires_at: Optional[datetime] = None
    acknowledged_by: List[str] = None

@dataclass
class FleetStatus:
    """Fleet status information"""
    total_vehicles: int
    active_vehicles: int
    available_vehicles: int
    in_mission_vehicles: int
    maintenance_vehicles: int
    emergency_vehicles: int
    last_updated: datetime

@dataclass
class VehicleCapacity:
    """Vehicle capacity and resource information"""
    vehicle_id: str
    available_capacity: float  # 0-1 scale
    current_mission_load: float
    estimated_availability: datetime
    capabilities: List[str]
    maintenance_due: Optional[datetime]

class KafkaFleetCoordinator:
    """Kafka-based fleet coordination system"""
    
    def __init__(self, kafka_config: Dict[str, Any], fleet_id: str = "default"):
        self.kafka_config = kafka_config
        self.fleet_id = fleet_id
        self.logger = get_logger(__name__)
        
        # Kafka components
        self.producer: Optional[KafkaProducer] = None
        self.consumer: Optional[KafkaConsumer] = None
        
        # Topics
        self.fleet_commands_topic = f"fleet-commands-{fleet_id}"
        self.fleet_status_topic = f"fleet-status-{fleet_id}"
        self.vehicle_heartbeat_topic = f"vehicle-heartbeat-{fleet_id}"
        self.resource_allocation_topic = f"resource-allocation-{fleet_id}"
        
        # Fleet state
        self.fleet_vehicles: Dict[str, Dict[str, Any]] = {}
        self.active_commands: Dict[str, FleetCommand] = {}
        self.fleet_status = FleetStatus(0, 0, 0, 0, 0, 0, datetime.now())
        
        # Resource management
        self.vehicle_capacities: Dict[str, VehicleCapacity] = {}
        self.resource_allocation_strategy = "load_balancing"  # "load_balancing", "priority", "capability"
        
        # Callbacks
        self.command_callbacks: Dict[str, List[Callable]] = {}
        self.status_callbacks: List[Callable] = []
        
        # Configuration
        self.heartbeat_interval = 30  # seconds
        self.command_timeout = 300    # seconds
        self.max_retry_attempts = 3
        
        # State
        self.running = False
        self.coordinator_task = None
    
    async def start_coordinator(self) -> None:
        """Start fleet coordinator"""
        
        try:
            # Initialize Kafka producer
            self.producer = KafkaProducer(
                bootstrap_servers=self.kafka_config.get("bootstrap_servers", ["localhost:9092"]),
                value_serializer=lambda v: json.dumps(v, default=str).encode('utf-8'),
                key_serializer=lambda k: k.encode('utf-8') if k else None,
                acks='all',
                retries=self.max_retry_attempts
            )
            
            # Initialize Kafka consumer
            self.consumer = KafkaConsumer(
                self.vehicle_heartbeat_topic,
                self.fleet_status_topic,
                bootstrap_servers=self.kafka_config.get("bootstrap_servers", ["localhost:9092"]),
                value_deserializer=lambda m: json.loads(m.decode('utf-8')),
                key_deserializer=lambda k: k.decode('utf-8') if k else None,
                group_id=f'fleet-coordinator-{self.fleet_id}',
                auto_offset_reset='latest'
            )
            
            self.running = True
            self.coordinator_task = asyncio.create_task(self._coordination_loop())
            
            self.logger.info(f"Started fleet coordinator for fleet {self.fleet_id}")
            
        except Exception as e:
            self.logger.error(f"Failed to start fleet coordinator: {e}")
            raise
    
    async def stop_coordinator(self) -> None:
        """Stop fleet coordinator"""
        
        self.running = False
        
        if self.coordinator_task:
            self.coordinator_task.cancel()
            try:
                await self.coordinator_task
            except asyncio.CancelledError:
                pass
        
        if self.producer:
            self.producer.close()
        
        if self.consumer:
            self.consumer.close()
        
        self.logger.info(f"Stopped fleet coordinator for fleet {self.fleet_id}")
    
    async def _coordination_loop(self) -> None:
        """Main coordination loop"""
        
        while self.running:
            try:
                # Process incoming messages
                message_batch = self.consumer.poll(timeout_ms=1000)
                
                for topic_partition, messages in message_batch.items():
                    topic_name = topic_partition.topic
                    
                    for message in messages:
                        if topic_name == self.vehicle_heartbeat_topic:
                            await self._process_vehicle_heartbeat(message)
                        elif topic_name == self.fleet_status_topic:
                            await self._process_fleet_status_update(message)
                
                # Update fleet status
                await self._update_fleet_status()
                
                # Check for expired commands
                await self._cleanup_expired_commands()
                
                # Publish heartbeat
                await self._publish_coordinator_heartbeat()
                
                # Small delay
                await asyncio.sleep(1)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Error in coordination loop: {e}")
                await asyncio.sleep(5)
    
    async def _process_vehicle_heartbeat(self, message) -> None:
        """Process vehicle heartbeat message"""
        
        try:
            vehicle_id = message.key
            heartbeat_data = message.value
            
            if not vehicle_id:
                return
            
            # Update vehicle information
            self.fleet_vehicles[vehicle_id] = {
                "vehicle_id": vehicle_id,
                "status": heartbeat_data.get("status", "unknown"),
                "location": heartbeat_data.get("location"),
                "battery_level": heartbeat_data.get("battery_level"),
                "mission_progress": heartbeat_data.get("mission_progress"),
                "capabilities": heartbeat_data.get("capabilities", []),
                "last_heartbeat": datetime.now(),
                "health_score": heartbeat_data.get("health_score", 1.0)
            }
            
            # Update vehicle capacity
            await self._update_vehicle_capacity(vehicle_id, heartbeat_data)
            
        except Exception as e:
            self.logger.error(f"Error processing vehicle heartbeat: {e}")
    
    async def _update_vehicle_capacity(self, vehicle_id: str, heartbeat_data: Dict[str, Any]) -> None:
        """Update vehicle capacity information"""
        
        # Calculate available capacity based on current status
        status = heartbeat_data.get("status", "unknown")
        battery_level = heartbeat_data.get("battery_level", 100)
        mission_progress = heartbeat_data.get("mission_progress", 0)
        
        # Base capacity calculation
        if status == "idle":
            base_capacity = 1.0
        elif status == "in_mission":
            base_capacity = max(0, 1.0 - mission_progress)
        elif status in ["maintenance", "error"]:
            base_capacity = 0.0
        else:
            base_capacity = 0.5  # Unknown status
        
        # Adjust for battery level
        battery_factor = min(1.0, max(0.0, (battery_level - 20) / 80))  # Scale from 20-100%
        available_capacity = base_capacity * battery_factor
        
        # Estimate availability
        if status == "in_mission" and mission_progress > 0:
            # Rough estimate based on mission progress
            estimated_time_remaining = (1 - mission_progress) * 3600  # Assume 1 hour missions
            estimated_availability = datetime.now() + timedelta(seconds=estimated_time_remaining)
        else:
            estimated_availability = datetime.now()
        
        self.vehicle_capacities[vehicle_id] = VehicleCapacity(
            vehicle_id=vehicle_id,
            available_capacity=available_capacity,
            current_mission_load=mission_progress,
            estimated_availability=estimated_availability,
            capabilities=heartbeat_data.get("capabilities", []),
            maintenance_due=None  # Would be determined by maintenance scheduling system
        )
    
    async def _update_fleet_status(self) -> None:
        """Update overall fleet status"""
        
        now = datetime.now()
        stale_threshold = timedelta(minutes=5)
        
        total_vehicles = len(self.fleet_vehicles)
        active_vehicles = 0
        available_vehicles = 0
        in_mission_vehicles = 0
        maintenance_vehicles = 0
        emergency_vehicles = 0
        
        for vehicle_id, vehicle_data in self.fleet_vehicles.items():
            last_heartbeat = vehicle_data.get("last_heartbeat", now)
            
            # Check if vehicle is still active (recent heartbeat)
            if (now - last_heartbeat) < stale_threshold:
                active_vehicles += 1
                
                status = vehicle_data.get("status", "unknown")
                
                if status == "idle":
                    available_vehicles += 1
                elif status == "in_mission":
                    in_mission_vehicles += 1
                elif status in ["maintenance", "error"]:
                    maintenance_vehicles += 1
                elif status == "emergency":
                    emergency_vehicles += 1
        
        self.fleet_status = FleetStatus(
            total_vehicles=total_vehicles,
            active_vehicles=active_vehicles,
            available_vehicles=available_vehicles,
            in_mission_vehicles=in_mission_vehicles,
            maintenance_vehicles=maintenance_vehicles,
            emergency_vehicles=emergency_vehicles,
            last_updated=now
        )
        
        # Notify status callbacks
        for callback in self.status_callbacks:
            try:
                await callback(self.fleet_status)
            except Exception as e:
                self.logger.error(f"Error in status callback: {e}")
    
    async def issue_fleet_command(self, command_type: str, target_vehicles: List[str],
                                command_data: Dict[str, Any], priority: int = 5,
                                expires_in_seconds: int = None) -> str:
        """Issue command to fleet vehicles"""
        
        command_id = str(uuid.uuid4())
        expires_at = None
        
        if expires_in_seconds:
            expires_at = datetime.now() + timedelta(seconds=expires_in_seconds)
        
        command = FleetCommand(
            command_id=command_id,
            command_type=command_type,
            target_vehicles=target_vehicles,
            command_data=command_data,
            priority=priority,
            issued_at=datetime.now(),
            expires_at=expires_at,
            acknowledged_by=[]
        )
        
        # Store command
        self.active_commands[command_id] = command
        
        # Publish command to Kafka
        try:
            command_message = {
                "command_id": command_id,
                "command_type": command_type,
                "target_vehicles": target_vehicles,
                "command_data": command_data,
                "priority": priority,
                "issued_at": command.issued_at.isoformat(),
                "expires_at": expires_at.isoformat() if expires_at else None
            }
            
            future = self.producer.send(
                self.fleet_commands_topic,
                key=command_id,
                value=command_message
            )
            
            await asyncio.wrap_future(future)
            
            self.logger.info(f"Issued fleet command {command_id} to {len(target_vehicles)} vehicles")
            
            return command_id
            
        except Exception as e:
            self.logger.error(f"Failed to issue fleet command: {e}")
            # Remove from active commands if failed to publish
            del self.active_commands[command_id]
            raise
    
    async def allocate_resources(self, resource_requirements: Dict[str, Any]) -> Dict[str, Any]:
        """Allocate vehicles based on resource requirements"""
        
        required_vehicles = resource_requirements.get("vehicle_count", 1)
        required_capabilities = resource_requirements.get("capabilities", [])
        mission_duration = resource_requirements.get("duration_minutes", 60)
        priority = resource_requirements.get("priority", 5)
        
        # Find suitable vehicles
        suitable_vehicles = []
        
        for vehicle_id, capacity in self.vehicle_capacities.items():
            # Check if vehicle meets requirements
            if capacity.available_capacity < 0.5:  # Need at least 50% capacity
                continue
            
            # Check capabilities
            if required_capabilities:
                if not all(cap in capacity.capabilities for cap in required_capabilities):
                    continue
            
            # Check availability timing
            mission_start = resource_requirements.get("start_time", datetime.now())
            if isinstance(mission_start, str):
                mission_start = datetime.fromisoformat(mission_start)
            
            if capacity.estimated_availability > mission_start:
                continue
            
            suitable_vehicles.append({
                "vehicle_id": vehicle_id,
                "capacity": capacity.available_capacity,
                "capabilities": capacity.capabilities,
                "estimated_availability": capacity.estimated_availability
            })
        
        # Sort by allocation strategy
        if self.resource_allocation_strategy == "load_balancing":
            # Prefer vehicles with highest available capacity
            suitable_vehicles.sort(key=lambda v: v["capacity"], reverse=True)
        elif self.resource_allocation_strategy == "capability":
            # Prefer vehicles with most matching capabilities
            suitable_vehicles.sort(
                key=lambda v: len(set(v["capabilities"]) & set(required_capabilities)),
                reverse=True
            )
        
        # Select vehicles
        allocated_vehicles = suitable_vehicles[:required_vehicles]
        
        allocation_result = {
            "allocation_id": str(uuid.uuid4()),
            "requested_vehicles": required_vehicles,
            "allocated_vehicles": len(allocated_vehicles),
            "vehicles": [v["vehicle_id"] for v in allocated_vehicles],
            "allocation_success": len(allocated_vehicles) >= required_vehicles,
            "allocation_score": len(allocated_vehicles) / required_vehicles if required_vehicles > 0 else 0,
            "allocated_at": datetime.now().isoformat()
        }
        
        # Publish allocation to Kafka
        try:
            future = self.producer.send(
                self.resource_allocation_topic,
                key=allocation_result["allocation_id"],
                value=allocation_result
            )
            
            await asyncio.wrap_future(future)
            
        except Exception as e:
            self.logger.error(f"Failed to publish resource allocation: {e}")
        
        self.logger.info(f"Allocated {len(allocated_vehicles)}/{required_vehicles} vehicles")
        
        return allocation_result
    
    async def request_formation(self, formation_type: str, leader_vehicle: str,
                              follower_vehicles: List[str], 
                              formation_params: Dict[str, Any]) -> str:
        """Request formation flight"""
        
        formation_command = {
            "formation_type": formation_type,
            "leader_vehicle": leader_vehicle,
            "follower_vehicles": follower_vehicles,
            "formation_parameters": formation_params
        }
        
        target_vehicles = [leader_vehicle] + follower_vehicles
        
        return await self.issue_fleet_command(
            "formation",
            target_vehicles,
            formation_command,
            priority=8,  # High priority
            expires_in_seconds=600
        )
    
    async def emergency_recall(self, reason: str, affected_vehicles: List[str] = None) -> str:
        """Issue emergency recall command"""
        
        if affected_vehicles is None:
            # Recall all active vehicles
            affected_vehicles = [
                vehicle_id for vehicle_id, vehicle_data in self.fleet_vehicles.items()
                if vehicle_data.get("status") in ["in_mission", "idle"]
            ]
        
        emergency_command = {
            "action": "emergency_recall",
            "reason": reason,
            "return_to_base": True,
            "immediate": True
        }
        
        return await self.issue_fleet_command(
            "emergency",
            affected_vehicles,
            emergency_command,
            priority=10,  # Maximum priority
            expires_in_seconds=60
        )
    
    async def _cleanup_expired_commands(self) -> None:
        """Clean up expired commands"""
        
        now = datetime.now()
        expired_commands = []
        
        for command_id, command in self.active_commands.items():
            if command.expires_at and now > command.expires_at:
                expired_commands.append(command_id)
        
        for command_id in expired_commands:
            del self.active_commands[command_id]
            self.logger.info(f"Removed expired command {command_id}")
    
    async def _publish_coordinator_heartbeat(self) -> None:
        """Publish coordinator heartbeat"""
        
        # Publish every heartbeat interval
        if not hasattr(self, '_last_heartbeat') or \
           (datetime.now() - self._last_heartbeat).total_seconds() >= self.heartbeat_interval:
            
            heartbeat_data = {
                "coordinator_id": f"fleet-coordinator-{self.fleet_id}",
                "fleet_id": self.fleet_id,
                "status": "active",
                "fleet_status": asdict(self.fleet_status),
                "active_commands": len(self.active_commands),
                "timestamp": datetime.now().isoformat()
            }
            
            try:
                future = self.producer.send(
                    f"coordinator-heartbeat-{self.fleet_id}",
                    key=f"coordinator-{self.fleet_id}",
                    value=heartbeat_data
                )
                
                future.add_errback(lambda e: self.logger.error(f"Failed to send heartbeat: {e}"))
                
                self._last_heartbeat = datetime.now()
                
            except Exception as e:
                self.logger.error(f"Error publishing coordinator heartbeat: {e}")
    
    def add_command_callback(self, command_type: str, callback: Callable) -> None:
        """Add callback for specific command type"""
        
        if command_type not in self.command_callbacks:
            self.command_callbacks[command_type] = []
        
        self.command_callbacks[command_type].append(callback)
    
    def add_status_callback(self, callback: Callable) -> None:
        """Add callback for fleet status updates"""
        self.status_callbacks.append(callback)
    
    def get_fleet_status(self) -> Dict[str, Any]:
        """Get current fleet status"""
        return asdict(self.fleet_status)
    
    def get_vehicle_status(self, vehicle_id: str = None) -> Dict[str, Any]:
        """Get vehicle status"""
        
        if vehicle_id:
            return self.fleet_vehicles.get(vehicle_id, {})
        else:
            return self.fleet_vehicles.copy()
    
    def get_active_commands(self) -> List[Dict[str, Any]]:
        """Get active commands"""
        
        return [asdict(command) for command in self.active_commands.values()]
    
    def get_resource_utilization(self) -> Dict[str, Any]:
        """Get fleet resource utilization"""
        
        if not self.vehicle_capacities:
            return {"total_capacity": 0, "utilized_capacity": 0, "utilization_rate": 0}
        
        total_capacity = len(self.vehicle_capacities)
        utilized_capacity = sum(
            1 - capacity.available_capacity 
            for capacity in self.vehicle_capacities.values()
        )
        
        utilization_rate = utilized_capacity / total_capacity if total_capacity > 0 else 0
        
        return {
            "total_capacity": total_capacity,
            "utilized_capacity": utilized_capacity,
            "utilization_rate": utilization_rate,
            "available_capacity": total_capacity - utilized_capacity,
            "vehicle_breakdown": {
                vehicle_id: {
                    "available_capacity": capacity.available_capacity,
                    "current_load": capacity.current_mission_load
                }
                for vehicle_id, capacity in self.vehicle_capacities.items()
            }
        }