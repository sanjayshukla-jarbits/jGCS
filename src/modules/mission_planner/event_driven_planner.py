"""
Event-driven Mission Planner - Reactive mission planning based on real-time events

This module provides:
- Event-driven mission updates
- Reactive planning triggers
- Context-aware mission adaptation
- Real-time mission state management
"""

import asyncio
import json
from typing import Dict, List, Any, Optional, Callable
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from enum import Enum
import uuid

from src.utils.logger import get_logger

class EventType(Enum):
    """Types of events that trigger mission replanning"""
    WEATHER_CHANGE = "weather_change"
    OBSTACLE_DETECTED = "obstacle_detected"
    VEHICLE_FAILURE = "vehicle_failure"
    BATTERY_LOW = "battery_low"
    GEOFENCE_BREACH = "geofence_breach"
    CONFLICT_DETECTED = "conflict_detected"
    MISSION_COMPLETE = "mission_complete"
    EMERGENCY = "emergency"
    OPTIMIZATION_OPPORTUNITY = "optimization_opportunity"
    USER_REQUEST = "user_request"

class PlanningStrategy(Enum):
    """Mission planning strategies"""
    CONSERVATIVE = "conservative"  # Prioritize safety
    AGGRESSIVE = "aggressive"      # Prioritize mission completion
    BALANCED = "balanced"          # Balance safety and efficiency
    ADAPTIVE = "adaptive"          # Adapt based on context

@dataclass
class PlanningEvent:
    """Event that triggers mission planning"""
    event_id: str
    event_type: EventType
    priority: int  # 1-10, 10 being highest
    source: str
    data: Dict[str, Any]
    timestamp: datetime
    expires_at: Optional[datetime] = None
    processed: bool = False

@dataclass
class PlanningContext:
    """Context for mission planning decisions"""
    vehicle_id: str
    current_mission: Optional[Dict[str, Any]]
    vehicle_state: Dict[str, Any]
    environmental_conditions: Dict[str, Any]
    constraints: Dict[str, Any]
    planning_strategy: PlanningStrategy
    historical_events: List[PlanningEvent]

@dataclass
class PlanningResponse:
    """Response from event-driven planning"""
    response_id: str
    trigger_event_id: str
    action_type: str  # "continue", "modify", "abort", "replan"
    new_mission: Optional[Dict[str, Any]]
    modifications: List[Dict[str, Any]]
    reasoning: str
    confidence: float  # 0-1
    estimated_impact: Dict[str, Any]

class EventDrivenPlanner:
    """Event-driven mission planner"""
    
    def __init__(self):
        self.logger = get_logger(__name__)
        
        # Event handling
        self.event_queue: List[PlanningEvent] = []
        self.event_handlers: Dict[EventType, List[Callable]] = {}
        self.planning_contexts: Dict[str, PlanningContext] = {}
        
        # Planning rules and strategies
        self.planning_rules = self._initialize_planning_rules()
        self.decision_tree = self._initialize_decision_tree()
        
        # State tracking
        self.active_missions: Dict[str, Dict[str, Any]] = {}
        self.planning_history: List[PlanningResponse] = []
        
        # Configuration
        self.max_queue_size = 1000
        self.event_expiry_hours = 24
        self.confidence_threshold = 0.7
        
    def _initialize_planning_rules(self) -> Dict[EventType, Dict[str, Any]]:
        """Initialize planning rules for different event types"""
        
        return {
            EventType.WEATHER_CHANGE: {
                "priority_multiplier": 1.5,
                "immediate_response": True,
                "safety_critical": True,
                "default_action": "modify_altitude_speed",
                "conditions": {
                    "wind_speed": {"threshold": 15, "action": "reduce_speed"},
                    "visibility": {"threshold": 1000, "action": "return_home"},
                    "precipitation": {"threshold": 5, "action": "land_immediately"}
                }
            },
            
            EventType.OBSTACLE_DETECTED: {
                "priority_multiplier": 2.0,
                "immediate_response": True,
                "safety_critical": True,
                "default_action": "avoid_obstacle",
                "conditions": {
                    "distance": {"threshold": 100, "action": "immediate_avoidance"},
                    "size": {"threshold": 50, "action": "path_modification"}
                }
            },
            
            EventType.BATTERY_LOW: {
                "priority_multiplier": 1.8,
                "immediate_response": True,
                "safety_critical": True,
                "default_action": "return_to_base",
                "conditions": {
                    "percentage": {"threshold": 20, "action": "immediate_rtb"},
                    "time_remaining": {"threshold": 5, "action": "emergency_land"}
                }
            },
            
            EventType.CONFLICT_DETECTED: {
                "priority_multiplier": 1.7,
                "immediate_response": True,
                "safety_critical": True,
                "default_action": "separation_maneuver",
                "conditions": {
                    "separation": {"threshold": 50, "action": "altitude_change"},
                    "time_to_conflict": {"threshold": 60, "action": "speed_adjustment"}
                }
            },
            
            EventType.GEOFENCE_BREACH: {
                "priority_multiplier": 1.9,
                "immediate_response": True,
                "safety_critical": True,
                "default_action": "return_to_safe_zone",
                "conditions": {
                    "breach_type": {"restricted": "immediate_exit", "altitude": "altitude_adjust"}
                }
            },
            
            EventType.OPTIMIZATION_OPPORTUNITY: {
                "priority_multiplier": 0.5,
                "immediate_response": False,
                "safety_critical": False,
                "default_action": "optimize_path",
                "conditions": {
                    "efficiency_gain": {"threshold": 10, "action": "replan_route"},
                    "time_savings": {"threshold": 5, "action": "speed_optimization"}
                }
            },
            
            EventType.EMERGENCY: {
                "priority_multiplier": 3.0,
                "immediate_response": True,
                "safety_critical": True,
                "default_action": "emergency_response",
                "conditions": {
                    "severity": {"critical": "land_immediately", "high": "return_home"}
                }
            }
        }
    
    def _initialize_decision_tree(self) -> Dict[str, Any]:
        """Initialize decision tree for planning responses"""
        
        return {
            "safety_critical": {
                "condition": "event.safety_critical",
                "true": {
                    "immediate": {
                        "condition": "event.immediate_response", 
                        "true": "execute_immediate_response",
                        "false": "plan_safety_response"
                    }
                },
                "false": {
                    "optimization": {
                        "condition": "event_type == OPTIMIZATION_OPPORTUNITY",
                        "true": "evaluate_optimization",
                        "false": "standard_planning"
                    }
                }
            }
        }
    
    async def process_event(self, event: PlanningEvent) -> Optional[PlanningResponse]:
        """Process a planning event and generate response"""
        
        try:
            self.logger.info(f"Processing planning event: {event.event_type} for {event.source}")
            
            # Add to queue if not already processed
            if not event.processed:
                await self._add_event_to_queue(event)
            
            # Get planning context
            vehicle_id = event.data.get("vehicle_id", event.source)
            context = await self._get_planning_context(vehicle_id, event)
            
            # Apply planning rules
            response = await self._generate_planning_response(event, context)
            
            if response:
                # Execute response
                await self._execute_planning_response(response)
                
                # Store in history
                self.planning_history.append(response)
                
                # Mark event as processed
                event.processed = True
                
                self.logger.info(f"Generated planning response: {response.action_type}")
                
            return response
            
        except Exception as e:
            self.logger.error(f"Error processing planning event: {e}")
            return None
    
    async def _add_event_to_queue(self, event: PlanningEvent) -> None:
        """Add event to processing queue"""
        
        # Check queue size
        if len(self.event_queue) >= self.max_queue_size:
            # Remove oldest non-critical events
            self.event_queue = [e for e in self.event_queue[-self.max_queue_size//2:] 
                              if e.priority >= 7]
        
        # Insert event in priority order
        inserted = False
        for i, queued_event in enumerate(self.event_queue):
            if event.priority > queued_event.priority:
                self.event_queue.insert(i, event)
                inserted = True
                break
        
        if not inserted:
            self.event_queue.append(event)
    
    async def _get_planning_context(self, vehicle_id: str, event: PlanningEvent) -> PlanningContext:
        """Get or create planning context for vehicle"""
        
        if vehicle_id not in self.planning_contexts:
            self.planning_contexts[vehicle_id] = PlanningContext(
                vehicle_id=vehicle_id,
                current_mission=self.active_missions.get(vehicle_id),
                vehicle_state={},
                environmental_conditions={},
                constraints={},
                planning_strategy=PlanningStrategy.BALANCED,
                historical_events=[]
            )
        
        context = self.planning_contexts[vehicle_id]
        
        # Update context with event data
        if "vehicle_state" in event.data:
            context.vehicle_state.update(event.data["vehicle_state"])
        
        if "environmental_conditions" in event.data:
            context.environmental_conditions.update(event.data["environmental_conditions"])
        
        # Add event to history
        context.historical_events.append(event)
        
        # Keep only recent events
        cutoff_time = datetime.now() - timedelta(hours=self.event_expiry_hours)
        context.historical_events = [
            e for e in context.historical_events 
            if e.timestamp > cutoff_time
        ]
        
        return context
    
    async def _generate_planning_response(self, event: PlanningEvent, 
                                        context: PlanningContext) -> Optional[PlanningResponse]:
        """Generate planning response based on event and context"""
        
        # Get planning rule for event type
        rule = self.planning_rules.get(event.event_type)
        if not rule:
            self.logger.warning(f"No planning rule for event type: {event.event_type}")
            return None
        
        # Calculate effective priority
        effective_priority = event.priority * rule["priority_multiplier"]
        
        # Determine action based on event data and rules
        action_type, modifications = await self._determine_action(event, context, rule)
        
        # Generate new mission if needed
        new_mission = None
        if action_type in ["replan", "modify"]:
            new_mission = await self._generate_modified_mission(context, modifications)
        
        # Calculate confidence
        confidence = self._calculate_confidence(event, context, action_type)
        
        # Estimate impact
        impact = await self._estimate_impact(event, context, action_type, modifications)
        
        # Generate reasoning
        reasoning = self._generate_reasoning(event, context, action_type, rule)
        
        return PlanningResponse(
            response_id=str(uuid.uuid4()),
            trigger_event_id=event.event_id,
            action_type=action_type,
            new_mission=new_mission,
            modifications=modifications,
            reasoning=reasoning,
            confidence=confidence,
            estimated_impact=impact
        )
    
    async def _determine_action(self, event: PlanningEvent, context: PlanningContext, 
                              rule: Dict[str, Any]) -> tuple[str, List[Dict[str, Any]]]:
        """Determine appropriate action based on event and context"""
        
        modifications = []
        
        # Safety-critical events get immediate response
        if rule["safety_critical"]:
            if event.event_type == EventType.EMERGENCY:
                return "abort", [{"type": "emergency_land", "reason": "emergency_event"}]
            
            elif event.event_type == EventType.WEATHER_CHANGE:
                weather_data = event.data.get("weather", {})
                
                if weather_data.get("wind_speed", 0) > 15:
                    modifications.append({
                        "type": "reduce_speed",
                        "factor": 0.7,
                        "reason": "high_wind"
                    })
                
                if weather_data.get("visibility", 10000) < 1000:
                    return "abort", [{"type": "return_home", "reason": "low_visibility"}]
                
                return "modify", modifications
            
            elif event.event_type == EventType.BATTERY_LOW:
                battery_level = event.data.get("battery_percentage", 100)
                
                if battery_level < 20:
                    return "abort", [{"type": "return_to_base", "reason": "low_battery"}]
                else:
                    return "modify", [{"type": "reduce_mission_scope", "reason": "battery_conservation"}]
            
            elif event.event_type == EventType.OBSTACLE_DETECTED:
                obstacle_data = event.data.get("obstacle", {})
                distance = obstacle_data.get("distance", 1000)
                
                if distance < 100:
                    modifications.append({
                        "type": "immediate_avoidance",
                        "obstacle_location": obstacle_data.get("location"),
                        "avoidance_distance": 150
                    })
                    return "modify", modifications
                
                return "replan", [{"type": "path_avoidance", "obstacle": obstacle_data}]
            
            elif event.event_type == EventType.CONFLICT_DETECTED:
                conflict_data = event.data.get("conflict", {})
                separation = conflict_data.get("min_distance", 1000)
                
                if separation < 50:
                    modifications.append({
                        "type": "altitude_separation",
                        "altitude_change": 50,
                        "reason": "collision_avoidance"
                    })
                    return "modify", modifications
                
                return "modify", [{"type": "speed_adjustment", "reason": "maintain_separation"}]
            
            elif event.event_type == EventType.GEOFENCE_BREACH:
                return "modify", [{"type": "return_to_safe_zone", "reason": "geofence_compliance"}]
        
        # Non-safety-critical events
        else:
            if event.event_type == EventType.OPTIMIZATION_OPPORTUNITY:
                optimization_data = event.data.get("optimization", {})
                potential_savings = optimization_data.get("efficiency_gain", 0)
                
                if potential_savings > 10:  # 10% efficiency gain
                    return "replan", [{"type": "route_optimization", "expected_gain": potential_savings}]
                
                return "continue", []
            
            elif event.event_type == EventType.USER_REQUEST:
                return "modify", [{"type": "user_modification", "request": event.data.get("request")}]
        
        # Default action
        return "continue", []
    
    async def _generate_modified_mission(self, context: PlanningContext, 
                                       modifications: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        """Generate modified mission based on modifications"""
        
        if not context.current_mission:
            return None
        
        modified_mission = context.current_mission.copy()
        
        for modification in modifications:
            mod_type = modification.get("type")
            
            if mod_type == "reduce_speed":
                factor = modification.get("factor", 0.8)
                waypoints = modified_mission.get("waypoints", [])
                
                for waypoint in waypoints:
                    if "speed" in waypoint:
                        waypoint["speed"] *= factor
            
            elif mod_type == "altitude_separation":
                altitude_change = modification.get("altitude_change", 20)
                waypoints = modified_mission.get("waypoints", [])
                
                for waypoint in waypoints:
                    waypoint["altitude"] = waypoint.get("altitude", 100) + altitude_change
            
            elif mod_type == "immediate_avoidance":
                # Add avoidance waypoint
                obstacle_location = modification.get("obstacle_location")
                avoidance_distance = modification.get("avoidance_distance", 100)
                
                if obstacle_location:
                    # Insert avoidance waypoint (simplified)
                    avoidance_waypoint = {
                        "latitude": obstacle_location[0] + 0.001,  # Offset north
                        "longitude": obstacle_location[1],
                        "altitude": modified_mission["waypoints"][0].get("altitude", 100) + 20,
                        "waypoint_type": "avoidance",
                        "speed": 8  # Slower speed for avoidance
                    }
                    
                    modified_mission["waypoints"].insert(1, avoidance_waypoint)
            
            elif mod_type == "reduce_mission_scope":
                # Remove some waypoints to conserve battery
                waypoints = modified_mission.get("waypoints", [])
                if len(waypoints) > 4:  # Keep takeoff, some mission points, landing
                    # Remove middle waypoints
                    keep_count = max(4, len(waypoints) // 2)
                    modified_mission["waypoints"] = waypoints[:keep_count//2] + waypoints[-keep_count//2:]
        
        # Update mission metadata
        modified_mission["modified_at"] = datetime.now().isoformat()
        modified_mission["modification_reason"] = [mod.get("reason", "unknown") for mod in modifications]
        
        return modified_mission
    
    def _calculate_confidence(self, event: PlanningEvent, context: PlanningContext, 
                            action_type: str) -> float:
        """Calculate confidence in planning decision"""
        
        base_confidence = 0.7  # Base confidence
        
        # Adjust based on event priority
        priority_factor = min(event.priority / 10.0, 1.0)
        confidence = base_confidence + (priority_factor * 0.2)
        
        # Adjust based on data quality
        if "confidence" in event.data:
            data_confidence = event.data["confidence"]
            confidence = (confidence + data_confidence) / 2
        
        # Adjust based on historical success
        recent_responses = [r for r in self.planning_history[-10:] 
                          if r.trigger_event_id == event.event_id]
        if recent_responses:
            avg_success = sum(r.confidence for r in recent_responses) / len(recent_responses)
            confidence = (confidence + avg_success) / 2
        
        # Safety-critical events get higher confidence
        rule = self.planning_rules.get(event.event_type, {})
        if rule.get("safety_critical", False):
            confidence = min(confidence + 0.1, 1.0)
        
        return round(confidence, 3)
    
    async def _estimate_impact(self, event: PlanningEvent, context: PlanningContext,
                             action_type: str, modifications: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Estimate impact of planning response"""
        
        impact = {
            "mission_completion": 1.0,  # Probability of completing mission
            "safety_improvement": 0.0,  # Safety improvement score
            "efficiency_change": 0.0,   # Efficiency change (positive is better)
            "time_impact": 0,           # Time change in seconds
            "resource_impact": 0        # Resource usage change
        }
        
        if action_type == "abort":
            impact["mission_completion"] = 0.0
            impact["safety_improvement"] = 0.8
            impact["time_impact"] = -3600  # Save time by aborting
        
        elif action_type == "modify":
            for mod in modifications:
                mod_type = mod.get("type")
                
                if mod_type == "reduce_speed":
                    impact["safety_improvement"] += 0.3
                    impact["efficiency_change"] -= 0.1
                    impact["time_impact"] += 300  # Extra 5 minutes
                
                elif mod_type == "altitude_separation":
                    impact["safety_improvement"] += 0.5
                    impact["efficiency_change"] -= 0.05
                
                elif mod_type == "immediate_avoidance":
                    impact["safety_improvement"] += 0.7
                    impact["efficiency_change"] -= 0.2
                    impact["time_impact"] += 180  # Extra 3 minutes
        
        elif action_type == "replan":
            impact["safety_improvement"] += 0.4
            impact["efficiency_change"] += 0.1  # Better planning
            impact["time_impact"] += 600  # Extra 10 minutes for replanning
        
        return impact
    
    def _generate_reasoning(self, event: PlanningEvent, context: PlanningContext,
                          action_type: str, rule: Dict[str, Any]) -> str:
        """Generate human-readable reasoning for planning decision"""
        
        reasoning_parts = []
        
        # Event description
        reasoning_parts.append(f"Event: {event.event_type.value} detected")
        
        # Priority explanation
        if event.priority >= 8:
            reasoning_parts.append("High priority event requiring immediate attention")
        elif event.priority >= 5:
            reasoning_parts.append("Medium priority event")
        else:
            reasoning_parts.append("Low priority event")
        
        # Safety consideration
        if rule.get("safety_critical", False):
            reasoning_parts.append("Safety-critical situation identified")
        
        # Action justification
        if action_type == "abort":
            reasoning_parts.append("Mission abort recommended due to safety concerns")
        elif action_type == "modify":
            reasoning_parts.append("Mission modification recommended to address event")
        elif action_type == "replan":
            reasoning_parts.append("Complete mission replanning recommended")
        else:
            reasoning_parts.append("No immediate action required, continuing mission")
        
        # Context factors
        if context.vehicle_state.get("battery_level", 100) < 30:
            reasoning_parts.append("Low battery level considered in decision")
        
        if context.environmental_conditions.get("wind_speed", 0) > 10:
            reasoning_parts.append("High wind conditions considered")
        
        return ". ".join(reasoning_parts) + "."
    
    async def _execute_planning_response(self, response: PlanningResponse) -> None:
        """Execute planning response"""
        
        try:
            # Update active mission if modified
            if response.new_mission and response.action_type in ["modify", "replan"]:
                vehicle_id = response.new_mission.get("vehicle_id")
                if vehicle_id:
                    self.active_missions[vehicle_id] = response.new_mission
            
            # Log execution
            self.logger.info(f"Executed planning response: {response.action_type}")
            
        except Exception as e:
            self.logger.error(f"Error executing planning response: {e}")
    
    def register_event_handler(self, event_type: EventType, handler: Callable) -> None:
        """Register event handler for specific event type"""
        
        if event_type not in self.event_handlers:
            self.event_handlers[event_type] = []
        
        self.event_handlers[event_type].append(handler)
    
    async def create_event(self, event_type: EventType, source: str, 
                         data: Dict[str, Any], priority: int = 5) -> PlanningEvent:
        """Create a new planning event"""
        
        event = PlanningEvent(
            event_id=str(uuid.uuid4()),
            event_type=event_type,
            priority=priority,
            source=source,
            data=data,
            timestamp=datetime.now(),
            expires_at=datetime.now() + timedelta(hours=self.event_expiry_hours)
        )
        
        return event
    
    def get_planning_statistics(self) -> Dict[str, Any]:
        """Get planning statistics"""
        
        total_events = len(self.planning_history)
        
        if total_events == 0:
            return {"total_events": 0}
        
        # Event type breakdown
        event_types = {}
        action_types = {}
        avg_confidence = 0
        
        for response in self.planning_history:
            # Count event types (would need to track trigger event type)
            action_type = response.action_type
            action_types[action_type] = action_types.get(action_type, 0) + 1
            avg_confidence += response.confidence
        
        avg_confidence /= total_events
        
        return {
            "total_events": total_events,
            "action_type_breakdown": action_types,
            "average_confidence": round(avg_confidence, 3),
            "active_missions": len(self.active_missions),
            "queue_size": len(self.event_queue)
        }
    
    def clear_expired_events(self) -> int:
        """Clear expired events from queue"""
        
        current_time = datetime.now()
        original_count = len(self.event_queue)
        
        self.event_queue = [
            event for event in self.event_queue
            if not event.expires_at or event.expires_at > current_time
        ]
        
        removed_count = original_count - len(self.event_queue)
        
        if removed_count > 0:
            self.logger.info(f"Removed {removed_count} expired events from queue")
        
        return removed_count