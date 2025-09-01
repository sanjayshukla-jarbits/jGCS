"""
Streaming Mission Optimizer - Continuous optimization of missions via streaming data

This module provides:
- Real-time mission optimization
- Streaming data analysis
- Performance metric tracking
- Adaptive optimization strategies
"""

import asyncio
import numpy as np
from typing import Dict, List, Any, Optional, Callable
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from collections import deque
import json

from src.utils.logger import get_logger

@dataclass
class OptimizationMetric:
    """Optimization metric tracking"""
    name: str
    current_value: float
    target_value: float
    improvement_potential: float
    priority: float
    last_updated: datetime

@dataclass
class OptimizationSuggestion:
    """Optimization suggestion"""
    suggestion_id: str
    optimization_type: str
    expected_improvement: float
    implementation_cost: float
    risk_level: str  # "low", "medium", "high"
    description: str
    parameters: Dict[str, Any]
    created_at: datetime

class StreamingOptimizer:
    """Continuous mission optimization using streaming data"""
    
    def __init__(self):
        self.logger = get_logger(__name__)
        
        # Optimization state
        self.active_optimizations: Dict[str, Dict[str, Any]] = {}
        self.optimization_history: List[Dict[str, Any]] = []
        self.performance_metrics: Dict[str, OptimizationMetric] = {}
        
        # Streaming data buffers
        self.telemetry_buffer: deque = deque(maxlen=1000)
        self.performance_buffer: deque = deque(maxlen=500)
        self.environmental_buffer: deque = deque(maxlen=200)
        
        # Optimization strategies
        self.optimization_strategies = {
            "fuel_efficiency": self._optimize_fuel_efficiency,
            "time_optimization": self._optimize_time,
            "coverage_optimization": self._optimize_coverage,
            "safety_optimization": self._optimize_safety,
            "adaptive_optimization": self._adaptive_optimization
        }
        
        # Configuration
        self.optimization_interval = 30  # seconds
        self.min_data_points = 10
        self.confidence_threshold = 0.75
        
        # Callbacks
        self.optimization_callbacks: List[Callable] = []
        
        # Running state
        self.is_running = False
        self.optimization_task = None
    
    async def start_optimization(self) -> None:
        """Start continuous optimization process"""
        
        if self.is_running:
            return
        
        self.is_running = True
        self.optimization_task = asyncio.create_task(self._optimization_loop())
        self.logger.info("Started streaming optimization")
    
    async def stop_optimization(self) -> None:
        """Stop continuous optimization process"""
        
        self.is_running = False
        
        if self.optimization_task:
            self.optimization_task.cancel()
            try:
                await self.optimization_task
            except asyncio.CancelledError:
                pass
        
        self.logger.info("Stopped streaming optimization")
    
    async def _optimization_loop(self) -> None:
        """Main optimization loop"""
        
        while self.is_running:
            try:
                # Update performance metrics
                await self._update_performance_metrics()
                
                # Analyze optimization opportunities
                suggestions = await self._analyze_optimization_opportunities()
                
                # Process suggestions
                for suggestion in suggestions:
                    await self._process_optimization_suggestion(suggestion)
                
                # Clean up old data
                self._cleanup_old_data()
                
                # Wait for next iteration
                await asyncio.sleep(self.optimization_interval)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Error in optimization loop: {e}")
                await asyncio.sleep(5)  # Back off on error
    
    async def add_telemetry_data(self, vehicle_id: str, telemetry: Dict[str, Any]) -> None:
        """Add telemetry data to optimization analysis"""
        
        timestamped_data = {
            "timestamp": datetime.now(),
            "vehicle_id": vehicle_id,
            **telemetry
        }
        
        self.telemetry_buffer.append(timestamped_data)
    
    async def add_performance_data(self, vehicle_id: str, performance: Dict[str, Any]) -> None:
        """Add performance data to optimization analysis"""
        
        timestamped_data = {
            "timestamp": datetime.now(),
            "vehicle_id": vehicle_id,
            **performance
        }
        
        self.performance_buffer.append(timestamped_data)
    
    async def add_environmental_data(self, environmental: Dict[str, Any]) -> None:
        """Add environmental data to optimization analysis"""
        
        timestamped_data = {
            "timestamp": datetime.now(),
            **environmental
        }
        
        self.environmental_buffer.append(timestamped_data)
    
    async def _update_performance_metrics(self) -> None:
        """Update current performance metrics"""
        
        if len(self.performance_buffer) < self.min_data_points:
            return
        
        recent_data = list(self.performance_buffer)[-50:]  # Last 50 data points
        
        # Calculate fuel efficiency metric
        fuel_consumptions = [d.get("fuel_consumption", 0) for d in recent_data if "fuel_consumption" in d]
        if fuel_consumptions:
            avg_fuel_consumption = np.mean(fuel_consumptions)
            target_fuel_consumption = min(fuel_consumptions) * 1.1  # 10% above best
            
            self.performance_metrics["fuel_efficiency"] = OptimizationMetric(
                name="fuel_efficiency",
                current_value=avg_fuel_consumption,
                target_value=target_fuel_consumption,
                improvement_potential=max(0, (avg_fuel_consumption - target_fuel_consumption) / avg_fuel_consumption),
                priority=0.8,
                last_updated=datetime.now()
            )
        
        # Calculate time efficiency metric
        flight_times = [d.get("flight_time", 0) for d in recent_data if "flight_time" in d]
        if flight_times:
            avg_flight_time = np.mean(flight_times)
            target_flight_time = min(flight_times) * 1.05  # 5% above best
            
            self.performance_metrics["time_efficiency"] = OptimizationMetric(
                name="time_efficiency",
                current_value=avg_flight_time,
                target_value=target_flight_time,
                improvement_potential=max(0, (avg_flight_time - target_flight_time) / avg_flight_time),
                priority=0.7,
                last_updated=datetime.now()
            )
        
        # Calculate coverage efficiency metric
        coverage_rates = [d.get("coverage_rate", 0) for d in recent_data if "coverage_rate" in d]
        if coverage_rates:
            avg_coverage = np.mean(coverage_rates)
            target_coverage = max(coverage_rates) * 0.95  # 95% of best
            
            self.performance_metrics["coverage_efficiency"] = OptimizationMetric(
                name="coverage_efficiency",
                current_value=avg_coverage,
                target_value=target_coverage,
                improvement_potential=max(0, (target_coverage - avg_coverage) / target_coverage),
                priority=0.6,
                last_updated=datetime.now()
            )
    
    async def _analyze_optimization_opportunities(self) -> List[OptimizationSuggestion]:
        """Analyze current data for optimization opportunities"""
        
        suggestions = []
        
        # Analyze each performance metric
        for metric_name, metric in self.performance_metrics.items():
            if metric.improvement_potential > 0.1:  # 10% improvement potential
                
                if metric_name == "fuel_efficiency":
                    suggestions.extend(await self._analyze_fuel_optimization())
                
                elif metric_name == "time_efficiency":
                    suggestions.extend(await self._analyze_time_optimization())
                
                elif metric_name == "coverage_efficiency":
                    suggestions.extend(await self._analyze_coverage_optimization())
        
        # Analyze environmental conditions for opportunities
        suggestions.extend(await self._analyze_environmental_optimization())
        
        # Analyze flight patterns for opportunities
        suggestions.extend(await self._analyze_pattern_optimization())
        
        # Sort by expected improvement
        suggestions.sort(key=lambda s: s.expected_improvement, reverse=True)
        
        return suggestions
    
    async def _analyze_fuel_optimization(self) -> List[OptimizationSuggestion]:
        """Analyze fuel efficiency optimization opportunities"""
        
        suggestions = []
        
        if len(self.telemetry_buffer) < self.min_data_points:
            return suggestions
        
        recent_telemetry = list(self.telemetry_buffer)[-100:]
        
        # Analyze altitude efficiency
        altitude_fuel = {}
        for data in recent_telemetry:
            altitude = data.get("altitude", 0)
            fuel_rate = data.get("fuel_consumption_rate", 0)
            
            if altitude and fuel_rate:
                alt_bucket = int(altitude // 50) * 50  # Bucket by 50m
                if alt_bucket not in altitude_fuel:
                    altitude_fuel[alt_bucket] = []
                altitude_fuel[alt_bucket].append(fuel_rate)
        
        # Find most efficient altitude
        if len(altitude_fuel) > 2:
            avg_fuel_by_alt = {alt: np.mean(rates) for alt, rates in altitude_fuel.items()}
            best_altitude = min(avg_fuel_by_alt.keys(), key=avg_fuel_by_alt.get)
            current_avg_fuel = np.mean([rate for rates in altitude_fuel.values() for rate in rates])
            best_fuel_rate = avg_fuel_by_alt[best_altitude]
            
            if (current_avg_fuel - best_fuel_rate) / current_avg_fuel > 0.05:  # 5% improvement
                suggestions.append(OptimizationSuggestion(
                    suggestion_id=f"fuel_alt_{datetime.now().timestamp()}",
                    optimization_type="altitude_adjustment",
                    expected_improvement=(current_avg_fuel - best_fuel_rate) / current_avg_fuel,
                    implementation_cost=0.2,  # Low cost
                    risk_level="low",
                    description=f"Adjust cruise altitude to {best_altitude}m for better fuel efficiency",
                    parameters={"target_altitude": best_altitude, "fuel_savings": best_fuel_rate},
                    created_at=datetime.now()
                ))
        
        # Analyze speed efficiency
        speed_fuel = {}
        for data in recent_telemetry:
            speed = data.get("ground_speed", 0)
            fuel_rate = data.get("fuel_consumption_rate", 0)
            
            if speed and fuel_rate:
                speed_bucket = int(speed // 2) * 2  # Bucket by 2 m/s
                if speed_bucket not in speed_fuel:
                    speed_fuel[speed_bucket] = []
                speed_fuel[speed_bucket].append(fuel_rate)
        
        if len(speed_fuel) > 2:
            avg_fuel_by_speed = {speed: np.mean(rates) for speed, rates in speed_fuel.items()}
            best_speed = min(avg_fuel_by_speed.keys(), key=avg_fuel_by_speed.get)
            current_avg_fuel = np.mean([rate for rates in speed_fuel.values() for rate in rates])
            best_fuel_rate = avg_fuel_by_speed[best_speed]
            
            if (current_avg_fuel - best_fuel_rate) / current_avg_fuel > 0.03:  # 3% improvement
                suggestions.append(OptimizationSuggestion(
                    suggestion_id=f"fuel_speed_{datetime.now().timestamp()}",
                    optimization_type="speed_adjustment",
                    expected_improvement=(current_avg_fuel - best_fuel_rate) / current_avg_fuel,
                    implementation_cost=0.1,  # Very low cost
                    risk_level="low",
                    description=f"Adjust cruise speed to {best_speed}m/s for better fuel efficiency",
                    parameters={"target_speed": best_speed, "fuel_savings": best_fuel_rate},
                    created_at=datetime.now()
                ))
        
        return suggestions
    
    async def _analyze_time_optimization(self) -> List[OptimizationSuggestion]:
        """Analyze time efficiency optimization opportunities"""
        
        suggestions = []
        
        if len(self.performance_buffer) < self.min_data_points:
            return suggestions
        
        recent_performance = list(self.performance_buffer)[-50:]
        
        # Analyze waypoint transition times
        transition_times = [d.get("waypoint_transition_time", 0) for d in recent_performance 
                           if "waypoint_transition_time" in d]
        
        if transition_times:
            avg_transition = np.mean(transition_times)
            min_transition = min(transition_times)
            
            if (avg_transition - min_transition) / avg_transition > 0.15:  # 15% improvement potential
                suggestions.append(OptimizationSuggestion(
                    suggestion_id=f"time_transition_{datetime.now().timestamp()}",
                    optimization_type="waypoint_optimization",
                    expected_improvement=(avg_transition - min_transition) / avg_transition,
                    implementation_cost=0.3,  # Medium cost (requires replanning)
                    risk_level="low",
                    description="Optimize waypoint transitions to reduce flight time",
                    parameters={"target_transition_time": min_transition * 1.1},
                    created_at=datetime.now()
                ))
        
        # Analyze turn radius optimization
        turn_data = [d for d in recent_performance if "turn_radius" in d and "turn_time" in d]
        if len(turn_data) > 5:
            turn_efficiency = [(d["turn_time"] / d["turn_radius"]) for d in turn_data]
            avg_efficiency = np.mean(turn_efficiency)
            best_efficiency = min(turn_efficiency)
            
            if (avg_efficiency - best_efficiency) / avg_efficiency > 0.1:  # 10% improvement
                suggestions.append(OptimizationSuggestion(
                    suggestion_id=f"time_turns_{datetime.now().timestamp()}",
                    optimization_type="turn_optimization",
                    expected_improvement=(avg_efficiency - best_efficiency) / avg_efficiency,
                    implementation_cost=0.4,  # Medium-high cost
                    risk_level="medium",
                    description="Optimize turn profiles for faster transitions",
                    parameters={"target_turn_efficiency": best_efficiency},
                    created_at=datetime.now()
                ))
        
        return suggestions
    
    async def _analyze_coverage_optimization(self) -> List[OptimizationSuggestion]:
        """Analyze coverage optimization opportunities"""
        
        suggestions = []
        
        recent_performance = list(self.performance_buffer)[-50:]
        
        # Analyze overlap efficiency
        overlap_data = [d.get("coverage_overlap", 0) for d in recent_performance 
                       if "coverage_overlap" in d]
        
        if overlap_data:
            avg_overlap = np.mean(overlap_data)
            target_overlap = 0.75  # 75% target
            
            if abs(avg_overlap - target_overlap) > 0.1:  # 10% deviation
                if avg_overlap > target_overlap:
                    # Too much overlap
                    suggestions.append(OptimizationSuggestion(
                        suggestion_id=f"coverage_overlap_{datetime.now().timestamp()}",
                        optimization_type="reduce_overlap",
                        expected_improvement=(avg_overlap - target_overlap) * 0.5,
                        implementation_cost=0.3,
                        risk_level="low",
                        description=f"Reduce coverage overlap from {avg_overlap:.1%} to {target_overlap:.1%}",
                        parameters={"target_overlap": target_overlap, "current_overlap": avg_overlap},
                        created_at=datetime.now()
                    ))
                else:
                    # Too little overlap
                    suggestions.append(OptimizationSuggestion(
                        suggestion_id=f"coverage_overlap_{datetime.now().timestamp()}",
                        optimization_type="increase_overlap",
                        expected_improvement=(target_overlap - avg_overlap) * 0.3,
                        implementation_cost=0.4,
                        risk_level="medium",
                        description=f"Increase coverage overlap from {avg_overlap:.1%} to {target_overlap:.1%}",
                        parameters={"target_overlap": target_overlap, "current_overlap": avg_overlap},
                        created_at=datetime.now()
                    ))
        
        return suggestions
    
    async def _analyze_environmental_optimization(self) -> List[OptimizationSuggestion]:
        """Analyze environmental condition optimization opportunities"""
        
        suggestions = []
        
        if len(self.environmental_buffer) < self.min_data_points:
            return suggestions
        
        recent_env = list(self.environmental_buffer)[-20:]
        
        # Analyze wind utilization
        wind_data = [(d.get("wind_speed", 0), d.get("wind_direction", 0)) 
                    for d in recent_env if "wind_speed" in d]
        
        if wind_data:
            avg_wind_speed = np.mean([w[0] for w in wind_data])
            
            if avg_wind_speed > 5:  # Significant wind
                suggestions.append(OptimizationSuggestion(
                    suggestion_id=f"wind_optimization_{datetime.now().timestamp()}",
                    optimization_type="wind_utilization",
                    expected_improvement=0.08,  # 8% potential improvement
                    implementation_cost=0.5,    # High cost (route replanning)
                    risk_level="medium",
                    description=f"Optimize flight path to utilize {avg_wind_speed:.1f}m/s winds",
                    parameters={"wind_speed": avg_wind_speed, "wind_directions": [w[1] for w in wind_data]},
                    created_at=datetime.now()
                ))
        
        # Analyze thermal conditions
        temperature_data = [d.get("temperature", 20) for d in recent_env if "temperature" in d]
        if temperature_data:
            avg_temp = np.mean(temperature_data)
            
            # High temperature affects performance
            if avg_temp > 30:
                suggestions.append(OptimizationSuggestion(
                    suggestion_id=f"thermal_optimization_{datetime.now().timestamp()}",
                    optimization_type="thermal_adjustment", 
                    expected_improvement=0.05,  # 5% improvement
                    implementation_cost=0.2,
                    risk_level="low",
                    description=f"Adjust altitude for better performance in {avg_temp:.1f}°C conditions",
                    parameters={"temperature": avg_temp, "suggested_altitude_adjustment": 50},
                    created_at=datetime.now()
                ))
        
        return suggestions
    
    async def _analyze_pattern_optimization(self) -> List[OptimizationSuggestion]:
        """Analyze flight pattern optimization opportunities"""
        
        suggestions = []
        
        if len(self.telemetry_buffer) < 50:
            return suggestions
        
        recent_telemetry = list(self.telemetry_buffer)[-100:]
        
        # Analyze path efficiency
        positions = [(d.get("latitude", 0), d.get("longitude", 0)) 
                    for d in recent_telemetry if "latitude" in d]
        
        if len(positions) > 10:
            # Calculate path length
            total_distance = 0
            for i in range(1, len(positions)):
                # Simple distance calculation
                lat_diff = positions[i][0] - positions[i-1][0]
                lon_diff = positions[i][1] - positions[i-1][1]
                distance = (lat_diff**2 + lon_diff**2)**0.5 * 111320  # Rough conversion to meters
                total_distance += distance
            
            # Calculate direct distance
            if len(positions) >= 2:
                lat_diff = positions[-1][0] - positions[0][0]
                lon_diff = positions[-1][1] - positions[0][1]
                direct_distance = (lat_diff**2 + lon_diff**2)**0.5 * 111320
                
                if direct_distance > 0:
                    path_efficiency = direct_distance / total_distance
                    
                    if path_efficiency < 0.7:  # Less than 70% efficient
                        suggestions.append(OptimizationSuggestion(
                            suggestion_id=f"path_optimization_{datetime.now().timestamp()}",
                            optimization_type="path_smoothing",
                            expected_improvement=(0.8 - path_efficiency),  # Target 80% efficiency
                            implementation_cost=0.6,  # High cost (major replanning)
                            risk_level="medium",
                            description=f"Optimize flight path (current efficiency: {path_efficiency:.1%})",
                            parameters={"current_efficiency": path_efficiency, "target_efficiency": 0.8},
                            created_at=datetime.now()
                        ))
        
        return suggestions
    
    async def _process_optimization_suggestion(self, suggestion: OptimizationSuggestion) -> None:
        """Process an optimization suggestion"""
        
        # Check if suggestion meets criteria
        if suggestion.expected_improvement < 0.03:  # Less than 3% improvement
            return
        
        if suggestion.risk_level == "high" and suggestion.expected_improvement < 0.1:  # High risk needs >10% improvement
            return
        
        # Check if similar optimization is already active
        for active_opt in self.active_optimizations.values():
            if active_opt["optimization_type"] == suggestion.optimization_type:
                return  # Skip duplicate optimization type
        
        # Store optimization
        self.active_optimizations[suggestion.suggestion_id] = {
            "suggestion": asdict(suggestion),
            "status": "pending",
            "created_at": datetime.now()
        }
        
        # Notify callbacks
        for callback in self.optimization_callbacks:
            try:
                await callback(suggestion)
            except Exception as e:
                self.logger.error(f"Error in optimization callback: {e}")
        
        self.logger.info(f"Generated optimization suggestion: {suggestion.optimization_type}")
    
    def add_optimization_callback(self, callback: Callable) -> None:
        """Add callback for optimization suggestions"""
        self.optimization_callbacks.append(callback)
    
    def remove_optimization_callback(self, callback: Callable) -> None:
        """Remove optimization callback"""
        if callback in self.optimization_callbacks:
            self.optimization_callbacks.remove(callback)
    
    def get_optimization_status(self) -> Dict[str, Any]:
        """Get current optimization status"""
        
        return {
            "is_running": self.is_running,
            "active_optimizations": len(self.active_optimizations),
            "total_suggestions": len(self.optimization_history),
            "performance_metrics": {
                name: {
                    "current_value": metric.current_value,
                    "target_value": metric.target_value,
                    "improvement_potential": metric.improvement_potential
                }
                for name, metric in self.performance_metrics.items()
            },
            "buffer_sizes": {
                "telemetry": len(self.telemetry_buffer),
                "performance": len(self.performance_buffer),
                "environmental": len(self.environmental_buffer)
            }
        }
    
    def get_suggestions_history(self, hours: int = 24) -> List[Dict[str, Any]]:
        """Get optimization suggestions history"""
        
        cutoff_time = datetime.now() - timedelta(hours=hours)
        
        recent_suggestions = [
            opt for opt in self.optimization_history
            if datetime.fromisoformat(opt["created_at"]) > cutoff_time
        ]
        
        return recent_suggestions
    
    def _cleanup_old_data(self) -> None:
        """Clean up old optimization data"""
        
        # Clean up completed optimizations older than 24 hours
        cutoff_time = datetime.now() - timedelta(hours=24)
        
        completed_opts = []
        for opt_id, opt_data in self.active_optimizations.items():
            if (opt_data["status"] == "completed" and 
                opt_data["created_at"] < cutoff_time):
                completed_opts.append(opt_id)
        
        for opt_id in completed_opts:
            # Move to history
            opt_data = self.active_optimizations.pop(opt_id)
            self.optimization_history.append(opt_data)
        
        # Limit history size
        if len(self.optimization_history) > 1000:
            self.optimization_history = self.optimization_history[-500:]  # Keep last 500