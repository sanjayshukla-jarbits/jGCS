"""
Analytics Engine - Advanced analytics for swarm operations

This module provides:
- Mission performance analytics
- Swarm efficiency analysis
- Predictive analytics
- Resource optimization metrics
"""

import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass
from collections import defaultdict
import json

from src.utils.logger import get_logger

@dataclass
class MissionAnalytics:
    """Mission analytics results"""
    mission_id: str
    vehicles_involved: List[str]
    duration_seconds: float
    distance_covered: float
    efficiency_score: float
    coverage_percentage: float
    resource_utilization: float
    anomalies_detected: int
    cost_estimate: float

@dataclass
class SwarmMetrics:
    """Swarm-level metrics"""
    coordination_score: float
    formation_accuracy: float
    collision_avoidance_score: float
    communication_efficiency: float
    overall_performance: float

class AnalyticsEngine:
    """Advanced analytics engine for swarm operations"""
    
    def __init__(self):
        self.logger = get_logger(__name__)
        
        # Analytics data storage
        self.mission_data: Dict[str, Dict[str, Any]] = {}
        self.performance_history: List[Dict[str, Any]] = []
        self.swarm_metrics_history: List[SwarmMetrics] = []
        
        # Analytics models and parameters
        self.efficiency_weights = {
            "time": 0.3,
            "fuel": 0.25,
            "coverage": 0.25,
            "safety": 0.2
        }
        
        # Performance baselines
        self.performance_baselines = {
            "min_efficiency": 0.6,
            "target_coverage": 0.85,
            "max_mission_time": 3600,  # seconds
            "fuel_efficiency_target": 0.8
        }
    
    def analyze_mission_performance(self, mission_data: Dict[str, Any]) -> MissionAnalytics:
        """Analyze performance of completed mission"""
        
        mission_id = mission_data.get("mission_id", "unknown")
        vehicles = mission_data.get("vehicles_involved", [])
        
        # Calculate basic metrics
        duration = self._calculate_mission_duration(mission_data)
        distance = self._calculate_total_distance(mission_data)
        
        # Calculate efficiency score
        efficiency = self._calculate_efficiency_score(mission_data)
        
        # Calculate coverage
        coverage = self._calculate_coverage_percentage(mission_data)
        
        # Calculate resource utilization
        resource_util = self._calculate_resource_utilization(mission_data)
        
        # Detect anomalies
        anomalies = self._detect_mission_anomalies(mission_data)
        
        # Estimate cost
        cost = self._estimate_mission_cost(mission_data)
        
        analytics = MissionAnalytics(
            mission_id=mission_id,
            vehicles_involved=vehicles,
            duration_seconds=duration,
            distance_covered=distance,
            efficiency_score=efficiency,
            coverage_percentage=coverage,
            resource_utilization=resource_util,
            anomalies_detected=len(anomalies),
            cost_estimate=cost
        )
        
        # Store analytics
        self.mission_data[mission_id] = {
            "analytics": analytics,
            "raw_data": mission_data,
            "analyzed_at": datetime.now()
        }
        
        self.logger.info(f"Analyzed mission {mission_id}: efficiency={efficiency:.2f}, coverage={coverage:.2f}")
        
        return analytics
    
    def _calculate_mission_duration(self, mission_data: Dict[str, Any]) -> float:
        """Calculate total mission duration in seconds"""
        
        start_time = mission_data.get("start_time")
        end_time = mission_data.get("end_time")
        
        if not start_time or not end_time:
            # Fallback to telemetry data
            telemetry = mission_data.get("telemetry_data", [])
            if len(telemetry) >= 2:
                first_timestamp = datetime.fromisoformat(telemetry[0]["timestamp"])
                last_timestamp = datetime.fromisoformat(telemetry[-1]["timestamp"])
                return (last_timestamp - first_timestamp).total_seconds()
            return 0
        
        if isinstance(start_time, str):
            start_time = datetime.fromisoformat(start_time)
        if isinstance(end_time, str):
            end_time = datetime.fromisoformat(end_time)
        
        return (end_time - start_time).total_seconds()
    
    def _calculate_total_distance(self, mission_data: Dict[str, Any]) -> float:
        """Calculate total distance covered by all vehicles"""
        
        total_distance = 0
        
        for vehicle_data in mission_data.get("vehicle_data", []):
            vehicle_distance = 0
            waypoints = vehicle_data.get("waypoints", [])
            
            for i in range(len(waypoints) - 1):
                wp1 = waypoints[i]
                wp2 = waypoints[i + 1]
                
                # Simple distance calculation
                lat_diff = wp2.get("latitude", 0) - wp1.get("latitude", 0)
                lon_diff = wp2.get("longitude", 0) - wp1.get("longitude", 0)
                alt_diff = wp2.get("altitude", 0) - wp1.get("altitude", 0)
                
                segment_distance = ((lat_diff * 111320)**2 + (lon_diff * 111320)**2 + alt_diff**2)**0.5
                vehicle_distance += segment_distance
            
            total_distance += vehicle_distance
        
        return total_distance
    
    def _calculate_efficiency_score(self, mission_data: Dict[str, Any]) -> float:
        """Calculate overall mission efficiency score (0-1)"""
        
        # Time efficiency
        planned_duration = mission_data.get("planned_duration", 3600)
        actual_duration = self._calculate_mission_duration(mission_data)
        time_efficiency = min(1.0, planned_duration / max(actual_duration, 1))
        
        # Fuel efficiency
        fuel_efficiency = self._calculate_fuel_efficiency(mission_data)
        
        # Coverage efficiency
        coverage_efficiency = self._calculate_coverage_percentage(mission_data)
        
        # Safety efficiency (inverse of incidents)
        safety_incidents = len(self._detect_mission_anomalies(mission_data))
        safety_efficiency = max(0, 1.0 - safety_incidents * 0.1)
        
        # Weighted average
        efficiency = (
            time_efficiency * self.efficiency_weights["time"] +
            fuel_efficiency * self.efficiency_weights["fuel"] +
            coverage_efficiency * self.efficiency_weights["coverage"] +
            safety_efficiency * self.efficiency_weights["safety"]
        )
        
        return min(1.0, max(0.0, efficiency))
    
    def _calculate_fuel_efficiency(self, mission_data: Dict[str, Any]) -> float:
        """Calculate fuel efficiency score"""
        
        total_fuel_used = 0
        planned_fuel = 0
        
        for vehicle_data in mission_data.get("vehicle_data", []):
            fuel_used = vehicle_data.get("fuel_consumed", 0)
            planned_consumption = vehicle_data.get("planned_fuel_consumption", fuel_used * 1.2)
            
            total_fuel_used += fuel_used
            planned_fuel += planned_consumption
        
        if planned_fuel > 0:
            return min(1.0, planned_fuel / max(total_fuel_used, 1))
        
        return 0.8  # Default efficiency if no fuel data
    
    def _calculate_coverage_percentage(self, mission_data: Dict[str, Any]) -> float:
        """Calculate area coverage percentage"""
        
        planned_coverage = mission_data.get("planned_coverage_area", 0)
        actual_coverage = mission_data.get("actual_coverage_area", 0)
        
        if planned_coverage > 0:
            return min(1.0, actual_coverage / planned_coverage)
        
        # Fallback calculation based on waypoints
        coverage_data = mission_data.get("coverage_analysis", {})
        return coverage_data.get("coverage_percentage", 0.85)  # Default assumption
    
    def _calculate_resource_utilization(self, mission_data: Dict[str, Any]) -> float:
        """Calculate resource utilization efficiency"""
        
        total_vehicles = len(mission_data.get("vehicles_involved", []))
        if total_vehicles == 0:
            return 0
        
        # Calculate average vehicle utilization
        utilizations = []
        
        for vehicle_data in mission_data.get("vehicle_data", []):
            active_time = vehicle_data.get("active_time", 0)
            total_time = vehicle_data.get("total_time", 1)
            utilization = active_time / max(total_time, 1)
            utilizations.append(utilization)
        
        if utilizations:
            return np.mean(utilizations)
        
        return 0.7  # Default utilization
    
    def _detect_mission_anomalies(self, mission_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Detect anomalies in mission execution"""
        
        anomalies = []
        
        # Check for significant deviations from plan
        for vehicle_data in mission_data.get("vehicle_data", []):
            vehicle_id = vehicle_data.get("vehicle_id", "unknown")
            
            # Check for large detours
            planned_distance = vehicle_data.get("planned_distance", 0)
            actual_distance = vehicle_data.get("actual_distance", 0)
            
            if actual_distance > planned_distance * 1.3:  # 30% more than planned
                anomalies.append({
                    "type": "route_deviation",
                    "vehicle_id": vehicle_id,
                    "severity": "medium",
                    "details": f"Route 30% longer than planned"
                })
            
            # Check for speed anomalies
            avg_speed = vehicle_data.get("average_speed", 0)
            planned_speed = vehicle_data.get("planned_speed", avg_speed)
            
            if avg_speed < planned_speed * 0.7:  # 30% slower than planned
                anomalies.append({
                    "type": "speed_anomaly",
                    "vehicle_id": vehicle_id,
                    "severity": "low",
                    "details": f"Average speed 30% below planned"
                })
            
            # Check for altitude violations
            altitude_violations = vehicle_data.get("altitude_violations", 0)
            if altitude_violations > 0:
                anomalies.append({
                    "type": "altitude_violation",
                    "vehicle_id": vehicle_id,
                    "severity": "high",
                    "details": f"{altitude_violations} altitude violations"
                })
        
        return anomalies
    
    def _estimate_mission_cost(self, mission_data: Dict[str, Any]) -> float:
        """Estimate mission cost in arbitrary units"""
        
        base_cost = 100  # Base cost per mission
        
        # Vehicle hours
        duration_hours = self._calculate_mission_duration(mission_data) / 3600
        vehicle_count = len(mission_data.get("vehicles_involved", []))
        vehicle_cost = vehicle_count * duration_hours * 50  # 50 units per vehicle-hour
        
        # Fuel cost
        total_fuel = sum(
            vd.get("fuel_consumed", 0) 
            for vd in mission_data.get("vehicle_data", [])
        )
        fuel_cost = total_fuel * 2  # 2 units per fuel unit
        
        # Distance cost
        distance_km = self._calculate_total_distance(mission_data) / 1000
        distance_cost = distance_km * 5  # 5 units per km
        
        total_cost = base_cost + vehicle_cost + fuel_cost + distance_cost
        
        return round(total_cost, 2)
    
    def analyze_swarm_coordination(self, swarm_data: Dict[str, Any]) -> SwarmMetrics:
        """Analyze swarm coordination metrics"""
        
        # Coordination score based on formation maintenance
        coordination_score = self._calculate_coordination_score(swarm_data)
        
        # Formation accuracy
        formation_accuracy = self._calculate_formation_accuracy(swarm_data)
        
        # Collision avoidance effectiveness
        collision_score = self._calculate_collision_avoidance_score(swarm_data)
        
        # Communication efficiency
        comm_efficiency = self._calculate_communication_efficiency(swarm_data)
        
        # Overall performance
        overall = np.mean([coordination_score, formation_accuracy, collision_score, comm_efficiency])
        
        metrics = SwarmMetrics(
            coordination_score=coordination_score,
            formation_accuracy=formation_accuracy,
            collision_avoidance_score=collision_score,
            communication_efficiency=comm_efficiency,
            overall_performance=overall
        )
        
        self.swarm_metrics_history.append(metrics)
        
        return metrics
    
    def _calculate_coordination_score(self, swarm_data: Dict[str, Any]) -> float:
        """Calculate swarm coordination effectiveness"""
        
        formation_events = swarm_data.get("formation_events", [])
        successful_formations = len([e for e in formation_events if e.get("success", False)])
        
        if len(formation_events) == 0:
            return 1.0  # No formations attempted
        
        formation_success_rate = successful_formations / len(formation_events)
        
        # Factor in response times
        avg_response_time = swarm_data.get("average_response_time", 5)  # seconds
        response_score = max(0, 1.0 - (avg_response_time - 2) / 10)  # Penalize slow responses
        
        return (formation_success_rate + response_score) / 2
    
    def _calculate_formation_accuracy(self, swarm_data: Dict[str, Any]) -> float:
        """Calculate formation maintenance accuracy"""
        
        formation_deviations = swarm_data.get("formation_deviations", [])
        
        if not formation_deviations:
            return 1.0
        
        # Calculate average deviation from planned formation
        avg_deviation = np.mean(formation_deviations)
        target_tolerance = 10  # meters
        
        accuracy = max(0, 1.0 - avg_deviation / target_tolerance)
        
        return accuracy
    
    def _calculate_collision_avoidance_score(self, swarm_data: Dict[str, Any]) -> float:
        """Calculate collision avoidance effectiveness"""
        
        near_misses = swarm_data.get("near_misses", 0)
        collision_events = swarm_data.get("collision_events", 0)
        total_interactions = swarm_data.get("total_interactions", 1)
        
        # Severe penalty for actual collisions
        collision_penalty = collision_events * 0.5
        
        # Moderate penalty for near misses
        near_miss_penalty = near_misses * 0.1
        
        # Base score
        base_score = 1.0
        
        # Calculate final score
        score = base_score - collision_penalty - near_miss_penalty
        
        return max(0, min(1.0, score))
    
    def _calculate_communication_efficiency(self, swarm_data: Dict[str, Any]) -> float:
        """Calculate communication system efficiency"""
        
        message_success_rate = swarm_data.get("message_success_rate", 0.95)
        avg_latency = swarm_data.get("average_message_latency", 100)  # milliseconds
        
        # Latency score (target: <100ms)
        latency_score = max(0, 1.0 - (avg_latency - 50) / 200)
        
        # Combined score
        return (message_success_rate + latency_score) / 2
    
    def generate_performance_report(self, time_period_days: int = 30) -> Dict[str, Any]:
        """Generate comprehensive performance report"""
        
        cutoff_date = datetime.now() - timedelta(days=time_period_days)
        
        # Filter recent missions
        recent_missions = [
            data for data in self.mission_data.values()
            if data["analyzed_at"] > cutoff_date
        ]
        
        if not recent_missions:
            return {"error": "No missions found in specified time period"}
        
        # Calculate aggregate statistics
        analytics_list = [data["analytics"] for data in recent_missions]
        
        avg_efficiency = np.mean([a.efficiency_score for a in analytics_list])
        avg_coverage = np.mean([a.coverage_percentage for a in analytics_list])
        avg_duration = np.mean([a.duration_seconds for a in analytics_list])
        total_distance = sum(a.distance_covered for a in analytics_list)
        total_cost = sum(a.cost_estimate for a in analytics_list)
        total_anomalies = sum(a.anomalies_detected for a in analytics_list)
        
        # Performance trends
        efficiency_trend = self._calculate_trend([a.efficiency_score for a in analytics_list])
        coverage_trend = self._calculate_trend([a.coverage_percentage for a in analytics_list])
        
        # Vehicle utilization
        vehicle_usage = defaultdict(int)
        for analytics in analytics_list:
            for vehicle_id in analytics.vehicles_involved:
                vehicle_usage[vehicle_id] += 1
        
        # Swarm metrics
        recent_swarm_metrics = [m for m in self.swarm_metrics_history if len(self.swarm_metrics_history) > 0]
        avg_swarm_performance = np.mean([m.overall_performance for m in recent_swarm_metrics]) if recent_swarm_metrics else 0
        
        return {
            "report_period": f"{time_period_days} days",
            "missions_analyzed": len(recent_missions),
            "performance_summary": {
                "average_efficiency": round(avg_efficiency, 3),
                "average_coverage": round(avg_coverage, 3),
                "average_duration_minutes": round(avg_duration / 60, 1),
                "total_distance_km": round(total_distance / 1000, 1),
                "total_cost": round(total_cost, 2),
                "total_anomalies": total_anomalies
            },
            "performance_trends": {
                "efficiency_trend": efficiency_trend,
                "coverage_trend": coverage_trend
            },
            "vehicle_utilization": dict(vehicle_usage),
            "swarm_performance": {
                "average_coordination": round(avg_swarm_performance, 3),
                "metrics_available": len(recent_swarm_metrics)
            },
            "recommendations": self._generate_recommendations(analytics_list),
            "generated_at": datetime.now().isoformat()
        }
    
    def _calculate_trend(self, values: List[float]) -> str:
        """Calculate trend direction from list of values"""
        
        if len(values) < 3:
            return "insufficient_data"
        
        # Simple linear regression slope
        x = np.arange(len(values))
        slope = np.polyfit(x, values, 1)[0]
        
        if slope > 0.01:
            return "improving"
        elif slope < -0.01:
            return "declining"
        else:
            return "stable"
    
    def _generate_recommendations(self, analytics_list: List[MissionAnalytics]) -> List[str]:
        """Generate performance improvement recommendations"""
        
        recommendations = []
        
        # Efficiency recommendations
        avg_efficiency = np.mean([a.efficiency_score for a in analytics_list])
        if avg_efficiency < self.performance_baselines["min_efficiency"]:
            recommendations.append("Consider optimizing mission planning to improve overall efficiency")
        
        # Coverage recommendations
        avg_coverage = np.mean([a.coverage_percentage for a in analytics_list])
        if avg_coverage < self.performance_baselines["target_coverage"]:
            recommendations.append("Review survey patterns to improve area coverage")
        
        # Duration recommendations
        avg_duration = np.mean([a.duration_seconds for a in analytics_list])
        if avg_duration > self.performance_baselines["max_mission_time"]:
            recommendations.append("Consider breaking long missions into shorter segments")
        
        # Anomaly recommendations
        total_anomalies = sum(a.anomalies_detected for a in analytics_list)
        if total_anomalies > len(analytics_list) * 0.5:  # More than 0.5 anomalies per mission
            recommendations.append("Investigate frequent anomalies and improve mission planning")
        
        if not recommendations:
            recommendations.append("Performance is within acceptable ranges")
        
        return recommendations
    
    def get_mission_analytics(self, mission_id: str) -> Optional[MissionAnalytics]:
        """Get analytics for specific mission"""
        
        mission_data = self.mission_data.get(mission_id)
        return mission_data["analytics"] if mission_data else None
    
    def compare_missions(self, mission_ids: List[str]) -> Dict[str, Any]:
        """Compare analytics between multiple missions"""
        
        missions = []
        for mission_id in mission_ids:
            analytics = self.get_mission_analytics(mission_id)
            if analytics:
                missions.append(analytics)
        
        if len(missions) < 2:
            return {"error": "Need at least 2 valid missions for comparison"}
        
        comparison = {
            "missions_compared": len(missions),
            "efficiency_comparison": [m.efficiency_score for m in missions],
            "coverage_comparison": [m.coverage_percentage for m in missions],
            "duration_comparison": [m.duration_seconds for m in missions],
            "cost_comparison": [m.cost_estimate for m in missions],
            "best_efficiency": max(m.efficiency_score for m in missions),
            "best_coverage": max(m.coverage_percentage for m in missions),
            "most_efficient_mission": max(missions, key=lambda m: m.efficiency_score).mission_id
        }
        
        return comparison