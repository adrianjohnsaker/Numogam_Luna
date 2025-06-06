"""
Enhanced Feedback Sensitivity Adjuster with improved performance, 
error handling, and additional features for AI girlfriend app integration.
"""

import json
import asyncio
import numpy as np
import traceback
import time
import threading
from pathlib import Path
from typing import Dict, List, Any, Optional, Union, Tuple, Callable
from dataclasses import dataclass, asdict, field
from enum import Enum
from collections import defaultdict, deque
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TrendDirection(Enum):
    """Enumeration for trend directions"""
    INCREASING = "increasing"
    DECREASING = "decreasing"
    STABLE = "stable"
    INSUFFICIENT_DATA = "insufficient_data"
    VOLATILE = "volatile"


class ZoneType(Enum):
    """Predefined zone types with descriptions"""
    CRITICAL = ("critical", 3.0, "High-priority feedback requiring immediate attention")
    IMPORTANT = ("important", 2.0, "Significant feedback that affects user experience")
    NORMAL = ("normal", 1.0, "Standard feedback for general interactions")
    MINOR = ("minor", 0.5, "Low-priority feedback for minor adjustments")
    EMOTIONAL = ("emotional", 2.5, "Emotional feedback requiring sensitive handling")
    LEARNING = ("learning", 1.8, "Educational feedback for system improvement")
    
    def __init__(self, zone_name: str, weight: float, description: str):
        self.zone_name = zone_name
        self.weight = weight
        self.description = description


@dataclass
class SensitivityConfig:
    """Enhanced configuration parameters for the Feedback Sensitivity Adjuster"""
    base_sensitivity: float = 1.0
    learning_rate: float = 0.05
    max_adjustment: float = 5.0
    min_adjustment: float = 0.1
    decay_factor: float = 0.98
    smoothing_window: int = 5
    auto_save_interval: float = 30.0  # seconds
    max_memory_mb: float = 50.0  # Maximum memory usage in MB
    adaptive_learning: bool = True  # Enable adaptive learning rate
    volatility_threshold: float = 0.8  # Threshold for detecting volatile patterns
    trend_sensitivity: float = 0.05  # Sensitivity for trend detection
    
    def __post_init__(self):
        """Validate configuration parameters"""
        if not 0.001 <= self.learning_rate <= 1.0:
            raise ValueError("Learning rate must be between 0.001 and 1.0")
        if not 0.1 <= self.max_adjustment <= 100.0:
            raise ValueError("Max adjustment must be between 0.1 and 100.0")
        if self.min_adjustment >= self.max_adjustment:
            raise ValueError("Min adjustment must be less than max adjustment")
        if self.smoothing_window < 1:
            raise ValueError("Smoothing window must be at least 1")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert the config to a dictionary"""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'SensitivityConfig':
        """Create config from dictionary"""
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


@dataclass
class FeedbackItem:
    """Enhanced feedback item with additional metadata"""
    timestamp: float
    value: float
    zone: str
    weighted_value: float
    adjustment: float
    user_id: Optional[str] = None
    context: Optional[Dict[str, Any]] = None
    confidence: float = 1.0
    source: str = "user"
    session_id: Optional[str] = None
    processed: bool = False
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'FeedbackItem':
        """Create from dictionary"""
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


class PerformanceMetrics:
    """Track performance metrics for the system"""
    
    def __init__(self):
        self.start_time = time.time()
        self.operation_counts = defaultdict(int)
        self.operation_times = defaultdict(list)
        self.memory_usage = deque(maxlen=100)
        self.error_counts = defaultdict(int)
        self._lock = threading.Lock()
    
    def record_operation(self, operation: str, duration: float):
        """Record operation performance"""
        with self._lock:
            self.operation_counts[operation] += 1
            self.operation_times[operation].append(duration)
            if len(self.operation_times[operation]) > 1000:
                self.operation_times[operation] = self.operation_times[operation][-500:]
    
    def record_error(self, error_type: str):
        """Record error occurrence"""
        with self._lock:
            self.error_counts[error_type] += 1
    
    def get_stats(self) -> Dict[str, Any]:
        """Get performance statistics"""
        with self._lock:
            stats = {
                "uptime_seconds": time.time() - self.start_time,
                "operation_counts": dict(self.operation_counts),
                "error_counts": dict(self.error_counts),
                "average_times": {}
            }
            
            for op, times in self.operation_times.items():
                if times:
                    stats["average_times"][op] = {
                        "mean": np.mean(times),
                        "min": np.min(times),
                        "max": np.max(times),
                        "std": np.std(times)
                    }
            
            return stats


class FeedbackSensitivityAdjuster:
    """
    Enhanced Feedback Sensitivity Adjuster with improved performance,
    error handling, and AI girlfriend app integration features.
    """
    
    def __init__(
        self,
        config: Optional[SensitivityConfig] = None,
        zone_weights: Optional[Dict[str, float]] = None,
        memory_capacity: int = 1000,
        auto_save_path: Optional[str] = None,
        enable_threading: bool = True
    ):
        """
        Initialize the Enhanced Feedback Sensitivity Adjuster.
        
        Args:
            config: Configuration parameters for sensitivity adjustment
            zone_weights: Dictionary mapping zones to their importance weights
            memory_capacity: Maximum number of feedback items to remember
            auto_save_path: Path for automatic state saving
            enable_threading: Whether to enable background processing
        """
        self.config = config or SensitivityConfig()
        self.zone_weights = zone_weights or self._get_default_zone_weights()
        self.memory_capacity = memory_capacity
        self.auto_save_path = auto_save_path
        self.enable_threading = enable_threading
        
        # Enhanced state management
        self.current_sensitivity: float = self.config.base_sensitivity
        self.feedback_history: deque = deque(maxlen=memory_capacity)
        self.adjustment_history: deque = deque(maxlen=memory_capacity)
        self.last_updated: float = time.time()
        self._cached_results: Dict[str, Tuple[Any, float]] = {}  # (result, timestamp)
        self.cache_ttl: float = 60.0  # Cache TTL in seconds
        
        # Performance tracking
        self.metrics = PerformanceMetrics()
        
        # Thread safety
        self._lock = threading.RLock()
        self._background_tasks: List[asyncio.Task] = []
        
        # Advanced features
        self.adaptive_learning_rate = self.config.learning_rate
        self.volatility_detector = VolatilityDetector()
        self.pattern_analyzer = PatternAnalyzer()
        
        # Auto-save functionality
        if self.auto_save_path and self.enable_threading:
            self._start_auto_save_task()
    
    def _get_default_zone_weights(self) -> Dict[str, float]:
        """Get default zone weights from ZoneType enum"""
        return {zone.zone_name: zone.weight for zone in ZoneType}
    
    def _start_auto_save_task(self):
        """Start background auto-save task"""
        if self.enable_threading:
            try:
                loop = asyncio.get_event_loop()
            except RuntimeError:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
            
            task = loop.create_task(self._auto_save_loop())
            self._background_tasks.append(task)
    
    async def _auto_save_loop(self):
        """Background auto-save loop"""
        while True:
            try:
                await asyncio.sleep(self.config.auto_save_interval)
                if self.auto_save_path:
                    await self.save_state_async(self.auto_save_path)
            except Exception as e:
                logger.error(f"Auto-save failed: {e}")
                self.metrics.record_error("auto_save_error")
    
    def _measure_performance(self, operation_name: str):
        """Decorator for measuring operation performance"""
        def decorator(func):
            def wrapper(*args, **kwargs):
                start_time = time.time()
                try:
                    result = func(*args, **kwargs)
                    duration = time.time() - start_time
                    self.metrics.record_operation(operation_name, duration)
                    return result
                except Exception as e:
                    self.metrics.record_error(f"{operation_name}_error")
                    raise
            return wrapper
        return decorator
    
    def to_json(self) -> str:
        """
        Convert the current state to JSON for Kotlin interoperability.
        
        Returns:
            JSON string representation of the current state
        """
        with self._lock:
            result = self._prepare_result_for_kotlin_bridge()
            return json.dumps(result, indent=2, ensure_ascii=False)
    
    @classmethod
    def from_json(cls, json_data: str, **kwargs) -> 'FeedbackSensitivityAdjuster':
        """
        Create a module instance from JSON data with enhanced error handling.
        
        Args:
            json_data: JSON string with module configuration
            **kwargs: Additional constructor arguments
            
        Returns:
            New module instance
        """
        try:
            data = json.loads(json_data)
            
            # Parse config with validation
            config_data = data.get("config", {})
            config = SensitivityConfig.from_dict(config_data)
            
            # Create instance with enhanced parameters
            instance = cls(
                config=config,
                zone_weights=data.get("zone_weights"),
                memory_capacity=data.get("memory_capacity", 1000),
                **kwargs
            )
            
            # Restore state if provided
            if "state_data" in data:
                instance._restore_state(data["state_data"])
            
            logger.info("Successfully created FeedbackSensitivityAdjuster from JSON")
            return instance
            
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON data: {e}")
        except Exception as e:
            raise ValueError(f"Failed to create module from JSON: {e}")
    
    def _restore_state(self, state_data: Dict[str, Any]):
        """Restore internal state from data"""
        with self._lock:
            self.current_sensitivity = state_data.get("current_sensitivity", self.config.base_sensitivity)
            self.last_updated = state_data.get("last_updated", time.time())
            
            # Restore feedback history
            history_data = state_data.get("feedback_history", [])
            self.feedback_history.clear()
            for item_data in history_data:
                try:
                    item = FeedbackItem.from_dict(item_data)
                    self.feedback_history.append(item)
                except Exception as e:
                    logger.warning(f"Failed to restore feedback item: {e}")
            
            # Restore adjustment history
            self.adjustment_history = deque(
                state_data.get("adjustment_history", []),
                maxlen=self.memory_capacity
            )
    
    def _prepare_result_for_kotlin_bridge(self) -> Dict[str, Any]:
        """
        Prepare results in a format optimized for Kotlin bridge transmission.
        
        Returns:
            Dictionary with results formatted for Kotlin
        """
        with self._lock:
            # Get simplified history for efficient transmission
            simplified_history = [
                {
                    "timestamp": item.timestamp,
                    "value": item.value,
                    "zone": item.zone,
                    "adjustment": item.adjustment,
                    "confidence": item.confidence
                }
                for item in list(self.feedback_history)[-50:]  # Last 50 items
            ]
            
            # Get performance metrics
            perf_stats = self.metrics.get_stats()
            
            return {
                "status": "success",
                "timestamp": time.time(),
                "config": self.config.to_dict(),
                "zone_weights": self.zone_weights,
                "memory_capacity": self.memory_capacity,
                "state_data": {
                    "current_sensitivity": self.current_sensitivity,
                    "adaptive_learning_rate": self.adaptive_learning_rate,
                    "feedback_history": simplified_history,
                    "adjustment_history": list(self.adjustment_history)[-50:],
                    "last_updated": self.last_updated,
                    "total_feedback_count": len(self.feedback_history)
                },
                "analytics": {
                    "performance_metrics": perf_stats,
                    "volatility_score": self.volatility_detector.get_current_volatility(),
                    "recent_patterns": self.pattern_analyzer.get_recent_patterns()
                },
                "metadata": {
                    "version": "2.0.0",
                    "zones": list(self.zone_weights.keys()),
                    "features": ["adaptive_learning", "volatility_detection", "pattern_analysis"],
                    "cache_stats": {
                        "cache_size": len(self._cached_results),
                        "cache_hit_rate": self._calculate_cache_hit_rate()
                    }
                }
            }
    
    def _calculate_cache_hit_rate(self) -> float:
        """Calculate cache hit rate"""
        hit_count = self.metrics.operation_counts.get("cache_hit", 0)
        miss_count = self.metrics.operation_counts.get("cache_miss", 0)
        total = hit_count + miss_count
        return hit_count / total if total > 0 else 0.0
    
    @_measure_performance("add_feedback")
    def add_feedback(self,
                    value: float,
                    zone: str = "normal",
                    user_id: Optional[str] = None,
                    context: Optional[Dict[str, Any]] = None,
                    confidence: float = 1.0,
                    source: str = "user",
                    session_id: Optional[str] = None) -> float:
        """
        Enhanced feedback addition with validation and advanced processing.
        
        Args:
            value: Feedback value (typically -1.0 to 1.0)
            zone: The feedback zone
            user_id: Optional user identifier
            context: Optional context information
            confidence: Confidence level (0.0 to 1.0)
            source: Source of feedback (user, system, auto)
            session_id: Optional session identifier
            
        Returns:
            New sensitivity value after adjustment
        """
        # Input validation
        if not isinstance(value, (int, float)):
            raise ValueError("Feedback value must be numeric")
        if not 0.0 <= confidence <= 1.0:
            raise ValueError("Confidence must be between 0.0 and 1.0")
        
        # Clamp value to reasonable range
        value = max(-10.0, min(10.0, float(value)))
        
        with self._lock:
            timestamp = time.time()
            
            # Apply zone weight with confidence adjustment
            zone_weight = self.zone_weights.get(zone, 1.0)
            weighted_value = value * zone_weight * confidence
            
            # Adaptive learning rate adjustment
            if self.config.adaptive_learning:
                volatility = self.volatility_detector.update(value)
                self.adaptive_learning_rate = self._adjust_learning_rate(volatility)
            
            # Calculate adjustment
            adjustment = weighted_value * self.adaptive_learning_rate
            
            # Apply adjustment with limits
            old_sensitivity = self.current_sensitivity
            self.current_sensitivity += adjustment
            self.current_sensitivity = max(
                self.config.min_adjustment,
                min(self.config.max_adjustment, self.current_sensitivity)
            )
            
            # Create enhanced feedback item
            feedback_item = FeedbackItem(
                timestamp=timestamp,
                value=value,
                zone=zone,
                weighted_value=weighted_value,
                adjustment=adjustment,
                user_id=user_id,
                context=context,
                confidence=confidence,
                source=source,
                session_id=session_id
            )
            
            # Store feedback
            self.feedback_history.append(feedback_item)
            self.adjustment_history.append(adjustment)
            self.last_updated = timestamp
            
            # Update pattern analyzer
            self.pattern_analyzer.add_data_point(value, zone, timestamp)
            
            # Clear relevant cache entries
            self._invalidate_cache()
            
            logger.debug(f"Added feedback: value={value}, zone={zone}, "
                        f"old_sensitivity={old_sensitivity:.3f}, "
                        f"new_sensitivity={self.current_sensitivity:.3f}")
            
            return self.current_sensitivity
    
    def _adjust_learning_rate(self, volatility: float) -> float:
        """Adjust learning rate based on volatility"""
        base_rate = self.config.learning_rate
        
        if volatility > self.config.volatility_threshold:
            # Reduce learning rate in volatile conditions
            return base_rate * 0.5
        elif volatility < 0.2:
            # Increase learning rate in stable conditions
            return base_rate * 1.5
        else:
            return base_rate
    
    def _invalidate_cache(self):
        """Invalidate cache entries"""
        current_time = time.time()
        # Remove expired entries
        expired_keys = [
            key for key, (_, timestamp) in self._cached_results.items()
            if current_time - timestamp > self.cache_ttl
        ]
        for key in expired_keys:
            del self._cached_results[key]
    
    def get_current_sensitivity(self) -> float:
        """Get the current sensitivity value with thread safety"""
        with self._lock:
            return self.current_sensitivity
    
    def reset_sensitivity(self) -> None:
        """Reset sensitivity to base value"""
        with self._lock:
            self.current_sensitivity = self.config.base_sensitivity
            self.adaptive_learning_rate = self.config.learning_rate
            self._cached_results.clear()
            logger.info("Sensitivity reset to base value")
    
    @_measure_performance("get_smoothed_sensitivity")
    def get_smoothed_sensitivity(self, method: str = "exponential") -> float:
        """
        Get sensitivity smoothed using various methods.
        
        Args:
            method: Smoothing method ("moving_average", "exponential", "median")
            
        Returns:
            Smoothed sensitivity value
        """
        cache_key = f"smoothed_sensitivity_{method}"
        
        # Check cache
        if cache_key in self._cached_results:
            result, timestamp = self._cached_results[cache_key]
            if time.time() - timestamp < self.cache_ttl:
                self.metrics.record_operation("cache_hit", 0)
                return result
        
        self.metrics.record_operation("cache_miss", 0)
        
        with self._lock:
            if not self.adjustment_history:
                return self.current_sensitivity
            
            window = min(self.config.smoothing_window, len(self.adjustment_history))
            recent_adjustments = list(self.adjustment_history)[-window:]
            
            if method == "moving_average":
                smoothed = self._moving_average_smooth(recent_adjustments)
            elif method == "exponential":
                smoothed = self._exponential_smooth(recent_adjustments)
            elif method == "median":
                smoothed = self._median_smooth(recent_adjustments)
            else:
                raise ValueError(f"Unknown smoothing method: {method}")
            
            # Cache result
            self._cached_results[cache_key] = (smoothed, time.time())
            
            return smoothed
    
    def _moving_average_smooth(self, adjustments: List[float]) -> float:
        """Apply moving average smoothing"""
        if not adjustments:
            return self.current_sensitivity
        
        avg_adjustment = sum(adjustments) / len(adjustments)
        smoothed = self.current_sensitivity - avg_adjustment * 0.3
        return max(self.config.min_adjustment,
                  min(self.config.max_adjustment, smoothed))
    
    def _exponential_smooth(self, adjustments: List[float]) -> float:
        """Apply exponential smoothing"""
        if not adjustments:
            return self.current_sensitivity
        
        alpha = 0.3  # Smoothing factor
        smoothed_adjustment = adjustments[0]
        
        for adj in adjustments[1:]:
            smoothed_adjustment = alpha * adj + (1 - alpha) * smoothed_adjustment
        
        smoothed = self.current_sensitivity - smoothed_adjustment * 0.5
        return max(self.config.min_adjustment,
                  min(self.config.max_adjustment, smoothed))
    
    def _median_smooth(self, adjustments: List[float]) -> float:
        """Apply median smoothing"""
        if not adjustments:
            return self.current_sensitivity
        
        median_adjustment = np.median(adjustments)
        smoothed = self.current_sensitivity - median_adjustment * 0.4
        return max(self.config.min_adjustment,
                  min(self.config.max_adjustment, smoothed))
    
    async def process_feedback_batch(self,
                                   feedbacks: List[Dict[str, Any]],
                                   max_concurrent: int = 10) -> Dict[str, Any]:
        """
        Process a batch of feedback items asynchronously with concurrency control.
        
        Args:
            feedbacks: List of feedback dictionaries
            max_concurrent: Maximum concurrent processing tasks
            
        Returns:
            Dictionary with processing results
        """
        if not feedbacks:
            return {"status": "success", "processed_count": 0}
        
        # Validate batch size
        if len(feedbacks) > 10000:
            raise ValueError("Batch size too large (max 10000)")
        
        semaphore = asyncio.Semaphore(max_concurrent)
        processed_results = []
        failed_items = []
        
        async def process_single_feedback(feedback_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
            async with semaphore:
                try:
                    # Extract parameters with defaults
                    value = feedback_data.get("value", 0.0)
                    zone = feedback_data.get("zone", "normal")
                    user_id = feedback_data.get("user_id")
                    context = feedback_data.get("context")
                    confidence = feedback_data.get("confidence", 1.0)
                    source = feedback_data.get("source", "user")
                    session_id = feedback_data.get("session_id")
                    
                    # Process feedback
                    new_sensitivity = self.add_feedback(
                        value, zone, user_id, context, confidence, source, session_id
                    )
                    
                    return {
                        "original_value": value,
                        "zone": zone,
                        "resulting_sensitivity": new_sensitivity,
                        "confidence": confidence,
                        "processed_at": time.time()
                    }
                    
                except Exception as e:
                    logger.error(f"Failed to process feedback item: {e}")
                    failed_items.append({"feedback": feedback_data, "error": str(e)})
                    return None
        
        # Process all feedbacks concurrently
        start_time = time.time()
        tasks = [process_single_feedback(fb) for fb in feedbacks]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Filter successful results
        processed_results = [r for r in results if r is not None and not isinstance(r, Exception)]
        processing_time = time.time() - start_time
        
        return {
            "status": "success",
            "processed_count": len(processed_results),
            "failed_count": len(failed_items),
            "processing_time_seconds": processing_time,
            "final_sensitivity": self.get_current_sensitivity(),
            "smoothed_sensitivity": self.get_smoothed_sensitivity(),
            "results": processed_results,
            "failed_items": failed_items[:10],  # Limit failed items for response size
            "performance": {
                "items_per_second": len(feedbacks) / processing_time if processing_time > 0 else 0,
                "average_item_time": processing_time / len(feedbacks) if feedbacks else 0
            }
        }
    
    @_measure_performance("analyze_feedback_patterns")
    def analyze_feedback_patterns(self,
                                zone: Optional[str] = None,
                                time_window_hours: Optional[float] = None,
                                include_predictions: bool = True) -> Dict[str, Any]:
        """
        Enhanced feedback pattern analysis with time windows and predictions.
        
        Args:
            zone: Optional zone to filter analysis by
            time_window_hours: Optional time window in hours
            include_predictions: Whether to include trend predictions
            
        Returns:
            Dictionary with comprehensive analysis results
        """
        cache_key = f"analysis_{zone}_{time_window_hours}_{include_predictions}"
        
        # Check cache
        if cache_key in self._cached_results:
            result, timestamp = self._cached_results[cache_key]
            if time.time() - timestamp < self.cache_ttl:
                return result
        
        with self._lock:
            if not self.feedback_history:
                return {"status": "no_data", "message": "No feedback data available"}
            
            # Filter data by time window
            current_time = time.time()
            data = list(self.feedback_history)
            
            if time_window_hours:
                cutoff_time = current_time - (time_window_hours * 3600)
                data = [item for item in data if item.timestamp >= cutoff_time]
            
            # Filter by zone if specified
            if zone:
                data = [item for item in data if item.zone == zone]
            
            if not data:
                result = {"status": "no_data_for_criteria", "zone": zone, "time_window_hours": time_window_hours}
                self._cached_results[cache_key] = (result, current_time)
                return result
            
            # Extract values and perform comprehensive analysis
            values = [item.value for item in data]
            timestamps = [item.timestamp for item in data]
            zones = [item.zone for item in data]
            confidences = [item.confidence for item in data]
            
            # Basic statistics
            analysis = {
                "status": "success",
                "data_points": len(values),
                "time_span_hours": (max(timestamps) - min(timestamps)) / 3600 if len(timestamps) > 1 else 0,
                "basic_stats": {
                    "mean": float(np.mean(values)),
                    "median": float(np.median(values)),
                    "std": float(np.std(values)),
                    "min": float(np.min(values)),
                    "max": float(np.max(values)),
                    "range": float(np.max(values) - np.min(values))
                },
                "confidence_stats": {
                    "mean_confidence": float(np.mean(confidences)),
                    "min_confidence": float(np.min(confidences)),
                    "low_confidence_count": sum(1 for c in confidences if c < 0.5)
                },
                "zone_distribution": self._analyze_zone_distribution(zones),
                "trend_analysis": self._enhanced_trend_analysis(values, timestamps),
                "volatility": {
                    "current_volatility": self.volatility_detector.get_current_volatility(),
                    "volatility_trend": self.volatility_detector.get_trend()
                }
            }
            
            # Add predictions if requested
            if include_predictions and len(values) >= 5:
                analysis["predictions"] = self._generate_predictions(values, timestamps)
            
            # Add pattern analysis
            analysis["patterns"] = self.pattern_analyzer.analyze_patterns(zone)
            
            # Cache result
            self._cached_results[cache_key] = (analysis, current_time)
            
            return analysis
    
    def _analyze_zone_distribution(self, zones: List[str]) -> Dict[str, Any]:
        """Analyze distribution of feedback across zones"""
        from collections import Counter
        zone_counts = Counter(zones)
        total = len(zones)
        
        return {
            "counts": dict(zone_counts),
            "percentages": {zone: (count / total) * 100 for zone, count in zone_counts.items()},
            "most_active_zone": zone_counts.most_common(1)[0][0] if zone_counts else None,
            "zone_diversity": len(zone_counts)  # Number of different zones used
        }
    
    def _enhanced_trend_analysis(self, values: List[float], timestamps: List[float]) -> Dict[str, Any]:
        """Enhanced trend analysis with multiple methods"""
        if len(values) < 2:
            return {"trend": TrendDirection.INSUFFICIENT_DATA.value}
        
        # Linear regression trend
        slope, r_squared = self._calculate_linear_trend(values, timestamps)
        
        # Recent vs older comparison
        mid_point = len(values) // 2
        older_mean = np.mean(values[:mid_point]) if mid_point > 0 else 0
        recent_mean = np.mean(values[mid_point:])
        
        # Volatility-aware trend classification
        volatility = np.std(values) / (np.mean(np.abs(values)) + 1e-8)
        
        if volatility > self.config.volatility_threshold:
            trend = TrendDirection.VOLATILE
        elif abs(slope) < self.config.trend_sensitivity:
            trend = TrendDirection.STABLE
        elif slope > 0:
            trend = TrendDirection.INCREASING
        else:
            trend = TrendDirection.DECREASING
        
        return {
            "trend": trend.value,
            "slope": slope,
            "r_squared": r_squared,
            "volatility": volatility,
            "recent_vs_older": {
                "recent_mean": recent_mean,
                "older_mean": older_mean,
                "change_percentage": ((recent_mean - older_mean) / (abs(older_mean) + 1e-8)) * 100,
                "improvement": "positive" if recent_mean > older_mean else "negative" if recent_mean < older_mean else "stable"
            },
            "confidence": min(r_squared, 1.0 - volatility),
            "trend_strength": abs(slope) * r_squared,
            "stability_score": 1.0 - volatility
        }
    
    def _calculate_linear_trend(self, values: List[float], timestamps: List[float]) -> Tuple[float, float]:
        """Calculate linear trend using least squares regression"""
        if len(values) < 2:
            return 0.0, 0.0
        
        # Normalize timestamps to avoid numerical issues
        t_norm = np.array(timestamps) - timestamps[0]
        y = np.array(values)
        
        # Calculate slope using least squares
        n = len(values)
        sum_t = np.sum(t_norm)
        sum_y = np.sum(y)
        sum_tt = np.sum(t_norm * t_norm)
        sum_ty = np.sum(t_norm * y)
        
        denominator = n * sum_tt - sum_t * sum_t
        if abs(denominator) < 1e-10:
            return 0.0, 0.0
        
        slope = (n * sum_ty - sum_t * sum_y) / denominator
        
        # Calculate R-squared
        y_mean = np.mean(y)
        ss_tot = np.sum((y - y_mean) ** 2)
        
        if ss_tot < 1e-10:
            r_squared = 1.0
        else:
            y_pred = slope * t_norm + (sum_y - slope * sum_t) / n
            ss_res = np.sum((y - y_pred) ** 2)
            r_squared = 1 - (ss_res / ss_tot)
        
        return slope, max(0.0, min(1.0, r_squared))
    
    def _generate_predictions(self, values: List[float], timestamps: List[float]) -> Dict[str, Any]:
        """Generate predictions for future values"""
        try:
            slope, r_squared = self._calculate_linear_trend(values, timestamps)
            
            current_time = timestamps[-1]
            current_value = values[-1]
            
            # Predict next few values (1 hour, 6 hours, 24 hours ahead)
            prediction_intervals = [3600, 21600, 86400]  # seconds
            predictions = []
            
            for interval in prediction_intervals:
                future_time = current_time + interval
                predicted_value = current_value + slope * interval
                
                # Add uncertainty based on historical volatility
                uncertainty = np.std(values[-10:]) if len(values) >= 10 else np.std(values)
                
                predictions.append({
                    "time_ahead_hours": interval / 3600,
                    "predicted_value": predicted_value,
                    "uncertainty": uncertainty,
                    "confidence": r_squared * (1.0 - min(uncertainty / (abs(predicted_value) + 1e-8), 1.0))
                })
            
            return {
                "method": "linear_extrapolation",
                "base_slope": slope,
                "base_r_squared": r_squared,
                "predictions": predictions
            }
            
        except Exception as e:
            logger.warning(f"Prediction generation failed: {e}")
            return {"method": "failed", "error": str(e)}
    
    async def save_state_async(self, filepath: str) -> bool:
        """Asynchronously save current state to file"""
        try:
            state_data = self.to_json()
            
            # Ensure directory exists
            Path(filepath).parent.mkdir(parents=True, exist_ok=True)
            
            # Write to temporary file first, then rename (atomic operation)
            temp_filepath = f"{filepath}.tmp"
            
            def write_file():
                with open(temp_filepath, 'w', encoding='utf-8') as f:
                    f.write(state_data)
                Path(temp_filepath).rename(filepath)
            
            # Run file I/O in thread pool to avoid blocking
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(None, write_file)
            
            logger.info(f"State saved successfully to {filepath}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to save state: {e}")
            self.metrics.record_error("save_state_error")
            return False
    
    async def load_state_async(self, filepath: str) -> bool:
        """Asynchronously load state from file"""
        try:
            def read_file():
                with open(filepath, 'r', encoding='utf-8') as f:
                    return f.read()
            
            # Run file I/O in thread pool
            loop = asyncio.get_event_loop()
            json_data = await loop.run_in_executor(None, read_file)
            
            # Parse and restore state
            data = json.loads(json_data)
            if "state_data" in data:
                self._restore_state(data["state_data"])
            
            logger.info(f"State loaded successfully from {filepath}")
            return True
            
        except FileNotFoundError:
            logger.warning(f"State file not found: {filepath}")
            return False
        except Exception as e:
            logger.error(f"Failed to load state: {e}")
            self.metrics.record_error("load_state_error")
            return False
    
    def get_zone_performance(self) -> Dict[str, Any]:
        """Analyze performance across different zones"""
        zone_stats = defaultdict(lambda: {"count": 0, "values": [], "adjustments": []})
        
        with self._lock:
            for item in self.feedback_history:
                zone = item.zone
                zone_stats[zone]["count"] += 1
                zone_stats[zone]["values"].append(item.value)
                zone_stats[zone]["adjustments"].append(item.adjustment)
        
        # Calculate statistics for each zone
        performance = {}
        for zone, stats in zone_stats.items():
            if stats["count"] > 0:
                values = stats["values"]
                adjustments = stats["adjustments"]
                
                performance[zone] = {
                    "feedback_count": stats["count"],
                    "average_value": np.mean(values),
                    "value_std": np.std(values),
                    "average_adjustment": np.mean(adjustments),
                    "total_adjustment": sum(adjustments),
                    "efficiency_score": abs(np.mean(adjustments)) / (np.std(adjustments) + 1e-8)
                }
        
        return performance
    
    def optimize_zone_weights(self, target_sensitivity: float = None) -> Dict[str, float]:
        """Optimize zone weights based on historical performance"""
        if target_sensitivity is None:
            target_sensitivity = self.config.base_sensitivity
        
        zone_performance = self.get_zone_performance()
        optimized_weights = self.zone_weights.copy()
        
        for zone, perf in zone_performance.items():
            if perf["feedback_count"] >= 5:  # Minimum data points
                # Adjust weight based on efficiency and impact
                efficiency = perf.get("efficiency_score", 1.0)
                current_weight = self.zone_weights.get(zone, 1.0)
                
                # Higher efficiency zones get slightly higher weights
                adjustment_factor = 1.0 + (efficiency - 1.0) * 0.1
                new_weight = current_weight * adjustment_factor
                
                # Keep weights within reasonable bounds
                optimized_weights[zone] = max(0.1, min(5.0, new_weight))
        
        return optimized_weights
    
    def safe_execute(self, function_name: str, **kwargs) -> Dict[str, Any]:
        """
        Safely execute a function with comprehensive error handling.
        
        Args:
            function_name: Name of the method to execute
            **kwargs: Arguments to pass to the method
            
        Returns:
            Dictionary with execution results or error information
        """
        start_time = time.time()
        
        try:
            if not hasattr(self, function_name):
                return {
                    "status": "error",
                    "error_type": "AttributeError",
                    "error_message": f"Function '{function_name}' not found",
                    "available_functions": [name for name in dir(self) 
                                          if not name.startswith('_') and callable(getattr(self, name))]
                }
            
            method = getattr(self, function_name)
            
            # Check if method is callable
            if not callable(method):
                return {
                    "status": "error",
                    "error_type": "TypeError",
                    "error_message": f"'{function_name}' is not callable"
                }
            
            # Execute the method
            result = method(**kwargs)
            execution_time = time.time() - start_time
            
            self.metrics.record_operation(f"safe_execute_{function_name}", execution_time)
            
            return {
                "status": "success",
                "function": function_name,
                "execution_time": execution_time,
                "data": result
            }
            
        except TypeError as e:
            return {
                "status": "error",
                "error_type": "TypeError",
                "error_message": f"Invalid arguments for {function_name}: {str(e)}",
                "function": function_name
            }
        except Exception as e:
            self.metrics.record_error(f"safe_execute_{function_name}_error")
            return {
                "status": "error",
                "error_type": type(e).__name__,
                "error_message": str(e),
                "function": function_name,
                "traceback": traceback.format_exc()
            }
    
    def clear_history(self, keep_recent: int = 10) -> Dict[str, Any]:
        """
        Clear feedback history while optionally keeping recent items.
        
        Args:
            keep_recent: Number of recent items to keep
            
        Returns:
            Dictionary with operation results
        """
        with self._lock:
            original_count = len(self.feedback_history)
            
            if keep_recent > 0 and self.feedback_history:
                # Keep the most recent items
                recent_items = list(self.feedback_history)[-keep_recent:]
                self.feedback_history.clear()
                self.feedback_history.extend(recent_items)
                
                recent_adjustments = list(self.adjustment_history)[-keep_recent:]
                self.adjustment_history.clear()
                self.adjustment_history.extend(recent_adjustments)
            else:
                self.feedback_history.clear()
                self.adjustment_history.clear()
            
            # Clear cache
            self._cached_results.clear()
            
            final_count = len(self.feedback_history)
            
            logger.info(f"History cleared: {original_count} -> {final_count} items")
            
            return {
                "status": "success",
                "original_count": original_count,
                "final_count": final_count,
                "items_removed": original_count - final_count,
                "items_kept": final_count
            }
    
    def get_system_health(self) -> Dict[str, Any]:
        """Get comprehensive system health metrics"""
        with self._lock:
            current_time = time.time()
            
            # Calculate memory usage estimate
            history_size = len(self.feedback_history)
            cache_size = len(self._cached_results)
            
            # Get recent activity
            recent_feedback = sum(1 for item in self.feedback_history 
                                if current_time - item.timestamp < 3600)  # Last hour
            
            # Calculate average processing time
            perf_stats = self.metrics.get_stats()
            avg_processing_time = 0
            if "add_feedback" in perf_stats.get("average_times", {}):
                avg_processing_time = perf_stats["average_times"]["add_feedback"]["mean"]
            
            return {
                "status": "healthy" if history_size < self.memory_capacity * 0.9 else "warning",
                "uptime_seconds": current_time - (self.last_updated if hasattr(self, 'start_time') else current_time),
                "memory_usage": {
                    "feedback_history_size": history_size,
                    "memory_capacity": self.memory_capacity,
                    "usage_percentage": (history_size / self.memory_capacity) * 100,
                    "cache_size": cache_size
                },
                "activity": {
                    "total_feedback_items": history_size,
                    "recent_feedback_count": recent_feedback,
                    "last_updated": self.last_updated
                },
                "performance": {
                    "average_processing_time_ms": avg_processing_time * 1000,
                    "current_sensitivity": self.current_sensitivity,
                    "adaptive_learning_rate": getattr(self, 'adaptive_learning_rate', self.config.learning_rate),
                    "volatility_score": self.volatility_detector.get_current_volatility()
                },
                "errors": dict(self.metrics.error_counts) if self.metrics.error_counts else {}
            }
    
    def export_data(self, format_type: str = "json", include_full_history: bool = False) -> str:
        """Export data in various formats"""
        with self._lock:
            if format_type.lower() == "json":
                if include_full_history:
                    return self.to_json()
                else:
                    # Export summarized data
                    summary = {
                        "current_state": {
                            "sensitivity": self.current_sensitivity,
                            "last_updated": self.last_updated,
                            "total_feedback_count": len(self.feedback_history)
                        },
                        "configuration": self.config.to_dict(),
                        "zone_weights": self.zone_weights,
                        "recent_patterns": self.pattern_analyzer.get_recent_patterns(),
                        "performance_summary": self.get_zone_performance()
                    }
                    return json.dumps(summary, indent=2)
            
            elif format_type.lower() == "csv":
                # Export feedback history as CSV
                if not self.feedback_history:
                    return "timestamp,value,zone,adjustment,user_id,confidence\n"
                
                csv_lines = ["timestamp,value,zone,adjustment,user_id,confidence"]
                for item in self.feedback_history:
                    line = f"{item.timestamp},{item.value},{item.zone},{item.adjustment},{item.user_id or ''},{item.confidence}"
                    csv_lines.append(line)
                
                return "\n".join(csv_lines)
            
            else:
                raise ValueError(f"Unsupported format: {format_type}")
    
    def __enter__(self):
        """Context manager entry"""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        self.cleanup()
```
    
    def cleanup(self) -> None:
        """Release resources and stop background tasks"""
        # Cancel background tasks
        for task in self._background_tasks:
            if not task.done():
                task.cancel()
        
        # Clear data structures
        with self._lock:
            self.feedback_history.clear()
            self.adjustment_history.clear()
            self._cached_results.clear()
        
        logger.info("FeedbackSensitivityAdjuster cleanup completed")


# Example usage
if __name__ == "__main__":
    async def main():
        """Main example function"""
        print("🚀 Enhanced Feedback Sensitivity Adjuster Demo")
        
        config = SensitivityConfig(
            base_sensitivity=1.5,
            learning_rate=0.08,
            max_adjustment=6.0,
            adaptive_learning=True
        )
        
        with FeedbackSensitivityAdjuster(config=config) as adjuster:
            # Add feedback
            adjuster.add_feedback(0.8, zone="emotional", user_id="user123", confidence=0.9)
            adjuster.add_feedback(-0.3, zone="minor", user_id="user123", confidence=0.7)
            
            # Analyze patterns
            patterns = adjuster.analyze_feedback_patterns(include_predictions=True)
            print(f"Analysis: {patterns}")
            
            print("✅ Demo completed!")
    
    # Run the demo
    asyncio.run(main())
