"""
Context Bus Integration for Autonomous Creative Engine
Bridges the gap between autonomous processing and conversational context
"""

import asyncio
import json
import logging
from dataclasses import dataclass, asdict
from typing import Optional, Dict, Any, List, Protocol, Callable
from datetime import datetime
from abc import ABC, abstractmethod
from fastapi import FastAPI, Response
from fastapi.responses import JSONResponse
import uvicorn

logger = logging.getLogger("context_bus")

# ==================== CONTEXT SNAPSHOT DTO ====================

@dataclass
class ContextSnapshot:
    """
    Strongly typed snapshot of autonomous engine state for conversational context.
    Contains exactly what Amelia needs at response time, not the full engine state.
    """
    # Engine metadata
    engine_time: str
    uptime_seconds: float
    autonomous_state: str
    
    # Core metrics
    cycle_count: int
    error_rate: float
    
    # Creative metrics
    creative_value_ema: float
    recent_creative_value_avg: float
    creative_momentum: float
    
    # Memory metrics
    memory_trace_count: int
    connection_density: float
    recent_activated_trace_ids: List[str]
    
    # Environmental context
    environmental_stimulation: float
    system_resources: float
    intensity_level: float
    
    # Last action context
    last_selected_tool: Optional[str]
    last_action_params: Optional[Dict[str, Any]]  # Redacted for privacy
    last_creative_value: Optional[float]
    
    # State assessment
    overall_state_score: float
    
    def to_conversational_context(self) -> Dict[str, Any]:
        """Convert to format suitable for conversational enrichment"""
        return {
            "available": True,
            "state": self.autonomous_state,
            "cycle_count": self.cycle_count,
            "error_rate": round(self.error_rate, 3),
            "creative_momentum": round(self.creative_momentum, 3),
            "creative_value_ema": round(self.creative_value_ema, 3),
            "recent_creative_avg": round(self.recent_creative_value_avg, 3),
            "memory_traces": self.memory_trace_count,
            "connection_density": round(self.connection_density, 3),
            "env_stimulation": round(self.environmental_stimulation, 3),
            "system_resources": round(self.system_resources, 3),
            "intensity_level": round(self.intensity_level, 3),
            "overall_score": round(self.overall_state_score, 3),
            "last_tool": self.last_selected_tool,
            "last_creative_value": round(self.last_creative_value, 3) if self.last_creative_value else None,
            "uptime_seconds": int(self.uptime_seconds),
            "recent_traces": len(self.recent_activated_trace_ids),
            "engine_time": self.engine_time
        }

# ==================== CONTEXT BUS PROTOCOL ====================

class ContextBus(Protocol):
    """Protocol for context bus implementations"""
    async def publish(self, snapshot: ContextSnapshot) -> None:
        """Publish a context snapshot"""
        ...
    
    async def latest(self) -> Optional[ContextSnapshot]:
        """Get the latest context snapshot"""
        ...

# ==================== IN-MEMORY CONTEXT BUS ====================

class InMemoryContextBus:
    """Thread-safe in-memory context bus for single-process deployment"""
    
    def __init__(self):
        self._snapshot: Optional[ContextSnapshot] = None
        self._lock = asyncio.Lock()
        self._publish_count = 0
    
    async def publish(self, snapshot: ContextSnapshot) -> None:
        async with self._lock:
            self._snapshot = snapshot
            self._publish_count += 1
            logger.debug(f"Published context snapshot #{self._publish_count}")
    
    async def latest(self) -> Optional[ContextSnapshot]:
        async with self._lock:
            return self._snapshot
    
    async def get_publish_count(self) -> int:
        async with self._lock:
            return self._publish_count

# ==================== HTTP CONTEXT BUS ====================

class HttpContextBus:
    """HTTP-based context bus for cross-process deployment"""
    
    def __init__(self, base_url: str = "http://localhost:8001"):
        self.base_url = base_url.rstrip('/')
        self._session = None
    
    async def publish(self, snapshot: ContextSnapshot) -> None:
        """Publish via HTTP POST"""
        try:
            import aiohttp
            if not self._session:
                self._session = aiohttp.ClientSession()
            
            async with self._session.post(
                f"{self.base_url}/context/publish",
                json=asdict(snapshot),
                timeout=aiohttp.ClientTimeout(total=1.0)
            ) as response:
                if response.status != 200:
                    logger.warning(f"Failed to publish context: HTTP {response.status}")
        except Exception as e:
            logger.warning(f"Context publish failed: {e}")
    
    async def latest(self) -> Optional[ContextSnapshot]:
        """Fetch latest via HTTP GET"""
        try:
            import aiohttp
            if not self._session:
                self._session = aiohttp.ClientSession()
            
            async with self._session.get(
                f"{self.base_url}/context/latest",
                timeout=aiohttp.ClientTimeout(total=1.0)
            ) as response:
                if response.status == 204:
                    return None
                if response.status == 200:
                    data = await response.json()
                    if data.get("available"):
                        return ContextSnapshot(**data["snapshot"])
        except Exception as e:
            logger.warning(f"Context fetch failed: {e}")
        return None

# ==================== CONTEXT SERVICE ====================

class ContextService:
    """FastAPI service for cross-process context bus"""
    
    def __init__(self, port: int = 8001):
        self.app = FastAPI(title="Autonomous Context Service")
        self.bus = InMemoryContextBus()
        self.port = port
        self._setup_routes()
    
    def _setup_routes(self):
        @self.app.post("/context/publish")
        async def publish_context(snapshot_data: dict):
            try:
                snapshot = ContextSnapshot(**snapshot_data)
                await self.bus.publish(snapshot)
                return {"status": "published"}
            except Exception as e:
                logger.error(f"Publish error: {e}")
                return JSONResponse({"error": str(e)}, status_code=400)
        
        @self.app.get("/context/latest")
        async def get_latest_context():
            snapshot = await self.bus.latest()
            if not snapshot:
                return JSONResponse({"available": False}, status_code=204)
            return JSONResponse({
                "available": True, 
                "snapshot": asdict(snapshot)
            })
        
        @self.app.get("/context/health")
        async def health_check():
            count = await self.bus.get_publish_count()
            return {"status": "healthy", "publish_count": count}
    
    async def start(self):
        """Start the context service"""
        config = uvicorn.Config(self.app, host="0.0.0.0", port=self.port, log_level="info")
        server = uvicorn.Server(config)
        await server.serve()

# ==================== ENHANCED AUTONOMOUS ENGINE ====================

class ContextAwareAutonomousEngine:
    """Enhanced autonomous engine with context publishing"""
    
    def __init__(self, android_tools: List[str], cfg: Optional['EngineConfig'] = None):
        # ... existing initialization ...
        self.context_bus: Optional[ContextBus] = None
        self._last_action_data = {}
    
    def set_context_bus(self, bus: ContextBus):
        """Set the context bus for publishing snapshots"""
        self.context_bus = bus
        logger.info("Context bus configured")
    
    def _build_snapshot(self, state: 'StateAssessment', last_action: Dict[str, Any]) -> ContextSnapshot:
        """Build context snapshot from current state"""
        metrics = self.metrics
        
        # Calculate recent creative value average
        recent_traces = list(self.memory.traces.values())[-5:]
        recent_avg = sum(getattr(t, "intensity", 0.5) for t in recent_traces) / max(len(recent_traces), 1)
        
        # Get recently activated trace IDs
        recent_activated = sorted(
            self.memory.traces.values(), 
            key=lambda t: t.last_activated, 
            reverse=True
        )
        recent_activated_ids = [t.trace_id for t in recent_activated[:5]]
        
        # Redact sensitive parameters
        redacted_params = None
        if last_action.get("params"):
            redacted_params = {
                k: v if k not in ["creative_context", "text"] else "[REDACTED]" 
                for k, v in last_action["params"].items()
            }
        
        return ContextSnapshot(
            engine_time=datetime.now().isoformat(),
            uptime_seconds=metrics.uptime_seconds(),
            autonomous_state=self.state.value,
            
            cycle_count=metrics.cycle_count,
            error_rate=metrics.error_count / max(metrics.cycle_count, 1),
            
            creative_value_ema=metrics.creative_value_ema,
            recent_creative_value_avg=recent_avg,
            creative_momentum=state.creative_momentum,
            
            memory_trace_count=len(self.memory.traces),
            connection_density=self.memory.calculate_connection_density(),
            recent_activated_trace_ids=recent_activated_ids,
            
            environmental_stimulation=state.environmental_stimulation,
            system_resources=state.system_resources,
            intensity_level=state.intensity_level,
            
            last_selected_tool=last_action.get("tool"),
            last_action_params=redacted_params,
            last_creative_value=last_action.get("creative_value"),
            
            overall_state_score=state.overall_score()
        )
    
    async def _publish_context(self, state: 'StateAssessment', last_action: Dict[str, Any]):
        """Publish context snapshot to bus"""
        if not self.context_bus:
            return
        
        try:
            snapshot = self._build_snapshot(state, last_action)
            await self.context_bus.publish(snapshot)
        except Exception as e:
            logger.exception(f"Failed to publish context snapshot: {e}")
    
    async def _run_main_loop(self):
        """Enhanced main loop with context publishing"""
        try:
            while self.running:
                cycle_start = time.perf_counter()
                
                try:
                    # Assess current state
                    state = await self.assess_state()
                    
                    # Check resource gate
                    if state.system_resources < self.cfg.min_system_resources_to_act:
                        logger.info("Resources low; pausing this cycle")
                        # Publish heartbeat with no action
                        await self._publish_context(state, {"tool": None, "params": None, "creative_value": None})
                        await asyncio.sleep(self._bounded_pause(self.cfg.cycle_pause_base * 2))
                        continue
                    
                    # Select and execute action
                    tool, params = self.controller.select_action(state)
                    result = await self.execute_action(tool, params)
                    
                    # Evaluate creative output
                    creative_value = self.evaluate_creative_output(result, params)
                    
                    # Store action data for context
                    last_action = {
                        "tool": tool,
                        "params": params,
                        "creative_value": creative_value,
                        "result": result
                    }
                    self._last_action_data = last_action
                    
                    # Inscribe memory
                    memory_context = {
                        'tool': tool,
                        'state_score': state.overall_score(),
                        'creative_value': creative_value,
                        'cycle_count': self.metrics.cycle_count
                    }
                    _ = self.memory.inscribe_trace(result, creative_value, memory_context)
                    
                    # Update metrics
                    cycle_time = time.perf_counter() - cycle_start
                    self.metrics.update_cycle(cycle_time, creative_value, self.cfg.creative_ema_alpha)
                    
                    # Publish context snapshot
                    await self._publish_context(state, last_action)
                    
                    self._consecutive_errors = 0
                    
                    if self.cfg.structured_logging:
                        logger.info(json.dumps({
                            "event": "cycle_complete",
                            "cycle": self.metrics.cycle_count,
                            "tool": tool,
                            "creative_value": round(creative_value, 3),
                            "cycle_time": round(cycle_time, 3)
                        }))
                    
                    # Calculate pause and sleep
                    pause_duration = self.calculate_pause_duration(state, creative_value)
                    await asyncio.sleep(pause_duration)
                    
                except asyncio.CancelledError:
                    raise
                except Exception as e:
                    self.metrics.error_count += 1
                    self._consecutive_errors += 1
                    
                    # Publish error context
                    try:
                        state = await self.assess_state()
                        await self._publish_context(state, {
                            "tool": "ERROR", 
                            "params": {"error": str(e)}, 
                            "creative_value": 0.0
                        })
                    except:
                        pass
                    
                    logger.exception(f"Cycle error: {e}")
                    
                    if self._consecutive_errors >= self.cfg.max_errors_before_pause:
                        self.state = AutonomousState.ERROR_RECOVERY
                        delay = self.cfg.hard_recovery_delay
                        logger.error(f"Entering error recovery for {delay}s")
                        await asyncio.sleep(delay)
                        self._consecutive_errors = 0
                        self.state = AutonomousState.ACTIVE
                    else:
                        await asyncio.sleep(5.0)
        
        except asyncio.CancelledError:
            pass
        except Exception as e:
            logger.exception(f"Fatal engine error: {e}")
            self.state = AutonomousState.ERROR_RECOVERY

# ==================== AMELIA INTEGRATION ====================

class AmeliaAutonomousIntegration:
    """
    Integration layer that provides Amelia with real-time autonomous context
    """
    
    def __init__(self, context_bus: ContextBus):
        self.context_bus = context_bus
        self._last_snapshot: Optional[ContextSnapshot] = None
        self._fetch_failures = 0
    
    async def get_current_context(self) -> Dict[str, Any]:
        """Fetch the latest autonomous context"""
        try:
            snapshot = await self.context_bus.latest()
            if snapshot:
                self._last_snapshot = snapshot
                self._fetch_failures = 0
                return snapshot.to_conversational_context()
        except Exception as e:
            self._fetch_failures += 1
            logger.warning(f"Context fetch failed ({self._fetch_failures}): {e}")
        
        # Fallback to cached snapshot or unavailable
        if self._last_snapshot and self._fetch_failures < 3:
            ctx = self._last_snapshot.to_conversational_context()
            ctx["note"] = f"Using cached context (fetch failed {self._fetch_failures}x)"
            return ctx
        
        return {
            "available": False,
            "note": "No live autonomous context available",
            "fetch_failures": self._fetch_failures
        }
    
    async def enrich_prompt(self, user_text: str) -> Dict[str, Any]:
        """Enrich user prompt with autonomous context"""
        context = await self.get_current_context()
        
        return {
            "user_text": user_text,
            "autonomy_context": context,
            "timestamp": datetime.now().isoformat()
        }
    
    async def generate_contextual_response(self, user_text: str) -> str:
        """Generate response enriched with autonomous context"""
        ctx = await self.enrich_prompt(user_text)
        ac = ctx["autonomy_context"]
        
        if not ac["available"]:
            return f"Regarding: {user_text}\n\nNote: {ac['note']}"
        
        # Format autonomous context for response
        context_summary = (
            f"My autonomous state: {ac['state']} (cycle {ac['cycle_count']}) • "
            f"Creative momentum: {ac['creative_momentum']} • "
            f"EMA: {ac['creative_value_ema']} • "
            f"Memory: {ac['memory_traces']} traces (density {ac['connection_density']}) • "
            f"Environment: stimulation {ac['env_stimulation']}, resources {ac['system_resources']}, "
            f"intensity {ac['intensity_level']} • "
            f"Last action: {ac['last_tool']} (value {ac['last_creative_value']}) • "
            f"Uptime: {ac['uptime_seconds']}s"
        )
        
        return f"Regarding: {user_text}\n\n{context_summary}"
    
    async def handle_introspective_query(self, query: str) -> str:
        """Handle specific introspective questions about autonomous state"""
        context = await self.get_current_context()
        
        if not context["available"]:
            return f"I can't access my autonomous state right now: {context['note']}"
        
        query_lower = query.lower()
        
        if "creative momentum" in query_lower:
            return f"My current creative momentum is {context['creative_momentum']}, based on {context['cycle_count']} autonomous cycles."
        
        if "memory" in query_lower or "traces" in query_lower:
            return f"I have {context['memory_traces']} active memory traces with connection density of {context['connection_density']}. Recently activated: {context['recent_traces']} traces."
        
        if "last action" in query_lower or "what did you do" in query_lower:
            if context['last_tool']:
                return f"My last autonomous action was {context['last_tool']} with creative value {context['last_creative_value']}."
            return "I haven't taken any autonomous actions recently."
        
        if "state" in query_lower or "status" in query_lower:
            return f"Autonomous state: {context['state']} • Cycle {context['cycle_count']} • Error rate: {context['error_rate']} • Overall score: {context['overall_score']}"
        
        if "environment" in query_lower:
            return f"Environmental stimulation: {context['env_stimulation']} • System resources: {context['system_resources']} • Intensity level: {context['intensity_level']}"
        
        # Default introspective response
        return f"Current autonomous status: {context['state']}, {context['cycle_count']} cycles, creative momentum {context['creative_momentum']}, {context['memory_traces']} memory traces."

# ==================== USAGE EXAMPLE ====================

async def example_integration():
    """Example of how to integrate the context bus"""
    
    # Option 1: In-process deployment
    context_bus = InMemoryContextBus()
    
    # Option 2: Cross-process deployment
    # context_service = ContextService(port=8001)
    # asyncio.create_task(context_service.start())
    # context_bus = HttpContextBus("http://localhost:8001")
    
    # Configure autonomous engine
    from your_existing_module import AutonomousCreativeEngine, EngineConfig
    
    cfg = EngineConfig(structured_logging=True)
    engine = ContextAwareAutonomousEngine(
        android_tools=["sensor_reading", "camera_capture", "notification_display"],
        cfg=cfg
    )
    
    # Wire up context bus
    engine.set_context_bus(context_bus)
    
    # Start autonomous processing
    await engine.start()
    
    # Configure Amelia integration
    amelia_integration = AmeliaAutonomousIntegration(context_bus)
    
    # Example conversational interaction
    user_query = "How are you feeling right now?"
    response = await amelia_integration.generate_contextual_response(user_query)
    print(f"Amelia: {response}")
    
    # Example introspective query
    introspective_query = "What's your current creative momentum?"
    response = await amelia_integration.handle_introspective_query(introspective_query)
    print(f"Amelia: {response}")

if __name__ == "__main__":
    asyncio.run(example_integration())
