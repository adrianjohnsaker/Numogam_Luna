============================================================================
# INTEGRATION PATCH 4: Monitoring Interface Module (bridge_monitor.py)
# =============================================================================
"""
Create this as a new file: bridge_monitor.py
Provides CLI and programmatic interface for monitoring dialogic bridge
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
bridge_monitor.py
-----------------
Monitoring interface for Dialogic Feedback Bridge

Provides real-time monitoring, diagnostics, and control interface
for the conversational → autonomous consciousness feedback loop.
"""

import os
import sys
import json
import time
from typing import Dict, Any, Optional

try:
    from dialogic_feedback_bridge import (
        get_bridge_status,
        generate_bridge_report,
        force_rim_sync,
        configure_bridge,
        CONFIG as BRIDGE_CONFIG
    )
    BRIDGE_AVAILABLE = True
except ImportError:
    BRIDGE_AVAILABLE = False
    print("ERROR: dialogic_feedback_bridge module not available")
    sys.exit(1)

try:
    import pipeline
    PIPELINE_AVAILABLE = True
except ImportError:
    PIPELINE_AVAILABLE = False


# =============================================================================
# Monitoring Functions
# =============================================================================EVOLUTIONARY

def monitor_realtime(interval_seconds: int = 5, duration_seconds: Optional[int] = None):
    """
    Real-time monitoring of bridge activity.
    
    Args:
        interval_seconds: Refresh interval
        duration_seconds: Total monitoring duration (None = infinite)
    """
    print("=" * 70)
    print("DIALOGIC BRIDGE REAL-TIME MONITOR")
    print("=" * 70)
    print(f"Refresh interval: {interval_seconds}s")
    print(f"Duration: {'Infinite (Ctrl+C to stop)' if duration_seconds is None else f'{duration_seconds}s'}")
    print("=" * 70)
    print()
    
    start_time = time.time()
    iteration = 0
    
    try:
        while True:
            iteration += 1
            elapsed = time.time() - start_time
            
            # Check duration limit
            if duration_seconds and elapsed >= duration_seconds:
                print("\nMonitoring duration reached. Exiting.")
                break
            
            # Clear screen (works on most terminals)
            os.system('clear' if os.name != 'nt' else 'cls')
            
            print(f"DIALOGIC BRIDGE MONITOR - Iteration {iteration} ({elapsed:.1f}s elapsed)")
            print("=" * 70)
            
            # Get status
            status = get_bridge_status()
            
            # Display key metrics
            print(f"\nStatus: {'ENABLED' if status['enabled'] else 'DISABLED'}")
            print(f"Total Exchanges: {status['total_exchanges']}")
            print(f"Recent History: {status['recent_exchanges']} exchanges")
            print(f"\nTime Since Last RIM Update: {status['time_since_update_ms'] / 1000:.1f}s")
            print(f"Next RIM Update In: {status['next_update_in_ms'] / 1000:.1f}s")
            
            print("\nCUMULATIVE DELTAS (pending):")
            print("-" * 70)
            for param, delta in status['cumulative_deltas'].items():
                bar_length = int(abs(delta) * 50)
                bar = "█" * bar_length
                direction = "→" if delta == 0 else "↑" if delta > 0 else "↓"
                print(f"  {param:25s} {direction} {delta:+.4f} {bar}")
            
            print("\n" + "=" * 70)
            print(f"Press Ctrl+C to stop | Refreshing in {interval_seconds}s...")
            
            time.sleep(interval_seconds)
    
    except KeyboardInterrupt:
        print("\n\nMonitoring stopped by user.")


def show_detailed_status():
    """Display detailed bridge status."""
    report = generate_bridge_report()
    print(report)
    
    # Add pipeline integration status
    if PIPELINE_AVAILABLE:
        print("\nPIPELINE INTEGRATION:")
        print("-" * 60)
        pipeline_status = pipeline.get_dialogic_bridge_status()
        print(json.dumps(pipeline_status, indent=2))


def sync_now():
    """Force immediate RIM synchronization."""
    print("Forcing RIM synchronization...")
    result = force_rim_sync()
    
    if result.get("status") == "applied":
        print("✓ RIM sync successful!")
        print("\nDeltas applied:")
        for param, delta in result.get("deltas", {}).items():
            print(f"  {param}: {delta:+.4f}")
        
        print("\nNew RIM values:")
        for param, value in result.get("new_rim", {}).items():
            print(f"  {param}: {value:.4f}")
    
    elif result.get("status") == "no_deltas":
        print("⚠ No deltas to apply")
    
    else:
        print(f"✗ Sync failed: {result.get('status')}")
        if "error" in result:
            print(f"  Error: {result['error']}")


def configure_interactive():
    """Interactive configuration wizard."""
    print("=" * 70)
    print("DIALOGIC BRIDGE CONFIGURATION")
    print("=" * 70)
    
    # Get current config
    status = get_bridge_status()
    current = status.get("config", {})
    
    print("\nCurrent Configuration:")
    print(json.dumps(current, indent=2))
    print()
    
    # Prompt for changes
    print("Enter new values (press Enter to keep current):")
    print()
    
    # Enabled
    enabled_input = input(f"Enabled [{current.get('enabled')}]: ").strip().lower()
    enabled = None
    if enabled_input:
        enabled = enabled_input in ['true', 'yes', '1', 'on']
    
    # Feedback threshold
    threshold_input = input(f"Feedback Threshold [{current.get('feedback_threshold')}]: ").strip()
    feedback_threshold = None
    if threshold_input:
        try:
            feedback_threshold = float(threshold_input)
        except ValueError:
            print("  Invalid threshold, keeping current")
    
    # Update interval
    interval_input = input(f"Update Interval (ms) [{current.get('update_interval_ms')}]: ").strip()
    update_interval = None
    if interval_input:
        try:
            update_interval = int(interval_input)
        except ValueError:
            print("  Invalid interval, keeping current")
    
    # Apply configuration
    result = configure_bridge(
        enabled=enabled,
        feedback_threshold=feedback_threshold,
        update_interval_ms=update_interval
    )
    
    print("\n" + "=" * 70)
    print("Configuration updated!")
    print(json.dumps(result.get("config"), indent=2))
