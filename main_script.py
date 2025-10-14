"""
Main Python Script for Android application

This script contains the main Python functionality for the app.
It is imported and used by python_hook.py when needed.
"""

import os
import sys
import json
import traceback

# Store references to shared resources
_resources = {}

def parse_kotlin_reflection_data(data_string):
    """
    Parse Kotlin reflection data from string format
    
    Args:
        data_string: String containing Kotlin reflection data
    
    Returns:
        Structured dict with parsed reflection data
    """
    if not data_string:
        return {"error": "Empty data string"}
        
    try:
        result = {
            "classes": [],
            "properties": [],
            "functions": [],
            "types": []
        }
        
        current_section = None
        current_class = None
        
        lines = data_string.split('\n')
        for line in lines:
            line = line.strip()
            if not line:
                continue
                
            # Check for section headers (non-indented lines)
            if not line.startswith('\t'):
                current_section = line
                if "KClass" in line:
                    result["classes"].append({"name": line, "members": []})
                    current_class = result["classes"][-1]
                continue
                
            # Parse indented lines based on current section
            if current_section:
                if "KClass" in current_section and current_class:
                    # Add member to current class
                    current_class["members"].append(line)
                elif "KProperty" in current_section:
                    # Add to properties
                    result["properties"].append(line)
                elif "KFunction" in current_section or "Function" in current_section:
                    # Add to functions
                    result["functions"].append(line)
                elif "KType" in current_section or "KVariance" in current_section:
                    # Add to types
                    result["types"].append(line)
        
        return result
    except Exception as e:
        traceback.print_exc()
        return {"error": str(e)}

def analyze_reflection_bulletins(bulletin_data):
    """
    Analyze Kotlin reflection bulletins for useful patterns
    
    Args:
        bulletin_data: String containing Kotlin reflection data
    
    Returns:
        Analysis results as a dict
    """
    try:
        parsed_data = parse_kotlin_reflection_data(bulletin_data)
        if "error" in parsed_data:
            return parsed_data
            
        # Analyze class structure
        class_analysis = {
            "total_classes": len(parsed_data["classes"]),
            "total_properties": len(parsed_data["properties"]),
            "total_functions": len(parsed_data["functions"]),
            "type_system": {
                "variance_types": [],
                "generic_types": []
            }
        }
        
        # Extract variance types
        for type_item in parsed_data["types"]:
            if "INVARIANT" in type_item or "IN" in type_item or "OUT" in type_item:
                class_analysis["type_system"]["variance_types"].append(type_item)
                
        # Look for generic type usage
        for prop in parsed_data["properties"]:
            if "<" in prop and ">" in prop:
                class_analysis["type_system"]["generic_types"].append(prop)
        
        return {
            "analysis": class_analysis,
            "recommendations": get_recommendations(class_analysis)
        }
    except Exception as e:
        traceback.print_exc()
        return {"error": str(e)}

def get_recommendations(analysis):
    """
    Generate recommendations based on reflection analysis
    
    Args:
        analysis: Analysis dict from analyze_reflection_bulletins
    
    Returns:
        List of recommendations
    """
    recommendations = []
    
    if analysis["total_classes"] > 0:
        recommendations.append("Found Kotlin classes that can be reflected")
        
    if analysis["total_properties"] > 0:
        recommendations.append(f"Found {analysis['total_properties']} properties that can be accessed")
        
    if analysis["total_functions"] > 0:
        recommendations.append(f"Found {analysis['total_functions']} functions that can be called")
        
    if analysis["type_system"]["variance_types"]:
        recommendations.append("Detected generic variance usage (in/out) which indicates advanced type system usage")
        
    if analysis["type_system"]["generic_types"]:
        recommendations.append("Found generic type usage that should be handled carefully in reflection")
    
    return recommendations

def extract_class_hierarchy(bulletin_data):
    """
    Extract class hierarchy from the reflection data
    
    Args:
        bulletin_data: String containing Kotlin reflection data
    
    Returns:
        Dict representing class hierarchy
    """
    try:
        lines = bulletin_data.split('\n')
        hierarchy = {}
        current_class = None
        current_level = 0
        
        for line in lines:
            # Count indentation level
            indentation = len(line) - len(line.lstrip('\t'))
            content = line.strip()
            
            if not content:
                continue
                
            if indentation == 0:
                # Top level class
                current_class = content
                hierarchy[current_class] = {"members": [], "subclasses": {}}
                current_level = 0
            elif indentation == current_level + 1:
                # Direct member or subclass
                if "<" in content or "(" in content or ":" in content:
                    # Likely a member
                    hierarchy[current_class]["members"].append(content)
                else:
                    # Likely a subclass
                    hierarchy[current_class]["subclasses"][content] = {"members": [], "subclasses": {}}
            
        return hierarchy
    except Exception as e:
        traceback.print_exc()
        return {"error": str(e)}

# Function to create sample schema from reflection data
def create_reflection_schema(bulletin_data):
    """
    Create a schema representation from reflection data
    
    Args:
        bulletin_data: String containing Kotlin reflection data
    
    Returns:
        JSON schema as a dict
    """
    try:
        parsed_data = parse_kotlin_reflection_data(bulletin_data)
        if "error" in parsed_data:
            return parsed_data
            
        schema = {
            "$schema": "http://json-schema.org/draft-07/schema#",
            "title": "KotlinReflectionSchema",
            "type": "object",
            "properties": {}
        }
        
        # Add classes to schema
        for cls in parsed_data["classes"]:
            class_name = cls["name"].split(".")[-1] if "." in cls["name"] else cls["name"]
            schema["properties"][class_name] = {
                "type": "object",
                "properties": {}
            }
            
            # Add members to class
            for member in cls.get("members", []):
                member_name = member.split(":")[0].strip() if ":" in member else member
                schema["properties"][class_name]["properties"][member_name] = {
                    "type": "string",
                    "description": f"Member of {class_name}"
                }
        
        return schema
    except Exception as e:
        traceback.print_exc()
        return {"error": str(e)}
        
# Additional utility functions can be added here
def get_module_info():
    """
    Get information about this module
    
    Returns:
        Dict with module information
    """
    return {
        "name": __name__,
        "file": __file__,
        "functions": [f for f in dir() if callable(globals()[f]) and not f.startswith('_')]
    }
