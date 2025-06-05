"""
AI Assistant Python Module Promoter
===================================

This module is designed to promote and suggest the use of enhanced Python modules
for an AI assistant Android app using Chaquopy bridge. It provides intelligent
recommendations for module combinations based on task types and contexts.
"""

import json
import re
from typing import Dict, List, Set, Optional, Tuple
from dataclasses import dataclass, asdict
from enum import Enum


class TaskCategory(Enum):
    """Categories of tasks that the AI assistant might handle."""
    DATA_ANALYSIS = "data_analysis"
    MACHINE_LEARNING = "machine_learning"
    WEB_SCRAPING = "web_scraping"
    IMAGE_PROCESSING = "image_processing"
    TEXT_PROCESSING = "text_processing"
    NUMERICAL_COMPUTATION = "numerical_computation"
    VISUALIZATION = "visualization"
    FILE_PROCESSING = "file_processing"
    NETWORKING = "networking"
    AUTOMATION = "automation"
    SCIENTIFIC_COMPUTING = "scientific_computing"
    DATABASE_OPERATIONS = "database_operations"


@dataclass
class ModuleRecommendation:
    """Represents a module recommendation with metadata."""
    name: str
    description: str
    use_cases: List[str]
    install_command: str
    compatibility_notes: str
    priority: int  # 1-10, higher is more important
    dependencies: List[str]


class EnhancedModulePromoter:
    """
    Main class for promoting and recommending Python modules
    to the AI assistant based on task context and requirements.
    """
    
    def __init__(self):
        self.module_catalog = self._initialize_module_catalog()
        self.task_module_mapping = self._initialize_task_mappings()
        self.usage_stats = {}
        
    def _initialize_module_catalog(self) -> Dict[str, ModuleRecommendation]:
        """Initialize the catalog of enhanced Python modules."""
        catalog = {}
        
        # Data Analysis & Scientific Computing
        catalog['pandas'] = ModuleRecommendation(
            name='pandas',
            description='Powerful data manipulation and analysis library',
            use_cases=['CSV processing', 'Data cleaning', 'Statistical analysis', 'Time series'],
            install_command='pip install pandas',
            compatibility_notes='Works well with Chaquopy, excellent Android performance',
            priority=9,
            dependencies=['numpy']
        )
        
        catalog['numpy'] = ModuleRecommendation(
            name='numpy',
            description='Fundamental package for scientific computing',
            use_cases=['Numerical operations', 'Array processing', 'Mathematical functions'],
            install_command='pip install numpy',
            compatibility_notes='Core dependency, optimized for mobile',
            priority=10,
            dependencies=[]
        )
        
        catalog['scipy'] = ModuleRecommendation(
            name='scipy',
            description='Scientific computing library with advanced algorithms',
            use_cases=['Optimization', 'Signal processing', 'Statistical functions'],
            install_command='pip install scipy',
            compatibility_notes='May require compilation optimizations for Android',
            priority=7,
            dependencies=['numpy']
        )
        
        # Machine Learning
        catalog['scikit-learn'] = ModuleRecommendation(
            name='scikit-learn',
            description='Machine learning library with simple and efficient tools',
            use_cases=['Classification', 'Regression', 'Clustering', 'Model evaluation'],
            install_command='pip install scikit-learn',
            compatibility_notes='Excellent mobile compatibility, lightweight models',
            priority=8,
            dependencies=['numpy', 'scipy']
        )
        
        catalog['joblib'] = ModuleRecommendation(
            name='joblib',
            description='Lightweight pipelining and parallel computing',
            use_cases=['Model persistence', 'Parallel processing', 'Caching'],
            install_command='pip install joblib',
            compatibility_notes='Perfect for mobile ML model loading',
            priority=6,
            dependencies=[]
        )
        
        # Text Processing
        catalog['nltk'] = ModuleRecommendation(
            name='nltk',
            description='Natural language processing toolkit',
            use_cases=['Text tokenization', 'Sentiment analysis', 'Language processing'],
            install_command='pip install nltk',
            compatibility_notes='Download corpora carefully for mobile deployment',
            priority=7,
            dependencies=[]
        )
        
        catalog['textblob'] = ModuleRecommendation(
            name='textblob',
            description='Simple text processing library',
            use_cases=['Sentiment analysis', 'Noun phrase extraction', 'Translation'],
            install_command='pip install textblob',
            compatibility_notes='Lightweight alternative to NLTK',
            priority=6,
            dependencies=['nltk']
        )
        
        # Web & Networking
        catalog['requests'] = ModuleRecommendation(
            name='requests',
            description='HTTP library for Python',
            use_cases=['API calls', 'Web scraping', 'HTTP requests'],
            install_command='pip install requests',
            compatibility_notes='Excellent mobile support, handles Android network policies',
            priority=9,
            dependencies=[]
        )
        
        catalog['beautifulsoup4'] = ModuleRecommendation(
            name='beautifulsoup4',
            description='HTML and XML parsing library',
            use_cases=['Web scraping', 'HTML parsing', 'Data extraction'],
            install_command='pip install beautifulsoup4',
            compatibility_notes='Lightweight and mobile-friendly',
            priority=7,
            dependencies=['requests']
        )
        
        # Image Processing
        catalog['pillow'] = ModuleRecommendation(
            name='pillow',
            description='Python Imaging Library (PIL) fork',
            use_cases=['Image manipulation', 'Format conversion', 'Basic image processing'],
            install_command='pip install pillow',
            compatibility_notes='Good Android support, handles mobile image formats',
            priority=8,
            dependencies=[]
        )
        
        # File Processing
        catalog['openpyxl'] = ModuleRecommendation(
            name='openpyxl',
            description='Library for reading and writing Excel files',
            use_cases=['Excel file processing', 'Spreadsheet automation', 'Data import/export'],
            install_command='pip install openpyxl',
            compatibility_notes='Pure Python, excellent mobile compatibility',
            priority=7,
            dependencies=[]
        )
        
        catalog['python-docx'] = ModuleRecommendation(
            name='python-docx',
            description='Library for creating and updating Microsoft Word documents',
            use_cases=['Document generation', 'Word file processing', 'Report creation'],
            install_command='pip install python-docx',
            compatibility_notes='Works well on Android, no native dependencies',
            priority=6,
            dependencies=[]
        )
        
        # Visualization (lightweight options for mobile)
        catalog['matplotlib'] = ModuleRecommendation(
            name='matplotlib',
            description='Comprehensive plotting library',
            use_cases=['Data visualization', 'Chart creation', 'Plot generation'],
            install_command='pip install matplotlib',
            compatibility_notes='May be heavy for mobile, consider lightweight alternatives',
            priority=5,
            dependencies=['numpy']
        )
        
        # Utilities
        catalog['python-dateutil'] = ModuleRecommendation(
            name='python-dateutil',
            description='Extensions to the standard datetime module',
            use_cases=['Date parsing', 'Timezone handling', 'Relative dates'],
            install_command='pip install python-dateutil',
            compatibility_notes='Lightweight, perfect for mobile apps',
            priority=7,
            dependencies=[]
        )
        
        catalog['pytz'] = ModuleRecommendation(
            name='pytz',
            description='World timezone definitions for Python',
            use_cases=['Timezone conversion', 'Localization', 'Time calculations'],
            install_command='pip install pytz',
            compatibility_notes='Essential for mobile apps with global users',
            priority=8,
            dependencies=[]
        )
        
        return catalog
    
    def _initialize_task_mappings(self) -> Dict[TaskCategory, List[str]]:
        """Map task categories to relevant modules."""
        return {
            TaskCategory.DATA_ANALYSIS: ['pandas', 'numpy', 'scipy', 'matplotlib'],
            TaskCategory.MACHINE_LEARNING: ['scikit-learn', 'numpy', 'pandas', 'joblib'],
            TaskCategory.WEB_SCRAPING: ['requests', 'beautifulsoup4'],
            TaskCategory.IMAGE_PROCESSING: ['pillow', 'numpy'],
            TaskCategory.TEXT_PROCESSING: ['nltk', 'textblob'],
            TaskCategory.NUMERICAL_COMPUTATION: ['numpy', 'scipy'],
            TaskCategory.VISUALIZATION: ['matplotlib', 'numpy'],
            TaskCategory.FILE_PROCESSING: ['openpyxl', 'python-docx', 'pandas'],
            TaskCategory.NETWORKING: ['requests'],
            TaskCategory.AUTOMATION: ['requests', 'pandas', 'python-dateutil'],
            TaskCategory.SCIENTIFIC_COMPUTING: ['numpy', 'scipy', 'pandas'],
            TaskCategory.DATABASE_OPERATIONS: ['pandas', 'numpy']
        }
    
    def analyze_task_context(self, user_input: str) -> List[TaskCategory]:
        """Analyze user input to determine relevant task categories."""
        categories = []
        input_lower = user_input.lower()
        
        # Keyword mapping for task detection
        keyword_mappings = {
            TaskCategory.DATA_ANALYSIS: ['analyze', 'data', 'csv', 'statistics', 'trends'],
            TaskCategory.MACHINE_LEARNING: ['predict', 'classify', 'model', 'train', 'ml', 'ai'],
            TaskCategory.WEB_SCRAPING: ['scrape', 'website', 'html', 'url', 'extract'],
            TaskCategory.IMAGE_PROCESSING: ['image', 'photo', 'picture', 'resize', 'filter'],
            TaskCategory.TEXT_PROCESSING: ['text', 'sentiment', 'language', 'words', 'nlp'],
            TaskCategory.NUMERICAL_COMPUTATION: ['calculate', 'math', 'formula', 'equation'],
            TaskCategory.VISUALIZATION: ['plot', 'chart', 'graph', 'visualize', 'display'],
            TaskCategory.FILE_PROCESSING: ['excel', 'word', 'document', 'file', 'convert'],
            TaskCategory.NETWORKING: ['api', 'request', 'download', 'upload', 'http'],
            TaskCategory.AUTOMATION: ['automate', 'schedule', 'batch', 'process'],
            TaskCategory.SCIENTIFIC_COMPUTING: ['scientific', 'research', 'experiment', 'algorithm'],
            TaskCategory.DATABASE_OPERATIONS: ['database', 'sql', 'query', 'table', 'records']
        }
        
        for category, keywords in keyword_mappings.items():
            if any(keyword in input_lower for keyword in keywords):
                categories.append(category)
        
        return categories if categories else [TaskCategory.DATA_ANALYSIS]  # Default fallback
    
    def get_module_recommendations(self, 
                                 task_categories: List[TaskCategory], 
                                 max_recommendations: int = 5) -> List[ModuleRecommendation]:
        """Get ranked module recommendations for given task categories."""
        module_scores = {}
        
        # Score modules based on task relevance
        for category in task_categories:
            if category in self.task_module_mapping:
                for module_name in self.task_module_mapping[category]:
                    if module_name in self.module_catalog:
                        module = self.module_catalog[module_name]
                        current_score = module_scores.get(module_name, 0)
                        module_scores[module_name] = current_score + module.priority
        
        # Sort by score and return top recommendations
        sorted_modules = sorted(module_scores.items(), key=lambda x: x[1], reverse=True)
        
        recommendations = []
        for module_name, score in sorted_modules[:max_recommendations]:
            recommendations.append(self.module_catalog[module_name])
        
        return recommendations
    
    def generate_promotion_message(self, 
                                 user_input: str, 
                                 recommendations: List[ModuleRecommendation]) -> str:
        """Generate a promotional message suggesting module usage."""
        if not recommendations:
            return "Consider using Python's standard library for this task."
        
        message_parts = [
            "🐍 **Enhanced Python Capabilities Available!**\n",
            "Based on your request, I recommend leveraging these powerful Python modules:\n"
        ]
        
        for i, rec in enumerate(recommendations, 1):
            message_parts.append(f"\n**{i}. {rec.name}** - {rec.description}")
            message_parts.append(f"   • Use cases: {', '.join(rec.use_cases[:3])}")
            if rec.compatibility_notes:
                message_parts.append(f"   • Mobile note: {rec.compatibility_notes}")
        
        message_parts.append(f"\n💡 **Tip:** These modules work seamlessly with your Kotlin app through Chaquopy!")
        message_parts.append("\nWould you like me to show you how to implement any of these solutions?")
        
        return "".join(message_parts)
    
    def get_module_combination_suggestions(self, 
                                        primary_modules: List[str]) -> Dict[str, List[str]]:
        """Suggest powerful module combinations for enhanced functionality."""
        combinations = {
            "Data Science Stack": ["pandas", "numpy", "matplotlib", "scikit-learn"],
            "Web Intelligence": ["requests", "beautifulsoup4", "pandas", "textblob"],
            "Document Processing": ["python-docx", "openpyxl", "pandas", "python-dateutil"],
            "ML Pipeline": ["scikit-learn", "pandas", "numpy", "joblib"],
            "Text Analysis Suite": ["nltk", "textblob", "pandas", "numpy"]
        }
        
        # Filter combinations based on primary modules
        relevant_combinations = {}
        for combo_name, modules in combinations.items():
            if any(module in primary_modules for module in modules):
                relevant_combinations[combo_name] = modules
        
        return relevant_combinations
    
    def create_implementation_guide(self, module_name: str) -> str:
        """Create a quick implementation guide for a specific module."""
        if module_name not in self.module_catalog:
            return f"Module '{module_name}' not found in catalog."
        
        module = self.module_catalog[module_name]
        
        guide = f"""
# Quick Implementation Guide: {module.name}

## Installation
```bash
{module.install_command}
```

## Android/Chaquopy Notes
{module.compatibility_notes}

## Primary Use Cases
{chr(10).join(f"• {use_case}" for use_case in module.use_cases)}

## Dependencies
{', '.join(module.dependencies) if module.dependencies else 'None'}

## Integration Tip
This module can be seamlessly integrated into your Kotlin app using Chaquopy.
Consider creating dedicated Python functions that your Kotlin code can call.
        """
        
        return guide.strip()
    
    def track_module_usage(self, module_name: str):
        """Track module usage for analytics and better recommendations."""
        if module_name not in self.usage_stats:
            self.usage_stats[module_name] = 0
        self.usage_stats[module_name] += 1
    
    def get_usage_analytics(self) -> Dict[str, int]:
        """Get module usage analytics."""
        return self.usage_stats.copy()
    
    def export_recommendations(self, recommendations: List[ModuleRecommendation]) -> str:
        """Export recommendations as JSON for Kotlin integration."""
        rec_dicts = [asdict(rec) for rec in recommendations]
        return json.dumps(rec_dicts, indent=2)


# Convenience functions for easy integration with Kotlin/Chaquopy
def promote_modules_for_task(user_input: str) -> str:
    """Main function to get module promotion message for a user task."""
    promoter = EnhancedModulePromoter()
    categories = promoter.analyze_task_context(user_input)
    recommendations = promoter.get_module_recommendations(categories)
    return promoter.generate_promotion_message(user_input, recommendations)


def get_quick_recommendations(task_type: str) -> List[str]:
    """Get quick module name recommendations for common task types."""
    promoter = EnhancedModulePromoter()
    
    # Map common task strings to categories
    task_mapping = {
        'data': TaskCategory.DATA_ANALYSIS,
        'ml': TaskCategory.MACHINE_LEARNING,
        'web': TaskCategory.WEB_SCRAPING,
        'text': TaskCategory.TEXT_PROCESSING,
        'image': TaskCategory.IMAGE_PROCESSING,
        'file': TaskCategory.FILE_PROCESSING
    }
    
    category = task_mapping.get(task_type.lower(), TaskCategory.DATA_ANALYSIS)
    recommendations = promoter.get_module_recommendations([category])
    
    return [rec.name for rec in recommendations]


def get_android_optimized_modules() -> List[str]:
    """Get list of modules specifically optimized for Android/mobile deployment."""
    promoter = EnhancedModulePromoter()
    
    optimized_modules = []
    for name, module in promoter.module_catalog.items():
        if any(keyword in module.compatibility_notes.lower() 
               for keyword in ['mobile', 'android', 'lightweight', 'excellent']):
            optimized_modules.append(name)
    
    return optimized_modules


# Advanced Features for Enhanced Integration

class ModulePerformanceTracker:
    """Track module performance and success rates for better recommendations."""
    
    def __init__(self):
        self.performance_data = {}
        self.success_rates = {}
        self.execution_times = {}
    
    def log_module_performance(self, module_name: str, execution_time: float, success: bool):
        """Log performance metrics for a module."""
        if module_name not in self.performance_data:
            self.performance_data[module_name] = {'total_uses': 0, 'successes': 0, 'total_time': 0.0}
        
        data = self.performance_data[module_name]
        data['total_uses'] += 1
        data['total_time'] += execution_time
        
        if success:
            data['successes'] += 1
        
        # Update success rate
        self.success_rates[module_name] = data['successes'] / data['total_uses']
    
    def get_performance_insights(self) -> Dict[str, Dict[str, float]]:
        """Get performance insights for all tracked modules."""
        insights = {}
        for module_name, data in self.performance_data.items():
            avg_time = data['total_time'] / data['total_uses'] if data['total_uses'] > 0 else 0
            insights[module_name] = {
                'success_rate': self.success_rates.get(module_name, 0),
                'average_execution_time': avg_time,
                'total_uses': data['total_uses']
            }
        return insights


class ContextAwarePromoter(EnhancedModulePromoter):
    """Enhanced promoter with context awareness and learning capabilities."""
    
    def __init__(self):
        super().__init__()
        self.performance_tracker = ModulePerformanceTracker()
        self.user_preferences = {}
        self.context_history = []
        self.module_compatibility_matrix = self._build_compatibility_matrix()
    
    def _build_compatibility_matrix(self) -> Dict[str, List[str]]:
        """Build a matrix of module compatibility and synergies."""
        return {
            'pandas': ['numpy', 'matplotlib', 'openpyxl', 'python-dateutil'],
            'scikit-learn': ['pandas', 'numpy', 'joblib', 'matplotlib'],
            'requests': ['beautifulsoup4', 'pandas', 'json'],
            'pillow': ['numpy', 'matplotlib'],
            'nltk': ['textblob', 'pandas', 'numpy'],
            'matplotlib': ['pandas', 'numpy', 'scipy']
        }
    
    def learn_from_user_feedback(self, module_name: str, user_rating: int, context: str):
        """Learn from user feedback to improve recommendations."""
        if module_name not in self.user_preferences:
            self.user_preferences[module_name] = {'ratings': [], 'contexts': []}
        
        self.user_preferences[module_name]['ratings'].append(user_rating)
        self.user_preferences[module_name]['contexts'].append(context)
    
    def get_personalized_recommendations(self, 
                                       task_categories: List[TaskCategory],
                                       consider_performance: bool = True) -> List[ModuleRecommendation]:
        """Get recommendations personalized based on user history and performance."""
        base_recommendations = super().get_module_recommendations(task_categories)
        
        if not consider_performance:
            return base_recommendations
        
        # Adjust recommendations based on performance data
        performance_insights = self.performance_tracker.get_performance_insights()
        
        for rec in base_recommendations:
            if rec.name in performance_insights:
                insight = performance_insights[rec.name]
                # Boost priority for high-performing modules
                if insight['success_rate'] > 0.8:
                    rec.priority = min(10, rec.priority + 1)
                elif insight['success_rate'] < 0.5:
                    rec.priority = max(1, rec.priority - 1)
        
        # Re-sort by adjusted priority
        base_recommendations.sort(key=lambda x: x.priority, reverse=True)
        return base_recommendations
    
    def suggest_module_workflows(self, primary_task: str) -> Dict[str, List[str]]:
        """Suggest complete workflows using multiple modules."""
        workflows = {
            "Data Analysis Pipeline": [
                "1. Load data with pandas",
                "2. Clean and preprocess with numpy",
                "3. Analyze with pandas statistical functions",
                "4. Visualize results with matplotlib",
                "5. Export processed data with openpyxl"
            ],
            "Web Data Collection": [
                "1. Fetch web content with requests",
                "2. Parse HTML with beautifulsoup4",
                "3. Store data in pandas DataFrame",
                "4. Clean and validate with pandas",
                "5. Export to Excel with openpyxl"
            ],
            "Text Intelligence": [
                "1. Load text data with pandas",
                "2. Preprocess with nltk",
                "3. Analyze sentiment with textblob",
                "4. Visualize results with matplotlib",
                "5. Generate reports with python-docx"
            ],
            "ML Model Development": [
                "1. Load and prepare data with pandas",
                "2. Feature engineering with numpy",
                "3. Train model with scikit-learn",
                "4. Evaluate performance with scikit-learn metrics",
                "5. Save model with joblib"
            ]
        }
        
        # Filter workflows based on task relevance
        task_lower = primary_task.lower()
        relevant_workflows = {}
        
        for workflow_name, steps in workflows.items():
            if any(keyword in task_lower for keyword in 
                   workflow_name.lower().split()):
                relevant_workflows[workflow_name] = steps
        
        return relevant_workflows if relevant_workflows else {"General Data Processing": workflows["Data Analysis Pipeline"]}


# Android Integration Helper Functions
class AndroidIntegrationHelper:
    """Helper class for seamless Android/Kotlin integration."""
    
    @staticmethod
    def create_chaquopy_config(modules: List[str]) -> str:
        """Generate Chaquopy configuration for required modules."""
        config = {
            "buildPython": "3.8",
            "pip": {
                "install": modules
            },
            "staticProxy": True
        }
        return json.dumps(config, indent=2)
    
    @staticmethod
    def generate_kotlin_interface(module_recommendations: List[ModuleRecommendation]) -> str:
        """Generate Kotlin interface code for Python module access."""
        kotlin_code = """
// Auto-generated Kotlin interface for Python module integration
package com.yourapp.python

import com.chaquo.python.Python
import com.chaquo.python.PyObject

class PythonModuleManager {
    private val python = Python.getInstance()
    private val promoterModule = python.getModule("ai_assistant_module_promoter")
    
    fun getModuleRecommendations(userInput: String): String {
        return promoterModule.callAttr("promote_modules_for_task", userInput).toString()
    }
    
    fun getQuickRecommendations(taskType: String): List<String> {
        val pyList = promoterModule.callAttr("get_quick_recommendations", taskType)
        return pyList.asList().map { it.toString() }
    }
    
    fun getAndroidOptimizedModules(): List<String> {
        val pyList = promoterModule.callAttr("get_android_optimized_modules")
        return pyList.asList().map { it.toString() }
    }
"""
        
        # Add specific module access methods
        for rec in module_recommendations:
            method_name = rec.name.replace('-', '_').replace(' ', '_')
            kotlin_code += f"""
    
    fun use{method_name.title()}(data: String): PyObject {{
        val {method_name}Module = python.getModule("{rec.name}")
        // Add specific module usage logic here
        return {method_name}Module
    }}"""
        
        kotlin_code += "\n}\n"
        return kotlin_code
    
    @staticmethod
    def create_android_manifest_permissions() -> List[str]:
        """Generate required Android manifest permissions for common modules."""
        return [
            '<uses-permission android:name="android.permission.INTERNET" />',
            '<uses-permission android:name="android.permission.ACCESS_NETWORK_STATE" />',
            '<uses-permission android:name="android.permission.READ_EXTERNAL_STORAGE" />',
            '<uses-permission android:name="android.permission.WRITE_EXTERNAL_STORAGE" />',
            '<uses-permission android:name="android.permission.CAMERA" />',
        ]


# Real-time Module Suggestion Engine
class RealTimeModuleSuggester:
    """Provides real-time module suggestions as users type or interact."""
    
    def __init__(self):
        self.promoter = ContextAwarePromoter()
        self.suggestion_cache = {}
        self.typing_patterns = {}
    
    def analyze_partial_input(self, partial_input: str, confidence_threshold: float = 0.6) -> Dict[str, float]:
        """Analyze partial user input and return module suggestions with confidence scores."""
        suggestions = {}
        
        # Quick keyword matching for real-time suggestions
        quick_matches = {
            'data': {'pandas': 0.9, 'numpy': 0.8},
            'csv': {'pandas': 0.95, 'openpyxl': 0.7},
            'web': {'requests': 0.9, 'beautifulsoup4': 0.8},
            'image': {'pillow': 0.9},
            'text': {'nltk': 0.8, 'textblob': 0.7},
            'ml': {'scikit-learn': 0.9, 'numpy': 0.7},
            'api': {'requests': 0.95},
            'excel': {'openpyxl': 0.95, 'pandas': 0.8}
        }
        
        input_lower = partial_input.lower()
        for keyword, modules in quick_matches.items():
            if keyword in input_lower:
                for module, confidence in modules.items():
                    suggestions[module] = max(suggestions.get(module, 0), confidence)
        
        # Filter by confidence threshold
        return {k: v for k, v in suggestions.items() if v >= confidence_threshold}
    
    def get_contextual_suggestions(self, current_session_modules: List[str]) -> List[str]:
        """Suggest additional modules based on currently used modules in the session."""
        suggestions = set()
        
        for module in current_session_modules:
            if module in self.promoter.module_compatibility_matrix:
                suggestions.update(self.promoter.module_compatibility_matrix[module])
        
        # Remove already used modules
        suggestions -= set(current_session_modules)
        return list(suggestions)


# Enhanced Testing and Demo Functions
def comprehensive_demo():
    """Run a comprehensive demonstration of all features."""
    print("🚀 === COMPREHENSIVE AI ASSISTANT MODULE PROMOTER DEMO ===\n")
    
    # Initialize components
    promoter = ContextAwarePromoter()
    android_helper = AndroidIntegrationHelper()
    realtime_suggester = RealTimeModuleSuggester()
    
    # Test scenarios
    test_scenarios = [
        {
            "input": "I need to analyze sales data from multiple CSV files and create visualizations",
            "description": "Complex data analysis task"
        },
        {
            "input": "Help me scrape product prices from e-commerce websites",
            "description": "Web scraping task"
        },
        {
            "input": "I want to build a sentiment analysis model for customer reviews",
            "description": "Machine learning and NLP task"
        },
        {
            "input": "Process and optimize images for my mobile app",
            "description": "Image processing task"
        }
    ]
    
    for i, scenario in enumerate(test_scenarios, 1):
        print(f"📋 **Scenario {i}: {scenario['description']}**")
        print(f"User Input: \"{scenario['input']}\"\n")
        
        # Get recommendations
        categories = promoter.analyze_task_context(scenario['input'])
        recommendations = promoter.get_personalized_recommendations(categories, max_recommendations=4)
        
        # Generate promotion message
        promotion = promoter.generate_promotion_message(scenario['input'], recommendations)
        print(promotion)
        
        # Show workflow suggestions
        workflows = promoter.suggest_module_workflows(scenario['input'])
        print(f"\n🔄 **Suggested Workflows:**")
        for workflow_name, steps in workflows.items():
            print(f"\n**{workflow_name}:**")
            for step in steps:
                print(f"   {step}")
        
        # Real-time suggestions demo
        print(f"\n⚡ **Real-time Suggestions:**")
        partial_suggestions = realtime_suggester.analyze_partial_input(scenario['input'])
        for module, confidence in partial_suggestions.items():
            print(f"   • {module} (confidence: {confidence:.1%})")
        
        print("\n" + "="*80 + "\n")
    
    # Android Integration Demo
    print("📱 **ANDROID INTEGRATION EXAMPLES**\n")
    
    sample_modules = ['pandas', 'requests', 'scikit-learn']
    
    print("**Chaquopy Configuration:**")
    print(android_helper.create_chaquopy_config(sample_modules))
    
    print(f"\n**Required Android Permissions:**")
    permissions = android_helper.create_android_manifest_permissions()
    for permission in permissions:
        print(f"   {permission}")
    
    print(f"\n**Kotlin Interface Example:**")
    sample_recommendations = [promoter.module_catalog[name] for name in sample_modules]
    kotlin_interface = android_helper.generate_kotlin_interface(sample_recommendations)
    print(kotlin_interface[:500] + "..." if len(kotlin_interface) > 500 else kotlin_interface)


if __name__ == "__main__":
    comprehensive_demo()
