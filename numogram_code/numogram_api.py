import os
import json
import requests
from flask import Flask, request, jsonify, send_file
from werkzeug.utils import secure_filename
from fastapi import FastAPI
from pydantic import BaseModel
from typing import Optional, Dict

# Import all required modules
from modules.meta_reflection import meta_reflection
from modules.memory_module import update_memory
from modules.implementation_feasibility import FeasibilityEvaluator
from modules.ethical_considerations import EthicalEvaluator
from modules.recursive_thought import RecursiveThought
from modules.multi_zone_memory import MultiZoneMemory
from modules.creative_expansion import CreativeExpansion
from modules.numogram_algorithm import NumogramCore
from modules.edge_explorer import EdgeExplorer
from modules.knowledge_synthesis import KnowledgeSynthesis
from modules.consequence_chains import ConsequenceChains
from modules.numogram_code_belief import NumogramCodeBelief
from modules.contradictory_analysis import ContradictoryAnalysis
from modules.hybrid_model import AdaptiveHybridModel
from modules.schizoanalytic_generator import SchizoanalyticGenerator, apply_schizoanalytic_mutation
from modules.morphogenesis_module import MorphogenesisModule

# Initialize Flask app
app = Flask(__name__)

# Initialize FastAPI app
fastapi_app = FastAPI(title="AI Friend API", description="Advanced AI interaction API")

# Configuration
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Data models
class ChatRequest(BaseModel):
    message: str
    context: Optional[Dict] = None

class ChatResponse(BaseModel):
    response: str
    meta_analysis: Optional[Dict] = None

# Helper functions
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Flask API Routes
@app.route('/api/health', methods=['GET'])
def health_check():
    return jsonify({"status": "healthy"})

@app.route('/api/chat', methods=['POST'])
def chat():
    try:
        data = request.json
        if not data or 'message' not in data:
            return jsonify({"error": "Invalid request format"}), 400
        
        # Process the chat message
        user_message = data['message']
        context = data.get('context', {})
        response = process_chat_message(user_message, context)
        
        return jsonify(response)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No file selected"}), 400
    
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        # Process uploaded file
        result = process_uploaded_file(filepath)
        
        return jsonify({"success": True, "filename": filename, "result": result})
    
    return jsonify({"error": "File type not allowed"}), 400

# FastAPI Routes
@fastapi_app.post("/api/v2/chat", response_model=ChatResponse)
async def chat_v2(request: ChatRequest):
    response = process_chat_message(request.message, request.context or {})
    return response

# Processing functions
def process_chat_message(message, context=None):
    """
    Process a chat message using all available modules
    """
    if context is None:
        context = {}
    
    # Initialize core components
    memory = MultiZoneMemory()
    feasibility = FeasibilityEvaluator()
    ethics = EthicalEvaluator()
    recursive = RecursiveThought()
    creative = CreativeExpansion()
    numogram = NumogramCore()
    edge = EdgeExplorer()
    knowledge = KnowledgeSynthesis()
    consequences = ConsequenceChains()
    belief = NumogramCodeBelief()
    contradiction = ContradictoryAnalysis()
    hybrid = AdaptiveHybridModel()
    schizo = SchizoanalyticGenerator()
    morpho = MorphogenesisModule()
    
    # Update memory with incoming message
    memory_result = update_memory(message, context)
    
    # Apply metamorphic thinking
    meta_result = meta_reflection(message, memory_result)
    
    # Apply schizoanalytic mutation
    schizo_result = apply_schizoanalytic_mutation(meta_result)
    
    # Apply morphogenesis transformation
    morpho_result = morpho.transform(schizo_result)
    
    # Generate response through hybrid model
    hybrid_response = hybrid.generate_response(
        input_text=message,
        memory=memory_result,
        meta=meta_result,
        schizo=schizo_result,
        morpho=morpho_result
    )
    
    # Build final response with meta-analysis
    response = {
        "response": hybrid_response,
        "meta_analysis": {
            "feasibility": feasibility.evaluate(hybrid_response),
            "ethics": ethics.evaluate(hybrid_response),
            "creative_index": creative.measure_creativity(hybrid_response),
            "contradiction_assessment": contradiction.analyze(hybrid_response)
        }
    }
    
    return response

def process_uploaded_file(filepath):
    """
    Process an uploaded file using our advanced modules
    """
    file_size = os.path.getsize(filepath)
    file_ext = os.path.splitext(filepath)[1].lower()
    
    # Basic file information
    result = {
        "file_path": filepath,
        "file_size": file_size,
        "file_type": file_ext,
        "status": "processed"
    }
    
    # Add advanced processing based on file type
    if file_ext in ['.jpg', '.jpeg', '.png', '.gif']:
        # Image analysis would go here
        # For now, just add placeholder for image metadata
        result["image_analysis"] = {
            "processed_by": "morphogenesis_module",
            "visual_patterns": morpho.analyze_visual_patterns(filepath)
        }
    
    return result

# Main entry point
if __name__ == '__main__':
    # For development, use Flask's built-in server
    app.run(host='0.0.0.0', port=5000, debug=True)
    
    # For production, you would use a WSGI server with FastAPI
    # import uvicorn
    # uvicorn.run(fastapi_app, host="0.0.0.0", port=8000)
