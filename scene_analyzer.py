import requests
import json
import logging
import time
from typing import List, Dict, Optional
from dataclasses import dataclass
import threading
import queue

@dataclass
class SceneAnalysis:
    threat_level: str
    description: str
    recommendations: List[str]
    confidence: float
    reasoning: str

class LocalLLMAnalyzer:
    """Scene analyzer using local LLM (Ollama) for intelligent threat assessment."""
    
    def __init__(self, config: dict):
        self.config = config
        self.llm_config = config.get('llm', {})
        self.ollama_url = self.llm_config.get('ollama_url', 'http://localhost:11434')
        self.model = self.llm_config.get('model', 'llama2:7b')
        self.enabled = self.llm_config.get('enabled', True)
        
        self.analysis_queue = queue.Queue(maxsize=10)
        self.result_queue = queue.Queue()
        self.worker_thread = None
        self.running = False
        
        if self.enabled:
            self.start_analyzer()
    
    def start_analyzer(self):
        """Start the background analysis worker."""
        if self.check_ollama_connection():
            self.running = True
            self.worker_thread = threading.Thread(target=self._analysis_worker, daemon=True)
            self.worker_thread.start()
            logging.info("Local LLM analyzer started")
        else:
            logging.warning("Ollama not available, LLM analysis disabled")
            self.enabled = False
    
    def stop_analyzer(self):
        """Stop the background analysis worker."""
        self.running = False
        if self.worker_thread:
            self.worker_thread.join(timeout=2.0)
        logging.info("Local LLM analyzer stopped")
    
    def check_ollama_connection(self) -> bool:
        """Check if Ollama is running and accessible."""
        try:
            response = requests.get(f"{self.ollama_url}/api/tags", timeout=5)
            return response.status_code == 200
        except requests.exceptions.RequestException:
            return False
    
    def analyze_scene_async(self, detections: List[Dict], frame_info: Dict):
        """Queue scene for analysis (non-blocking)."""
        if not self.enabled:
            return
        
        analysis_data = {
            'detections': detections,
            'frame_info': frame_info,
            'timestamp': time.time()
        }
        
        try:
            self.analysis_queue.put_nowait(analysis_data)
        except queue.Full:
            logging.warning("Analysis queue full, skipping frame")
    
    def get_latest_analysis(self) -> Optional[SceneAnalysis]:
        """Get the latest analysis result (non-blocking)."""
        try:
            return self.result_queue.get_nowait()
        except queue.Empty:
            return None
    
    def _analysis_worker(self):
        """Background worker for processing scene analysis."""
        while self.running:
            try:
                analysis_data = self.analysis_queue.get(timeout=0.5)
                
                result = self._analyze_scene_with_llm(
                    analysis_data['detections'],
                    analysis_data['frame_info']
                )
                
                try:
                    self.result_queue.get_nowait()  # Remove old result
                except queue.Empty:
                    pass
                
                self.result_queue.put_nowait(result)
                
                self.analysis_queue.task_done()
                
            except queue.Empty:
                continue
            except Exception as e:
                logging.error(f"Error in analysis worker: {e}")
    
    def _analyze_scene_with_llm(self, detections: List[Dict], frame_info: Dict) -> SceneAnalysis:
        """Analyze scene using local LLM."""
        try:
            scene_description = self._create_scene_description(detections, frame_info)
            
            prompt = self._create_analysis_prompt(scene_description)
            
            response = self._query_ollama(prompt)
            
            analysis = self._parse_llm_response(response, detections)
            
            return analysis
            
        except Exception as e:
            logging.error(f"Error in LLM analysis: {e}")
            return self._fallback_analysis(detections)
    
    def _create_scene_description(self, detections: List[Dict], frame_info: Dict) -> str:
        """Create a textual description of the current scene."""
        if not detections:
            return "Empty scene with no detections."
        
        humans = [d for d in detections if d.get('type') == 'human' or d.get('class_name') == 'person']
        dangerous_objects = [d for d in detections if d.get('is_dangerous', False)]
        regular_objects = [d for d in detections if not d.get('is_dangerous', False) and d.get('type') != 'human']
        
        description_parts = []
        
        if humans:
            description_parts.append(f"{len(humans)} person(s) detected")
            for i, human in enumerate(humans[:3]):  # Limit to first 3 for brevity
                confidence = human.get('confidence', 0)
                description_parts.append(f"  - Person {i+1}: confidence {confidence:.2f}")
        
        if dangerous_objects:
            description_parts.append(f"{len(dangerous_objects)} dangerous object(s) detected:")
            for obj in dangerous_objects:
                class_name = obj.get('class_name', 'unknown')
                confidence = obj.get('confidence', 0)
                weapon_score = obj.get('weapon_score', confidence)
                method = obj.get('detection_method', 'standard')
                description_parts.append(f"  - {class_name}: confidence {confidence:.2f}, weapon_score {weapon_score:.2f}, method: {method}")
        
        if regular_objects:
            top_objects = sorted(regular_objects, key=lambda x: x.get('confidence', 0), reverse=True)[:5]
            object_names = [obj.get('class_name', 'unknown') for obj in top_objects]
            description_parts.append(f"Other objects: {', '.join(object_names)}")
        
        fps = frame_info.get('fps', 0)
        timestamp = frame_info.get('timestamp', time.strftime('%H:%M:%S'))
        description_parts.append(f"Frame info: {fps:.1f} FPS at {timestamp}")
        
        return '\n'.join(description_parts)
    
    def _create_analysis_prompt(self, scene_description: str) -> str:
        """Create analysis prompt for the LLM."""
        return f"""You are an AI security analyst. Analyze the following security camera scene and provide a threat assessment.

Scene Description:
{scene_description}

Please analyze this scene and respond with a JSON object containing:
- threat_level: "none", "low", "medium", "high", or "critical"
- description: Brief description of what you observe
- recommendations: Array of specific actions to take
- confidence: Float between 0.0 and 1.0
- reasoning: Explanation of your assessment

Focus on:
1. Actual security threats vs normal activities
2. Context of dangerous objects (kitchen knife vs weapon)
3. Human behavior and positioning
4. Urgency of response needed

Response format:
{{
  "threat_level": "...",
  "description": "...", 
  "recommendations": ["...", "..."],
  "confidence": 0.0,
  "reasoning": "..."
}}"""
    
    def _query_ollama(self, prompt: str) -> str:
        """Query the local Ollama LLM."""
        try:
            payload = {
                "model": self.model,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": 0.3,  # Lower temperature for more consistent security analysis
                    "top_p": 0.9,
                    "num_predict": 500   # Limit response length
                }
            }
            
            response = requests.post(
                f"{self.ollama_url}/api/generate",
                json=payload,
                timeout=10  # 10 second timeout
            )
            
            if response.status_code == 200:
                result = response.json()
                return result.get('response', '')
            else:
                logging.error(f"Ollama API error: {response.status_code}")
                return ""
                
        except requests.exceptions.RequestException as e:
            logging.error(f"Error querying Ollama: {e}")
            return ""
    
    def _parse_llm_response(self, response: str, detections: List[Dict]) -> SceneAnalysis:
        """Parse LLM response into SceneAnalysis object."""
        try:
            start_idx = response.find('{')
            end_idx = response.rfind('}') + 1
            
            if start_idx >= 0 and end_idx > start_idx:
                json_str = response[start_idx:end_idx]
                data = json.loads(json_str)
                
                return SceneAnalysis(
                    threat_level=data.get('threat_level', 'unknown'),
                    description=data.get('description', 'No description available'),
                    recommendations=data.get('recommendations', []),
                    confidence=float(data.get('confidence', 0.5)),
                    reasoning=data.get('reasoning', 'No reasoning provided')
                )
            else:
                return self._parse_text_response(response, detections)
                
        except (json.JSONDecodeError, ValueError) as e:
            logging.warning(f"Error parsing LLM response: {e}")
            return self._fallback_analysis(detections)
    
    def _parse_text_response(self, response: str, detections: List[Dict]) -> SceneAnalysis:
        """Parse non-JSON text response."""
        dangerous_count = len([d for d in detections if d.get('is_dangerous', False)])
        
        if dangerous_count > 0:
            threat_level = "high" if dangerous_count > 1 else "medium"
            description = f"LLM detected potential threats: {response[:100]}..."
            recommendations = ["Review footage", "Consider security response"]
        else:
            threat_level = "low"
            description = f"Scene appears normal: {response[:100]}..."
            recommendations = ["Continue monitoring"]
        
        return SceneAnalysis(
            threat_level=threat_level,
            description=description,
            recommendations=recommendations,
            confidence=0.7,
            reasoning="Parsed from text response"
        )
    
    def _fallback_analysis(self, detections: List[Dict]) -> SceneAnalysis:
        """Fallback analysis when LLM is unavailable."""
        dangerous_objects = [d for d in detections if d.get('is_dangerous', False)]
        humans = [d for d in detections if d.get('type') == 'human']
        
        if dangerous_objects:
            if len(dangerous_objects) > 1:
                threat_level = "high"
                description = f"Multiple dangerous objects detected: {len(dangerous_objects)} items"
                recommendations = ["Immediate security response required", "Lock down area", "Contact authorities"]
            else:
                threat_level = "medium"
                obj_name = dangerous_objects[0].get('class_name', 'unknown object')
                description = f"Dangerous object detected: {obj_name}"
                recommendations = ["Investigate further", "Monitor closely", "Prepare security response"]
        elif humans:
            threat_level = "low"
            description = f"Normal activity: {len(humans)} person(s) present"
            recommendations = ["Continue monitoring", "No immediate action required"]
        else:
            threat_level = "none"
            description = "No significant activity detected"
            recommendations = ["Routine monitoring"]
        
        return SceneAnalysis(
            threat_level=threat_level,
            description=description,
            recommendations=recommendations,
            confidence=0.8,
            reasoning="Rule-based fallback analysis"
        )

class SmartReportGenerator:
    """Generate intelligent reports from scene analysis."""
    
    def __init__(self):
        self.incident_history = []
        self.max_history = 100
    
    def generate_incident_report(self, analysis: SceneAnalysis, detections: List[Dict], 
                               timestamp: str) -> Dict:
        """Generate a detailed incident report."""
        report = {
            'timestamp': timestamp,
            'threat_level': analysis.threat_level,
            'summary': analysis.description,
            'ai_reasoning': analysis.reasoning,
            'confidence_score': analysis.confidence,
            'recommendations': analysis.recommendations,
            'detection_details': self._format_detection_details(detections),
            'incident_id': f"INC_{int(time.time())}_{len(self.incident_history):03d}"
        }
        
        self.incident_history.append(report)
        if len(self.incident_history) > self.max_history:
            self.incident_history.pop(0)
        
        return report
    
    def _format_detection_details(self, detections: List[Dict]) -> List[Dict]:
        """Format detection details for the report."""
        formatted = []
        for detection in detections:
            formatted.append({
                'object': detection.get('class_name', 'unknown'),
                'confidence': detection.get('confidence', 0),
                'dangerous': detection.get('is_dangerous', False),
                'detection_method': detection.get('detection_method', 'standard'),
                'weapon_score': detection.get('weapon_score', 0)
            })
        return formatted
    
    def get_recent_incidents(self, count: int = 10) -> List[Dict]:
        """Get recent incident reports."""
        return self.incident_history[-count:] if self.incident_history else []
    
    def get_threat_statistics(self) -> Dict:
        """Get threat level statistics."""
        if not self.incident_history:
            return {}
        
        threat_counts = {}
        for incident in self.incident_history:
            level = incident.get('threat_level', 'unknown')
            threat_counts[level] = threat_counts.get(level, 0) + 1
        
        return {
            'total_incidents': len(self.incident_history),
            'threat_distribution': threat_counts,
            'high_threat_incidents': threat_counts.get('high', 0) + threat_counts.get('critical', 0)
        }
