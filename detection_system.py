import cv2
import numpy as np
import yaml
import logging
import time
import os
import pygame
from datetime import datetime
from ultralytics import YOLO
from typing import List, Tuple, Dict, Any
import threading
import queue
from detection.pipeline import DetectionPipeline
from paths import resolve_weapon_model_path
from weapon_detector import WeaponDetector
from alert_system import AlertSystem
from scene_analyzer import LocalLLMAnalyzer, SmartReportGenerator

class AIDetectionSystem:
    def __init__(self, config_path: str = "config.yaml"):
        """Initialize the AI Detection System with configuration."""
        self.config = self.load_config(config_path)
        self.setup_logging()
        
        self.human_model = None
        self.object_model = None
        self.weapon_model = None
        self.load_models()
        
        self.weapon_detector = WeaponDetector(self.config)
        self.pipeline = DetectionPipeline(
            self.config,
            self.object_model,
            self.weapon_detector,
            self.human_model,
        )
        self.alert_system = AlertSystem(self.config)
        self.llm_analyzer = LocalLLMAnalyzer(self.config)
        self.report_generator = SmartReportGenerator()
        
        self.camera = None
        self.setup_camera()
        
        self.last_llm_analysis = 0
        self.llm_analysis_interval = self.config.get('llm', {}).get('analysis_interval', 2.0)
        self.current_scene_analysis = None
        
        self.last_alert_time = 0
        self.detection_history = []
        self.frame_count = 0
        self.start_time = time.time()
        
        os.makedirs(self.config['logging']['output_dir'], exist_ok=True)
        
    def load_config(self, config_path: str) -> Dict[str, Any]:
        """Load configuration from YAML file."""
        try:
            with open(config_path, 'r') as file:
                return yaml.safe_load(file)
        except FileNotFoundError:
            logging.error(f"Config file {config_path} not found. Using default settings.")
            return self.get_default_config()
    
    def get_default_config(self) -> Dict[str, Any]:
        """Return default configuration if config file is not found."""
        return {
            'detection': {
                'confidence_threshold': 0.5,
                'nms_threshold': 0.4,
                'input_size': 640,
                'camera_index': 0,
                'frame_width': 1280,
                'frame_height': 720,
                'fps': 30,
                'human_model': 'yolov8n.pt',
                'object_model': 'yolov8n.pt',
                'weapon_model': 'yolov8n.pt'
            },
            'alerts': {
                'enable_audio': True,
                'enable_visual': True,
                'alert_duration': 3.0,
                'dangerous_objects': ['knife', 'gun', 'rifle', 'pistol', 'weapon']
            },
            'logging': {
                'level': 'INFO',
                'save_detections': True,
                'output_dir': 'detections'
            },
            'display': {
                'window_title': 'AI Security Detection System',
                'show_confidence': True,
                'show_fps': True,
                'bbox_thickness': 2,
                'font_scale': 0.6
            }
        }
    
    def setup_logging(self):
        """Setup logging configuration."""
        log_level = getattr(logging, self.config['logging']['level'])
        logging.basicConfig(
            level=log_level,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('detection.log'),
                logging.StreamHandler()
            ]
        )
        
    def load_models(self):
        """Load YOLO models for detection."""
        try:
            logging.info("Loading detection models...")
            
            self.human_model = YOLO(self.config['detection']['human_model'])
            self.object_model = YOLO(self.config['detection']['object_model'])
            
            weapon_path = resolve_weapon_model_path(self.config)
            self.weapon_model = YOLO(weapon_path)
            logging.info('Weapon model: %s', weapon_path)
            logging.info("Models loaded successfully")
            
        except Exception as e:
            logging.error(f"Error loading models: {e}")
            raise
    
    def setup_camera(self):
        """Initialize camera capture."""
        try:
            self.camera = cv2.VideoCapture(self.config['detection']['camera_index'])
            self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, self.config['detection']['frame_width'])
            self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, self.config['detection']['frame_height'])
            self.camera.set(cv2.CAP_PROP_FPS, self.config['detection']['fps'])
            
            if not self.camera.isOpened():
                raise RuntimeError("Could not open camera")
                
            logging.info("Camera initialized successfully")
            
        except Exception as e:
            logging.error(f"Error initializing camera: {e}")
            raise
    
    def create_alert_sound(self):
        """Create an alert sound for dangerous object detection."""
        duration = 0.5
        sample_rate = 22050
        frames = int(duration * sample_rate)
        arr = np.zeros(frames)
        
        for i in range(frames):
            arr[i] = np.sin(2 * np.pi * 800 * i / sample_rate) * 0.5
        
        arr = (arr * 32767).astype(np.int16)
        sound = pygame.sndarray.make_sound(arr)
        return sound
    
    def detect_objects(self, frame: np.ndarray) -> List[Dict]:
        try:
            self.pipeline.tick_fps()
            return self.pipeline.process_frame(frame)
        except Exception as e:
            logging.error(f"Error during detection: {e}")
            return []
    
    def categorize_detection(self, class_name: str) -> str:
        """Categorize detected objects."""
        if class_name == 'person':
            return 'human'
        elif class_name in ['knife', 'scissors']:
            return 'dangerous_object'
        else:
            return 'object'
    
    def is_dangerous_object(self, class_name: str) -> bool:
        """Check if detected object is dangerous."""
        dangerous_objects = self.config['alerts']['dangerous_objects']
        return any(dangerous_item.lower() in class_name.lower() for dangerous_item in dangerous_objects)
    
    def draw_detections(self, frame: np.ndarray, detections: List[Dict]) -> np.ndarray:
        """Draw bounding boxes and labels on the frame."""
        for detection in detections:
            x1, y1, x2, y2 = detection['bbox']
            confidence = detection['confidence']
            class_name = detection['class_name']
            is_dangerous = detection.get('is_dangerous', False)
            detection_method = detection.get('detection_method', 'standard')
            
            if is_dangerous:
                if 'multi_angle' in detection_method:
                    color = (0, 0, 200)  # Dark red for multi-angle weapon detections
                    thickness = 4
                elif 'multi_scale' in detection_method:
                    color = (0, 50, 255)  # Bright red for multi-scale weapon detections
                    thickness = 4
                else:
                    color = (0, 0, 255)  # Red for dangerous objects
                    thickness = 3
            elif detection.get('type') == 'human':
                color = (0, 255, 0)  # Green for humans
                thickness = 2
            else:
                color = (255, 0, 0)  # Blue for other objects
                thickness = 2
            
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
            
            if self.config['display']['show_confidence']:
                label = f"{class_name}: {confidence:.2f}"
            else:
                label = class_name
            
            if 'multi_angle' in detection_method:
                angle_type = detection_method.split('_')[-1] if '_' in detection_method else ''
                label += f" [{angle_type[:3]}]"
            elif 'multi_scale' in detection_method:
                scale = detection_method.split('_')[-1] if '_' in detection_method else ''
                label += f" [{scale}]"
            elif 'region_focused' in detection_method:
                label += " [focus]"
            
            if is_dangerous:
                label = f"ALERT {label}"
            
            font_scale = self.config['display']['font_scale']
            font_thickness = 1
            (label_w, label_h), baseline = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness
            )
            
            padding = 5
            bg_y1 = max(0, y1 - label_h - padding * 2)
            bg_y2 = y1
            bg_x2 = min(frame.shape[1], x1 + label_w + padding)
            
            cv2.rectangle(frame, (x1, bg_y1), (bg_x2, bg_y2), color, -1)
            
            text_y = y1 - padding
            cv2.putText(
                frame, label, (x1 + 2, text_y),
                cv2.FONT_HERSHEY_SIMPLEX, font_scale,
                (255, 255, 255), font_thickness
            )
        
        return frame
    
    def handle_dangerous_detection(self, detections: List[Dict]):
        """Handle detection of dangerous objects."""
        dangerous_detections = [d for d in detections if d['is_dangerous']]
        
        if dangerous_detections and time.time() - self.last_alert_time > self.config['alerts']['alert_duration']:
            self.last_alert_time = time.time()
            
            for detection in dangerous_detections:
                logging.warning(f"DANGEROUS OBJECT DETECTED: {detection['class_name']} "
                              f"(confidence: {detection['confidence']:.2f})")
            
            if self.config['alerts']['enable_audio'] and hasattr(self, 'alert_sound'):
                try:
                    self.alert_sound.play()
                except:
                    pass
            
            if self.config['logging']['save_detections']:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"dangerous_detection_{timestamp}.jpg"
                filepath = os.path.join(self.config['logging']['output_dir'], filename)
    
    def add_info_overlay(self, frame: np.ndarray) -> np.ndarray:
        """Add information overlay to the frame."""
        height, width = frame.shape[:2]
        
        if self.config['display']['show_fps']:
            self.frame_count += 1
            elapsed_time = time.time() - self.start_time
            fps = self.frame_count / elapsed_time if elapsed_time > 0 else 0
            
            cv2.putText(
                frame, f"FPS: {fps:.1f}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2
            )
        
        title = self.config['display']['window_title']
        cv2.putText(
            frame, title, (10, height - 20),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2
        )
        
        status_color = (0, 255, 0)  # Green for normal
        cv2.circle(frame, (width - 30, 30), 10, status_color, -1)
        
        return frame
    
    def run(self):
        """Main detection loop."""
        logging.info("Starting AI Detection System...")
        
        self.alert_system.start()
        
        try:
            while True:
                ret, frame = self.camera.read()
                if not ret:
                    logging.error("Failed to capture frame from camera")
                    break
                
                detections = self.detect_objects(frame)
                
                current_time = time.time()
                if current_time - self.last_llm_analysis > self.llm_analysis_interval:
                    self.last_llm_analysis = current_time
                    frame_info = {
                        'fps': self.frame_count / (current_time - self.start_time) if current_time > self.start_time else 0,
                        'timestamp': datetime.now().strftime('%H:%M:%S')
                    }
                    self.llm_analyzer.analyze_scene_async(detections, frame_info)
                
                latest_analysis = self.llm_analyzer.get_latest_analysis()
                if latest_analysis:
                    self.current_scene_analysis = latest_analysis
                    
                    if latest_analysis.threat_level in ['high', 'critical']:
                        report = self.report_generator.generate_incident_report(
                            latest_analysis, detections, datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                        )
                        logging.info(f"Incident report generated: {report['incident_id']}")
                
                dangerous_detections = self.pipeline.get_dangerous(detections)
                if dangerous_detections and self.pipeline.should_fire_alert(detections):
                    self.alert_system.trigger_alert(dangerous_detections, frame)
                    self.pipeline.save_threat(frame, dangerous_detections)

                frame = self.pipeline.draw_frame(frame, detections)
                frame = self.alert_system.draw_alert_overlay(frame, detections)
                
                cv2.imshow(self.config['display']['window_title'], frame)
                
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    logging.info("Exit requested by user")
                    break
                    
        except KeyboardInterrupt:
            logging.info("Detection stopped by user")
        except Exception as e:
            logging.error(f"Error in main loop: {e}")
        finally:
            self.cleanup()
    
    def cleanup(self):
        """Clean up resources."""
        logging.info("Cleaning up resources...")
        
        if hasattr(self, 'alert_system'):
            self.alert_system.stop()
        
        if hasattr(self, 'llm_analyzer'):
            self.llm_analyzer.stop_analyzer()
        
        if self.camera:
            self.camera.release()
        
        try:
            cv2.destroyAllWindows()
        except cv2.error:
            pass
        
        if hasattr(self, 'alert_sound'):
            pygame.mixer.quit()
        
        logging.info("Cleanup completed")

if __name__ == "__main__":
    try:
        detection_system = AIDetectionSystem()
        detection_system.run()
    except Exception as e:
        logging.error(f"Failed to start detection system: {e}")
        print(f"Error: {e}")
