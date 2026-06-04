import cv2
import numpy as np
import yaml
import logging
import time
import os
import json
import requests
import threading
import queue
from datetime import datetime
from ultralytics import YOLO
from typing import List, Dict, Any
from io import BytesIO

from alert_system import AlertSystem
from paths import resolve_weapon_model_path
from scene_analyzer import LocalLLMAnalyzer, SmartReportGenerator
from weapon_detector import WeaponDetector


class RemoteVideoStream:
    """Handles streaming from ESP32-CAM or other remote video sources"""
    def __init__(self, stream_url: str, reconnect_attempts: int = 5):
        self.stream_url = stream_url
        self.reconnect_attempts = reconnect_attempts
        self.running = False
        self.frame_buffer = queue.Queue(maxsize=5)
        self.current_frame = None
        self.stream_thread = None
        self.connection_active = False
        self.frame_timeout = 0.1
        
    def start(self):
        """Start the video stream thread"""
        self.running = True
        self.stream_thread = threading.Thread(target=self._stream_worker, daemon=True)
        self.stream_thread.start()
        logging.info(f"Remote video stream started: {self.stream_url}")
    
    def stop(self):
        """Stop the video stream"""
        self.running = False
        if self.stream_thread:
            self.stream_thread.join(timeout=5)
        logging.info("Remote video stream stopped")
    
    def read(self) -> tuple:
        try:
            self.current_frame = self.frame_buffer.get(timeout=self.frame_timeout)
            return True, self.current_frame
        except queue.Empty:
            if self.current_frame is not None:
                return True, self.current_frame
            return False, None
    
    def _stream_worker(self):
        """Worker thread for streaming video"""
        retry_count = 0
        
        while self.running:
            try:
                logging.info(f"Connecting to stream... (attempt {retry_count + 1}/{self.reconnect_attempts})")
                logging.info(f"Stream URL: {self.stream_url}")
                response = requests.get(self.stream_url, stream=True, timeout=(10, 30))
                bytes_data = b''
                retry_count = 0
                self.connection_active = True
                logging.info("Stream connection established successfully")
                
                while self.running:
                    try:
                        for chunk in response.iter_content(chunk_size=4096):
                            if not self.running:
                                break
                            
                            if chunk:
                                bytes_data += chunk
                                
                                a = bytes_data.find(b'\xff\xd8')
                                b = bytes_data.find(b'\xff\xd9')
                                
                                if a != -1 and b != -1 and b > a:
                                    jpg = bytes_data[a:b+2]
                                    bytes_data = bytes_data[b+2:]
                                    
                                    if len(jpg) > 100:
                                        frame = cv2.imdecode(np.frombuffer(jpg, dtype=np.uint8), cv2.IMREAD_COLOR)
                                        
                                        if frame is not None:
                                            try:
                                                self.frame_buffer.put_nowait(frame)
                                            except queue.Full:
                                                try:
                                                    self.frame_buffer.get_nowait()
                                                    self.frame_buffer.put_nowait(frame)
                                                except:
                                                    pass
                    
                    except (ConnectionError, ConnectionResetError, BrokenPipeError):
                        logging.warning("Connection lost, reconnecting...")
                        self.connection_active = False
                        break
            
            except requests.exceptions.Timeout:
                retry_count += 1
                self.connection_active = False
                logging.warning(f"Timeout, retrying... ({retry_count}/{self.reconnect_attempts})")
                time.sleep(2)
            
            except requests.exceptions.ConnectionError:
                retry_count += 1
                self.connection_active = False
                logging.warning(f"Connection failed, retrying... ({retry_count}/{self.reconnect_attempts})")
                time.sleep(2)
            
            except Exception as e:
                retry_count += 1
                self.connection_active = False
                logging.error(f"Stream error: {e}")
                time.sleep(2)
            
            if retry_count >= self.reconnect_attempts and self.running:
                logging.error("Max retries reached for remote stream")
                self.running = False
                break
    
    def isOpened(self) -> bool:
        """Check if stream is active (compatible with cv2.VideoCapture interface)"""
        return self.connection_active
    
    def set(self, prop_id: int, value: float) -> bool:
        """Dummy method for compatibility with cv2.VideoCapture interface"""
        return True
    
    def release(self):
        """Release the stream"""
        self.stop()


class AIDetectionSystem:
    def __init__(self, config_path: str = "config.yaml"):
        self.config = self.load_config(config_path)
        self.setup_logging()
        
        self.human_model = None
        self.object_model = None
        self.weapon_model = None
        self.load_models()
        
        self.weapon_detector = WeaponDetector(self.config)
        self.alert_system = AlertSystem(self.config)
        self.llm_analyzer = LocalLLMAnalyzer(self.config)
        self.report_generator = SmartReportGenerator()
        
        self.camera = None
        self.setup_camera()
        
        self.last_llm_analysis = 0
        self.llm_analysis_interval = self.config.get('llm', {}).get('analysis_interval', 2.0)
        self.frame_count = 0
        self.start_time = time.time()
        
        os.makedirs(self.config['logging']['output_dir'], exist_ok=True)
    
    def load_config(self, config_path: str) -> Dict[str, Any]:
        try:
            with open(config_path, 'r') as file:
                return yaml.safe_load(file)
        except FileNotFoundError:
            logging.error(f"Config file {config_path} not found")
            return self.get_default_config()
    
    def get_default_config(self) -> Dict[str, Any]:
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
                'weapon_model': 'yolov8n.pt',
                'use_remote_stream': True,
                'remote_stream_url': 'http://YOUR_ESP32_IP:80/stream',
                'remote_reconnect_attempts': 5
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
        try:
            logging.info("Loading detection models...")
            self.human_model = YOLO(self.config['detection']['human_model'])
            self.object_model = YOLO(self.config['detection']['object_model'])
            weapon_path = resolve_weapon_model_path(self.config)
            self.weapon_model = YOLO(weapon_path)
            logging.info('Models loaded (weapon: %s)', weapon_path)
        except Exception as e:
            logging.error(f"Error loading models: {e}")
            raise
    
    def setup_camera(self):
        try:
            if os.environ.get("DRONEAI_SKIP_CAMERA", "").strip() in ("1", "true", "yes"):
                logging.info("DRONEAI_SKIP_CAMERA set - skipping camera init (tests)")
                self.camera = None
                return

            use_remote = self.config['detection'].get('use_remote_stream', False)
            
            if use_remote:
                stream_url = self.config['detection'].get(
                    'remote_stream_url', 'http://YOUR_ESP32_IP:80/stream'
                )
                reconnect_attempts = self.config['detection'].get('remote_reconnect_attempts', 5)
                self.camera = RemoteVideoStream(stream_url, reconnect_attempts)
                self.camera.start()
                logging.info(f"Remote camera stream initialized: {stream_url}")
                
                connection_timeout = 10
                start_time = time.time()
                while not self.camera.isOpened() and (time.time() - start_time) < connection_timeout:
                    time.sleep(0.5)
                    logging.info("Waiting for remote stream connection...")
                
                if not self.camera.isOpened():
                    raise RuntimeError(f"Could not connect to remote stream after {connection_timeout} seconds")
            else:
                self.camera = cv2.VideoCapture(self.config['detection']['camera_index'])
                self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, self.config['detection']['frame_width'])
                self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, self.config['detection']['frame_height'])
                self.camera.set(cv2.CAP_PROP_FPS, self.config['detection']['fps'])
                logging.info("Local camera initialized")
                
                if not self.camera.isOpened():
                    raise RuntimeError("Could not open local camera")
            
            logging.info("Camera/Stream initialized successfully")
        except Exception as e:
            logging.error(f"Error initializing camera: {e}")
            raise
    
    def detect_objects(self, frame: np.ndarray) -> List[Dict]:
        detections = []
        try:
            results = self.object_model(frame, conf=self.config['detection']['confidence_threshold'])
            for result in results:
                if result.boxes is not None:
                    for box in result.boxes:
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                        confidence = float(box.conf[0].cpu().numpy())
                        class_id = int(box.cls[0].cpu().numpy())
                        class_name = self.object_model.names[class_id]
                        detection_type = 'human' if class_name == 'person' else 'object'
                        detections.append({
                            'bbox': (int(x1), int(y1), int(x2), int(y2)),
                            'confidence': confidence,
                            'class_name': class_name,
                            'type': detection_type,
                            'is_dangerous': self.is_dangerous_object(class_name)
                        })
            
            weapon_detections = self.weapon_detector.detect_weapons(frame)
            for weapon_detection in weapon_detections:
                weapon_detection['is_dangerous'] = weapon_detection.get('is_weapon', False)
                detections.append(weapon_detection)
        except Exception as e:
            logging.error(f"Error during detection: {e}")
        
        return detections
    
    def is_dangerous_object(self, class_name: str) -> bool:
        dangerous_objects = self.config['alerts']['dangerous_objects']
        return any(dangerous_item.lower() in class_name.lower() for dangerous_item in dangerous_objects)
    
    def draw_detections(self, frame: np.ndarray, detections: List[Dict]) -> np.ndarray:
        for detection in detections:
            x1, y1, x2, y2 = detection['bbox']
            confidence = detection['confidence']
            class_name = detection['class_name']
            is_dangerous = detection.get('is_dangerous', False)
            
            color = (0, 0, 255) if is_dangerous else (0, 255, 0) if detection.get('type') == 'human' else (255, 0, 0)
            thickness = 3 if is_dangerous else 2
            
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
            
            label = f"{class_name}: {confidence:.2f}" if self.config['display']['show_confidence'] else class_name
            if is_dangerous:
                label = f"ALERT {label}"
            
            font_scale = self.config['display']['font_scale']
            (label_w, label_h), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 1)
            
            cv2.rectangle(frame, (x1, y1 - label_h - 10), (x1 + label_w + 10, y1), color, -1)
            cv2.putText(frame, label, (x1 + 5, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), 1)
        
        return frame
    
    def add_info_overlay(self, frame: np.ndarray) -> np.ndarray:
        height, width = frame.shape[:2]
        
        if self.config['display']['show_fps']:
            self.frame_count += 1
            elapsed_time = time.time() - self.start_time
            fps = self.frame_count / elapsed_time if elapsed_time > 0 else 0
            cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        title = self.config['display']['window_title']
        cv2.putText(frame, title, (10, height - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        cv2.circle(frame, (width - 30, 30), 10, (0, 255, 0), -1)
        
        return frame
    
    def run(self):
        logging.info("Starting AI Detection System")
        if self.camera is None:
            logging.error("No camera/stream configured. Cannot run detection loop.")
            return
        self.alert_system.start()
        
        consecutive_failures = 0
        max_consecutive_failures = 30
        
        try:
            while True:
                ret, frame = self.camera.read()
                if not ret or frame is None:
                    consecutive_failures += 1
                    if consecutive_failures > max_consecutive_failures:
                        logging.error(f"Failed to capture frame {consecutive_failures} times, stopping")
                        break
                    logging.warning(f"Failed to capture frame (attempt {consecutive_failures}/{max_consecutive_failures})")
                    time.sleep(0.1)
                    continue
                
                consecutive_failures = 0
                
                detections = self.detect_objects(frame)
                
                current_time = time.time()
                if current_time - self.last_llm_analysis > self.llm_analysis_interval:
                    self.last_llm_analysis = current_time
                    frame_info = {'fps': self.frame_count / (current_time - self.start_time) if current_time > self.start_time else 0, 'timestamp': datetime.now().strftime('%H:%M:%S')}
                    self.llm_analyzer.analyze_scene_async(detections, frame_info)
                
                latest_analysis = self.llm_analyzer.get_latest_analysis()
                if latest_analysis and latest_analysis.threat_level in ['high', 'critical']:
                    report = self.report_generator.generate_incident_report(latest_analysis, detections, datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
                    logging.critical(f"INCIDENT REPORT GENERATED: {report['incident_id']}")
                    self._save_incident_evidence(frame, detections, report)
                
                dangerous_detections = [d for d in detections if d.get('is_dangerous', False)]
                if dangerous_detections:
                    self.alert_system.trigger_alert(dangerous_detections, frame)
                    self._log_threat_details(dangerous_detections)
                
                frame = self.draw_detections(frame, detections)
                frame = self.alert_system.draw_alert_overlay(frame, detections)
                frame = self.add_info_overlay(frame)
                
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
    
    def _log_threat_details(self, dangerous_detections: List[Dict]):
        for detection in dangerous_detections:
            class_name = detection.get('class_name', 'unknown')
            confidence = detection.get('confidence', 0)
            weapon_score = detection.get('weapon_score', 0)
            bbox = detection.get('bbox', (0, 0, 0, 0))
            
            logging.critical(f"THREAT DETECTED: {class_name}")
            logging.critical(f"  Confidence: {confidence:.2%}")
            logging.critical(f"  Weapon Score: {weapon_score:.2%}")
            logging.critical(f"  Location: {bbox}")
    
    def _save_incident_evidence(self, frame: np.ndarray, detections: List[Dict], report: Dict):
        try:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            evidence_dir = os.path.join(self.config['logging']['output_dir'], 'incidents')
            os.makedirs(evidence_dir, exist_ok=True)
            
            frame_path = os.path.join(evidence_dir, f"incident_{report['incident_id']}_frame.jpg")
            report_path = os.path.join(evidence_dir, f"incident_{report['incident_id']}_report.json")
            
            annotated = frame.copy()
            for detection in detections:
                if detection.get('is_dangerous', False):
                    x1, y1, x2, y2 = detection['bbox']
                    cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 0, 255), 3)
                    cv2.putText(annotated, f"THREAT: {detection['class_name']}", (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            
            cv2.imwrite(frame_path, annotated)
            with open(report_path, 'w') as f:
                json.dump(report, f, indent=2)
            
            logging.critical(f"EVIDENCE SAVED: {frame_path}")
            logging.critical(f"REPORT SAVED: {report_path}")
        except Exception as e:
            logging.error(f"Failed to save incident evidence: {e}")
    
    def cleanup(self):
        logging.info("Detection system shutdown")
        if hasattr(self, 'alert_system'):
            self.alert_system.stop()
        if hasattr(self, 'llm_analyzer'):
            self.llm_analyzer.stop_analyzer()
        if self.camera:
            if isinstance(self.camera, RemoteVideoStream):
                self.camera.stop()
            else:
                self.camera.release()
        try:
            cv2.destroyAllWindows()
        except cv2.error:
            pass
        logging.info("System shutdown complete")

if __name__ == "__main__":
    try:
        detection_system = AIDetectionSystem()
        detection_system.run()
    except Exception as e:
        logging.error(f"Failed to start: {e}")
        print(f"Error: {e}")