import cv2
import numpy as np
import yaml
import logging
import time
import os
import requests
import threading
import queue
from datetime import datetime
from ultralytics import YOLO
from typing import List, Dict, Any
from io import BytesIO

from alert_system import AlertSystem
from detection.pipeline import DetectionPipeline
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
        try:
            self.pipeline.tick_fps()
            return self.pipeline.process_frame(frame)
        except Exception as e:
            logging.error(f"Error during detection: {e}")
            return []
    
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
                    self.pipeline.save_incident(frame, detections, report)
                
                dangerous_detections = self.pipeline.get_dangerous(detections)
                if dangerous_detections and self.pipeline.should_fire_alert(detections):
                    self.alert_system.trigger_alert(dangerous_detections, frame)
                    self._log_threat_details(dangerous_detections)
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