import cv2
import numpy as np
import pygame
import logging
import time
import threading
import queue
from datetime import datetime
from typing import Dict, List, Optional
import os

class AlertSystem:
    """Advanced alert system for dangerous object detection."""
    
    def __init__(self, config: dict):
        self.config = config
        self.alert_config = config['alerts']
        
        self.last_alert_time = 0
        self.alert_duration = self.alert_config.get('alert_duration', 3.0)
        self.is_alerting = False
        self.alert_queue = queue.Queue()
        
        if self.alert_config.get('enable_audio', True):
            self.init_audio()
        
        self.alert_history = []
        self.max_history = 100
        
        self.alert_thread = None
        self.running = False
        
        self.alert_sounds = {}
        self.create_alert_sounds()
        
    def init_audio(self):
        """Initialize pygame audio system."""
        try:
            pygame.mixer.init(frequency=22050, size=-16, channels=2, buffer=512)
            logging.info("Audio system initialized")
        except Exception as e:
            logging.error(f"Failed to initialize audio: {e}")
            self.alert_config['enable_audio'] = False
    
    def create_alert_sounds(self):
        """Create different alert sounds for different threat levels."""
        if not self.alert_config.get('enable_audio', True):
            return
        
        try:
            self.alert_sounds['high'] = self.generate_alert_sound(800, 0.5, 'high')
            
            self.alert_sounds['medium'] = self.generate_alert_sound(600, 0.3, 'medium')
            
            self.alert_sounds['low'] = self.generate_alert_sound(400, 0.2, 'low')
            
            logging.info("Alert sounds created")
            
        except Exception as e:
            logging.error(f"Error creating alert sounds: {e}")
    
    def generate_alert_sound(self, frequency: int, duration: float, alert_type: str) -> Optional[pygame.mixer.Sound]:
        """Generate an alert sound with specific characteristics."""
        try:
            sample_rate = 22050
            frames = int(duration * sample_rate)
            arr = np.zeros(frames)
            
            if alert_type == 'high':
                for i in range(frames):
                    t = i / sample_rate
                    freq_mod = frequency + 200 * np.sin(2 * np.pi * 5 * t)
                    arr[i] = 0.7 * np.sin(2 * np.pi * freq_mod * t)
                    
            elif alert_type == 'medium':
                for i in range(frames):
                    t = i / sample_rate
                    arr[i] = 0.5 * np.sin(2 * np.pi * frequency * t)
                    
            elif alert_type == 'low':
                for i in range(frames):
                    t = i / sample_rate
                    arr[i] = 0.3 * np.sin(2 * np.pi * frequency * t) * np.exp(-2 * t)
            
            arr = (arr * 32767).astype(np.int16)
            
            stereo_arr = np.zeros((frames, 2), dtype=np.int16)
            stereo_arr[:, 0] = arr
            stereo_arr[:, 1] = arr
            
            sound = pygame.sndarray.make_sound(stereo_arr)
            return sound
            
        except Exception as e:
            logging.error(f"Error generating {alert_type} alert sound: {e}")
            return None
    
    def start(self):
        """Start the alert system."""
        self.running = True
        self.alert_thread = threading.Thread(target=self._alert_worker, daemon=True)
        self.alert_thread.start()
        logging.info("Alert system started")
    
    def stop(self):
        """Stop the alert system."""
        self.running = False
        if self.alert_thread:
            self.alert_thread.join(timeout=1.0)
        logging.info("Alert system stopped")
    
    def trigger_alert(self, detections: List[Dict], frame: Optional[np.ndarray] = None):
        """Trigger alerts based on detections."""
        if not detections:
            return
        
        current_time = time.time()
        
        if current_time - self.last_alert_time < self.alert_duration:
            return
        
        threat_level = self._assess_threat_level(detections)
        
        if threat_level == 'none':
            return
        
        alert_data = {
            'timestamp': current_time,
            'threat_level': threat_level,
            'detections': detections,
            'frame': frame.copy() if frame is not None else None
        }
        
        try:
            self.alert_queue.put_nowait(alert_data)
            self.last_alert_time = current_time
        except queue.Full:
            logging.warning("Alert queue is full")
    
    def _assess_threat_level(self, detections: List[Dict]) -> str:
        """Assess the threat level based on detections."""
        max_weapon_score = 0.0
        dangerous_count = 0
        
        for detection in detections:
            if detection.get('is_dangerous', False) or detection.get('is_weapon', False):
                dangerous_count += 1
                weapon_score = detection.get('weapon_score', detection.get('confidence', 0))
                max_weapon_score = max(max_weapon_score, weapon_score)
        
        if max_weapon_score > 0.8 or dangerous_count > 2:
            return 'high'
        elif max_weapon_score > 0.5 or dangerous_count > 0:
            return 'medium'
        elif max_weapon_score > 0.3:
            return 'low'
        else:
            return 'none'
    
    def _alert_worker(self):
        """Worker thread for processing alerts."""
        while self.running:
            try:
                alert_data = self.alert_queue.get(timeout=0.1)
                
                self._process_alert(alert_data)
                
                self._add_to_history(alert_data)
                
                self.alert_queue.task_done()
                
            except queue.Empty:
                continue
            except Exception as e:
                logging.error(f"Error in alert worker: {e}")
    
    def _process_alert(self, alert_data: Dict):
        """Process a single alert."""
        threat_level = alert_data['threat_level']
        detections = alert_data['detections']
        timestamp = alert_data['timestamp']
        
        self._log_alert(alert_data)
        
        if self.alert_config.get('enable_audio', True):
            self._play_audio_alert(threat_level)
        
        if self.config['logging'].get('save_detections', True) and alert_data['frame'] is not None:
            self._save_alert_image(alert_data)
        
        self._send_notifications(alert_data)
    
    def _log_alert(self, alert_data: Dict):
        """Log alert information."""
        threat_level = alert_data['threat_level']
        detections = alert_data['detections']
        timestamp = datetime.fromtimestamp(alert_data['timestamp'])
        
        log_message = f"SECURITY ALERT [{threat_level.upper()}] at {timestamp}: "
        
        for detection in detections:
            if detection.get('is_dangerous', False) or detection.get('is_weapon', False):
                class_name = detection.get('class_name', 'unknown')
                confidence = detection.get('confidence', 0)
                weapon_score = detection.get('weapon_score', confidence)
                log_message += f"{class_name} (confidence: {confidence:.2f}, weapon_score: {weapon_score:.2f}); "
        
        if threat_level == 'high':
            logging.critical(log_message)
        elif threat_level == 'medium':
            logging.warning(log_message)
        else:
            logging.info(log_message)
    
    def _play_audio_alert(self, threat_level: str):
        """Play audio alert based on threat level."""
        try:
            sound = self.alert_sounds.get(threat_level)
            if sound:
                sound.play()
                
                if threat_level == 'high':
                    threading.Timer(0.6, lambda: sound.play()).start()
                    threading.Timer(1.2, lambda: sound.play()).start()
                    
        except Exception as e:
            logging.error(f"Error playing audio alert: {e}")
    
    def _save_alert_image(self, alert_data: Dict):
        """Save the frame that triggered the alert."""
        try:
            frame = alert_data['frame']
            timestamp = datetime.fromtimestamp(alert_data['timestamp'])
            threat_level = alert_data['threat_level']
            
            filename = f"alert_{threat_level}_{timestamp.strftime('%Y%m%d_%H%M%S')}.jpg"
            filepath = os.path.join(self.config['logging']['output_dir'], filename)
            
            annotated_frame = self._annotate_alert_frame(frame, alert_data['detections'])
            
            cv2.imwrite(filepath, annotated_frame)
            logging.info(f"Alert image saved: {filepath}")
            
        except Exception as e:
            logging.error(f"Error saving alert image: {e}")
    
    def _annotate_alert_frame(self, frame: np.ndarray, detections: List[Dict]) -> np.ndarray:
        """Annotate frame with alert information."""
        annotated = frame.copy()
        
        timestamp_str = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        cv2.putText(annotated, f"ALERT: {timestamp_str}", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
        
        for detection in detections:
            if detection.get('is_dangerous', False) or detection.get('is_weapon', False):
                x1, y1, x2, y2 = detection['bbox']
                
                cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 0, 255), 4)
                
                label = f"DANGER: {detection.get('class_name', 'unknown')}"
                cv2.putText(annotated, label, (x1, y1 - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        
        return annotated
    
    def _send_notifications(self, alert_data: Dict):
        """Send notifications (placeholder for future expansion)."""
        pass
    
    def _add_to_history(self, alert_data: Dict):
        """Add alert to history."""
        history_entry = {
            'timestamp': alert_data['timestamp'],
            'threat_level': alert_data['threat_level'],
            'detection_count': len(alert_data['detections']),
            'detections_summary': [
                {
                    'class_name': d.get('class_name', 'unknown'),
                    'confidence': d.get('confidence', 0),
                    'weapon_score': d.get('weapon_score', 0)
                }
                for d in alert_data['detections']
                if d.get('is_dangerous', False) or d.get('is_weapon', False)
            ]
        }
        
        self.alert_history.append(history_entry)
        
        if len(self.alert_history) > self.max_history:
            self.alert_history.pop(0)
    
    def get_recent_alerts(self, count: int = 10) -> List[Dict]:
        """Get recent alerts from history."""
        return self.alert_history[-count:] if self.alert_history else []
    
    def get_alert_stats(self) -> Dict:
        """Get alert statistics."""
        if not self.alert_history:
            return {'total': 0, 'high': 0, 'medium': 0, 'low': 0}
        
        stats = {'total': len(self.alert_history), 'high': 0, 'medium': 0, 'low': 0}
        
        for alert in self.alert_history:
            threat_level = alert.get('threat_level', 'low')
            stats[threat_level] = stats.get(threat_level, 0) + 1
        
        return stats
    
    def draw_alert_overlay(self, frame: np.ndarray, current_detections: List[Dict]) -> np.ndarray:
        """Draw alert status overlay on frame."""
        if not self.alert_config.get('enable_visual', True):
            return frame
        
        overlay = frame.copy()
        height, width = frame.shape[:2]
        
        dangerous_detections = [d for d in current_detections 
                              if d.get('is_dangerous', False) or d.get('is_weapon', False)]
        
        if dangerous_detections:
            if int(time.time() * 4) % 2:  # Flash at 2Hz
                overlay_color = (0, 0, 255)  # Red
                alpha = 0.2
                overlay[:] = cv2.addWeighted(overlay, 1-alpha, 
                                           np.full_like(overlay, overlay_color), alpha, 0)
            
            warning_text = "DANGER DETECTED"
            text_size = cv2.getTextSize(warning_text, cv2.FONT_HERSHEY_SIMPLEX, 1.5, 3)[0]
            text_x = (width - text_size[0]) // 2
            text_y = 60
            
            cv2.rectangle(overlay, (text_x - 10, text_y - 40), 
                         (text_x + text_size[0] + 10, text_y + 10), (0, 0, 0), -1)
            
            cv2.putText(overlay, warning_text, (text_x, text_y),
                       cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)
        
        recent_alerts = self.get_recent_alerts(5)
        if recent_alerts:
            status_color = (0, 255, 255)  # Yellow for recent activity
            status_text = f"Recent alerts: {len(recent_alerts)}"
        else:
            status_color = (0, 255, 0)  # Green for no recent alerts
            status_text = "System normal"
        
        cv2.putText(overlay, status_text, (10, height - 50),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, status_color, 2)
        
        return overlay
