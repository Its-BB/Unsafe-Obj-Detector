import cv2
import numpy as np
import logging
from typing import List, Dict, Tuple
from ultralytics import YOLO
import torch

from paths import (
    has_trained_weapon_model,
    resolve_weapon_model_path,
    resolve_weapon_secondary_path,
)

class WeaponDetector:
    
    def __init__(self, config: dict):
        self.config = config
        self.weapon_classes = [
            'knife', 'pistol', 'rifle', 'gun', 'handgun', 'firearm',
            'blade', 'sword', 'dagger', 'machete', 'axe', 'hammer',
            'scissors', 'box_cutter', 'razor', 'weapon'
        ]
        
        self.gun_like_objects = [
            'remote', 'cell phone', 'bottle', 'umbrella', 'flashlight',
            'drill', 'spray bottle', 'hair dryer', 'stapler', 'glue gun'
        ]
        
        self.primary_model = None
        self.secondary_model = None
        self.load_models()
        
        self.confidence_threshold = config['detection']['confidence_threshold']
        
        self.using_custom_model = has_trained_weapon_model(config)
        if self.using_custom_model:
            weapon_cfg = config.get('weapon', {})
            self.danger_threshold = weapon_cfg.get('danger_threshold', 0.6)
            self.custom_knife_threshold = weapon_cfg.get('knife_threshold', 0.35)
            self.custom_pistol_threshold = weapon_cfg.get('pistol_threshold', 0.3)
        else:
            self.danger_threshold = 0.2
            self.custom_knife_threshold = 0.2
            self.custom_pistol_threshold = 0.2

        self.gun_detection_enabled = not self.using_custom_model
        self.shape_detection_enabled = not self.using_custom_model
        self.suspicious_object_threshold = 0.5

    def load_models(self):
        try:
            primary = resolve_weapon_model_path(self.config)
            secondary = resolve_weapon_secondary_path(self.config)
            self.primary_model = YOLO(primary)
            self.secondary_model = YOLO(secondary)
            logging.info('Weapon model: %s', primary)
            logging.info('Model classes: %s', self.primary_model.names)
        except Exception as e:
            logging.error('Error loading weapon detection models: %s', e)
            raise
    
    def detect_weapons(self, frame: np.ndarray) -> List[Dict]:
        """Detect weapons using multiple detection strategies with enhanced angle robustness."""
        all_detections = []
        
        angle_detections = self._detect_multi_angle(frame)
        all_detections.extend(angle_detections)
        
        person_regions = self._get_person_regions(frame)
        for region in person_regions:
            region_detections = self._detect_in_region(frame, region)
            all_detections.extend(region_detections)
        
        scale_detections = self._detect_multi_scale(frame)
        all_detections.extend(scale_detections)
        
        if self.shape_detection_enabled:
            shape_detections = self._detect_by_shape(frame)
            all_detections.extend(shape_detections)
        
        if self.gun_detection_enabled and len(person_regions) > 0:  # Only check for guns near people
            gun_detections = self._detect_gun_patterns(frame)
            all_detections.extend(gun_detections)
        
        context_detections = self._detect_suspicious_objects(frame)
        all_detections.extend(context_detections)
        
        edge_detections = self._detect_weapons_edge_enhanced(frame)
        all_detections.extend(edge_detections)
        
        filtered_detections = self._filter_detections(all_detections)
        
        return filtered_detections
    
    def _detect_with_model(self, frame: np.ndarray, model: YOLO, model_name: str) -> List[Dict]:
        """Detect objects using YOLO model."""
        detections = []
        
        try:
            results = model(frame, conf=self.danger_threshold)
            
            for result in results:
                boxes = result.boxes
                if boxes is not None:
                    for box in boxes:
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                        confidence = box.conf[0].cpu().numpy()
                        class_id = int(box.cls[0].cpu().numpy())
                        class_name = model.names[class_id]
                        
                        if self._should_accept_detection(class_name, confidence):
                            weapon_score = self._calculate_weapon_score(class_name, confidence)
                            
                            if weapon_score > 0:
                                detection = {
                                    'bbox': (int(x1), int(y1), int(x2), int(y2)),
                                    'confidence': float(confidence),
                                    'weapon_score': weapon_score,
                                    'class_name': class_name,
                                    'detection_method': f'{model_name}_model',
                                    'is_weapon': weapon_score > 0.5
                                }
                                detections.append(detection)
                            
        except Exception as e:
            logging.error(f"Error in {model_name} model detection: {e}")
        
        return detections
    
    def _detect_multi_angle(self, frame: np.ndarray) -> List[Dict]:
        """Detect weapons from multiple angles by applying rotations and flips."""
        all_detections = []
        height, width = frame.shape[:2]
        
        original_detections = self._detect_with_model(frame, self.primary_model, "original")
        all_detections.extend(original_detections)
        
        transformations = [
            ('horizontal_flip', cv2.flip(frame, 1)),
            ('vertical_flip', cv2.flip(frame, 0)),
            ('rotate_90', cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)),
            ('rotate_270', cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)),
            ('rotate_180', cv2.rotate(frame, cv2.ROTATE_180)),
            ('rotate_45', self._rotate_frame(frame, 45)),
            ('rotate_315', self._rotate_frame(frame, -45)),
            ('rotate_30', self._rotate_frame(frame, 30)),
            ('rotate_330', self._rotate_frame(frame, -30))
        ]
        
        for transform_name, transformed_frame in transformations:
            try:
                transformed_detections = self._detect_with_model(transformed_frame, self.primary_model, f"transform_{transform_name}")
                
                for detection in transformed_detections:
                    if transform_name.startswith('rotate_'):
                        angle = int(transform_name.split('_')[1]) if transform_name.split('_')[1].lstrip('-').isdigit() else 0
                        original_bbox = self._transform_rotated_bbox(detection['bbox'], angle, width, height)
                    else:
                        original_bbox = self._transform_bbox_back(detection['bbox'], transform_name, width, height)
                    
                    if original_bbox:  # Only add if transformation was successful
                        detection['bbox'] = original_bbox
                        detection['detection_method'] = f"multi_angle_{transform_name}"
                        
                        class_name = detection.get('class_name', '').lower()
                        if class_name in ['knife', 'pistol'] or any(word in class_name for word in ['gun', 'firearm', 'scissors']):
                            detection['confidence'] *= 0.98  # Minimal reduction for weapons
                            detection['weapon_score'] *= 0.98
                        else:
                            detection['confidence'] *= 0.95
                            detection['weapon_score'] *= 0.95
                        
                        all_detections.append(detection)
                        
            except Exception as e:
                logging.warning(f"Error in {transform_name} transformation: {e}")
                continue
        
        return all_detections
    
    def _detect_multi_scale(self, frame: np.ndarray) -> List[Dict]:
        """Detect weapons at multiple scales to catch different sizes."""
        all_detections = []
        original_height, original_width = frame.shape[:2]
        
        scale_factors = [0.7, 0.85, 1.15, 1.3, 1.6]  # More granular scaling for weapons
        
        for scale in scale_factors:
            try:
                new_width = int(original_width * scale)
                new_height = int(original_height * scale)
                scaled_frame = cv2.resize(frame, (new_width, new_height))
                
                scaled_detections = self._detect_with_model(scaled_frame, self.primary_model, f"scale_{scale}")
                
                for detection in scaled_detections:
                    x1, y1, x2, y2 = detection['bbox']
                    
                    original_x1 = int(x1 / scale)
                    original_y1 = int(y1 / scale)
                    original_x2 = int(x2 / scale)
                    original_y2 = int(y2 / scale)
                    
                    original_x1 = max(0, min(original_x1, original_width))
                    original_y1 = max(0, min(original_y1, original_height))
                    original_x2 = max(0, min(original_x2, original_width))
                    original_y2 = max(0, min(original_y2, original_height))
                    
                    detection['bbox'] = (original_x1, original_y1, original_x2, original_y2)
                    detection['detection_method'] = f"multi_scale_{scale}"
                    detection['confidence'] *= 0.9
                    detection['weapon_score'] *= 0.9
                    all_detections.append(detection)
                    
            except Exception as e:
                logging.warning(f"Error in scale {scale} detection: {e}")
                continue
        
        return all_detections
    
    def _transform_bbox_back(self, bbox: Tuple[int, int, int, int], transform_name: str, 
                           original_width: int, original_height: int) -> Tuple[int, int, int, int]:
        """Transform bounding box coordinates back to original frame."""
        x1, y1, x2, y2 = bbox
        
        try:
            if transform_name == 'horizontal_flip':
                new_x1 = original_width - x2
                new_x2 = original_width - x1
                return (new_x1, y1, new_x2, y2)
            
            elif transform_name == 'vertical_flip':
                new_y1 = original_height - y2
                new_y2 = original_height - y1
                return (x1, new_y1, x2, new_y2)
            
            elif transform_name == 'rotate_90':
                new_x1 = y1
                new_y1 = original_height - x2
                new_x2 = y2
                new_y2 = original_height - x1
                return (new_x1, new_y1, new_x2, new_y2)
            
            elif transform_name == 'rotate_270':
                new_x1 = original_width - y2
                new_y1 = x1
                new_x2 = original_width - y1
                new_y2 = x2
                return (new_x1, new_y1, new_x2, new_y2)
            
            elif transform_name == 'rotate_180':
                new_x1 = original_width - x2
                new_y1 = original_height - y2
                new_x2 = original_width - x1
                new_y2 = original_height - y1
                return (new_x1, new_y1, new_x2, new_y2)
            
        except Exception as e:
            logging.warning(f"Error transforming bbox for {transform_name}: {e}")
            return None
        
        return None
    
    def _transform_rotated_bbox(self, bbox: Tuple[int, int, int, int], angle: float, 
                              original_width: int, original_height: int) -> Tuple[int, int, int, int]:
        """Transform bounding box for arbitrary angle rotations."""
        try:
            x1, y1, x2, y2 = bbox
            
            center_x, center_y = original_width // 2, original_height // 2
            
            bbox_center_x = (x1 + x2) // 2 - center_x
            bbox_center_y = (y1 + y2) // 2 - center_y
            
            angle_rad = np.radians(-angle)  # Reverse rotation
            cos_a, sin_a = np.cos(angle_rad), np.sin(angle_rad)
            
            new_center_x = int(bbox_center_x * cos_a - bbox_center_y * sin_a) + center_x
            new_center_y = int(bbox_center_x * sin_a + bbox_center_y * cos_a) + center_y
            
            width = x2 - x1
            height = y2 - y1
            
            new_x1 = max(0, new_center_x - width // 2)
            new_y1 = max(0, new_center_y - height // 2)
            new_x2 = min(original_width, new_center_x + width // 2)
            new_y2 = min(original_height, new_center_y + height // 2)
            
            return (new_x1, new_y1, new_x2, new_y2)
            
        except Exception as e:
            logging.warning(f"Error transforming rotated bbox: {e}")
            return None
    
    def _rotate_frame(self, frame: np.ndarray, angle: float) -> np.ndarray:
        """Rotate frame by arbitrary angle for enhanced weapon detection."""
        try:
            height, width = frame.shape[:2]
            center = (width // 2, height // 2)
            
            rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
            
            cos_angle = abs(rotation_matrix[0, 0])
            sin_angle = abs(rotation_matrix[0, 1])
            new_width = int((height * sin_angle) + (width * cos_angle))
            new_height = int((height * cos_angle) + (width * sin_angle))
            
            rotation_matrix[0, 2] += (new_width / 2) - center[0]
            rotation_matrix[1, 2] += (new_height / 2) - center[1]
            
            rotated = cv2.warpAffine(frame, rotation_matrix, (new_width, new_height))
            
            rotated = cv2.resize(rotated, (width, height))
            
            return rotated
        except Exception as e:
            logging.warning(f"Error rotating frame by {angle} degrees: {e}")
            return frame
    
    def _should_accept_detection(self, class_name: str, confidence: float) -> bool:
        """Apply class-specific confidence thresholds to reduce false positives."""
        class_lower = class_name.lower()
        
        if self.using_custom_model:
            if class_lower == 'knife':
                return confidence >= 0.35  # Lowered for better knife detection at angles
            elif class_lower == 'pistol':
                return confidence >= 0.3   # Lowered for better pistol detection at angles
            else:
                return confidence >= 0.6
        else:
            if class_lower in ['knife', 'scissors']:
                return confidence >= 0.25  # Lower threshold for blade detection
            elif any(word in class_lower for word in ['gun', 'pistol', 'firearm']):
                return confidence >= 0.3   # Lower threshold for firearm detection
            else:
                return confidence >= self.danger_threshold
    
    def _get_person_regions(self, frame: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """Get regions where people are detected for focused weapon detection."""
        regions = []
        
        try:
            results = self.primary_model(frame, conf=0.3)
            
            for result in results:
                boxes = result.boxes
                if boxes is not None:
                    for box in boxes:
                        class_id = int(box.cls[0].cpu().numpy())
                        class_name = self.primary_model.names[class_id]
                        
                        if class_name == 'person':
                            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                            
                            h, w = frame.shape[:2]
                            expand_factor = 0.2
                            
                            x1 = max(0, int(x1 - (x2-x1) * expand_factor))
                            y1 = max(0, int(y1 - (y2-y1) * expand_factor))
                            x2 = min(w, int(x2 + (x2-x1) * expand_factor))
                            y2 = min(h, int(y2 + (y2-y1) * expand_factor))
                            
                            regions.append((x1, y1, x2, y2))
                            
        except Exception as e:
            logging.error(f"Error detecting person regions: {e}")
        
        return regions
    
    def _detect_in_region(self, frame: np.ndarray, region: Tuple[int, int, int, int]) -> List[Dict]:
        """Perform focused detection in a specific region."""
        x1, y1, x2, y2 = region
        roi = frame[y1:y2, x1:x2]
        
        if roi.size == 0:
            return []
        
        detections = []
        
        try:
            results = self.secondary_model(roi, conf=self.danger_threshold)
            
            for result in results:
                boxes = result.boxes
                if boxes is not None:
                    for box in boxes:
                        rx1, ry1, rx2, ry2 = box.xyxy[0].cpu().numpy()
                        confidence = box.conf[0].cpu().numpy()
                        class_id = int(box.cls[0].cpu().numpy())
                        class_name = self.secondary_model.names[class_id]
                        
                        weapon_score = self._calculate_weapon_score(class_name, confidence)
                        
                        if weapon_score > 0:
                            detection = {
                                'bbox': (int(x1 + rx1), int(y1 + ry1), int(x1 + rx2), int(y1 + ry2)),
                                'confidence': float(confidence),
                                'weapon_score': weapon_score,
                                'class_name': class_name,
                                'detection_method': 'region_focused',
                                'is_weapon': weapon_score > 0.5
                            }
                            detections.append(detection)
                            
        except Exception as e:
            logging.error(f"Error in region detection: {e}")
        
        return detections
    
    def _detect_by_shape(self, frame: np.ndarray) -> List[Dict]:
        """Detect potential weapons based on shape analysis."""
        detections = []
        
        try:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            edges = cv2.Canny(gray, 50, 150)
            
            contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            for contour in contours:
                area = cv2.contourArea(contour)
                if area < 1000 or area > 50000:  # Skip very small or very large objects
                    continue
                
                x, y, w, h = cv2.boundingRect(contour)
                
                aspect_ratio = float(w) / h
                extent = float(area) / (w * h)
                
                if (aspect_ratio > 3.0 or aspect_ratio < 0.33) and extent > 0.3:
                    weapon_score = self._calculate_shape_weapon_score(contour, aspect_ratio, extent)
                    
                    if weapon_score > 0.2:
                        detection = {
                            'bbox': (x, y, x + w, y + h),
                            'confidence': weapon_score,
                            'weapon_score': weapon_score,
                            'class_name': 'potential_blade',
                            'detection_method': 'shape_analysis',
                            'is_weapon': weapon_score > 0.4
                        }
                        detections.append(detection)
                        
        except Exception as e:
            logging.error(f"Error in shape-based detection: {e}")
        
        return detections
    
    def _calculate_weapon_score(self, class_name: str, confidence: float) -> float:
        """Calculate weapon likelihood score."""
        base_score = 0.0
        class_lower = class_name.lower()
        
        if self.using_custom_model:
            if class_lower in ['knife']:
                base_score = 1.0
                if confidence > 0.4:
                    base_score = 1.2  # Boost confidence for good knife detections
            elif class_lower in ['pistol']:
                base_score = 1.0
                if confidence > 0.35:
                    base_score = 1.15  # Boost confidence for good pistol detections
            else:
                base_score = 0.1
        else:
            if class_lower in ['knife']:
                base_score = 1.0
                if confidence > 0.3:
                    base_score = 1.3
            elif class_lower in ['scissors']:
                base_score = 0.95  # High score for scissors (knife-like)
                if confidence > 0.35:
                    base_score = 1.2
            elif any(word in class_lower for word in ['gun', 'pistol', 'rifle', 'firearm', 'handgun']):
                base_score = 1.0
                if confidence > 0.4:
                    base_score = 1.25
            elif class_lower in ['hammer', 'axe']:
                base_score = 0.7  # Increased from 0.6
            elif class_lower in ['bottle', 'baseball bat', 'tennis racket']:
                base_score = 0.3
            elif any(word in class_lower for word in ['remote', 'cell phone', 'mobile phone']):
                base_score = 0.45 if confidence > 0.6 else 0.25
            elif any(word in class_lower for word in ['bottle', 'flashlight', 'drill']):
                base_score = 0.55 if confidence > 0.5 else 0.25
            elif 'tool' in class_lower or 'device' in class_lower:
                base_score = 0.35
        
        weapon_score = base_score * confidence
        
        if confidence > 0.8 and base_score > 0:
            weapon_score = min(1.0, weapon_score * 1.2)
        
        return weapon_score
    
    def _calculate_shape_weapon_score(self, contour, aspect_ratio: float, extent: float) -> float:
        """Calculate weapon score based on shape characteristics."""
        score = 0.0
        
        if aspect_ratio > 4.0 or aspect_ratio < 0.25:
            score += 0.3
        
        if 0.3 < extent < 0.8:
            score += 0.2
        
        hull = cv2.convexHull(contour)
        hull_area = cv2.contourArea(hull)
        contour_area = cv2.contourArea(contour)
        
        if hull_area > 0:
            solidity = float(contour_area) / hull_area
            if solidity > 0.8:  # Fairly solid object
                score += 0.2
        
        return min(1.0, score)
    
    def _detect_gun_patterns(self, frame: np.ndarray) -> List[Dict]:
        """Detect gun-like objects using specialized pattern detection."""
        detections = []
        
        try:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            edges1 = cv2.Canny(gray, 30, 100)
            edges2 = cv2.Canny(gray, 50, 150)
            combined_edges = cv2.bitwise_or(edges1, edges2)
            
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
            edges_closed = cv2.morphologyEx(combined_edges, cv2.MORPH_CLOSE, kernel)
            
            contours, _ = cv2.findContours(edges_closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            for contour in contours:
                area = cv2.contourArea(contour)
                if area < 2000 or area > 100000:  # Filter by size
                    continue
                
                x, y, w, h = cv2.boundingRect(contour)
                aspect_ratio = float(w) / h
                
                gun_score = self._calculate_gun_pattern_score(contour, aspect_ratio, area)
                
                if gun_score > 0.3:
                    detection = {
                        'bbox': (x, y, x + w, y + h),
                        'confidence': gun_score,
                        'weapon_score': gun_score,
                        'class_name': 'potential_firearm',
                        'detection_method': 'gun_pattern',
                        'is_weapon': gun_score > 0.5
                    }
                    detections.append(detection)
                    
        except Exception as e:
            logging.error(f"Error in gun pattern detection: {e}")
        
        return detections
    
    def _detect_suspicious_objects(self, frame: np.ndarray) -> List[Dict]:
        """Detect objects that might be used to conceal weapons or are gun-like."""
        detections = []
        
        try:
            results = self.primary_model(frame, conf=0.3)
            
            for result in results:
                boxes = result.boxes
                if boxes is not None:
                    for box in boxes:
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                        confidence = box.conf[0].cpu().numpy()
                        class_id = int(box.cls[0].cpu().numpy())
                        class_name = self.primary_model.names[class_id]
                        
                        if class_name.lower() in [obj.lower() for obj in self.gun_like_objects]:
                            suspicion_score = self._calculate_suspicion_score(class_name, confidence, (x1, y1, x2, y2))
                            
                            if suspicion_score > 0.3:
                                detection = {
                                    'bbox': (int(x1), int(y1), int(x2), int(y2)),
                                    'confidence': float(confidence),
                                    'weapon_score': suspicion_score,
                                    'class_name': f'suspicious_{class_name}',
                                    'detection_method': 'context_suspicious',
                                    'is_weapon': suspicion_score > 0.6
                                }
                                detections.append(detection)
                                
        except Exception as e:
            logging.error(f"Error in suspicious object detection: {e}")
        
        return detections
    
    def _calculate_gun_pattern_score(self, contour, aspect_ratio: float, area: float) -> float:
        """Calculate gun likelihood based on shape patterns."""
        score = 0.0
        
        if 1.5 < aspect_ratio < 4.0:  # Handgun-like proportions
            score += 0.4
        elif 4.0 <= aspect_ratio < 8.0:  # Rifle-like proportions
            score += 0.5
        
        hull = cv2.convexHull(contour)
        hull_area = cv2.contourArea(hull)
        contour_area = cv2.contourArea(contour)
        
        if hull_area > 0:
            solidity = float(contour_area) / hull_area
            if 0.6 < solidity < 0.9:
                score += 0.3
        
        epsilon = 0.02 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)
        
        if 8 <= len(approx) <= 20:
            score += 0.2
        
        return min(1.0, score)
    
    def _calculate_suspicion_score(self, class_name: str, confidence: float, bbox: Tuple[float, float, float, float]) -> float:
        """Calculate suspicion score for potentially gun-like objects."""
        base_score = 0.0
        
        suspicion_levels = {
            'remote': 0.4,  # TV remotes can look like guns in surveillance
            'cell phone': 0.3,  # Phones held in certain ways
            'bottle': 0.3,  # Bottles can be used as weapons or look gun-like
            'umbrella': 0.5,  # Umbrellas can be mistaken for rifles
            'flashlight': 0.6,  # Flashlights are very gun-like
            'drill': 0.7,  # Power drills look very much like guns
            'spray bottle': 0.5,  # Can look like pistols
            'hair dryer': 0.6,  # Classic gun-like shape
            'stapler': 0.4,  # Can look like small pistols
            'glue gun': 0.8   # Literally called "gun"
        }
        
        class_lower = class_name.lower()
        for obj, level in suspicion_levels.items():
            if obj in class_lower:
                base_score = level
                break
        
        suspicion_score = base_score * confidence
        
        if confidence > 0.7:
            suspicion_score = min(1.0, suspicion_score * 1.3)
        
        return suspicion_score
    
    def _filter_detections(self, detections: List[Dict]) -> List[Dict]:
        """Filter and deduplicate detections."""
        if not detections:
            return []
        
        detections.sort(key=lambda x: x['weapon_score'], reverse=True)
        
        filtered = []
        for detection in detections:
            is_duplicate = False
            
            for existing in filtered:
                if self._calculate_iou(detection['bbox'], existing['bbox']) > 0.5:
                    is_duplicate = True
                    break
            
            if not is_duplicate and detection['weapon_score'] > 0.2:
                filtered.append(detection)
        
        return filtered
    
    def _calculate_iou(self, box1: Tuple[int, int, int, int], box2: Tuple[int, int, int, int]) -> float:
        """Calculate Intersection over Union (IoU) of two bounding boxes."""
        x1_1, y1_1, x2_1, y2_1 = box1
        x1_2, y1_2, x2_2, y2_2 = box2
        
        x1_i = max(x1_1, x1_2)
        y1_i = max(y1_1, y1_2)
        x2_i = min(x2_1, x2_2)
        y2_i = min(y2_1, y2_2)
        
        if x2_i <= x1_i or y2_i <= y1_i:
            return 0.0
        
        intersection = (x2_i - x1_i) * (y2_i - y1_i)
        
        area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
        area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
        union = area1 + area2 - intersection
        
        return intersection / union if union > 0 else 0.0
    
    def _detect_weapons_edge_enhanced(self, frame: np.ndarray) -> List[Dict]:
        """Enhanced edge detection specifically optimized for pistols and knives."""
        all_detections = []
        
        try:
            enhanced_frames = self._create_weapon_enhanced_frames(frame)
            
            for enhancement_name, enhanced_frame in enhanced_frames:
                enhanced_detections = self._detect_with_model(enhanced_frame, self.primary_model, f"edge_{enhancement_name}")
                
                for detection in enhanced_detections:
                    class_name = detection.get('class_name', '').lower()
                    if (class_name in ['knife', 'pistol'] or 
                        any(word in class_name for word in ['gun', 'firearm', 'scissors'])):
                        
                        detection['detection_method'] = f"edge_enhanced_{enhancement_name}"
                        detection['confidence'] *= 1.05
                        detection['weapon_score'] *= 1.05
                        all_detections.append(detection)
                        
        except Exception as e:
            logging.warning(f"Error in edge-enhanced weapon detection: {e}")
        
        return all_detections
    
    def _create_weapon_enhanced_frames(self, frame: np.ndarray) -> List[Tuple[str, np.ndarray]]:
        """Create edge-enhanced frames specifically for weapon detection."""
        enhanced_frames = []
        
        try:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            contrast_enhanced = cv2.convertScaleAbs(gray, alpha=1.8, beta=30)
            contrast_frame = cv2.cvtColor(contrast_enhanced, cv2.COLOR_GRAY2BGR)
            enhanced_frames.append(("high_contrast", contrast_frame))
            
            kernel_sharpen = np.array([[-1,-1,-1],
                                     [-1, 9,-1],
                                     [-1,-1,-1]])
            sharpened = cv2.filter2D(gray, -1, kernel_sharpen)
            sharpened = np.clip(sharpened, 0, 255).astype(np.uint8)
            sharp_frame = cv2.cvtColor(sharpened, cv2.COLOR_GRAY2BGR)
            enhanced_frames.append(("edge_sharp", sharp_frame))
            
            gaussian_blur = cv2.GaussianBlur(gray, (0, 0), 2.0)
            unsharp_mask = cv2.addWeighted(gray, 2.0, gaussian_blur, -1.0, 0)
            unsharp_frame = cv2.cvtColor(unsharp_mask, cv2.COLOR_GRAY2BGR)
            enhanced_frames.append(("unsharp_mask", unsharp_frame))
            
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
            morph_enhanced = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, kernel)
            morph_enhanced = cv2.morphologyEx(morph_enhanced, cv2.MORPH_OPEN, kernel)
            morph_frame = cv2.cvtColor(morph_enhanced, cv2.COLOR_GRAY2BGR)
            enhanced_frames.append(("morph_enhanced", morph_frame))
            
            hist_eq = cv2.equalizeHist(gray)
            hist_frame = cv2.cvtColor(hist_eq, cv2.COLOR_GRAY2BGR)
            enhanced_frames.append(("hist_equalized", hist_frame))
            
        except Exception as e:
            logging.warning(f"Error creating weapon-enhanced frames: {e}")
        
        return enhanced_frames
