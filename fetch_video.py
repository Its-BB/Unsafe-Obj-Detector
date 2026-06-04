#!/usr/bin/env python3
"""
ESP32-CAM Video Viewer with GUI
Displays live video from ESP32-CAM in a Tkinter window
"""

import cv2
import requests
import numpy as np
import threading
import tkinter as tk
from tkinter import ttk, messagebox
from PIL import Image, ImageTk
from io import BytesIO
import time
from pathlib import Path
from urllib.parse import urlparse

import yaml


def stream_host_from_config() -> str:
    config_path = Path(__file__).resolve().parent / 'config.yaml'
    with open(config_path, encoding='utf-8') as f:
        url = yaml.safe_load(f)['detection']['remote_stream_url']
    host = urlparse(url).hostname or 'YOUR_ESP32_IP'
    return host.replace('YOUR_ESP32_IP', '127.0.0.1')


class CameraGUI:
    def __init__(self, root, esp_ip):
        self.root = root
        self.esp_ip = esp_ip
        self.stream_url = f"http://{esp_ip}:80/stream"
        self.running = False
        self.frame_count = 0
        
        self.root.title("ESP32-CAM Live Stream")
        self.root.geometry("800x600")
        
        title_label = ttk.Label(root, text=" ESP32-CAM Live Stream", font=("Arial", 16, "bold"))
        title_label.pack(pady=10)
        
        self.video_label = ttk.Label(root, background="black")
        self.video_label.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        self.status_var = tk.StringVar(value=" Connecting...")
        status_label = ttk.Label(root, textvariable=self.status_var, relief=tk.SUNKEN)
        status_label.pack(fill=tk.X, padx=10, pady=5)
        
        button_frame = ttk.Frame(root)
        button_frame.pack(pady=10)
        
        self.start_btn = ttk.Button(button_frame, text=" Start Stream", command=self.start_stream)
        self.start_btn.pack(side=tk.LEFT, padx=5)
        
        self.stop_btn = ttk.Button(button_frame, text=" Stop Stream", command=self.stop_stream, state=tk.DISABLED)
        self.stop_btn.pack(side=tk.LEFT, padx=5)
        
        self.save_btn = ttk.Button(button_frame, text=" Save Video", command=self.save_video, state=tk.DISABLED)
        self.save_btn.pack(side=tk.LEFT, padx=5)
        
        self.stream_thread = None
        self.video_writer = None
        self.recording = False
        
    def start_stream(self):
        """Start streaming video"""
        self.running = True
        self.frame_count = 0
        self.start_btn.config(state=tk.DISABLED)
        self.stop_btn.config(state=tk.NORMAL)
        self.save_btn.config(state=tk.NORMAL)
        self.status_var.set(" Streaming...")
        
        self.stream_thread = threading.Thread(target=self._stream_loop, daemon=True)
        self.stream_thread.start()
    
    def stop_stream(self):
        """Stop streaming video"""
        self.running = False
        self.start_btn.config(state=tk.NORMAL)
        self.stop_btn.config(state=tk.DISABLED)
        self.save_btn.config(state=tk.DISABLED)
        self.status_var.set(" Stopped")
        
        if self.video_writer:
            self.video_writer.release()
            self.video_writer = None
            self.recording = False
    
    def save_video(self):
        """Toggle video recording"""
        if not self.recording:
            self.recording = True
            self.save_btn.config(text=" Stop Recording")
            self.status_var.set(" Recording...")
        else:
            self.recording = False
            self.save_btn.config(text=" Save Video")
            if self.video_writer:
                self.video_writer.release()
                self.video_writer = None
            self.status_var.set(" Video saved!")
    
    def _stream_loop(self):
        """Main streaming loop with reconnection"""
        retry_count = 0
        max_retries = 5
        
        while self.running:
            try:
                print(f" Connecting to stream... (attempt {retry_count + 1})")
                self.status_var.set(" Connecting...")
                
                response = requests.get(self.stream_url, stream=True, timeout=(10, 30))
                bytes_data = b''
                retry_count = 0  # Reset on successful connection
                
                while self.running:
                    try:
                        for chunk in response.iter_content(chunk_size=4096):
                            if not self.running:
                                break
                            
                            if chunk:
                                bytes_data += chunk
                                
                                a = bytes_data.find(b'\xff\xd8')  # JPEG start
                                b = bytes_data.find(b'\xff\xd9')  # JPEG end
                                
                                if a != -1 and b != -1 and b > a:
                                    jpg = bytes_data[a:b+2]
                                    bytes_data = bytes_data[b+2:]
                                    
                                    if len(jpg) > 100:  # Minimum JPEG size
                                        frame = cv2.imdecode(np.frombuffer(jpg, dtype=np.uint8), cv2.IMREAD_COLOR)
                                        
                                        if frame is not None:
                                            self.frame_count += 1
                                            
                                            if self.recording:
                                                if self.video_writer is None:
                                                    h, w = frame.shape[:2]
                                                    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                                                    self.video_writer = cv2.VideoWriter('esp32_video.mp4', fourcc, 5.0, (w, h))
                                                self.video_writer.write(frame)
                                            
                                            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                                            
                                            h, w = frame_rgb.shape[:2]
                                            scale = min(780/w, 550/h)
                                            new_w, new_h = int(w*scale), int(h*scale)
                                            frame_resized = cv2.resize(frame_rgb, (new_w, new_h))
                                            
                                            pil_image = Image.fromarray(frame_resized)
                                            photo = ImageTk.PhotoImage(pil_image)
                                            
                                            self.video_label.config(image=photo)
                                            self.video_label.image = photo
                                            
                                            status_text = f" Frame: {self.frame_count}"
                                            if self.recording:
                                                status_text += " |  Recording"
                                            self.status_var.set(status_text)
                                            
                                            self.root.update()
                    
                    except (ConnectionError, ConnectionResetError, BrokenPipeError):
                        print("Connection lost, reconnecting...")
                        break
        
            except requests.exceptions.Timeout:
                retry_count += 1
                self.status_var.set(f" Timeout, retrying... ({retry_count}/{max_retries})")
                print(f"Timeout, retrying... ({retry_count}/{max_retries})")
                time.sleep(2)
                
            except requests.exceptions.ConnectionError:
                retry_count += 1
                self.status_var.set(f" Connection failed ({retry_count}/{max_retries})")
                print(f"Connection failed, retrying... ({retry_count}/{max_retries})")
                time.sleep(2)
                
            except Exception as e:
                retry_count += 1
                self.status_var.set(f" Error: {type(e).__name__}")
                print(f"Stream error: {e}")
                time.sleep(2)
            
            if retry_count >= max_retries and self.running:
                self.status_var.set(" Max retries reached")
                self.running = False
                break

def main():
    root = tk.Tk()
    app = CameraGUI(root, stream_host_from_config())
    root.mainloop()

if __name__ == '__main__':
    main()