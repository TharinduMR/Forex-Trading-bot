import tkinter as tk
from tkinter import ttk, messagebox
import subprocess
import webbrowser
import os
import sys

try:
    import psutil
except ImportError:
    psutil = None

class AntigravityLauncher(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Antigravity Quant - System Launcher")
        self.geometry("400x250")
        self.resizable(False, False)
        self.configure(bg="#0f172a")  # Dark background matching dashboard
        
        # Keep track of running processes
        self.monitor_process = None
        self.engine_process = None
        self.unified_process = None
        
        self.create_widgets()
        self.check_status()
        
    def create_widgets(self):
        style = ttk.Style()
        style.theme_use('clam')
        style.configure("TFrame", background="#0f172a")
        style.configure("TLabel", background="#0f172a", foreground="#ffffff", font=("Segoe UI", 10))
        style.configure("TButton", font=("Segoe UI", 10, "bold"), padding=5)
        
        main_frame = ttk.Frame(self)
        main_frame.pack(expand=True, fill="both", padx=20, pady=20)
        
        # Title
        title_label = ttk.Label(main_frame, text="⚡ Antigravity Quant", font=("Segoe UI", 16, "bold"), foreground="#00f0ff")
        title_label.pack(pady=(0, 5))
        
        subtitle = ttk.Label(main_frame, text="Institutional Execution Engine", foreground="#8b9bb4", font=("Segoe UI", 9))
        subtitle.pack(pady=(0, 20))
        
        # Status Label
        self.status_var = tk.StringVar(value="Status: STOPPED")
        self.status_label = ttk.Label(main_frame, textvariable=self.status_var, font=("Segoe UI", 11, "bold"), foreground="#ff3366")
        self.status_label.pack(pady=(0, 15))
        
        # Buttons Frame
        btn_frame = ttk.Frame(main_frame)
        btn_frame.pack(fill="x")
        
        self.start_btn = tk.Button(btn_frame, text="▶ START SYSTEM", bg="#00ff88", fg="#000000", font=("Segoe UI", 10, "bold"), cursor="hand2", command=self.start_system)
        self.start_btn.pack(side="left", expand=True, fill="x", padx=5)
        
        self.stop_btn = tk.Button(btn_frame, text="■ STOP SYSTEM", bg="#ff3366", fg="#ffffff", font=("Segoe UI", 10, "bold"), cursor="hand2", state="disabled", command=self.stop_system)
        self.stop_btn.pack(side="right", expand=True, fill="x", padx=5)
        
        # Open Dashboard Button
        self.dash_btn = tk.Button(main_frame, text="🌐 OPEN DASHBOARD", bg="#00f0ff", fg="#000000", font=("Segoe UI", 10, "bold"), cursor="hand2", command=self.open_dashboard)
        self.dash_btn.pack(fill="x", pady=(15, 0))

    def check_status(self):
        """Check if our specific processes are running."""
        is_running = self.monitor_process is not None and self.monitor_process.poll() is None
        
        if is_running:
            self.status_var.set("Status: RUNNING")
            self.status_label.config(foreground="#00ff88")
            self.start_btn.config(state="disabled", bg="#333333")
            self.stop_btn.config(state="normal", bg="#ff3366")
        else:
            self.status_var.set("Status: STOPPED")
            self.status_label.config(foreground="#ff3366")
            self.start_btn.config(state="normal", bg="#00ff88")
            self.stop_btn.config(state="disabled", bg="#333333")
            
        self.after(1000, self.check_status)

    def start_system(self):
        try:
            # Ensure no orphaned processes are running first
            self.kill_orphans()
            
            # Use pythonw or python depending on if we want consoles
            # Creating separate process groups so they don't block the GUI
            env = os.environ.copy()
            cwd = os.path.dirname(os.path.abspath(__file__))
            
            # We want them to run in new command windows so the user can see logs
            CREATE_NEW_CONSOLE = 0x00000010
            
            # Start Monitor Server
            self.monitor_process = subprocess.Popen(
                [sys.executable, "monitor_server.py"],
                cwd=cwd,
                creationflags=CREATE_NEW_CONSOLE
            )
            
            # Start Live Engine for all supported timeframes
            self.engine_process = subprocess.Popen(
                [sys.executable, "main.py", "--mode", "live", "--timeframes", "1m", "5m", "15m", "30m", "1h", "4h", "1d"],
                cwd=cwd,
                creationflags=CREATE_NEW_CONSOLE
            )
            
            # Start Unified Global Trade Engine
            self.unified_process = subprocess.Popen(
                [sys.executable, "unified_engine.py"],
                cwd=cwd,
                creationflags=CREATE_NEW_CONSOLE
            )
            
            self.check_status()
            
            # Automatically open dashboard after 3 seconds
            self.after(3000, self.open_dashboard)
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to start system:\n{str(e)}")

    def stop_system(self):
        if self.monitor_process:
            try:
                self.monitor_process.terminate()
            except: pass
            self.monitor_process = None
            
        if self.engine_process:
            try:
                self.engine_process.terminate()
            except: pass
            self.engine_process = None
            
        if self.unified_process:
            try:
                self.unified_process.terminate()
            except: pass
            self.unified_process = None
            
        self.kill_orphans()
        self.check_status()
        
    def kill_orphans(self):
        """Kill any background python processes running main.py or monitor_server.py"""
        if psutil is None:
            # Fallback to taskkill
            os.system('taskkill /F /FI "WINDOWTITLE eq python main.py*" /IM python.exe >nul 2>&1')
            os.system('taskkill /F /FI "WINDOWTITLE eq python monitor_server.py*" /IM python.exe >nul 2>&1')
            os.system('taskkill /F /FI "WINDOWTITLE eq python unified_engine.py*" /IM python.exe >nul 2>&1')
            return
            
        for p in psutil.process_iter(['cmdline']):
            try:
                cmdline = p.info['cmdline']
                if cmdline and ('main.py' in ' '.join(cmdline) or 'monitor_server.py' in ' '.join(cmdline) or 'unified_engine.py' in ' '.join(cmdline)):
                    # Don't kill this launcher script itself
                    if 'launcher.py' not in ' '.join(cmdline):
                        p.kill()
            except:
                pass

    def open_dashboard(self):
        webbrowser.open("http://127.0.0.1:5000")

    def on_closing(self):
        if messagebox.askokcancel("Quit", "Do you want to quit the launcher?\n\nThis will also stop the trading engine if it's running."):
            self.stop_system()
            self.destroy()

if __name__ == "__main__":
    app = AntigravityLauncher()
    app.protocol("WM_DELETE_WINDOW", app.on_closing)
    app.mainloop()
