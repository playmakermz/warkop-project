"""
Pomodoro Timer Application
A beautiful and functional Pomodoro timer with two modes:
1. Manual Mode - Timer pauses when done, user clicks play to continue
2. Automatic Mode - Cycles automatically between work and break periods
"""

import tkinter as tk
from tkinter import ttk, messagebox
from enum import Enum


class TimerState(Enum):
    IDLE = "idle"
    RUNNING = "running"
    PAUSED = "paused"
    FINISHED = "finished"


class SessionType(Enum):
    WORK = "work"
    BREAK = "break"


class PomodoroTimer:
    def __init__(self, root):
        self.root = root
        self.root.title("Pomodoro Timer")
        self.root.geometry("480x780")  # Increased height
        self.root.resizable(False, False)
        self.root.configure(bg="#1a1a2e")
        
        # Timer variables
        self.time_left = 0
        self.timer_state = TimerState.IDLE
        self.session_type = SessionType.WORK
        self.is_automatic_mode = tk.BooleanVar(value=False)
        self.sessions_completed = 0
        
        # Default times (in minutes)
        self.work_time = tk.IntVar(value=25)
        self.break_time = tk.IntVar(value=5)
        
        # Colors
        self.colors = {
            'bg_dark': '#1a1a2e',
            'bg_medium': '#16213e',
            'accent_work': '#e94560',
            'accent_break': '#4ecca3',
            'text_light': '#eaeaea',
            'text_dim': '#a0a0a0',
            'button_bg': '#0f3460',
            'button_hover': '#1a4a7a'
        }
        
        self.setup_styles()
        self.create_widgets()
        
    def setup_styles(self):
        """Setup custom styles for ttk widgets"""
        self.style = ttk.Style()
        self.style.theme_use('clam')
        
        # Configure styles
        self.style.configure('TFrame', background=self.colors['bg_dark'])
        self.style.configure('TLabel', 
                           background=self.colors['bg_dark'], 
                           foreground=self.colors['text_light'],
                           font=('Segoe UI', 12))
        self.style.configure('Title.TLabel', 
                           font=('Segoe UI', 24, 'bold'),
                           foreground=self.colors['accent_work'])
        self.style.configure('Timer.TLabel', 
                           font=('Consolas', 64, 'bold'),
                           foreground=self.colors['text_light'])
        self.style.configure('Session.TLabel', 
                           font=('Segoe UI', 14),
                           foreground=self.colors['accent_work'])
        self.style.configure('TLabelframe',
                           background=self.colors['bg_dark'],
                           foreground=self.colors['text_light'])
        self.style.configure('TLabelframe.Label',
                           background=self.colors['bg_dark'],
                           foreground=self.colors['accent_work'],
                           font=('Segoe UI', 11, 'bold'))
        
    def create_widgets(self):
        """Create all GUI widgets"""
        # Main container with scrollable capability
        main_frame = ttk.Frame(self.root, padding=20)
        main_frame.pack(fill='both', expand=True)
        
        # ===== TITLE SECTION =====
        title_label = tk.Label(main_frame, text="🍅 Pomodoro Timer", 
                              font=('Segoe UI', 22, 'bold'),
                              bg=self.colors['bg_dark'],
                              fg=self.colors['accent_work'])
        title_label.pack(pady=(0, 8))
        
        # Session indicator
        self.session_label = tk.Label(main_frame, text="💼 Mode: Kerja",
                                     font=('Segoe UI', 14),
                                     bg=self.colors['bg_dark'],
                                     fg=self.colors['accent_work'])
        self.session_label.pack(pady=(0, 3))
        
        # Sessions counter
        self.counter_label = tk.Label(main_frame, text="Sesi selesai: 0", 
                                     font=('Segoe UI', 10),
                                     bg=self.colors['bg_dark'],
                                     fg=self.colors['text_dim'])
        self.counter_label.pack(pady=(0, 10))
        
        # ===== TIMER DISPLAY SECTION =====
        self.timer_canvas = tk.Canvas(main_frame, width=250, height=250,
                                      bg=self.colors['bg_dark'], highlightthickness=0)
        self.timer_canvas.pack(pady=5)
        
        # Draw initial timer
        self.draw_timer_circle(0)
        
        # Timer text on canvas
        self.timer_text_id = self.timer_canvas.create_text(
            125, 125, text="25:00",
            font=('Consolas', 52, 'bold'),
            fill=self.colors['text_light']
        )
        
        # Status label
        self.status_label = tk.Label(main_frame, text="⏸️ Siap untuk mulai",
                                    font=('Segoe UI', 11),
                                    bg=self.colors['bg_dark'],
                                    fg=self.colors['text_dim'])
        self.status_label.pack(pady=(8, 12))
        
        # ===== SETTINGS SECTION =====
        settings_frame = tk.LabelFrame(main_frame, text=" ⚙️ Pengaturan Waktu ",
                                      font=('Segoe UI', 11, 'bold'),
                                      bg=self.colors['bg_dark'],
                                      fg=self.colors['accent_work'],
                                      padx=15, pady=10)
        settings_frame.pack(fill='x', pady=8)
        
        # Work time setting
        work_frame = tk.Frame(settings_frame, bg=self.colors['bg_dark'])
        work_frame.pack(fill='x', pady=6)
        
        tk.Label(work_frame, text="⏱️ Waktu Kerja (menit):",
                font=('Segoe UI', 11),
                bg=self.colors['bg_dark'],
                fg=self.colors['accent_work']).pack(side='left')
        
        self.work_spinbox = tk.Spinbox(work_frame, from_=1, to=120, width=6,
                                       textvariable=self.work_time,
                                       font=('Segoe UI', 12),
                                       justify='center')
        self.work_spinbox.pack(side='right')
        
        # Break time setting
        break_frame = tk.Frame(settings_frame, bg=self.colors['bg_dark'])
        break_frame.pack(fill='x', pady=6)
        
        tk.Label(break_frame, text="☕ Waktu Istirahat (menit):",
                font=('Segoe UI', 11),
                bg=self.colors['bg_dark'],
                fg=self.colors['accent_break']).pack(side='left')
        
        self.break_spinbox = tk.Spinbox(break_frame, from_=1, to=60, width=6,
                                        textvariable=self.break_time,
                                        font=('Segoe UI', 12),
                                        justify='center')
        self.break_spinbox.pack(side='right')
        
        # ===== MODE SELECTION SECTION =====
        mode_frame = tk.LabelFrame(main_frame, text=" 🎛️ Mode Timer ",
                                  font=('Segoe UI', 11, 'bold'),
                                  bg=self.colors['bg_dark'],
                                  fg=self.colors['accent_work'],
                                  padx=15, pady=10)
        mode_frame.pack(fill='x', pady=8)
        
        # Mode radio buttons
        self.mode_var = tk.StringVar(value="manual")
        
        manual_rb = tk.Radiobutton(mode_frame, 
                                   text="📋 Mode Manual - Pause otomatis saat selesai",
                                   variable=self.mode_var, value="manual",
                                   bg=self.colors['bg_dark'],
                                   fg=self.colors['text_light'],
                                   selectcolor=self.colors['bg_medium'],
                                   activebackground=self.colors['bg_dark'],
                                   activeforeground=self.colors['text_light'],
                                   font=('Segoe UI', 10),
                                   command=self.on_mode_change)
        manual_rb.pack(anchor='w', pady=3)
        
        auto_rb = tk.Radiobutton(mode_frame,
                                 text="🔄 Mode Otomatis - Kerja ↔ Istirahat otomatis",
                                 variable=self.mode_var, value="automatic",
                                 bg=self.colors['bg_dark'],
                                 fg=self.colors['text_light'],
                                 selectcolor=self.colors['bg_medium'],
                                 activebackground=self.colors['bg_dark'],
                                 activeforeground=self.colors['text_light'],
                                 font=('Segoe UI', 10),
                                 command=self.on_mode_change)
        auto_rb.pack(anchor='w', pady=3)
        
        # ===== CONTROL BUTTONS SECTION =====
        buttons_frame = tk.Frame(main_frame, bg=self.colors['bg_dark'])
        buttons_frame.pack(pady=15)
        
        # Play/Pause button
        self.play_btn = tk.Button(buttons_frame, text="▶️ Mulai", 
                                  command=self.toggle_timer,
                                  bg=self.colors['accent_work'],
                                  fg='white',
                                  font=('Segoe UI', 11, 'bold'),
                                  width=10, height=2,
                                  relief='flat',
                                  cursor='hand2')
        self.play_btn.pack(side='left', padx=4)
        
        # Reset button
        self.reset_btn = tk.Button(buttons_frame, text="🔄 Reset",
                                   command=self.reset_timer,
                                   bg=self.colors['button_bg'],
                                   fg='white',
                                   font=('Segoe UI', 11, 'bold'),
                                   width=10, height=2,
                                   relief='flat',
                                   cursor='hand2')
        self.reset_btn.pack(side='left', padx=4)
        
        # Skip button
        self.skip_btn = tk.Button(buttons_frame, text="⏭️ Skip",
                                  command=self.skip_session,
                                  bg=self.colors['button_bg'],
                                  fg='white',
                                  font=('Segoe UI', 11, 'bold'),
                                  width=8, height=2,
                                  relief='flat',
                                  cursor='hand2')
        self.skip_btn.pack(side='left', padx=4)
        
        # Button hover effects
        self.setup_button_hover(self.play_btn, self.colors['accent_work'], '#ff6b8a')
        self.setup_button_hover(self.reset_btn, self.colors['button_bg'], self.colors['button_hover'])
        self.setup_button_hover(self.skip_btn, self.colors['button_bg'], self.colors['button_hover'])
        
        # Initialize timer display
        self.reset_timer()
        
    def setup_button_hover(self, button, normal_color, hover_color):
        """Setup hover effect for buttons"""
        def on_enter(e):
            button.configure(bg=hover_color)
        def on_leave(e):
            if button == self.play_btn:
                if self.session_type == SessionType.WORK:
                    button.configure(bg=self.colors['accent_work'])
                else:
                    button.configure(bg=self.colors['accent_break'])
            else:
                button.configure(bg=normal_color)
        
        button.bind('<Enter>', on_enter)
        button.bind('<Leave>', on_leave)
        
    def draw_timer_circle(self, progress):
        """Draw circular progress indicator"""
        self.timer_canvas.delete("arc")
        self.timer_canvas.delete("bg_circle")
        self.timer_canvas.delete("inner_circle")
        
        # Canvas dimensions
        width = 250
        height = 250
        center_x = width // 2
        center_y = height // 2
        radius = 110
        
        # Background circle
        self.timer_canvas.create_oval(
            center_x - radius, center_y - radius,
            center_x + radius, center_y + radius,
            outline=self.colors['bg_medium'], width=10,
            tags="bg_circle"
        )
        
        # Progress arc color based on session type
        arc_color = self.colors['accent_work'] if self.session_type == SessionType.WORK else self.colors['accent_break']
        
        # Progress arc (clockwise from top)
        if progress > 0:
            start_angle = 90
            extent = -360 * progress
            
            self.timer_canvas.create_arc(
                center_x - radius, center_y - radius,
                center_x + radius, center_y + radius,
                start=start_angle, extent=extent,
                outline=arc_color, width=10, style='arc',
                tags="arc"
            )
        
        # Inner decorative circle
        inner_radius = 90
        self.timer_canvas.create_oval(
            center_x - inner_radius, center_y - inner_radius,
            center_x + inner_radius, center_y + inner_radius,
            outline=self.colors['bg_medium'], width=2,
            tags="inner_circle"
        )
        
    def on_mode_change(self):
        """Handle mode change"""
        mode = self.mode_var.get()
        if mode == "automatic":
            self.status_label.configure(text="🔄 Mode Otomatis aktif")
        else:
            self.status_label.configure(text="📋 Mode Manual aktif")
            
    def toggle_timer(self):
        """Toggle between play and pause"""
        if self.timer_state == TimerState.IDLE or self.timer_state == TimerState.FINISHED:
            self.start_timer()
        elif self.timer_state == TimerState.RUNNING:
            self.pause_timer()
        elif self.timer_state == TimerState.PAUSED:
            self.resume_timer()
            
    def start_timer(self):
        """Start the timer"""
        if self.session_type == SessionType.WORK:
            self.time_left = self.work_time.get() * 60
            self.total_time = self.time_left
        else:
            self.time_left = self.break_time.get() * 60
            self.total_time = self.time_left
            
        self.timer_state = TimerState.RUNNING
        self.update_button_state()
        self.update_display()
        self.countdown()
        
        # Disable settings while running
        self.work_spinbox.configure(state='disabled')
        self.break_spinbox.configure(state='disabled')
        
    def pause_timer(self):
        """Pause the timer"""
        self.timer_state = TimerState.PAUSED
        self.update_button_state()
        self.status_label.configure(text="⏸️ Timer dijeda")
        
    def resume_timer(self):
        """Resume the timer"""
        self.timer_state = TimerState.RUNNING
        self.update_button_state()
        self.countdown()
        
    def reset_timer(self):
        """Reset the timer to initial state"""
        self.timer_state = TimerState.IDLE
        self.session_type = SessionType.WORK
        self.sessions_completed = 0
        self.time_left = self.work_time.get() * 60
        self.total_time = self.time_left
        
        self.update_display()
        self.update_button_state()
        self.update_session_display()
        self.counter_label.configure(text="Sesi selesai: 0")
        self.status_label.configure(text="⏸️ Siap untuk mulai")
        
        # Re-enable settings
        self.work_spinbox.configure(state='normal')
        self.break_spinbox.configure(state='normal')
        
        # Reset progress circle
        self.draw_timer_circle(0)
        
    def skip_session(self):
        """Skip current session and move to next"""
        if self.timer_state != TimerState.IDLE:
            self.time_left = 0
            self.on_timer_complete()
            
    def countdown(self):
        """Countdown logic"""
        if self.timer_state != TimerState.RUNNING:
            return
            
        if self.time_left > 0:
            self.time_left -= 1
            self.update_display()
            self.root.after(1000, self.countdown)
        else:
            self.on_timer_complete()
            
    def on_timer_complete(self):
        """Handle timer completion"""
        self.timer_state = TimerState.FINISHED
        
        # Play notification sound
        self.root.bell()
        
        mode = self.mode_var.get()
        
        if mode == "manual":
            # Manual mode - pause and wait for user
            if self.session_type == SessionType.WORK:
                self.sessions_completed += 1
                self.counter_label.configure(text=f"Sesi selesai: {self.sessions_completed}")
                messagebox.showinfo("⏰ Timer Selesai!", 
                                   f"Waktu kerja selesai! 🎉\n\nSesi ke-{self.sessions_completed} selesai.\nKlik OK lalu tekan 'Mulai' untuk istirahat.")
                self.session_type = SessionType.BREAK
            else:
                messagebox.showinfo("☕ Istirahat Selesai!", 
                                   "Waktu istirahat selesai!\n\nKlik OK lalu tekan 'Mulai' untuk kembali bekerja.")
                self.session_type = SessionType.WORK
                
            self.update_session_display()
            self.update_button_state()
            self.time_left = (self.work_time.get() if self.session_type == SessionType.WORK 
                            else self.break_time.get()) * 60
            self.total_time = self.time_left
            self.update_display()
            
        else:
            # Automatic mode - switch sessions automatically
            if self.session_type == SessionType.WORK:
                self.sessions_completed += 1
                self.counter_label.configure(text=f"Sesi selesai: {self.sessions_completed}")
                self.session_type = SessionType.BREAK
                self.status_label.configure(text="☕ Memulai waktu istirahat...")
            else:
                self.session_type = SessionType.WORK
                self.status_label.configure(text="💪 Kembali bekerja...")
                
            self.update_session_display()
            
            # Short delay before starting next session
            self.root.after(1500, self.start_timer)
            
    def update_display(self):
        """Update timer display"""
        minutes = self.time_left // 60
        seconds = self.time_left % 60
        
        time_str = f"{minutes:02d}:{seconds:02d}"
        self.timer_canvas.itemconfig(self.timer_text_id, text=time_str)
        
        # Update progress circle
        if self.total_time > 0:
            progress = 1 - (self.time_left / self.total_time)
            self.draw_timer_circle(progress)
            
        # Update status
        if self.timer_state == TimerState.RUNNING:
            session_text = "kerja" if self.session_type == SessionType.WORK else "istirahat"
            self.status_label.configure(text=f"▶️ Waktu {session_text} berjalan...")
            
    def update_button_state(self):
        """Update button text and colors based on state"""
        if self.timer_state == TimerState.IDLE or self.timer_state == TimerState.FINISHED:
            self.play_btn.configure(text="▶️ Mulai")
        elif self.timer_state == TimerState.RUNNING:
            self.play_btn.configure(text="⏸️ Jeda")
        elif self.timer_state == TimerState.PAUSED:
            self.play_btn.configure(text="▶️ Lanjut")
            
        # Update play button color based on session type
        if self.session_type == SessionType.WORK:
            self.play_btn.configure(bg=self.colors['accent_work'])
        else:
            self.play_btn.configure(bg=self.colors['accent_break'])
            
    def update_session_display(self):
        """Update session type display"""
        if self.session_type == SessionType.WORK:
            self.session_label.configure(text="💼 Mode: Kerja", 
                                        fg=self.colors['accent_work'])
        else:
            self.session_label.configure(text="☕ Mode: Istirahat",
                                        fg=self.colors['accent_break'])


def main():
    root = tk.Tk()
    
    app = PomodoroTimer(root)
    
    # Center window on screen
    root.update_idletasks()
    width = root.winfo_width()
    height = root.winfo_height()
    x = (root.winfo_screenwidth() // 2) - (width // 2)
    y = (root.winfo_screenheight() // 2) - (height // 2)
    root.geometry(f'+{x}+{y}')
    
    root.mainloop()


if __name__ == "__main__":
    main()
