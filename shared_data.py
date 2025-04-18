import threading

class SharedData:
    def __init__(self):
        self.lock = threading.Lock()
        self.latest_frame = None
        self.mediapipe_data = None
        self.midas_data = None
        self.running = True