from threading import Thread
import sys
import cv2
import platform

class Video:
    
    def __init__(self, path=0, size=(720, 480), output=(720, 480), camera=True):
        print("[DEBUG] Camera Input: ", size)
        print("[DEBUG] Camera Scaled Output: ", output)

        system = platform.system()
        if(system == 'Windows'):
            self.stream = cv2.VideoCapture(path | cv2.CAP_DSHOW) #TODO add DSHOW
        else:
            self.stream = cv2.VideoCapture(path)

        print("[DEBUG] Camera open")

        if(camera):
            self.stream.set(cv2.CAP_PROP_FRAME_WIDTH, size[0])
            self.stream.set(cv2.CAP_PROP_FRAME_HEIGHT, size[1])

        (self.grabbed, self.frame) = self.stream.read()
        self.stopped = False
        self.size = size
        self.output = output

    def start(self):
        t = Thread(target=self.update, args=())
        t.start()
        return self

    def update(self):
        while True:
            if self.stopped:
                return
            (self.grabbed, self.frame) = self.stream.read()

    def read(self):
        return self.frame

    def readScaled(self):
        if(self.frame is not None):
            return cv2.resize(self.frame, self.output)
        else:
            return None

    def stop(self):
        self.stopped = True


