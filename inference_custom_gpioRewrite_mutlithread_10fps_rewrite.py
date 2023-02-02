import ctypes
import threading
import time

import cv2
import imgui
#import Jetson.GPIO as GPIO
import OpenGL.GL as gl
import sdl2 as sdl
import torch
from imgui.integrations.sdl2 import SDL2Renderer
from influxdb_client import InfluxDBClient, Point
from influxdb_client.client.write_api import ASYNCHRONOUS, WritePrecision

VideoDevice = 1
webcam_frame_width = 1280
webcam_frame_height = 720
# GPIOLEDPin = 7


class CameraThread:
    def __init__(self, src, width, height, model):
        self.src = src
        self.model = model
        self.cap = cv2.VideoCapture(self.src, cv2.CAP_V4L2)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self.cap.set(cv2.CAP_PROP_FOURCC,
                     cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))
        self.texture = gl.glGenTextures(1)
        if not (self.cap.isOpened()):
            print("VideoCapture error.")
            return

        self.ret, self.image = self.cap.read()
        if not (self.ret):
            print("No more frames.")
            return
        self.img_rgb = cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB)
        self.output = self.model(self.img_rgb)

        self.thread = threading.Thread(target=self.update, daemon=True)
        self.started = False

    def start(self):
        self.started = True
        self.thread.start()
        return self

    def update(self):
        while self.started:
            self.ret, self.image = self.cap.read()

    def read(self):
        return self.image, self.output

    def bind(self, image):
        # opengl prepare textures
        gl.glBindTexture(gl.GL_TEXTURE_2D, self.texture)
        gl.glTexParameteri(
            gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MAG_FILTER, gl.GL_LINEAR)
        gl.glTexParameteri(
            gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MIN_FILTER, gl.GL_LINEAR)
        gl.glTexImage2D(gl.GL_TEXTURE_2D, 0, gl.GL_RGB, webcam_frame_width,
                        webcam_frame_height, 0, gl.GL_BGR, gl.GL_UNSIGNED_BYTE, image)

    def stop(self):
        self.started = False
        self.thread.join()

    def __exit__(self, exc_type, exc_value, exc_traceback):
        self.cap.release()


def impl_pysdl2_init():
    width, height = 1920, 1080
    window_name = "minimal ImGui/SDL2 example"

    if sdl.SDL_Init(sdl.SDL_INIT_VIDEO) < 0:
        print("Error: SDL could not initialize! SDL Error: " +
              sdl.SDL_GetError().decode("utf-8"))
        exit(1)

    sdl.SDL_GL_SetAttribute(sdl.SDL_GL_DOUBLEBUFFER, 1)
    sdl.SDL_GL_SetAttribute(sdl.SDL_GL_DEPTH_SIZE, 24)
    #sdl.SDL_GL_SetAttribute(sdl.SDL_GL_STENCIL_SIZE, 8)
    sdl.SDL_GL_SetAttribute(sdl.SDL_GL_ACCELERATED_VISUAL, 1)
    #sdl.SDL_GL_SetAttribute(sdl.SDL_GL_MULTISAMPLEBUFFERS, 1)
    #sdl.SDL_GL_SetAttribute(sdl.SDL_GL_MULTISAMPLESAMPLES, 16)
    sdl.SDL_GL_SetAttribute(sdl.SDL_GL_CONTEXT_FLAGS,
                            sdl.SDL_GL_CONTEXT_FORWARD_COMPATIBLE_FLAG)
    sdl.SDL_GL_SetAttribute(sdl.SDL_GL_CONTEXT_MAJOR_VERSION, 4)
    sdl.SDL_GL_SetAttribute(sdl.SDL_GL_CONTEXT_MINOR_VERSION, 6)
    sdl.SDL_GL_SetAttribute(sdl.SDL_GL_CONTEXT_PROFILE_MASK,
                            sdl.SDL_GL_CONTEXT_PROFILE_CORE)

    sdl.SDL_SetHint(sdl.SDL_HINT_MAC_CTRL_CLICK_EMULATE_RIGHT_CLICK, b"1")
    sdl.SDL_SetHint(sdl.SDL_HINT_VIDEO_HIGHDPI_DISABLED, b"1")

    window = sdl.SDL_CreateWindow(window_name.encode('utf-8'),
                                  sdl.SDL_WINDOWPOS_CENTERED, sdl.SDL_WINDOWPOS_CENTERED,
                                  width, height,
                                  sdl.SDL_WINDOW_OPENGL | sdl.SDL_WINDOW_RESIZABLE)

    if window is None:
        print("Error: Window could not be created! SDL Error: " +
              sdl.SDL_GetError().decode("utf-8"))
        exit(1)

    gl_context = sdl.SDL_GL_CreateContext(window)
    if gl_context is None:
        print("Error: Cannot create OpenGL Context! SDL Error: " +
              sdl.SDL_GetError().decode("utf-8"))
        exit(1)

    sdl.SDL_GL_MakeCurrent(window, gl_context)
    if sdl.SDL_GL_SetSwapInterval(1) < 0:
        print("Warning: Unable to set VSync! SDL Error: " +
              sdl.SDL_GetError().decode("utf-8"))
        exit(1)

    return window, gl_context


def loggingToInfluxDB(noMaskCount):
    bucket = "maskAI"
    client = InfluxDBClient.from_config_file("influxdb.ini")
    write_api = client.write_api(write_options=ASYNCHRONOUS)

    p = Point("no_mask").field("amount", noMaskCount)
    write_api.write(bucket=bucket, record=p, write_precision=WritePrecision.S)


def main():
    SDLwindow, glContext = impl_pysdl2_init()
    # Setup YOLOv5
    devices = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = torch.hub.load(
        'yolov5', 'custom', path='20221201-boxRecognition-300epoch-yolov5s.pt', source="local")
    #   Path to yolov5, 'custom', path to weight, source='local'
    model.to(devices)
    model.eval()

    # Setup GPIO
    #GPIO.setmode(GPIO.BOARD)
    #GPIO.setup(GPIOLEDPin, GPIO.OUT, initial=GPIO.LOW)

    # Setup Image Capture
    video = CameraThread(src=VideoDevice, width=webcam_frame_width,
                         height=webcam_frame_height, model=model)
    video.start()
    frame_height = webcam_frame_height
    frame_width = webcam_frame_width

    # Setup logging
    timeRetain = ""

    # Setup imgui
    imgui.create_context()  # type: ignore
    impl = SDL2Renderer(SDLwindow)
    sdlEvent = sdl.SDL_Event()

    io = imgui.get_io()  # type: ignore
    clearColorRGB = 1., 1., 1.
    newFont = io.fonts.add_font_from_file_ttf(
        'fonts/NotoSansMono-Regular.ttf', 36)
    impl.refresh_font_texture()

    # States and variables
    running = True
    showCustomWindow = True
    cBoxBoxClass = True
    boxThreshold = 0.4
    showloggingWindow = True
    cBoxLogToInfluxDB = False
    maxHeadCount = 0
    showImageTexture = True

    while (running):
        # read frame
        image, output = video.read()
        # print(output)
        img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        output = model(img_rgb)

        # TODO: move this to the imageProcessing thread.
        # print custom bounding box
        for box in output.xyxy[0]:
            xmin, ymin, xmax, ymax = int(box[0]), int(
                box[1]), int(box[2]), int(box[3])
            # boxes
            if cBoxBoxClass == True and box[5] == 0 and box[4] > boxThreshold:
                image = cv2.rectangle(
                    image, (xmin, ymin), (xmax, ymax), (97, 105, 255), 6)
                image = cv2.rectangle(
                    image, (xmin, ymin), (xmin+250, ymin-30), (97, 105, 255), -1)
                image = cv2.putText(image, f'Box {box[4]:.2f}', (int(xmin), int(
                    ymin)-5), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

        video.bind(image=image)

        # GPIO and logging stuff
        # counting if there's any no_mask
        boxCount = output.pandas().xyxy[0]['class'].tolist().count(0)
        if boxCount >= 1 and cBoxLogToInfluxDB == True:
            #GPIO.output(GPIOLEDPin, GPIO.HIGH)
            # maybe telegraf client here?
            # TODO:figure out this with thread(loggingToInfluxDB()
            #loggingThread = threading.Thread(target=loggingToInfluxDB, args=(noMaskCount,), daemon=True)
            # loggingThread.start()
            loggingToInfluxDB(boxCount)
        #else:
            #GPIO.output(GPIOLEDPin, GPIO.LOW)

        # SDL & imgui event polling
        while sdl.SDL_PollEvent(ctypes.byref(sdlEvent)) != 0:
            if sdlEvent.type == sdl.SDL_QUIT:
                running = False
                break
            impl.process_event(sdlEvent)
        impl.process_inputs()

        imgui.new_frame()          # type: ignore

        if (showCustomWindow):
            expandCustomWindow, showCustomWindow = imgui.begin(
                "sdlWindow", True)
            imgui.text(f"FPS: {io.framerate:.2f}")
            _, clearColorRGB = imgui.color_edit3(
                "Background Color", *clearColorRGB)
            imgui.new_line()
            imgui.text(f"Total Threads: {threading.active_count()}")
            _, cBoxLogToInfluxDB = imgui.checkbox(
                "Log to InfluxDB", cBoxLogToInfluxDB)
            imgui.new_line()
            imgui.text("Settings:")
            _, cBoxBoxClass = imgui.checkbox(
                "Box", cBoxBoxClass)
            _, boxThreshold = imgui.slider_float(
                "Box Threshold", boxThreshold,
                min_value=0.0, max_value=1.0,
                format="%.2f"
            )
            imgui.end()

        if (showImageTexture):
            expandImageTexture, showImageTexture = imgui.begin(
                "ImageTexture", False)
            imgui.image(video.texture, frame_width, frame_height)
            imgui.end()

        timeNow = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        nowHeadCount = len(output.pandas().xyxy[0].index)
        if (showloggingWindow):
            expandloggingWindow, showloggingWindow = imgui.begin(
                "logging", True)
            if boxCount > 0:
                timeRetain = timeNow
            if nowHeadCount > maxHeadCount:
                maxHeadCount = nowHeadCount
            with imgui.font(newFont):
                imgui.text(
                    f"Boxes in view: {nowHeadCount}\tRecorded max boxes in view: {maxHeadCount}")
                imgui.new_line()
                imgui.text(f"{timeRetain}: Box detected.")
            imgui.end()

        gl.glClearColor(clearColorRGB[0],
                        clearColorRGB[1], clearColorRGB[2], 1)
        gl.glClear(gl.GL_COLOR_BUFFER_BIT)

        imgui.render()
        impl.render(imgui.get_draw_data())
        sdl.SDL_GL_SwapWindow(SDLwindow)

    video.stop()
    # loggingThread.join()
    impl.shutdown()
    sdl.SDL_GL_DeleteContext(glContext)
    sdl.SDL_DestroyWindow(SDLwindow)
    sdl.SDL_Quit()


if __name__ == '__main__':
    main()
