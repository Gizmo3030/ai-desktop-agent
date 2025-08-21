try:
    import mss
    import base64
    from io import BytesIO
    from PIL import Image
except Exception:
    mss = None
    base64 = None
    BytesIO = None
    Image = None

try:
    from crewai.tools import BaseTool
except Exception:
    # Minimal fallback so tests can import without crewai.tools
    class BaseTool:
        name: str = "BaseTool"
        description: str = "Fallback BaseTool"


class ScreenCaptureTool(BaseTool):
    name: str = "Screen Capture Tool"
    description: str = "Captures the current screen and returns it as a base64 encoded image string."

    def _run(self) -> str:
        if mss is None or Image is None or base64 is None:
            return "Screen capture dependencies are not installed in this environment."
        try:
            with mss.mss() as sct:
                monitor_number = 1
                mon = sct.monitors[monitor_number]
                monitor = {
                    "top": mon["top"],
                    "left": mon["left"],
                    "width": mon["width"],
                    "height": mon["height"],
                    "mon": monitor_number,
                }
                sct_img = sct.grab(monitor)
                img = Image.frombytes("RGB", sct_img.size, sct_img.bgra, "raw", "BGRX")
                buffered = BytesIO()
                img.save(buffered, format="JPEG", quality=75)
                img_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')
                return f"Successfully captured the screen. Here is the base64 encoded image string: {img_base64}"
        except Exception as e:
            return f"Error capturing screen: {e}"
