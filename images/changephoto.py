from PIL import Image

img = Image.open("./logo.png")
img.save("./logo.ico", format="ICO", sizes=[(256, 256)])