from PIL import Image
import requests
from io import BytesIO
import base64

def load_pil_from_url(url):
    img_data = requests.get(url, stream=True).raw
    img_pil = Image.open(img_data)
    return img_pil

def load_bs64_from_url(url):
    img_data = requests.get(url, stream=True).raw
    if isinstance(img_data, base64):
        return img_data
    else:
        return base64.b64encode(img_data)

def bs64_to_pil(img_bs64):
    img_data = base64.b64decode(img_bs64)
    img_pil = Image.open(BytesIO(img_data))
    return img_pil

def img_box_crop(img_pil, box):
    x1, y1, x2, y2 = box
    return img_pil.crop((x1,y1,x2,y2))

def np_to_bs64(img_np):
    img = Image.fromarray(img_np.astype('uint8'))
    buffered = BytesIO()
    img.save(buffered, format="png")
    img_bytes = buffered.getvalue()
    img_base64 = base64.b64encode(img_bytes).decode()
    return img_base64

def save_image_to_local(img):
    Image.fromarray(img).save("./test_img.png")

