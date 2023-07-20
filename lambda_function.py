from inference import RemoveBackGround
import json
from utils import bs64_to_pil, img_box_crop, np_to_bs64

def respond(err, res):
    respond_msg = {'statusCode': 502 if err is not None else 200, 'body': json.dumps(res)}
    print(f'Respond Message: {respond_msg}')
    return respond_msg

def lambda_handler(event, context):

    print("Loading arguments")
    print(f"Event: {event}")
    args = json.loads(event)
    print(f"Arguments: {args}")

    print("Loading image")
    if "image_b64" in args:
        img_bs64 = args["image_b64"]
    else:
        print("Can not found image base64 format")
        raise AssertionError
    
    img_pil = bs64_to_pil(img_bs64)

    print("Cropping image")
    if "bbox" in args:
        box = args["bbox"]
        img_pil = img_box_crop(img_pil, box)
    
    print("Loading model")
    remover = RemoveBackGround(backbone="swinB", device="cpu")

    print("Processing model")
    output = remover.process(img_pil)

    print("Respond")
    output_b64 = np_to_bs64(output)
    result = {"mask":output_b64}

    return respond(None, result)
