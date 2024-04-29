import json
import os
from PIL import Image, ImageDraw, ImageFont
import json
from tqdm import tqdm

# The CTLABELS list for index-to-letter conversion
'''
CTLABELS = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's',
            't', 'u', 'v', 'w', 'x', 'y', 'z', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '']
'''
CTLABELS = [' ','!','"','#','$','%','&','\'','(',')','*','+',',','-','.','/',
                             '0','1','2','3','4','5','6','7','8','9',':',';','<','=','>','?','@',
                             'A','B','C','D','E','F','G','H','I','J','K','L','M','N','O','P','Q','R','S','T','U','V',
                             'W','X','Y','Z','[','\\',']','^','_','`','a','b','c','d','e','f','g','h','i','j','k','l',
                             'm','n','o','p','q','r','s','t','u','v','w','x','y','z','{','|','}','~']


# The JSON data you provided


json_file = '/home/steven/data/project/AdelaiDet/datasets/on_site/train.json'
img_root = '/home/steven/data/project/AdelaiDet/datasets/on_site/train_images'

def draw_annotations(json_file_path,img_root):
    with open(json_file_path, 'r') as file:
        data = json.load(file)

    for image_info in data['images']:
        # Load the image
        try:
            image = Image.open(os.path.join(img_root,image_info['file_name']))
        except FileNotFoundError:
            print(f"File {image_info['file_name']} not found. Skipping.")
            continue
        draw = ImageDraw.Draw(image)

        # Process each annotation corresponding to the current image
        for annotation in [a for a in data['annotations'] if a['image_id'] == image_info['id']]:
            # Draw bezier points as blue dots
            for i in range(0, len(annotation['bezier_pts']), 2):
                x, y = annotation['bezier_pts'][i], annotation['bezier_pts'][i + 1]
                draw.ellipse((x-2, y-2, x+2, y+2), fill='blue')

            # Draw bbox as a red rectangle
            bbox = annotation['bbox']
            draw.rectangle([bbox[0], bbox[1], bbox[0] + bbox[2], bbox[1] + bbox[3]], outline='red')

            # Convert 'rec' indices to text, handling index 37 as ''
            text = ''.join(CTLABELS[i] for i in annotation['rec'] if i < len(CTLABELS))
            draw.text((bbox[0] + 5, bbox[1] + 5), text, fill='green')

        # Save the modified image
        output_dir = './output/'
        output_path = output_dir + os.path.basename(image_info['file_name'])
        image.save(output_path)
        print(f"Image saved to {output_path}")
# Example usage
draw_annotations(json_file,img_root)
