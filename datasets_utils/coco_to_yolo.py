import json


class ConvertCOCOToYOLO:

    def __init__(self, json_path, target_path):
        self.target_path = target_path
        self.json_path = json_path

    def convert_labels(self, x, y, w, h, size=(640, 640)):

        dw = 1. / size[1]
        dh = 1. / size[0]

        x = x * dw
        w = w * dw
        y = y * dh
        h = h * dh
        return (x, y, w, h)

    def convert(self, annotation_key='annotations', img_id='image_id', cat_id='category_id'):
        # Enter directory to read JSON file
        data = json.load(open(self.json_path))

        check_set = set()

        # Retrieve data
        for i in range(len(data[annotation_key])):

            # Get required data
            image_id = data[annotation_key][i][img_id]
            category_id = data[annotation_key][i][cat_id]

            bbox = data[annotation_key][i]['bbox']

            yolo_bbox = self.convert_labels(*bbox)

            # Prepare for export

            image_info = data['images'][image_id]
            assert image_info['id'] == image_id

            image_name = image_info['file_name'].rsplit('.', maxsplit=1)[0]
            filename = f'{self.target_path}/{image_name}.txt'

            category_info = data['categories'][category_id]
            assert category_info['id'] == category_id

            content = f"{category_info['name']} {yolo_bbox[0]} {yolo_bbox[1]} {yolo_bbox[2]} {yolo_bbox[3]}"

            # Export
            if image_id in check_set:
                # Append to existing file as there can be more than one label in each image
                file = open(filename, "a")
                file.write("\n")
                file.write(content)
                file.close()

            elif image_id not in check_set:
                check_set.add(image_id)
                # Write files
                file = open(filename, "w")
                file.write(content)
                file.close()


if __name__ == '__main__':
    # ConvertCOCOToYOLO(json_path='../../car_dent_coco/car_dent_coco/train/_annotations.coco.json',
    #                   target_path='../../car_dent_coco/car_dent_coco/labels/train').convert()

    ConvertCOCOToYOLO(json_path='../../car_dent_coco/car_dent_coco/valid/_annotations.coco.json',
                      target_path='../../car_dent_coco/car_dent_coco/labels/valid').convert()
