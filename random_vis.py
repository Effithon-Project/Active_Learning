from matplotlib import pyplot as plt
import argparse
import os

def label_data_record(record):
    """
    Given a string 'record', this will
    output a dictionary of the appropriate labels
    and data types.
    """
    record_list = record.split(" ")
    record_dict = {
        "type":record_list.pop(0),
        "truncated":float(record_list.pop(0)),
        "occluded":int(record_list.pop(0)),
        "alpha":float(record_list.pop(0)),
        "left":float(record_list.pop(0)),
        "top":float(record_list.pop(0)),
        "right":float(record_list.pop(0)),
        "bottom":float(record_list.pop(0)),
        "height":float(record_list.pop(0)),
        "width":float(record_list.pop(0)),
        "length":float(record_list.pop(0)),
        "x":float(record_list.pop(0)),
        "y":float(record_list.pop(0)),
        "z":float(record_list.pop(0)),
    }
    return record_dict

def extract_bbox(labeled_record):
    """
    Given a labeled record from the training data, 
    this will output a 3-tuple with the bounding box parameters.
    This output can be passed into Rectangle()
    """
    width = labeled_record["right"] - labeled_record["left"]
    height = labeled_record["bottom"] - labeled_record["top"]
    bbox = ((labeled_record["left"], labeled_record["top"]), width, height)
    return bbox

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='This code is visualizing a random training example')
    
    parser.add_argument('--id', type=str,metavar='id',
                        help='training example id')
    args = parser.parse_args()
    example_id = args.id #string
    
    root = r"D:\kitti\object\training"
    img_path = os.path.join(root,"image_2", example_id+".png")
#     img_path = os.path.join(root, example_id+".png")

    test_image = plt.imread(img_path)
    
    filename = os.path.join(root,"label_2", example_id+".txt")
#     filename = os.path.join(root, example_id+".txt")

    with open(filename) as f:
        records = f.readlines()

    # Look at the labels
    for record in records:
        print(record)
        
        
    fig,ax = plt.subplots(figsize=(25, 10))


    ax.imshow(test_image)

    for record in records:
        # Add keys to the data point
        single_record = label_data_record(record)
        # Get the dimensions and location of the bounding box
        bbox_params = extract_bbox(single_record)
        # Create the bounding box
        bbox = plt.Rectangle(*bbox_params, fill=False,linewidth=1.5,edgecolor='g')
        # Add the bounding box to the plot
        ax.add_patch(bbox)
        # Add a label to the bounding box
        ax.annotate(s = single_record['type'], xy = bbox_params[0],color='#7CFC00', size=8)
        
    fig.savefig('./example_result.png')