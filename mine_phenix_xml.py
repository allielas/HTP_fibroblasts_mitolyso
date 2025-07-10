import os, re
from pathlib import Path 
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches

from matplotlib.collections import PatchCollection
import numpy as np
from IPython.display import display
import xml.etree.ElementTree as ET


xml_path = Path("/mnt/bigdisk1/AllieSpangaro/Morphology_Replicative_Age_Project/Raw Images/20250328_AS_MRC-5_Rep5__2025-03-28T16_23_25-Measurement 1/Images/Index.idx.xml")
tree = ET.parse(xml_path)
root = tree.getroot()

print(root.findall('.//Plates'))


# ...existing code...
ns = root.tag.split('}')[0].strip('{')
final_img_list  = []
img_data = []

test_id = "0202"

with open("index_idx_info.txt", "w") as out_txt:
    for wells in root.findall(f".//{{{ns}}}Wells"):
        for well in wells.findall(f"{{{ns}}}Well"):
            well_id = well.findtext(f"{{{ns}}}id")
            row = well.findtext(f"{{{ns}}}Row")
            col = well.findtext(f"{{{ns}}}Col")
            
            if well_id == "0202":
                images = well.findall(f"{{{ns}}}Image")
                img_list = [image.attrib.get('id', '') for image in images]
            
                
            #image_list = images.attrib.get('id', '').split(',') if images.attrib.get('id') else []
                out_txt.write(f"Well id: {well_id}, Row: {row}, Col: {col}\n Images: {img_list}\n")
                
                final_img_list = img_list
    for images in root.find(f".//{{{ns}}}Images").findall(f"{{{ns}}}Image"):
        image_id, pos_x, pos_y, pos_z = '', '', '', ''
        #print(images.tag, images.attrib)
        found = False
        for tags in images:
            #print(tags.tag, tags.attrib)
            
            if tags.tag == f"{{{ns}}}id":
                #print(tags.text)
                image_id = tags.text
                if image_id in final_img_list:
                    out_txt.write(f"Image id: {image_id}\n Attributes: {images.attrib}\n Text: {images.text}\n")
                    found = True
                else:
                    ValueError(f"Image id {image_id} not found in final image list.")
            if found == False:
                continue
            if tags.tag == f"{{{ns}}}PositionX":    
                out_txt.write(f"Position: {tags.text}\n")
                pos_x = tags.text
            if tags.tag == f"{{{ns}}}PositionY":
                out_txt.write(f"PositionY: {tags.text}\n")
                pos_y = tags.text
            if tags.tag == f"{{{ns}}}PositionZ":
                out_txt.write(f"PositionZ: {tags.text}\n")
                pos_z = tags.text
        if found:  
            field = re.search(pattern = r"F(\d{1,2})P", string = image_id).group(1)
            img_data.append({
            "Image_ID": image_id,
            "Well_ID": image_id[:4],  # Assuming well ID is the first 4 characters of image ID
            "Well": f"{chr(ord('@') + int(image_id[:2]))}{int(image_id[2:4]):02d}",
            "Row": int(image_id[:2]),  # Assuming row is the next
            "Col": int(image_id[2:4]),  # Assuming column is the next 2 characters of image ID
            "Field_ID": f"F{int(field):02}",  # Assuming field ID is in the format "FXXP"
            "Field": int(field),  # Matches "FXX" where XX are digits
            "PositionX": pos_x,
            "PositionY": pos_y,
            "PositionZ": pos_z
            })
coordinates_df = pd.DataFrame(img_data)

resolution = 9.4916838247105038E-02 # microns per pixel
well_size = 2160 * resolution # microns

def calculate_bbox(row):
    return {
        'min_bbox_PositionX (um)': row['PositionX (um)'] - well_size / 2, # Calculate min bbox PositionX, given that (X,Y) is the center of the well
        'max_bbox_PositionX (um)': row['PositionX (um)'] + well_size / 2,
        'min_bbox_PositionY (um)': row['PositionY (um)'] - well_size / 2,
        'max_bbox_PositionY (um)': row['PositionY (um)'] + well_size / 2
    }

for col in ['PositionX', 'PositionY', 'PositionZ']:
    coordinates_df[col] = pd.to_numeric(coordinates_df[col], errors='coerce')
    coordinates_df[f'{col} (um)'] = np.round(coordinates_df[col] * 1E6,5) # Convert to micrometers
    coordinates_df[f'{col} (pixels)'] = np.round(coordinates_df[f'{col} (um)'] / resolution,3)
    

coordinates_df[['min_bbox_PositionX (um)', 'max_bbox_PositionX (um)', 'min_bbox_PositionY (um)', 'max_bbox_PositionY (um)']] = coordinates_df.apply(calculate_bbox, axis=1, result_type='expand')
coordinates_df['bbox_coords'] = coordinates_df.apply(lambda row: f"{round(row['min_bbox_PositionX (um)'],1)},{round(row['min_bbox_PositionY (um)'],1)},{round(row['max_bbox_PositionX (um)'],1)},{round(row['max_bbox_PositionY (um)'],1)}", axis=1)

def plot_well_coordinates(coordinates_df, save = False):
    fig, ax = plt.subplots(figsize=(12, 12))
    ax.scatter(coordinates_df['PositionX (um)'], coordinates_df['PositionY (um)'], c='blue', s=30, alpha=0.5, zorder = 2, label='Image Position (center)')
    for i, row in coordinates_df.iterrows():
        bbox = patches.Rectangle((row['min_bbox_PositionX (um)'], row['min_bbox_PositionY (um)']), 
                                well_size , well_size, 
                                linewidth=1, edgecolor='c', facecolor='m', fill=False, alpha=0.5)
        
        ax.add_patch(bbox)
        ax.annotate(coordinates_df['Field'][i], (coordinates_df['PositionX (um)'][i] + well_size/2, coordinates_df['PositionY (um)'][i]+ well_size/2),
                    fontsize=10, ha='center', va='center', color='black')
        ax.annotate(coordinates_df['bbox_coords'][i], (coordinates_df['min_bbox_PositionX (um)'][i] , coordinates_df['min_bbox_PositionY (um)'][i]),
                    fontsize=8, ha='center', va='center', color='black')
        #bbox = Bbox.from_bounds(coordinates_df['min_bbox_PositionX (um)'][i], coordinates_df['min_bbox_PositionY (um)'][i], well_size, well_size)

#plt.annotate(coordinates_df['min_bbox_PositionX (um)'], coordinates_df['min_bbox_PositionY (um)'], c='red', s=10, alpha=0.5, label='Min BBox')
#plt.scatter(coordinates_df['max_bbox_PositionX (um)'], coordinates_df['max_bbox_PositionY (um)'], c='green', s=10, alpha=0.5, label='Max BBox')
    plt.legend()
    plt.title('Image Coordinates')
    plt.xlabel('PositionX (um)')
    plt.ylabel('PositionY (um)')
    plt.grid(True)
    plt.axis('equal')
    if save:
        plt.savefig(f"Well {coordinates_df['Well'][0]}.png")
    plt.show()

coordinates_df.to_csv("image_coordinates.csv", index=False) #should be 1120 rows for one well

def block_a_order_key(img_id):
    
    field_pattern = r"F(\d{1,2})P"  # Matches "FXX" where XX are digits
    match = re.search(field_pattern, img_id)
    if match:
        field = int(match.group(1))
        # Place f01 after f25 but before f26
        if field == 1:
            field_order = 25.5
        else:
            field_order = float(field)
    else:
        # If no match, put at the end
        field_order = float('inf')
    #print (filename, field_order)
    return field_order

unique_coordinates_df = coordinates_df.drop_duplicates(subset=['PositionX (um)','PositionY (um)'])
sorted_unique_coordinates_df = unique_coordinates_df.sort_values(by='Image_ID', key=lambda x: x.apply(block_a_order_key))
sorted_unique_coordinates_df.reset_index(drop=True, inplace=True)
readable_sorted_unique_coordinates_df = sorted_unique_coordinates_df[['Image_ID','Field_ID','PositionX (um)', 'PositionY (um)']]

plot_well_coordinates(coordinates_df)
plot_well_coordinates(sorted_unique_coordinates_df, save=True)

print(readable_sorted_unique_coordinates_df)
  
# ...existing code...   
#print('well', well)
#r02c02 = well.find('0202')
#print('well', r02c02)
# ...existing code...
# ...existing code...
'''
with open("index_idx_info.txt", "w") as out_txt:
    for elem in root.iter():
        if "Wells" in elem.tag:
            if 'id' in elem.attrib:
                for child in elem.iter():
                    out_txt.write(f"Element text: {elem.text}\n")
                    out_txt.write(f"Well attrib: {elem.attrib}\n")
                    out_txt.write(f"Well tags: {elem.tag}\n")
                    id = elem.attrib.get('id', None)
                    if id == "0202":
                        out_txt.write(f"Element tag: {elem.tag}\n")
                        out_txt.write(f"Well attrib: {child.attrib}\n")
                        out_txt.write(f"Well tags: {child.tag}\n")
                        out_txt.write(f"Well text: {child.text}\n")'''
                        
                
           # out_txt.write(ET.tostring(elem, encoding="unicode") + "\n")

'''for elem in root.iter():
    out_txt.write("Attributes:", elem.attrib)
    out_txt.write("\n Tags:", str(elem.tag) + "\n")
    # Example: print all child Image elements
    if elem.tag == "Well":
        print(ET.tostring(well, encoding="unicode"))
        out_txt.write(ET.tostring(well, encoding="unicode") + "\n")
    
    for image in well.findall("Image"):
        print("  Image attributes:", image.attrib)
# ...existing code...'''