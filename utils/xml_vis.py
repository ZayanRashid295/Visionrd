import cv2
import xml.etree.ElementTree as ET
import os
import argparse

def process_folder(xml_file, images_dir, output_video):
    # Parse the XML file
    tree = ET.parse(xml_file)
    root = tree.getroot()

    # Get all image entries
    images = root.findall("image")

    # Initialize video writer
    frame_width, frame_height = 720, 404  # Assuming fixed dimensions from the XML
    fps = 15
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(output_video, fourcc, fps, (frame_width, frame_height))

    # Process each image
    for image in images:
        image_name = image.get("name")
        # Update path construction to match the new structure
        folder_name = os.path.basename(os.path.dirname(xml_file))
        image_path = os.path.join(images_dir, folder_name, image_name.replace('frame_', '') + ".png")
        
        # Add fallback path if the first attempt fails
        if not os.path.exists(image_path):
            image_path = os.path.join(images_dir, image_name + ".png")
        
        frame = cv2.imread(image_path)

        if frame is None:
            print(f"Image not found: {image_path}")
            continue

        # Draw bounding boxes and polylines
        for box in image.findall("box"):
            xtl, ytl = int(float(box.get("xtl"))), int(float(box.get("ytl")))
            xbr, ybr = int(float(box.get("xbr"))), int(float(box.get("ybr")))
            label = box.get("label")
            cv2.rectangle(frame, (xtl, ytl), (xbr, ybr), (0, 255, 0), 2)
            cv2.putText(frame, label, (xtl, ytl - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        for polyline in image.findall("polyline"):
            points = polyline.get("points").split(";")
            points = [tuple(map(int, map(float, point.split(",")))) for point in points]
            label = polyline.get("label")
            for i in range(len(points) - 1):
                cv2.line(frame, points[i], points[i + 1], (255, 0, 0), 2)
            if points:
                cv2.putText(frame, label, points[0], cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

        # Write frame to video
        video_writer.write(frame)

    # Release video writer
    video_writer.release()
    print(f"Video saved at {output_video}")

def main():
    parser = argparse.ArgumentParser(description='Process a folder of images and save annotated frames.')
    # Update default paths
    parser.add_argument('--xml_folder', type=str, help='Path to the folder containing XML files', 
                       default="/home/user/AI-Hackathon24/data/xmls")
    parser.add_argument('--images_dir', type=str, help='Path to the directory containing PNG frames', 
                       default="/home/user/AI-Hackathon24/data/features/gtea_png/png")
    parser.add_argument('--output_dir', type=str, help='Path to the directory where processed videos will be saved', default="data/features/gtea_png/png_vis")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    for xml_file in os.listdir(args.xml_folder):
        if xml_file.endswith('.xml'):
            xml_path = os.path.join(args.xml_folder, xml_file)
            video_name = xml_file.replace('.xml', '_vis.mp4')
            output_video = os.path.join(args.output_dir, video_name)
            images_subdir = os.path.join(args.images_dir, xml_file.replace('.xml', ''))
            process_folder(xml_path, images_subdir, output_video)

if __name__ == '__main__':
    main()
