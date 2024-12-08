# **Utility Functions** 


## **Feature Manifold Visualization**

- **Description**:

This script performs t-SNE visualization on a set of input features from a  pre-saved `.npy` feature file. It uses the provided labels to generate a 2D scatter plot, where each point corresponds to a frame or feature, color-coded by its label. The resulting visualization is saved as a PNG image in the specified output directory.

- **How to Run**:

Use the following command to run the 'feature_manifold_visualization.py' file

```python

python feature_manifold_visualization.py -i path/to/features/file.npy -l path/to/groundTruth/file.txt

```

- **Example**:

```python

python feature_manifold_visualization.py -i ./data/features/S1_Cheese_C1.npy -l ./data/features/S1_Cheese_C1.txt

```

## **Labels Visualization**

- **Description**:

This script analyzes a dataset of action labels for video frames, generating histograms for action duration, label counts, and video durations. It also computes class weights for cross-entropy loss and position weights for boundary detection tasks in action recognition.

- **How to Run**:

First, you need to run the 'generate_all_files.py' file. This file will extract all the video names from the video directory and write into a text file.

Use the following commands to run the 'labels_visualization.py' file

```python

python labels_visualization.py /path/to/videos /path/to/labels /path/to/mapping.txt /path/to/output

```

- **Example**:

```python

python labels_visualization.py ./data/gtea/videos ./data/gtea/groundTruth ./data/mapping.txt /output

```

## **Data Visualization**

- **Description**:

This script visualizes hand annotations on a video by overlaying bounding boxes, keypoints, and segmentation masks on each frame. The annotations, stored in a JSON file, are processed to draw keypoints, finger connections, and segmentation masks on the video, which is then saved as a new output video. The tool supports real-time visualization and can be stopped by pressing 'q'.

- **How to Run**:

Use the following commands to run the 'data_visualization.py' file

```python

python data_visualization.py /path/to/input/file.mp4 path/to/annotations.json path/to/output.mp4

```

- **Example**:

```python

python data_visualization.py ./data/gtea/videos/S1_Cheese_C1.mp4 ./data/annotations.json output/S1_Cheese_C1.mp4

```

## **Action Visualization**

- **Description**:

This script displays the actions occuring at different frames in the video given.

- **How to Run**:

Use the following commands to run the 'data_visualization.py' file

```python

python action_visualization.py /path/to/videos/directory path/to/frame_labels/directory path/to/output/directory

```

- **Example**:

```python

python action_visualization.py ./data/gtea/videos ./data/gtea/frame_labels ./output


```

## **XML Visualization**

- **Description**:

This script processes an XML file containing annotations and visualizes them on the corresponding images, saving the result as a video.

- **How to Run**:

Use the following commands to run the 'xml_vis.py' file
