# The PERDAW Dataset v.1.0
![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54) ![scikit-learn](https://img.shields.io/badge/scikit--learn-%23F7931E.svg?style=for-the-badge&logo=scikit-learn&logoColor=white) ![Keras](https://img.shields.io/badge/Keras-%23D00000.svg?style=for-the-badge&logo=Keras&logoColor=white) ![TensorFlow](https://img.shields.io/badge/TensorFlow-%23FF6F00.svg?style=for-the-badge&logo=TensorFlow&logoColor=white) ![Google Cloud](https://img.shields.io/badge/GoogleCloud-%234285F4.svg?style=for-the-badge&logo=google-cloud&logoColor=white) ![Jupyter Notebook](https://img.shields.io/badge/jupyter-%23FA0F00.svg?style=for-the-badge&logo=jupyter&logoColor=white)

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.10823907.svg)](https://doi.org/10.5281/zenodo.10823907) 


The Perdaw Dataset, short for "Plan, and Elevation Representations of Doors And Windows," is a comprehensive dataset designed to facilitate research and development in the field of architectural element recognition and classification. This dataset comprises plan and elevation views of various doors and windows, providing rich and detailed representations suitable for machine learning tasks, particularly in the domain of computer vision.

The original inspiration for the doors and windows comes from a widespread building typology in the former USSR and its variations in former socialist countries, sometimes called 'Khrushchyovkas'. More information on the building typology and its history can be found [here](https://en.wikipedia.org/wiki/Khrushchevka).

#### TLDR:

* **Total Images:** 28.800
* **Object Types:** Doors and Windows
* **Representation Views:** Plan and Elevation
* **Number of Classes:** 40
* **Number of images per Classes:** 720
* **Object Variation:** Different sizes and features including glass panels and corner window variations
* **Data Augmentation:** Rotating, mirroring, splitting, resizing and shearing
* **Image Format:** JPG
* **Image Size:** 224x224 pixels 
* **Link to Dataset:** [PERDAW (Plan, and Elevation Representations of Doors And Windows)](https://zenodo.org/records/10823907) 

#### Table of Contents:
##### [Dataset Content and Structure](#dataset-content-and-structure)
##### [Development Process](#development-process)
##### [Publications](#publications)
##### [License](#license)
##### [Aknowledgements](#daknowledgements)
##### [Contact](#contact)
##### [Previous Work](#previous-work)
-----------

## Dataset Content and Structure

The dataset comprises four zip files designed to offer users access to various components, including the dataset content, different partitions, the original 3D models utilized for image extraction and the initial extracted images.

### original_3d_objects_revit.zip
This file contains the 20 3D models in a **.rtv** format, compatible with Autodesk's AEC software suite, such as Revit. 

Below, a more comprehensive description of each building component (object) can be found:

#### The Doors
| Name   | Size          | Location | Model          | Height | Length | Placement        |
|--------|---------------|----------|----------------|--------|--------|------------------|
| Door 0 | Double-door   | Exterior | With Glass     | 2100   | 2100   | Building Entrance |
| Door 1 | Double-door   | Exterior | Without Glass  | 2100   | 2100   | Building Entrance |
| Door 2 | One-and-a-half| Exterior | With Glass     | 2100   | 1500   | Building Entrance |
| Door 3 | One-and-a-half| Exterior | Without Glass  | 2100   | 1500   | Building Entrance |
| Door 4 | Single-door   | Exterior | With Glass     | 2100   | 700    | Balcony|
| Door 5 | Single-door   | Exterior | Without Glass  | 2100   | 700    | Roof & Basement |
| Door 6 | Single-door   | Interior | Without Glass  | 2100   | 750    |Apt. Entrance|
| Door 7 | Single-door   | Interior | With Glass     | 2100   | 700    |Living R. & Kitchen |
| Door 8 | Double-door   | Interior | Without Glass  | 2100   | 700    |Bedroom|
| Door 9 | Single-door   | Interior | Without Glass  | 2100   | 650    |Toilet|

#### The Windows
| Name    | Size    | Model                 | Type    | Height | Length | Location         |
|---------|---------|-----------------------|---------|--------|--------|------------------|
| Window 0| Single  | Without corner window | Hinged  | 600   | 600   | Toilet  |
| Window 1| Double  | Without corner window | Hinged  | 1500   | 1400   | Kitchen |
| Window 2| Double  | With corner window    | Hinged  | 1500   | 1400   | Kitchen |
| Window 3| Double  | Without corner window | Hinged  | 1500   | 1500   | Living & Bedroom |
| Window 4| Double  | With corner window    | Hinged | 1500    | 1500    |Living & Bedroom |
| Window 5| Triple  | Without corner window| Hinged   | 3000   | 2073   | Living & Bedroom|
| Window 6| Triple  | With corner window   | Hinged   | 3000   | 2073   |Living & Bedroom|
| Window 7| Single  | Without corner window| Awned   | 750    | 1500   | Staircase        |
| Window 8| Single  | Without corner window| Awned   | 2073   | 3000  | Staircase        |
| Window 9| Single  | With corner window   | Awned   | 2073   | 3000  | Staircase        |

### initial_2d_images_of_objects.zip

This file contains 2D images extracted from the 3D Revit files. 

Each 'Elevation View' folder of every object contains 12 images. Among these, six depict the object with thick-lined contours from frontal, rear, and four 45-degree angled perspectives, while the other six feature thinned-line contours. The examples below present 'Door 0' and its 2D representations:

#### Elevation Representations of 'Door 0':
##### Thick-lined (From front to back)

![Front View](/images/original_2d/door_0_elev/door_0_double_exterior_windowed_hinged_thick_front_2100_2100_elev_224x224.jpg) ![45-1](/images/original_2d/door_0_elev/door_0_double_exterior_windowed_hinged_thick_451_2100_2100_elev_224x224.jpg) ![45-2](/images/original_2d/door_0_elev/door_0_double_exterior_windowed_hinged_thick_452_2100_2100_elev_224x224.jpg) 

![Back View](/images/original_2d/door_0_elev/door_0_double_exterior_windowed_hinged_thick_back_2100_2100_elev_224x224.jpg) ![45-3](/images/original_2d/door_0_elev/door_0_double_exterior_windowed_hinged_thick_453_2100_2100_elev_224x224.jpg) ![45-4](/images/original_2d/door_0_elev/door_0_double_exterior_windowed_hinged_thick_454_2100_2100_elev_224x224.jpg)

##### Thin-lined
![Front View](/images/original_2d/door_0_elev/door_0_double_exterior_windowed_hinged_thin_front_2100_2100_elev_224x224.jpg) ![45-1](/images/original_2d/door_0_elev/door_0_double_exterior_windowed_hinged_thin_451_2100_2100_elev_224x224.jpg) ![45-2](/images/original_2d/door_0_elev/door_0_double_exterior_windowed_hinged_thin_452_2100_2100_elev_224x224.jpg) 

![Back View](/images/original_2d/door_0_elev/door_0_double_exterior_windowed_hinged_thin_back_2100_2100_elev_224x224.jpg) ![45-3](/images/original_2d/door_0_elev/door_0_double_exterior_windowed_hinged_thin_453_2100_2100_elev_224x224.jpg) ![45-4](/images/original_2d/door_0_elev/door_0_double_exterior_windowed_hinged_thin_454_2100_2100_elev_224x224.jpg)

#### Plan Representations of 'Door 0' (Thick and Thin-lined):
![Plan-Thick](/images/original_2d/door_0_plan/door_0_double_exterior_windowed_hinged_thick_2100_2100_plan_224x224.jpg) ![Plan-Thin](/images/original_2d/door_0_plan/door_0_double_exterior_windowed_hinged_thin_2100_2100_plan_224x224.jpg)

### perdaw_object_view_split.zip
This file contains the entire dataset which is grouped based on the 20 building conmponents (objects), and then their corresponding 2D views. Each folder contains 720 images. Below, an extraction of the folder structure can be seen:

```
/perdaw_v3_224x224px
    /door_0_double_exterior_windowed_hinged_2100_2100
        /door_0_double_exterior_windowed_hinged_2100_2100_elev
        /door_0_double_exterior_windowed_hinged_2100_2100_plan
    /door_1_double_exterior_windowless_hinged_2100_2100
        /door_1_double_exterior_windowless_hinged_2100_2100_elev
        /door_1_double_exterior_windowless_hinged_2100_2100_plan
    ...
```

### other_splits.zip
This file contains 12 partitions of the dataset, which were the basis for the "Towards Digitisation of Technical Drawings in Architecture: Evaluation of CNN Classification on the Perdaw Dataset" publication. Each partition has 3 sub-folders: train, test and evaluate with a split of 80/10/10 (%). More information on the paper can be found [here](#publications).

##### Partitions Featuring Both Plan and Elevation Views:
* Perdaw Dataset - 40 classes
* Perdaw Dataset - 20 classes
* Doors Plan and Elevations - 20 Classes
* Windows Plan and Elevations - 20 classes

##### Partitions with Only Elevation Views:
* Only Elevation Views - 20 Classes
* Doors - Elevation View - 10 classes
* Windows - Elevation View - 10 classes

##### Partitions with Only Plan Views:
* Only Plan Views - 20 classes 
* Doors - Plan View - 10 classes 
* Windows - Plan View - 10 classes 

##### Binary Partitions on Plan View Classifications:
* Doors versus Windows - Plan View - 2 classes
* Single versus Double Doors - Plan View - 2 classes

## Development Process

![Dev-Diagram](/images/perdaw_dev_diagram_v2.png)

## Publications
"Towards Digitisation of Technical Drawings in Architecture: Evaluation of CNN Classification on the Perdaw Dataset" is currently under review for the [Engineering Applications of Neural Networks](https://eannconf.org) which compares the performance of ResNet50, MobileNet V2 and Inception V3 on the PERDAW dataset splits (which were described under [Dataset Content and Structure](#dataset-content-and-structure)). The paper was authored by Alexandru Filip and Stella Grasshof. The code used in the development of this paper can be found [here](/code/publication_code/towards-digitisation).
<br>
**Note:** If there are other papers based on this dataset, please feel free to reach out to the authors so your paper will be added to this list.

## License
This repository containing text, images and code, as well as the [dataset](https://zenodo.org/records/10823907), are available under the [MIT License](https://opensource.org/license/mit).

## Contact

#### Alexandru Filip | [LinkedIn](https://www.linkedin.com/in/alfi/) | 

#### Stella Grasshof | [LinkedIn](https://www.linkedin.com/in/stella-gra%C3%9Fhof-15202579/) | 

## Previous Work
The PERDAW dataset comes as the result of more than 1,5 years of research into the field of 3D reconstruction in Architecture and Construction.

###### 3D reconstruction of buildings based on architectural floor plans, using Open Computer Vision and Optical Character Recognition

It started with a 7.5 ECTS point research paper, called "3D reconstruction of buildings based on architectural floor plans, using Open Computer Vision and Optical Character Recognition".

The paper written by Alexandru Filip investigates automated methods for 3D reconstruction from architectural floor plans. It emphasizes the integration of Computer Science with architectural, engineering, and construction (AEC) fields to explore new technologies and their interrelations within 3D reconstruction. 

The research is divided into a literature review to understand current technologies and methodologies, followed by experimentation with Computer Vision and Optical Character Recognition (OCR) libraries to prototype a 3D reconstruction software named ReconstructR. 

The aim is to convert 2D architectural drawings into 3D models efficiently, enhancing the digital transformation in the construction industry. 

The study critically analyzes existing solutions, challenges, and potential future advancements, concluding with the implementation outcomes and suggestions for further research in automating 3D model generation from architectural plans. The paper can be found [here](/previous_work/research_project_report_alfi.pdf).

###### Evaluating Convolutional Neural Network Classification Performance in Building Components from Architectural Drawing

Building on top of this paper, Alexandru wrote his master thesis titled "Evaluating Convolutional Neural Network Classification Performance in Building Components from Architectural Drawings". 

The paper explores the application of machine learning, specifically Convolutional Neural Networks (CNNs), in the classification of building components like doors and windows from architectural drawings. 

The study aims to assess the accuracy and efficiency of five different CNN models in this context, leveraging a dataset comprising images of windows and doors from both plan and elevation views developed by the author. The five models explored are:
* A "Vanilla" CNN
* VGG-16
* ResNet50
* MobileNet V2
* Inception V3

A more rudimentary version of the PERDAW dataset was used as a basis for this paper. The current version of PERDAW was redone from scratch. Therefore, even though there are major similarities between the two datasets, the current version cannot be considered a newer version of the one used in the paper. More information on the initial dataset and results can be found [here](/previous_work/research_project_report_alfi.pdf).

## Aknowledgements
Special thanks to Stella Grasshof, who, even though she wasn't directly involved in the development of the dataset, was a mentor throughout the entire research and development period, as well as for the papers that emerged from this endeavor.

## Resources:
[How to make a readme](https://www.makeareadme.com)
[Badges](https://github.com/Ileriayo/markdown-badges?tab=readme-ov-file#markdown-badges)