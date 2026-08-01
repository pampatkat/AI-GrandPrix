<p align="center">
  <a href="" rel="noopener">
 <img src="https://staticr1.blastingcdncf.com/media/photogallery/2026/1/28/660x290/b_1200x675x95/anduril-launches-the-ai-grand-prix-a-global-autonomous-drone-race-c-creative-commons_3182713.webp" alt="Project logo"></a>
</p>
<h1 align="center">🤖 Anduril AI GrandPrix Attempt</h1>

<div align="center">

  [![Hackathon](https://img.shields.io/badge/hackathon-AIGPX_2026-orange.svg)](https://www.theaigrandprix.com/)

</div>

This repository is the home for the code written by Oregon State University engineering students [Pam](https://github.com/pampatkat), [Allie](https://github.com/quetzlcoatlus) and [Chris](https://github.com/chrisbuild124) for the Anduril AI Grand Prix 2026. It is largely exploratory as we didn't pass the first qualifier. However, it was an enjoyable learning experience working together and trying autonomous drone software development. We learned some facets of machine learning, computer vision, networking, drone physics and parallel programming.

## 🛠 Problem Statement

Program a drone to fly through a set of gates while managing noise and obstacle avoidance. No manual intervention; only the drone interface and a software stack.

## 🤔 What we tried 

- Experimenting with AI tooling (i.e. which models did best) for writing the code
- YOLOv8 for object detection trained by Pam
- Different static computer vision strategies (see docs)
    - Morphological image opening and closing
    - Color filters, HSV color segmentation
    - PnP
    - Confidence threshold for approxPolyDP and minAreaRect
    - Quad extract corners
- Manual navigation for object detection testing


### 🌱 What was left to do

We figured out how to do relatively accurate object detection for the gates, but we weren't able to finish the qualifier in time because we didn't have an accurate path finding strategy given the positions of the objects.

## 🤝 How we coordinated

We had weekly meetings throughout the duration of the competition where we did:
- Code reviews
- Concept discussions
- Research presentations
- Collaborative coding
- Design documents (see docs)

## ⚙ Technology Stack

- Python
- YOLOv8
- MAVLink

## 🛟 Acknowledgements

Links to the resources we used in development:
- https://www.cs.cmu.edu/news/2020/cmu-team-trains-autonomous-drones-using-cross-modal-simulated-data
- https://duckduckgo.com/?q=ORB-SLAM3&ia=videos&iax=videos&iai=https%3A%2F%2Fwww.youtube.com%2Fwatch%3Fv%3DUOmfbaVbok0
- https://duckduckgo.com/?q=drone+computer+vision+image+stream+obstacle+detection&ia=web
- https://www.sciencedirect.com/science/article/pii/S2590005624000274
- https://codezup.com/computer-vision-autonomous-drones/
- https://www.mdpi.com/2504-446X/7/2/89
- https://www.ultralytics.com/blog/computer-vision-applications-ai-drone-uav-operations#seeing-the-bigger-picture-vision-ais-impact-on-drones
- https://docs.ultralytics.com/models/yolo11#citations-and-acknowledgments
- https://www.cs.cmu.edu/news/2020/cmu-team-trains-autonomous-drones-using-cross-modal-simulated-data
- https://duckduckgo.com/?q=imu+data+definition&ia=web
- https://www.youtube.com/watch?v=vDOkUHNdmKs
- https://lemlib.readthedocs.io/en/stable/about.html
- https://duckduckgo.com/?q=path+finding+algorithms+in+SLAM&ia=web
- https://path.jerryio.com/
- https://docs.path.jerryio.com/docs/getting-started
- https://en.wikipedia.org/wiki/Bézier_curve
- https://towardsdatascience.com/train-mask-rcnn-net-for-object-detection-in-60-lines-of-code-9b6bbff292c3/
- https://www.youtube.com/watch?v=vDOkUHNdmKs