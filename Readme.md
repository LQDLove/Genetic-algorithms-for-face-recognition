# Genetic algorithms for face recognition

Genetic algorithms for face recognition adjusting faces with ellipses. 

## Requirements
You need the following requirements in order to run the project:
- Python 2.7
- PIL
- CV

## Algorithm

Previous start of genetic algorithms, the following Image Treatment was applied:
1. Convert RGB Image to grayscale
2. Change the image size to 256x256 
3. Image noise reduction
4. Apply Canny algorithm 

Then the genetic algorithm was applied adjusting faces with ellipses. (Details at Reporte.pdf)


## Sample output

With 15 generations and 100 individuals, I get the following result:

![alt tag](https://github.com/cgcastro/Genetic-algorithms-for-face-recognition/blob/master/face.PNG)