# Image Processing Ex1

In this assigment I will going to do this fucntions:
1. read and display image 
2. convernt an image into 2 color spaces - RGB and YIQ
3. perform Histogram Equalization on images
4. perform Image Quantization
5. perform Gamma correction on Images.


# Basic read and display
* imReadAndConvert() :
    * param: img and represetation
    * retrun: the img as np.ndarray
* imDisplay():
    * param: img path and represetation
    * return: the img in the requested representation    

# Convert Color spaces
* transformRGB2YIQ():
    * param: imgRGB as np.ndarray
    * return: imgYIQ as np.ndarray

* transformYIQ2RGB():
    * param: imgYIQ as np.ndarray
    * return: imgRGB as np.ndarray

 ![Screenshot from 2023-04-17 08-43-24](https://user-images.githubusercontent.com/100192040/232402037-bb735355-3207-43aa-86ac-a1cf9824c8e4.png)
 ![Screenshot from 2023-04-17 10-28-21](https://user-images.githubusercontent.com/100192040/232415260-44ddfd78-1368-4274-bec8-4b394d6de46a.png)



# Histogram Equalization
* hsitogramEqualize():
   * param - imgOrig -> np.ndarray
   * return - tuple:
        * imgEq -> np.ndarray -> new equalized image.
        * histOrg -> np.ndarray -> histogram of the original image.
       * histEQ -> np.ndarray -> histogram of the equalized image.

  ![Screenshot from 2023-04-17 08-44-53](https://user-images.githubusercontent.com/100192040/232402087-b4bb5207-6d14-4650-b338-d3fbb8ccdc3c.png)
  
![Screenshot from 2023-04-17 08-46-47](https://user-images.githubusercontent.com/100192040/232402121-747969d1-7b74-433a-b71e-2ecf77af11f4.png)

  
#  Image Quantization
   * quantizeImage():
        * param:  
            1. imgOrig -> np.ndarray 
            2.  nQuant -> int -> amount of intensities to quantize.
            3. nIter -> int -> amount of iterations to perform.
        * return: tuple-
             1.Images_list -> List[np.ndarray] -> list of the images after each iteration 
             2.MSE_list -> List[float] -> list of the MSE after each iteration
![Uploading Screenshot from 2023-04-17 10-08-10.png…]()
![Screenshot from 2023-04-17 10-08-31](https://user-images.githubusercontent.com/100192040/232410543-fb2676e4-d5b1-44fe-86ea-d48b4bed2ac6.png)
![Screenshot from 2023-04-17 10-08-10](https://user-images.githubusercontent.com/100192040/232410526-a26a878a-93d1-47c3-8471-0557d01c627e.png)


# Gamma correction
    * gammaDisplay():
        * param: img_path 
        * return: rep as int - 1 for gray 2 for RGB
![image](https://user-images.githubusercontent.com/100192040/232410852-62371761-1a81-495e-90ac-2a0ae22ff254.png)
