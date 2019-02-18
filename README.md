# CT_Kernel_Conversion
<br/>
Source codes for performing image-based kernel conversion.<br/>
Data preparation, learning, and conversion are all coded in python. The NVCaffe library and DIGITS, which provides UI for the learning process, were used for the learning. These are provided by nvidia. <br/>
<br/>

## Environments
<br/>
ubuntu 14.04 LTS, python 2.7, NVCaffe 0.15.14, cuda 8.0, DIGITS 5.1, <br/>
<br/>
<br/>

## Step by Step for learning
<br/>

### 1. Create a database(LMDB) for learning
<br/>

Use the source of *converter_to_lmdb.py* to create an LMDB for image and label.<br/>
<br/>

### 2. Upload LMDB to DIGITS

<br/>
<img src="https://user-images.githubusercontent.com/17020746/52621696-5208a000-2eeb-11e9-9670-468176a36610.png">
<br/>

Upload image LMDB, label LMDB, and mean image created in **step1**.<br/>
<br/>
<img width="400" src="https://user-images.githubusercontent.com/17020746/52622299-d90a4800-2eec-11e9-84f8-d2c47d1aebe2.png"><br/>
<br/>

### 3. Learning with DIGITS
<br/>
<img src="https://user-images.githubusercontent.com/17020746/52621765-719fc880-2eeb-11e9-91ef-d2a6019c2220.png">
<br/>
See the example below for the solver options. The batch size must be a multiple of the number of GPUs within the range of not exceeding memory. When learning one 512x512 image, 1.68GB of memory was occupied.<br/>
<br/>
<center><img src="https://user-images.githubusercontent.com/17020746/52621868-b4fa3700-2eeb-11e9-84fe-1a32e7417b69.png" width="250"></center>      
<br/>

Enter the contents of *model.prototxt* on your custom network and start learning.<br/>
<br/>
<img src="https://user-images.githubusercontent.com/17020746/52622561-72d1f500-2eed-11e9-9f29-dfc2380ba4f1.png" width="500"><br/>
<br/>
<br/>

## Kernel conversion
<br/>

Convert the image using *generator.ipynb*<br/>
The following images are examples of B10f to B70f conversion in test1. From left to right are the input, ground-truth, and converted images.<br/>
<br/>
<p>
<img src="https://user-images.githubusercontent.com/17020746/52680913-7e6cfc80-2f7d-11e9-89f4-ea369d1c48f7.png" width="30%">    <img src="https://user-images.githubusercontent.com/17020746/52680965-9cd2f800-2f7d-11e9-9198-16764f56ecdb.png" width="30%">    <img src="https://user-images.githubusercontent.com/17020746/52681033-d4da3b00-2f7d-11e9-97ef-1721bca09e6c.png" width="30%">
</p>
<br/>

There is a learned weight of the conversion from B10f to B70f in the *weight* directory,<br/>
In the *data/noncontrast* directory, there are test2 example images in addition to test1. You can convert test2 directly.<br/>
<br/>
