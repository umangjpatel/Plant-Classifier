<h1>Plant-Classifier</h1>
An app that classifies a plant image using a Convolutional Neural Network trained with the help of Keras and deployed using TensorFlow Lite

<h2>Table of Contents</h2>
<ol>
  <li><a href="#abstract">Abstract</a></li>
  <li><a href="#project_overview">Project Overview</a></li>
  <li><a href="#scope">Scope</a></li>
  <li><a href="#user_characs">User Characteristics</a></li>
  <li><a href="#tools_n_tech">Tools and Technology</a></li>
  <li><a href="#sys_design">System Design</a></li>
  <li><a href="#imple">Implementation</a></li>
  <li><a href="#screens">App Recordings</a></li>
  <li><a href="#constraints">Constraints and Future Enhancements</a></li>
  <li><a href="#refs">References</a></li>
</ol>

<h2 id="abstract">Abstract</h2>
<p>The primary purpose of doing the project was to determine the species of a plant seedling from an image of the plant seedling using Deep Learning. The need for this project is related to improving farming techniques leading to better crop yields as well as better stewardship of the environment. Computer vision has grew recently in the past few years with the advent of deep learning algorithms. Convolutional Neural Networks have become the state-of-the-art algorithm for image detection and classification tasks. With the help of popular modern deep learning framework Keras, the process of building and deploying a DL model has become easy. As such, more and more focus can be made on solving the actual problem instead of dealing with the intricacies of working of our model.</p>
<p>The project involved preprocessing of plant images, masking those images, building the CNN model, validating the model, testing the model, and finally deploying the CNN model to an optimised version that can be used in an Android app. The key challenge in the project was preprocessing and masking the images. These images consisted of unwanted information in the background which was being unnecessarily computed while training the model. With the help of OpenCV library, we could cut out the sections of image containing the plant and darken the background of the image. These images were then directly fed to the CNN model, leading to greater accuracy in the results.</p>

<h2 id="project_overview">Project Overview</h2>
<p>The main objective of this project is to classify a plant seedling from weed using Convolutional Neural Networks. The ability to classify it effectively can mean better crop yields and better stewardship of the environment.</p>
<p>The images required for this project were obtained from the website of Aarhus University where they released a dataset containing images of approximately 960 unique plants belonging to 12 species at several growth stages.</p>
<p>The project involved preprocessing the images, building the model, tuning the model and deploying the trained optimised model to an Android app. The end product of this project is an Android app where the user provides the image of the plant and with the click of the button, the app feeds the image to the trained model and gives a result class displayed on the screen.</p>

<h2 id="scope">Scope</h2>
<p>It is a software system, which provide the functionality of clarifying the plant images to categories helping the farmers improve crop yields using just a smartphone and no advanced technological equipment. The android app takes in the plant images, and with a click of a button, the app feeds this image to the trained model loaded in the app, and displays the class to which the plant belongs to.</p>

<h2 id="user_characs">User Characteristics</h2>
<p>This system is divided in three parts. First one is the preprocessing part where the images are first preprocessed and masked according to the input requirements for the CNN model. Second one is the CNN model where we have implemented a custom CNN model from scratch. Building our model was an iterative process where we were required to take an idea, code it and experiment various parameters so that our model could get good accurate result. Third, is the android app through which the users can be able to fetch an image and feed it to the optimised model with just a click of the button without requiring an internet connection.</p>

<h2 id="tools_n_tech">Tools and Technology Used</h2>

<h3>Google Colab</h3>
<img src="/images/colab_logo.png" />
<p>Colaboratory is a free Jupyter notebook environment that requires no setup and runs entirely in the cloud. With Colaboratory you can write and execute code, save and share your analyses, and access powerful computing resources, all for free from your browser.</p>
<p>Colaboratory works with all the major browsers and is free to use. All Colaboratory notebooks are stored in Google Drive. Colaboratory notebooks can be shared just as you would do with Google Docs or Sheets with just clicking the Share button.</p>
<p>Colaboratory supports Python 2.7 and Python 3.6 and has all the popular machine learning and deep learning frameworks and packages already imported in the Colaboratory notebooks.</p>
<p>Code is executed in a virtual machine dedicated to your Google account. Virtual machines are recycled when idle for a while, have a maximum lifetime enforced by the system.</p>
<p>It is intended for interactive-use. Long-running background computations, particularly on GPUs, may be stopped.</p>

<img src="/images/colab_env.png" />

<h3>TensorFlow</h3>
<img src="/images/tf_logo.png" />
<p>TensorFlow is an end-to-end open source machine learning platform. It has a comprehensible, flexible ecosystem of tools, libraries and community resources that lets researchers push the state-of-the-art in ML and developers easily build and deploy ML powered applications.</p>
<p>Why TensorFlow :</p>
<ol>
<li><b>Easy Model Building</b><br>
TensorFlow offers multiple levels of abstraction so you can choose the right one for your needs. Build and train models by using the high-level Keras API, which makes getting started with Tensorflow and machine learning easy.</li>
<li><b>Robust ML production anywhere</b><br>
TensorFlow has always provided a direct path to production. Whether it’s on servers, edge devices, web or mobile, TensorFlow lets you train and deploy your model easily, no matter what language or platform you use.
For running the inference on mobile and edge devices, we have used TensorFlow Lite.</li>
<li><b>Powerful experimentation for research</b><br>
Build and train state-of-the-art models without sacrificing speed or performance. TensorFlow gives you the flexibility and control with features like the Keras Functional API and Model Subclassing API for creating of complex topologies. Eager execution is used for easy prototyping and fast debugging.</li>
</ol>

<h3>Keras</h3>
<p>Keras is a high-level neural networks API, written in Python and capable of running on top of TensorFlow, CNTK or Theano. It was developed with a focus on enabling fast experimentation. Being able to go from idea to result with the least possible delay is the motto of Keras.</p>
<p>Keras allows for easy and fast prototyping (through user friendliness, modularity and extensibility). It supports both convolutional neural networks and recurrent networks, as well as combinations of the two. It seamlessly runs on CPU and GPU architectures.</p>

<p>Guiding principles of Keras :-</p>
<ol>
<li><b>User Friendliness</b><br>
Keras is an API designed for human beings, not machines. It puts user experience front and center. Keras follows best practices for reducing cognitive load; it offers consistent and simple APIs, it minimises the number of user actions required for common use cases and it provided clear and actionable feedback upon user error.</li>
<li><b>Modularity</b><br>
A model is understood as a sequence or a graph of standalone, fully configurable modules that can be plugged together with as few restrictions as possible. In particular, neural layers, cost functions, optimizers, initialization schemes, activation functions and regularization schemes are all standalone modules that you can combine to create new models.</li>
</ol>

<h2 id="sys_design">System Design</h2>
<h3>Flow of system :-</h3>
<img src="/images/cnn_diagram.png" />

<p>The above figure represents the architecture of the convolutional neural network model.</p>
<p>The images present in the dataset are first loaded into the memory, fed to the CNN model with the appropriate input sizes, and then the model computes and processes the information present in the image and finally gives the output i.e which class the image belongs.</p>

<img src="/images/tflite.png" />
<p>The above figure represents the architecture of the deployment of the trained CNN model to the Android app using TensorFlow Lite.</p>
<p>After training the CNN model, we first save the model in .h5 format where the trained weights are saved or freezes with the model. To convert the TensorFlow model to TensorFlow Lite model, we use the TocoConverter tool which converts our model to the .tflite format and this model can then be bundled with our Android app. This model is optimised to work on mobile devices and can perform tasks without requiring an internet connection.</p>

<h3>Working flow :-</h3>
<img src="/images/working_flow.png" />
<p>The user first opens the app, chooses a plant image, clicks a button. After clicking the button, the image is fed to the TensorFlow Lite model already bundled with the app and generates a class label. This class label is simply displayed on the device’s screen.</p>


<h2 id="imple">Implementation</h2>
<p>To create the best possible classifier, we planned out the development process into three parts.</p>
<ol>
<li>Preprocessing the images</li>
<li>Building the model.</li>
<li>Building the app.</li>
</ol>

<h3>Preprocessing the image</h3>
<p>The dataset provided consists of a training set and testing set of images of plant seedlings at various growth stages. Each image has a filename that acts as its own unique ID. The dataset comprises of images of 12 plant species i.e Black-grass, Charlock, Cleavers, Common Chickweed, Common wheat, Fat Hen, Loose Silky-bent, Maize, Scentless Mayweed, Shepherds Purse, Small-flowered Cranesbill and Sugar beet.</p>

<img src="/images/dataset_directory.png" />

<p>Firstly, the images are stored in a directory structure where the directory name itself acts as the class label for a group of training images</p>

<img src="/images/training_set_directory.png" />

<p>This is how a training image looks like (below)</p>

<img src="/images/image_train.png" />

<p>Using Glob library, we read these images and allot these images to the class label (directory name) they belong. This can be done by fetching the directory name from the current directory name while reading the images.</p>

<img src="/images/image_props.png" />

<p>The images provided in the training set are of different dimensions as shown above. i.e one image is of the shape 1900 x 1900 whereas another image is of the shape 352 x 352.</p>
<p>To make the images of consistent dimensions, we resized them to 70 x 70 pixels using the OpenCV library. So we read the image from the directory, resize them to 70 x 70 pixels and put the resulting image in an ordinary Python list. This allows us not to make any changes to the original images present in the dataset.</p>

<p>Moreover, we are required to clean images as the images contained many unwanted information in the background such as barcodes, soil, etc. This hinders the plant in the image (green in colour).</p>

<img src="/images/barcode_bg.png" />

<p>Also, some plant seedlings in the images are too thin that they camouflage with the soil background (brown in colour) as shown below :</p>

<img src="/images/slim_plant.png" />

<p>To solve the mentioned issues, we use a mask that fetches the greenish parts of the plant seedling in the image and blackens out the unwanted background portions.</p>
<p>For a particular image, we convert the RGB / BRG image into HSV format as the OpenCV library performs better with images in HSV format. Then we blur the image to remove any noise present in it and finally, we create the mask removing the background from it.</p>

```python
blurr = cv2.GaussianBlur(i,(5,5),0);
hsv = cv2.cvtColor(blurr,cv2.COLOR_BGR2HSV)
#GREEN PARAMETERS
lower = (25,40,50)
upper = (75,255,255)
mask = cv2.inRange(hsv,lower,upper)
struc = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(11,11))
mask = cv2.morphologyEx(mask,cv2.MORPH_CLOSE,struc)
boolean = mask > 0
new = np.zeros_like(i,np.uint8)
new[boolean] = i[boolean]
new_train.append(new)
```

<img src="/images/preprocessing.png" />

<p>After properly processing the images, we get the new training images. These images are used to feed to the CNN model.</p>

<img src="/images/post_process.png" />

<h3>Building the CNN model</h3>
<p>The Convolutional Neural Network models are designed to map image data to an output variable. The benefit of using CNNs is their ability to develop an internal representation of a two-dimensional image. This allows the model to learn position and scale in variant structures in the data, which is important when working with images.</p>
<p>They have proven so effective that they are the go-to method for any type of classification problem involving image data as an input.</p>

<p>The CNN model we have built has the following characteristics :-</p>
<ul>
<li>6 Conv2D layers</li>
<li>6 MaxPool layers</li>
<li>2 Dense layers</li>
</ul>

<p>Using Keras’ high-level APIs and TensorFlow as the computation backend infrastructure, we could easily build a CNN model from scratch with a few lines of code.</p>

```python
model = Sequential()
model.add(Conv2D(...), activation='relu'))
model.add(BatchNormalization(axis=3))
model.add(Conv2D(...), activation='relu'))
model.add(MaxPooling2D(...))

...
```

<p>After compiling the model, we trained our model with 50 epochs / iterations. As we are working with gigabytes of data, deep learning models takes a large amount of time training on CPUs. Therefore, GPUs are used to speed up the process of training. Using free GPU compute resources like Google Colab and Kaggle Kernels, we were able to iterate and train various architectures of our model.</p>
<p>Now, after succesfully training and evaluating the model, we were required to convert the model suitable to make inferences on mobile devices. Keras provides an easy-to-use API to convert any tf.Keras model to .tflite model. This model then can be bundled in an Android app.</p>

```python
# Save tf.keras model in HDF5 format.
keras_file = "keras_model.h5"
model.save(keras_file)

# Convert to TensorFlow Lite model.
converter = tf.lite.TFLiteConverter.from_keras_model_file(keras_file)
tflite_model = converter.convert()
open("converted_model.tflite", "wb").write(tflite_model)
```
<p>Above lines of code are all it takes to convert the tf.keras model to a .tflite model. This model then can be used directly in an Android app.</p>

<h3>Building the app</h3>
<p>To build the app, we first ensure all the Android related dependencies are imported and working.</p>
<p>We first save the .tflite model in ‘assets’ directory in the project structure. Then we have to include the following dependency :</p>

```gradle
  implementation 'org.tensorflow:tensorflow-lite:1.13.1'
```

<p>This dependency allows us to utilise TensorFlow APIs with the Java language and therefore utilise the .tflite model bundled in the app.</p>
<p>We first setup the layout files and listeners on the widgets utilised in the user interface of the app.</p>
<p>When an image is loaded in the app, it first needs to be pre-processed before feeding it to the model. In Android ecosystem, we take the camera bitmap image and convert it to a Bytebuffer format for efficient processing. We pre-allocate the memory for ByteBuffer object based on the image dimensions because Bytebuffer objects can't infer the object shape.</p>
<p>Since this model is quantized 8-bit, we will put a single byte for each channel. The ByteBuffer object will contain an encoded Color value for each pixel in ARGB format, so we need to mask the least significant 8 bits to get blue, and next 8 bits to get green and next 8 bits to get blue, and we have an opaque image so alpha can be ignored.</p>
<p>After some preprocessing, we need to make an inference. To do so, we need to create the interpreter. To create the interpreter, we need to load the model file. In Android devices, we recommend pre-loading and memory mapping the model file to offer faster load times and reduce the dirty pages in memory. If your model file is compressed, then you will have to load the model as a file, as it cannot be directly mapped and used from memory.</p>

```java
// Memory-map the model file
AssetFileDescriptor fileDescriptor = assets.openFd(modelFilename);
FileInputStream inputStream = new FileInputStream(fileDescriptor.getFileDescriptor());
FileChannel fileChannel = inputStream.getChannel(); long startOffset = fileDescriptor.getStartOffset();
long declaredLength = fileDescriptor.getDeclaredLength();
return fileChannel.map(FileChannel.MapMode.READ_ONLY, startOffset, declaredLength);
```

<p>After loading the model, we need to provide the input shape for the input values and finally, pass it to the model. The model generates the output which needs to be processed in order to display relevant class labels on the screen.</p>

<h2>Notebook Link</h2>
<a href="https://www.kaggle.com/umangjpatel/plant-cnn" target="_blank">Plant CNN Jupyter Notebook</a>

<h2 id="screens">App Recordings</h2>

<video width="320" autoplay>
    <source src="/app_recordings/app_recording.webm"
            type="video/webm">
  Video not available...
</video>
<br><br>
Screenshots
<br>
<img width="250" src="/app_recordings/splash_screen.png" />
<img width="250" src="/app_recordings/first_screen.png" />
<img width="250" src="/app_recordings/first_class.png" />
<img width="250" src="/app_recordings/second_class.png" />
<img width="250" src="/app_recordings/third_class.png" />


<h2 id="constraints">Constraints and Future Enhancements</h2>
<p>The limitation of the current working of the app is that the images are required to be pre-processed and masked before feeding it to the CNN TensorFlow Lite model. To overcome this situation, we can upload the image to a web app which can do the required pre-processing and return the resultant image back to the device.</p>
<p>Moreover, the model can be hosted on Firebase MLKit which can improve the developers’ effort as bundling the model to the app seems a daunting task every time the .tflite model is updated. MLKit can help to update the model without requiring the users to update the app itself.</p>

<h2 id="refs">References</h2>
<ul>
  <li><a href="https://towardsdatascience.com/building-a-convolutional-neural-network-cnn-in-keras-329fbbadc5f5">Towards Data Science Blog</a></li>
  <li><a href="https://machinelearningmastery.com/blog/">Machine Learning Mastery Blog</a></li>
  <li><a href="https://keras.io">Keras Documentation</a></li>
  <li><a href="https://www.tensorflow.org/lite/">TensorFlowLite documentation</a></li>
</ul>
