# Covid-19 CNN Classification

## Project Description
This project aims to develop a Convolutional Neural Network (CNN) model to classify chest X-ray images into categories such as normal, bacterial pneumonia, and viral pneumonia, including Covid-19 cases. The primary goal is to assist in the rapid and accurate diagnosis of Covid-19 using chest X-ray images.

## Dataset
The dataset used in this project consists of labeled chest X-ray images. The categories include:

- Normal
- Bacterial Pneumonia
- Viral Pneumonia (including Covid-19)

## Data Source
The dataset can be obtained from various medical imaging repositories or online sources. Make sure to download the dataset and organize it into respective folders for training, validation, and testing.
 
## Installation
To run this project, you need to have Python installed along with the following libraries:

- TensorFlow
- Keras
- NumPy
- Pandas
- Matplotlib
- Seaborn
- Scikit-learn
You can install the required packages using the following command:
```
pip install tensorflow keras numpy pandas matplotlib seaborn scikit-learn
```

## Data Visualization
This important step for recogniaze data and understand data architecture.



![download Covid visual](https://github.com/Mahmedorabi/Covid_19_CNN_model/assets/105740465/b44408fa-9353-4ac6-a96d-390c15c1d15e)



## Data Preprocessing
The data preprocessing steps include resizing images, normalizing pixel values, and augmenting the data to improve model generalization.

```
# Define data generators for training and validation
data_generator=ImageDataGenerator(rescale=1/255)

# Load and preprocess the training data
train_data=data_generator.flow_from_directory(train_path,
                                              batch_size=16,
                                              target_size=(224,224))


# Load and preprocess the validation data
test_data=data_generator.flow_from_directory(test_path,
                                            target_size=(224,224),
                                            batch_size=1)
```

## Model Architecture
The CNN model architecture used in this project consists of several convolutional layers, max-pooling layers, and fully connected dense layers. The model is trained using the Adam optimizer and categorical cross-entropy loss function.

### Build Model 
```
model=Sequential()

model.add(Conv2D(filters=32,kernel_size=3,activation='relu',padding='same',input_shape=[224,224,3]))

model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Conv2D(filters=128,kernel_size=3,activation='relu',padding='same'))

model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Flatten())

model.add(Dense(128,activation='relu'))
# model.add(Dense(64,activation='relu'))

model.add(Dense(3,activation='softmax')
```
### Model Compile
```
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])
```



### Model Training
Train the CNN model using the training and validation data.
```
model_hist=model.fit(train_data,validation_data=test_data,epochs=5)
```
## Model Preformance Visualization
```
fig,ax=plt.subplots(1,2,figsize=(15,5))
fig.suptitle("Model preformance Visualization",fontsize=20)
ax[0].plot(model_hist.history['loss'],label='Training Loss')
ax[0].plot(model_hist.history['val_loss'],label='Testing Loss')
ax[0].set_title("Training Loss VS. Testing Loss")

ax[1].plot(model_hist.history['accuracy'],label='Training Accuracy')
ax[1].plot(model_hist.history['val_accuracy'],label='Testing Accuracy')
ax[1].set_title("Training Accuracy VS. Testing Accuracy")
plt.show()
```



![model pre covid](https://github.com/Mahmedorabi/Covid_19_CNN_model/assets/105740465/6d8094a5-a9d1-4d4f-ae25-fabe7a70e785)

## Model Evaluation
- Tranining
```
oos,acc=model.evaluate(train_data)
print(f'Accuracy of training is {acc*100}')

```
### Output 
Accuracy of training is 100.0

- Testing
  ```
  loos,acc=model.evaluate(test_data)
  print(f'Accuracy of testing is {acc*100}')
  ```
  ### Output
  
  Accuracy of testing is 92.42424368858337

## Predict a new Image 
- **1-Give path of testing image**
   - ```
     testing_img="D:/Project AI/Covid19-dataset/test/Viral Pneumonia/0103.jpeg"
     ```
- **2-Give Class map**
  - ```
    class_map=dict([value,key] for key,value in train_data.class_indices.items())
    class_map
    ```
    -**Function return predication image and label**
     - ```
       def predication (testing_img,actual_label):
       test_img=image.load_img(testing_img,target_size=(224,224))
       test_img_arr=image.img_to_array(test_img)/255
       test_img_input=test_img_arr.reshape((1,test_img_arr.shape[0],
                                         test_img_arr.shape[1],
                                         test_img_arr.shape[2]))
    
       # Make predication 
       predicate_class=np.argmax(model.predict(test_img_input))
       predicated_map=class_map[predicate_class]
    
       plt.figure(figsize=(10,5))
       plt.imshow(test_img_arr)
       plt.grid()
       plt.axis('off')
       plt.title(f"Actual Label is: {actual_label} | predict label is: {predicated_map}")

       ```
      
 
 ![predication covid](https://github.com/Mahmedorabi/Covid_19_CNN_model/assets/105740465/1569163b-bed3-457c-a0e5-1099be516436)


   
  











