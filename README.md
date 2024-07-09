# Celebrity_Face_Recognition

This project involves training a machine learning model to recognize images of five celebrities: Madonna, Mindy Kaling, Jerry Seinfeld, Elton John, and Ben Affleck. The model is trained on a dataset containing images of these celebrities and is capable of predicting the celebrity in new images with a reasonable degree of accuracy.

## Dataset Preparation
Download images from https://www.kaggle.com/datasets/dansbecker/5-celebrity-faces-dataset.
## Project Structure

- `train`: Directory containing training images of the celebrities.
- `val`: Directory containing validation images of the celebrities.
- `celebrity_recognition_model.h5`: The trained Keras model.
- `label_encoder.pkl`: The label encoder used for transforming class labels.

## Dependencies

The project requires the following Python packages:

- `tensorflow`
- `numpy`
- `opencv-python`
- `matplotlib`
- `scikit-learn`

These can be installed using the provided `requirements.txt` file.

## Usage

### Training the Model

1. Place the training images in the `train` directory with each celebrity's images in their respective folders.
2. Preprocess the images using data generators and train a convolutional neural network (CNN) model.
3. Save the trained model and the label encoder for future use.

### Predicting on Validation Data

1. Load the trained model and label encoder.
2. Use the validation data to test the model's performance.
3. Display the predictions for the validation images, showing the predicted celebrity names.

## Example Workflow

1. **Prepare the Dataset**: Organize the images into `train` and `val` directories with subdirectories for each celebrity.
2. **Train the Model**: Use a CNN to learn features from the training images and save the model.
3. **Validate the Model**: Load the model and evaluate it on the validation set to check its performance.
4. **Display Predictions**: Visualize the predictions by displaying validation images with the predicted celebrity labels.

## Requirements

- TensorFlow
- NumPy
- OpenCV
- Matplotlib
- Scikit-learn

## Results
![Screenshot 2024-07-04 092811](https://github.com/UppuluriKalyani/Celebrity_Face_Recognition/assets/105410881/bc82af27-1dd3-4485-8a6a-e582236414b3)
![Screenshot 2024-07-04 092713](https://github.com/UppuluriKalyani/Celebrity_Face_Recognition/assets/105410881/0874f96a-e370-4a32-b0eb-8708f3678358)
![Screenshot 2024-07-04 092752](https://github.com/UppuluriKalyani/Celebrity_Face_Recognition/assets/105410881/33743cb9-ca2a-41f4-bbff-a4db86447f6f)



## License

This project is licensed under the MIT License.
