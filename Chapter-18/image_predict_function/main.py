import functions_framework
from PIL import Image
import base64
import json
from PIL import Image
import numpy as np
from google.cloud import aiplatform
from google.cloud import storage
from io import BytesIO
import os

PROJECT_ID = os.environ.get("PROJECT_ID")
REGION = os.environ.get("REGION")
ENDPOINT_NAME = os.environ.get("ENDPOINT_NAME")
DATA_FOLDER_PATH = os.environ.get("DATA_FOLDER_PATH")
PRED_FOLDER_PATH = os.environ.get("PRED_FOLDER_PATH")

# Create a GCS client
storage_client = storage.Client()

@functions_framework.cloud_event 
def predict(cloud_event):
    try:
        # Extract data from the GCS event
        data = cloud_event.data
        bucket_name = data["bucket"]
        object_name = data["name"]
        image_path = f"gs://{bucket_name}/{object_name}"
        
        if not object_name.startswith(DATA_FOLDER_PATH):
            print(f"Skipping object outside of target folder: {object_name}")
            return "Object not in target folder", 200
        else: 
            # Get a reference to the GCS bucket and blob
            bucket = storage_client.bucket(bucket_name)
            blob = bucket.blob(object_name)

            # Download the image data as bytes
            image_bytes = blob.download_as_bytes()

            # Load the image using Pillow's Image.open
            image = Image.open(BytesIO(image_bytes))

            # Image preprocessing 
            resized_image = image.resize((32, 32), resample=Image.BILINEAR)
            rgb_image = resized_image.convert('RGB')
            image_array = np.array(rgb_image)
            normalized_image = image_array / 255.0

            # Create the request payload
            request_payload = {
                "instances": [normalized_image.tolist()]  # Reshape to [32, 32, 3]
            }

            # Get the endpoint resource name
            mlops_endpoint_list = aiplatform.Endpoint.list(
                filter=f'display_name={ENDPOINT_NAME}', order_by='create_time desc'
            )
            new_mlops_endpoint = mlops_endpoint_list[0]
            endpoint_resource_name = new_mlops_endpoint.resource_name
            print(endpoint_resource_name)

            # Send the inference request using the request payload
            response = aiplatform.Endpoint(endpoint_resource_name).predict(
                instances=request_payload["instances"]
            )

            # Extract predictions and find the highest probability class
            predictions = response.predictions[0]
            class_index = predictions.index(max(predictions))
            class_probability = max(predictions)

            # CIFAR-10 class labels
            class_labels = [
                "airplane",
                "automobile",
                "bird",
                "cat",
                "deer",
                "dog",
                "frog",
                "horse",
                "ship",
                "truck",
            ]
            predicted_label = class_labels[class_index]

            # Format the prediction results in JSON format
            prediction_result = json.dumps({"Predicted class": predicted_label, "probability": class_probability})

            # Upload prediction results to GCS
            result_blob_name = (
                f"{PRED_FOLDER_PATH}/{object_name}-prediction.txt"  # Store results in a subfolder with the image name
            )
            result_blob = bucket.blob(result_blob_name)
            result_blob.upload_from_string(prediction_result)

            print(f"Prediction results uploaded to: gs://{bucket_name}/{result_blob_name}")

            return prediction_result, 200
    except Exception as e:
        print(f"Error: {e}")
        return json.dumps({"error": str(e)}), 500  # Return error response with status code 500
