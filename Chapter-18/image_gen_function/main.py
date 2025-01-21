import functions_framework
from PIL import Image
import vertexai
from vertexai.preview.vision_models import ImageGenerationModel
from google.cloud import storage
import os

# Environment variables
PROJECT_ID = os.environ.get("PROJECT_ID")
REGION = os.environ.get("REGION")
BUCKET = os.environ.get("BUCKET")
DATA_FOLDER_PATH = os.environ.get("DATA_FOLDER_PATH")
PROMPT = os.environ.get("PROMPT")

BLOB_NAME = "func_generated_image.png"

@functions_framework.cloud_event  # Decorator for Cloud Events 
def generate_and_store_image(cloud_event):
    try:
        # Initialize the Vertex AI environment using our specified project and region
        vertexai.init(project=PROJECT_ID, location=REGION)
        
        # Load the Imagen `"imagegeneration@006"` image generation model
        model = ImageGenerationModel.from_pretrained("imagegeneration@006")

        # Generate an image
        images = model.generate_images(
            prompt=PROMPT,
            number_of_images=1,
            language="en",
            aspect_ratio="1:1",
            safety_filter_level="block_some",
            person_generation="allow_adult",
        )

        # Store the image data in GCS
        image_data = images[0]._image_bytes
        storage_client = storage.Client(project=PROJECT_ID)
        bucket = storage_client.bucket(BUCKET)
        blob_path = f"{DATA_FOLDER_PATH}/{BLOB_NAME}"
        blob = bucket.blob(blob_path)
        blob.upload_from_string(image_data, content_type="image/png")

        # Return status
        return "Image generated and stored successfully!", 200

    except Exception as e:
        print(f"Error: {e}")
        return f"An error occurred: {e}", 500
