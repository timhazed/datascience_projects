
from transformers import BlipProcessor, BlipForConditionalGeneration

class BlipManager:
    def __init__(self):
        """
        Initialize the BlipManager by creating the processor and model.
        """
        self.processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
        self.model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

    def generate_caption(self, image, max_new_tokens=50, prompt=None):
        """
        Generate a caption for the given image. Optionally, use a prompt for conditional captioning.

        Args:
            image (PIL.Image): The image to caption.
            max_new_tokens (int): The maximum number of tokens to generate.
            prompt (str, optional): The prompt for generating the caption.

        Returns:
            str: The generated caption.
        """
        inputs = self.processor(images=image, text=prompt if prompt else "", return_tensors="pt")
        outputs = self.model.generate(**inputs, max_new_tokens=max_new_tokens)
        caption = self.processor.decode(outputs[0], skip_special_tokens=True).strip()

        # Post-process the caption
        if prompt and prompt in caption:
            caption = caption.replace(prompt, "").strip()

        # Handle edge case where no meaningful caption is generated
        if not caption:
            caption = "No meaningful caption generated."

        return f"{prompt}: {caption}" if prompt else caption

    def caption_image(self, image, prompt=None):
        """
        Takes a PIL Image input and an optional prompt, and returns a caption.

        Args:
            image (PIL.Image): The image to caption.
            prompt (str, optional): The prompt for generating the caption.

        Returns:
            str: The generated caption.
        """
        return self.generate_caption(image, prompt=prompt)
