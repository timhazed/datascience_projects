
import gradio as gr
from BlipManager import BlipManager

# Instantiate the BlipManager
blip_manager = BlipManager()

# Function to clear the prompt and caption
def reset_prompt_and_caption():
    return "", ""

# Set up the Gradio interface using Blocks
with gr.Blocks(title="Image Captioning with BLIP") as iface:
    with gr.Row():
        image_input = gr.Image(type="pil", label="Upload Image", interactive=True)
        prompt_input = gr.Textbox(lines=1, placeholder="Optional prompt", label="Prompt (Optional)", interactive=True)
    
    output = gr.Textbox(label="Generated Caption")
    
    # Event to clear the prompt and caption when a new image is uploaded
    image_input.change(fn=reset_prompt_and_caption, inputs=[], outputs=[prompt_input, output])
    
    # Define what happens when both inputs are submitted
    submit_button = gr.Button("Generate Caption")
    submit_button.click(fn=blip_manager.caption_image, inputs=[image_input, prompt_input], outputs=output)

print("Launching Gradio interface...")
# Launch the interface
iface.launch(server_port=7860)
print("Gradio interface launched.")