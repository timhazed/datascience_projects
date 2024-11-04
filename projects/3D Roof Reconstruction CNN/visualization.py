import matplotlib.pyplot as plt

class Visualization:
    def __init__(self):
        pass

    def plot_component_losses(self, history, title=None):       
        # Plot losses if keys are present
        azimuth_loss = history.history.get('azimuth_loss', [])
        height_loss = history.history.get('height_loss', [])
        tilt_loss = history.history.get('tilt_loss', [])
        perimeter_loss = history.history.get('perimeter_loss', [])

        val_azimuth_loss = history.history.get('val_azimuth_loss', [])
        val_height_loss = history.history.get('val_height_loss', [])
        val_tilt_loss = history.history.get('val_tilt_loss', [])
        val_perimeter_loss = history.history.get('val_perimeter_loss', [])

        plt.figure(figsize=(12, 8))
        plt.subplot(2, 2, 1)
        plt.plot(azimuth_loss, label='Training Azimuth Loss')
        plt.plot(val_azimuth_loss, label='Validation Azimuth Loss')
        plt.title('Azimuth Loss')
        plt.legend()

        plt.subplot(2, 2, 2)
        plt.plot(height_loss, label='Training Height Loss')
        plt.plot(val_height_loss, label='Validation Height Loss')
        plt.title('Height Loss')
        plt.legend()

        plt.subplot(2, 2, 3)
        plt.plot(tilt_loss, label='Training Tilt Loss')
        plt.plot(val_tilt_loss, label='Validation Tilt Loss')
        plt.title('Tilt Loss')
        plt.legend()

        plt.subplot(2, 2, 4)
        plt.plot(perimeter_loss, label='Training Perimeter Loss')
        plt.plot(val_perimeter_loss, label='Validation Perimeter Loss')
        plt.title('Perimeter Loss')
        plt.legend()

        if title:
            plt.suptitle(title)
        else:
            plt.suptitle('Training and Validation Losses for All Components')

        plt.tight_layout()
        plt.show(block=True)


    def plot_training(self, history):
        plt.figure(figsize=(12, 6))
        plt.plot(history.history['loss'], label='Training Loss')
        plt.plot(history.history['val_loss'], label='Validation Loss')
        plt.title('Training and Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.show(block=True)
