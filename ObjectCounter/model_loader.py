from ultralytics import YOLO


class ModelLoader:
    def __init__(self, model_path, use_gpu=False):
        """
        Initializes the ModelLoader.

        :param model_path: Path to the YOLO model file.
        :param device: Device to load the model on ("cuda" or "cpu").
                       If None, it will use CUDA if available, otherwise CPU.
        """
        self.model_path = model_path
        self.model = None
        self.device = "cuda" if use_gpu else "cpu"
        self.class_labels = None


    def load_yolo_model(self):
        """
        Loads the YOLO model, sets it to evaluation mode, moves it to the selected device,
        and returns the model, class labels, and device.

        :return: Tuple having (model, class_labels, device)
        """
        try:
            self.model = YOLO(self.model_path)  # .to(self.device) Move model to device
            self.model.eval()  # Set model to evaluation mode
            return self.model
        except Exception as e:
            raise ValueError(f"Failed to load model: {str(e)}")