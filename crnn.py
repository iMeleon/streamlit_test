import torch
from model import PytorchConvRecurNN
import albumentations
from PIL import Image
import numpy as np
from sklearn import preprocessing


class RCNNModel:
    def __init__(self, label_encoder_save_path, model_save_path, device, image_width, image_height, mean, std):
        label_encoder_save_path = r"C:\Users\user\Desktop\streamlit\models\crnn\number_classes.npy"
        model_save_path = r"C:\Users\user\Desktop\streamlit\models\crnn\crnn_numbers_model.pth"
        self.lbl_enc = preprocessing.LabelEncoder()
        self.lbl_enc.classes_ = np.load(label_encoder_save_path)
        self.model = PytorchConvRecurNN(len(self.lbl_enc.classes_))
        self.model.load_state_dict(torch.load(model_save_path, map_location=torch.device('cpu')))
        self.model.eval()
        self.image_width = image_width
        self.image_height = image_height
        self.aug  = albumentations.Compose(
            [
                albumentations.Normalize(
                    mean, std, max_pixel_value=255.0, always_apply=True
                )
            ]
        )

    def predict(self, image):
        image = image.resize(
            (self.image_width, self.image_height), resample=Image.BILINEAR
        )

        image = np.array(image)
        augmented = self.aug(image=image)
        image = augmented["image"]
        image = np.transpose(image, (2, 0, 1)).astype(np.float32)
        image = torch.tensor(image, dtype=torch.float)
        image = image.reshape(1, *image.shape)
        res = self.model(image)[0]
        current_predictions = self.decode_predictions(res)
        return current_predictions

    def decode_predictions(self, predictions):
        predictions = predictions.permute(1, 0, 2)
        predictions = torch.softmax(predictions, 2)
        predictions = torch.argmax(predictions, 2)
        predictions = predictions.detach().cpu().numpy()
        cap_preds = []
        for j in range(predictions.shape[0]):
            temp = []
            for k in predictions[j, :]:
                k = k - 1
                if k == -1:
                    temp.append("§")
                else:
                    p = self.lbl_enc.inverse_transform([k])[0]
                    temp.append(p)
            tp = "".join(temp)
            cap_preds.append(float(self.remove_duplicates(tp).replace("§", "")))
        return cap_preds

    def remove_duplicates(self, x):
        if len(x) < 2:
            return x
        fin = ""
        for j in x:
            if fin == "":
                fin = j
            else:
                if j == fin[-1]:
                    continue
                else:
                    fin = fin + j
        return fin


if __name__ == '__main__':
    print('helloo from main')