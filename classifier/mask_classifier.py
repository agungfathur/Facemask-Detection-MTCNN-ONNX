import onnx
import onnxruntime as ort
import numpy as np
import cv2

with open('classifier/model/label.txt', 'r') as f:
    labels = [l.rstrip() for l in f]

model_path = 'classifier/model/mobilenetV2_224.onnx'
model = onnx.load(model_path)
session = ort.InferenceSession(model.SerializeToString())

def classifier(rois) :
    rois = preprocess(rois)
    # print(rois.shape)
    ort_inputs = {session.get_inputs()[0].name: rois}
    preds = session.run(None, ort_inputs)[0]
    preds = np.squeeze(preds)
    a = np.argsort(preds)[::-1]
    resp = labels[a[0]]
    # preds = list(zip(labels, preds))
    return resp

def preprocess(img):
    # img = img / 255.
    img = cv2.resize(img, (224, 224))
    # img = np.transpose(img, axes=[2, 0, 1])
    img = img.astype(np.float32)
    img = np.expand_dims(img, axis=0)
    return img