import json
from sklearn.svm import SVC
from sklearn.cross_validation import train_test_split
from sklearn.cross_validation import cross_val_score, KFold
from scipy.stats import sem
from sklearn import metrics
import cv2
import numpy as np
from scipy.ndimage import zoom
from sklearn import datasets

faces = datasets.fetch_olivetti_faces()

class Trainer:
    def __init__(self):
        self.results = {}
        self.imgs = faces.images
        self.index = 0

    def reset(self):
        self.results = {}
        self.imgs = faces.images
        self.index = 0

    def increment_face(self):
        if self.index + 1 >= len(self.imgs):
            return self.index
        else:
            while str(self.index) in self.results:
                self.index += 1
            return self.index

    def record_result(self, smile=True):
        print ("Image", self.index + 1, ":", "Happy" if smile is True else "Sad")
        self.results[str(self.index)] = smile


# Trained classifier's performance evaluation
def evaluate_cross_validation(clf, X, y, K):
    cv = KFold(len(y), K, shuffle=True, random_state=0)
    scores = cross_val_score(clf, X, y, cv=cv)
    print ("Scores: ",scores)
    print ("Mean score: ".format(np.mean(scores), sem(scores)))


# Confusion Matrix and Results
def train_and_evaluate(clf, X_train, X_test, y_train, y_test):
    clf.fit(X_train, y_train)
    print ("Accuracy on training set:")
    print (clf.score(X_train, y_train))
    print ("Accuracy on testing set:")
    print (clf.score(X_test, y_test))
    y_pred = clf.predict(X_test)
    print ("Classification Report:")
    print (metrics.classification_report(y_test, y_pred))
    print ("Confusion Matrix:")
    print (metrics.confusion_matrix(y_test, y_pred))


def detectFaces(frame):
    cascPath = "../data/haarcascade_frontalface_default.xml"
    faceCascade = cv2.CascadeClassifier(cascPath)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    detected_faces = faceCascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=6,
            minSize=(50, 50),
            flags=cv2.CASCADE_SCALE_IMAGE)
    return gray, detected_faces


def extract_face_features(gray, detected_face, offset_coefficients):
    (x, y, w, h) = detected_face
    horizontal_offset = int(offset_coefficients[0] * w)
    vertical_offset = int(offset_coefficients[1] * h)
    extracted_face = gray[y + vertical_offset:y + h,x + horizontal_offset:x - horizontal_offset + w]
    new_extracted_face = zoom(extracted_face, (64. / extracted_face.shape[0],64. / extracted_face.shape[1]))
    new_extracted_face = new_extracted_face.astype(np.float32)
    new_extracted_face /= float(new_extracted_face.max())
    return new_extracted_face


def predict_face_is_smiling(extracted_face):
    return True if svc_1.predict(extracted_face.reshape(1, -1)) else False

if __name__ == "__main__":

    svc_1 = SVC(kernel='linear')  # Initializing Classifier

    trainer = Trainer()
    results = json.load(open("../results/results.xml"))  # Loading the classification result
    trainer.results = results

    indices = [int(i) for i in trainer.results]  # Building the dataset now
    data = faces.data[indices, :]  # Image Data

    target = [trainer.results[i] for i in trainer.results]  # Target Vector
    target = np.array(target).astype(np.int32)

    # Train the classifier using 5 fold cross validation
    X_train, X_test, y_train, y_test = train_test_split(data, target, test_size=0.25, random_state=0)

    # Trained classifier's performance evaluation
    evaluate_cross_validation(svc_1, X_train, y_train, 5)

    # Confusion Matrix and Results
    train_and_evaluate(svc_1, X_train, X_test, y_train, y_test)

    video_capture = cv2.VideoCapture(0)

    while True:
        ret, frame = video_capture.read()

        # detect faces
        gray, detected_faces = detectFaces(frame)

        face_index = 0

        cv2.putText(frame, "Press Esc to QUIT", (15, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1)

        # predict output
        for face in detected_faces:
            (x, y, w, h) = face
            if w > 100:
                # draw rectangle around face
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

                extracted_face = extract_face_features(gray, face, (0.3, 0.05)) #(0.075, 0.05)

                # predict smile
                prediction_result = predict_face_is_smiling(extracted_face)

                # draw extracted face in the top right corner
                frame[face_index * 64: (face_index + 1) * 64, -65:-1, :] = cv2.cvtColor(extracted_face * 255, cv2.COLOR_GRAY2RGB)                
                if prediction_result is True:
                    cv2.putText(frame, "HAPPY",(x,y), cv2.FONT_HERSHEY_SIMPLEX, 2, 155, 5)
                else:
                    cv2.putText(frame, "SAD",(x,y), cv2.FONT_HERSHEY_SIMPLEX, 2, 155, 5)

                face_index += 1

        # Display the resulting frame
        cv2.imshow('Video', frame)
        if cv2.waitKey(10) & 0xFF == 27:
            break

    video_capture.release()
    cv2.destroyAllWindows()
