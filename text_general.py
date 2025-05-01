import pandas as pd
import re
import string
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay

csv_file = "E:\\DS_LAB\\final\\text\\spam.csv"  
text_column = "v2"          
label_column = "v1"         

# 1. Load dataset
df = pd.read_csv(csv_file,encoding='ISO-8859-1')

# Ensure only the necessary columns are used
df = df[[label_column, text_column]].dropna()
df.columns = ['Label', 'Message']

# 2. Label Encoding
le = LabelEncoder()
df['Label'] = le.fit_transform(df['Label'])
print(f"Label Mapping: {dict(zip(le.classes_, le.transform(le.classes_)))}")

# 3. Text Cleaning Function
def clean_text(text):
    text = text.lower()
    text = re.sub(r'\d+', '', text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    return text.strip()

df['Message'] = df['Message'].apply(clean_text)

# 4. Split data
X_train, X_test, y_train, y_test = train_test_split(df['Message'], df['Label'], test_size=0.2, random_state=42)

# 5. TF-IDF Vectorization
vectorizer = TfidfVectorizer(stop_words='english')
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# 6. Train SVM Classifier
svm = LinearSVC()
svm.fit(X_train_vec, y_train)
svm_pred = svm.predict(X_test_vec)

# 7. Evaluation
print("SVM Accuracy:", accuracy_score(y_test, svm_pred))
print(classification_report(y_test, svm_pred, target_names=le.classes_))

cm_svm = confusion_matrix(y_test, svm_pred)
ConfusionMatrixDisplay(cm_svm, display_labels=le.classes_).plot(cmap='Greens')
plt.title("SVM Confusion Matrix")
plt.show()

# 8. Predict user-input text
def predict_user_input():
    while True:
        user_text = input("\nEnter text to classify (or type 'exit' to quit): ")
        if user_text.lower() == 'exit':
            break
        cleaned = clean_text(user_text)
        vec = vectorizer.transform([cleaned])
        pred = svm.predict(vec)
        print("Predicted Label:", le.inverse_transform(pred)[0])

predict_user_input()