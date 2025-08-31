import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
import os
import tkinter as tk
from tkinter import ttk, scrolledtext, messagebox, filedialog
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import ttkbootstrap as ttk
from ttkbootstrap.constants import *
import nltk
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.tokenize import word_tokenize
import joblib
from tqdm import tqdm
import logging
import datetime
import json
import uuid
from ttkbootstrap.tooltip import ToolTip  # Changed from tooltip import
import warnings
warnings.filterwarnings('ignore')

# Safe NLTK downloads
try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('punkt', quiet=True)
    nltk.download('wordnet', quiet=True)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('classifier_advanced.log'),
        logging.StreamHandler()
    ]
)

class TextClassifier:
    def __init__(self):
        logging.info("Initializing Advanced Text Classifier...")
        self.vectorizer = TfidfVectorizer(max_features=3000, stop_words='english')
        self.models = {
            'Naive Bayes': MultinomialNB(),
            'Logistic Regression': LogisticRegression(max_iter=1000),
            'SVM': SVC(probability=True),
            'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, random_state=42)
        }
        self.current_model = 'Naive Bayes'
        self.labels = []
        self.stemmer = PorterStemmer()
        self.lemmatizer = WordNetLemmatizer()
        self.model_history = []
        self.trained = False  # Add trained flag
        
    def preprocess_text(self, text, use_stemming=False, use_lemmatization=False):
        """Preprocess text with stemming or lemmatization"""
        tokens = word_tokenize(text.lower())
        if use_stemming:
            tokens = [self.stemmer.stem(token) for token in tokens]
        if use_lemmatization:
            tokens = [self.lemmatizer.lemmatize(token) for token in tokens]
        return ' '.join(tokens)

    def load_data(self, filepath, use_stemming=False, use_lemmatization=False):
        """Load and prepare data from CSV file"""
        try:
            if not os.path.exists(filepath):
                raise FileNotFoundError(f"CSV file not found at: {filepath}")
                
            logging.info(f"Loading data from {filepath}")
            df = pd.read_csv(filepath)
            
            if df.empty:
                raise ValueError("The CSV file is empty")
                
            if 'title' not in df.columns or 'summary' not in df.columns or 'category' not in df.columns:
                raise ValueError("CSV file must contain 'title', 'summary', and 'category' columns")
                
            df['text'] = df['title'] + ' ' + df['summary']
            if use_stemming or use_lemmatization:
                df['text'] = df['text'].apply(lambda x: self.preprocess_text(x, use_stemming, use_lemmatization))
                
            logging.info(f"Loaded {len(df)} articles")
            return df['text'], df['category']
            
        except Exception as e:
            logging.error(f"Failed to load data: {str(e)}")
            raise

    def prepare_data(self, X, y):
        """Convert text to TF-IDF features and split data"""
        try:
            logging.info("Converting text to TF-IDF features...")
            if isinstance(X, pd.Series):
                X = X.fillna('')  # Handle NaN values
            X_transformed = self.vectorizer.fit_transform(X)
            logging.info(f"Created {X_transformed.shape[1]} features")
            
            self.labels = sorted(y.unique())
            logging.info(f"Found {len(self.labels)} unique categories: {', '.join(self.labels)}")
            
            return X_transformed, self.labels
        except Exception as e:
            logging.error(f"Error in prepare_data: {str(e)}")
            raise
    
    def train(self, X_train, y_train, model_name, hyperparameters=None):
        """Train the selected classifier with optional hyperparameters"""
        try:
            if model_name not in self.models:
                raise ValueError(f"Invalid model name: {model_name}")
                
            logging.info(f"Training {model_name} classifier...")
            model = self.models[model_name]
            
            if hyperparameters:
                # Filter valid hyperparameters
                valid_params = model.get_params().keys()
                filtered_params = {k: v for k, v in hyperparameters.items() 
                                if k in valid_params}
                model.set_params(**filtered_params)
            
            model.fit(X_train, y_train)
            self.trained = True
            
            self.model_history.append({
                'model': model_name,
                'timestamp': datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                'id': str(uuid.uuid4())
            })
            logging.info("Training completed")
        except Exception as e:
            logging.error(f"Training error: {str(e)}")
            raise
    
    def evaluate(self, X_test, y_test, model_name):
        """Evaluate the model and return predictions and metrics"""
        logging.info(f"Evaluating {model_name} on {X_test.shape[0]} test samples...")
        y_pred = self.models[model_name].predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred)
        logging.info(f"Test accuracy: {accuracy*100:.2f}%")
        return y_pred, accuracy, report
    
    def get_full_dataset_confusion_matrix(self, X_full, y_full, model_name):
        """Get confusion matrix for the full dataset"""
        logging.info("Generating confusion matrix for full dataset...")
        y_pred_full = self.models[model_name].predict(X_full)
        return confusion_matrix(y_full, y_pred_full)
    
    def save_model(self, filepath):
        """Save the current model and vectorizer"""
        try:
            joblib.dump({
                'model': self.models[self.current_model],
                'vectorizer': self.vectorizer,
                'labels': self.labels,
                'history': self.model_history
            }, filepath)
            logging.info(f"Model saved to {filepath}")
        except Exception as e:
            logging.error(f"Failed to save model: {str(e)}")
            raise
    
    def load_model(self, filepath):
        """Load a saved model and vectorizer"""
        try:
            data = joblib.load(filepath)
            self.models[self.current_model] = data['model']
            self.vectorizer = data['vectorizer']
            self.labels = data['labels']
            self.model_history = data.get('history', [])
            logging.info(f"Model loaded from {filepath}")
        except Exception as e:
            logging.error(f"Failed to load model: {str(e)}")
            raise
    
    def classify_text(self, text, model_name, use_stemming=False, use_lemmatization=False):
        """Classify a single text input with confidence score"""
        logging.info("Classifying input text...")
        
        processed_text = self.preprocess_text(text, use_stemming, use_lemmatization)
        X_input = self.vectorizer.transform([processed_text])
        
        try:
            probabilities = self.models[model_name].predict_proba(X_input)[0]
            prediction_idx = probabilities.argmax()
            confidence = probabilities[prediction_idx]
            prediction = self.labels[prediction_idx]
            
            logging.info(f"Predicted category: {prediction} (confidence: {confidence:.2%})")
            logging.info("Confidence scores for all categories:")
            for label, prob in zip(self.labels, probabilities):
                logging.info(f"{label}: {prob:.2%}")
                
            return prediction, confidence, probabilities
            
        except Exception as e:
            logging.error(f"Classification error: {str(e)}")
            return None, 0.0, []

class TextClassifierUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Elegant Text Classification System")
        self.root.geometry("1800x1000")
        
        self.classifier = TextClassifier()
        self.current_tooltip = None
        self.setup_ui()
        
    def setup_ui(self):
        style = ttk.Style("flatly")  # Light theme
        
        # Main container
        main_frame = ttk.Frame(self.root, padding="20")
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Configure grid
        main_frame.columnconfigure((0, 1, 2), weight=1)
        main_frame.rowconfigure(0, weight=1)
        
        # Left panel (Controls)
        left_panel = ttk.LabelFrame(main_frame, text="Control Panel", padding="15", bootstyle="primary")
        left_panel.grid(row=0, column=0, sticky="nsew", padx=10, pady=10)
        
        # Center panel (Input & Classification)
        center_panel = ttk.LabelFrame(main_frame, text="Classification", padding="15", bootstyle="primary")
        center_panel.grid(row=0, column=1, sticky="nsew", padx=10, pady=10)
        
        # Right panel (Results & Visualization)
        right_panel = ttk.LabelFrame(main_frame, text="Results & Visualization", padding="15", bootstyle="primary")
        right_panel.grid(row=0, column=2, sticky="nsew", padx=10, pady=10)
        
        # Setup panels
        self.setup_control_panel(left_panel)
        self.setup_classification_panel(center_panel)
        self.setup_results_panel(right_panel)
        
        # Configure styles
        style.configure('TButton', font=('Helvetica', 11))
        style.configure('TLabelframe.Label', font=('Helvetica', 13, 'bold'))
        style.configure('primary.TButton', padding=12)
        style.configure('success.TButton', padding=12)
        style.configure('warning.TButton', padding=12)
        style.configure('danger.TButton', padding=12)
        style.configure('info.TCheckbutton', font=('Helvetica', 10))
        
    def setup_control_panel(self, panel):
        # Model selection
        model_frame = ttk.Frame(panel)
        model_frame.pack(fill=tk.X, pady=5)
        
        ttk.Label(model_frame, text="Model:", font=('Helvetica', 11, 'bold')).pack(side=tk.LEFT, padx=5)
        self.model_var = tk.StringVar(value='Naive Bayes')
        model_dropdown = ttk.Combobox(
            model_frame,
            textvariable=self.model_var,
            values=list(self.classifier.models.keys()),
            state='readonly',
            width=20
        )
        model_dropdown.pack(side=tk.LEFT, padx=5)
        # Fix tooltip implementation
        ToolTip(model_dropdown, text="Select classification model")
        
        # Preprocessing options
        preprocess_frame = ttk.Frame(panel)
        preprocess_frame.pack(fill=tk.X, pady=5)
        
        self.stem_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(
            preprocess_frame,
            text="Enable Stemming",
            variable=self.stem_var,
            style='info.TCheckbutton'
        ).pack(side=tk.LEFT, padx=5)
        
        self.lemma_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(
            preprocess_frame,
            text="Enable Lemmatization",
            variable=self.lemma_var,
            style='info.TCheckbutton'
        ).pack(side=tk.LEFT, padx=5)
        
        # Hyperparameters
        hyper_frame = ttk.LabelFrame(panel, text="Hyperparameters", padding="10")
        hyper_frame.pack(fill=tk.X, pady=5)
        
        ttk.Label(hyper_frame, text="Max Features:").pack(anchor="w", padx=5)
        self.max_features_var = tk.StringVar(value="3000")
        ttk.Entry(hyper_frame, textvariable=self.max_features_var, width=10).pack(anchor="w", padx=5)
        
        # Training controls
        control_frame = ttk.Frame(panel)
        control_frame.pack(fill=tk.X, pady=5)
        
        self.train_button = ttk.Button(
            control_frame,
            text="Train Model",
            command=self.train_model,
            style="primary.TButton",
            width=15
        )
        self.train_button.pack(side=tk.LEFT, padx=5)
        
        ttk.Button(
            control_frame,
            text="Save Model",
            command=self.save_model,
            style="success.TButton",
            width=15
        ).pack(side=tk.LEFT, padx=5)
        
        ttk.Button(
            control_frame,
            text="Load Model",
            command=self.load_model,
            style="warning.TButton",
            width=15
        ).pack(side=tk.LEFT, padx=5)
        
        # Model history
        history_frame = ttk.LabelFrame(panel, text="Model History", padding="10")
        history_frame.pack(fill=tk.BOTH, expand=True, pady=5)
        
        self.history_text = scrolledtext.ScrolledText(
            history_frame,
            height=10,
            width=30,
            font=("Helvetica", 10),
            wrap=tk.WORD
        )
        self.history_text.pack(fill=tk.BOTH, expand=True)
        
    def setup_classification_panel(self, panel):
        # Text input
        ttk.Label(
            panel,
            text="Text Input",
            font=("Helvetica", 11, "bold")
        ).pack(anchor="w", pady=(5, 2))
        
        self.input_text = scrolledtext.ScrolledText(
            panel,
            height=10,
            width=50,
            font=("Helvetica", 10),
            wrap=tk.WORD
        )
        self.input_text.pack(fill=tk.X, pady=5)
        # Fix tooltip implementation
        ToolTip(self.input_text, text="Enter text to classify")
        
        self.classify_button = ttk.Button(
            panel,
            text="Classify Text",
            command=self.classify_text,
            style="success.TButton",
            width=15
        )
        self.classify_button.pack(pady=5)
        
        # Classification results
        results_frame = ttk.LabelFrame(panel, text="Classification Results", padding="10")
        results_frame.pack(fill=tk.BOTH, expand=True, pady=5)
        
        self.results_text = scrolledtext.ScrolledText(
            results_frame,
            height=12,
            width=50,
            font=("Helvetica", 10),
            wrap=tk.WORD
        )
        self.results_text.pack(fill=tk.BOTH, expand=True)
        
    def setup_results_panel(self, panel):
        # Confusion matrix
        self.plot_frame = ttk.Frame(panel)
        self.plot_frame.pack(fill=tk.BOTH, expand=True, pady=5)
        
        # Export options
        export_frame = ttk.Frame(panel)
        export_frame.pack(fill=tk.X, pady=5)
        
        ttk.Button(
            export_frame,
            text="Export Results",
            command=lambda: self.export_results('txt'),
            style="danger.TButton",
            width=15
        ).pack(side=tk.LEFT, padx=5)
        
        ttk.Button(
            export_frame,
            text="Export as CSV",
            command=lambda: self.export_results('csv'),
            style="danger.TButton",
            width=15
        ).pack(side=tk.LEFT, padx=5)
        
    def train_model(self):
        try:
            self.train_button.config(state='disabled')
            self.results_text.delete(1.0, tk.END)
            self.results_text.insert(tk.END, "Training model, please wait...\n")
            
            data_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'classification_data.csv')
            X, y = self.classifier.load_data(data_path, self.stem_var.get(), self.lemma_var.get())
            X_transformed, labels = self.classifier.prepare_data(X, y)
            X_train, X_test, y_train, y_test = train_test_split(X_transformed, y, test_size=0.2, random_state=42)
            
            self.classifier.current_model = self.model_var.get()
            hyperparameters = {'max_features': int(self.max_features_var.get())}
            self.classifier.vectorizer.set_params(**hyperparameters)
            
            self.classifier.train(X_train, y_train, self.model_var.get(), hyperparameters)
            y_pred, accuracy, report = self.classifier.evaluate(X_test, y_test, self.model_var.get())
            
            self.results_text.delete(1.0, tk.END)
            self.results_text.insert(tk.END, f"Model Accuracy: {accuracy*100:.2f}%\n\n")
            self.results_text.insert(tk.END, "Detailed Classification Report:\n")
            self.results_text.insert(tk.END, report)
            
            cm = self.classifier.get_full_dataset_confusion_matrix(X_transformed, y, self.model_var.get())
            self.plot_confusion_matrix(cm)
            
            self.update_history()
            
            messagebox.showinfo("Success", "Model training completed successfully!", parent=self.root)
            
        except Exception as e:
            messagebox.showerror("Error", f"An error occurred: {str(e)}", parent=self.root)
        finally:
            self.train_button.config(state='normal')
    
    def classify_text(self):
        if not self.classifier.trained:
            messagebox.showwarning("Warning", "Please train the model first!", parent=self.root)
            return
            
        text = self.input_text.get(1.0, tk.END).strip()
        if not text:
            messagebox.showwarning("Warning", "Please enter some text to classify", parent=self.root)
            return
            
        try:
            prediction, confidence, probabilities = self.classifier.classify_text(
                text, self.model_var.get(), self.stem_var.get(), self.lemma_var.get()
            )
            
            self.results_text.delete(1.0, tk.END)
            self.results_text.insert(tk.END, f"Predicted Category: {prediction}\n")
            self.results_text.insert(tk.END, f"Confidence Score: {confidence:.2%}\n\n")
            self.results_text.insert(tk.END, "Confidence Scores for All Categories:\n")
            for label, prob in zip(self.classifier.labels, probabilities):
                self.results_text.insert(tk.END, f"{label}: {prob:.2%}\n")
                
        except Exception as e:
            messagebox.showerror("Error", f"Classification error: {str(e)}", parent=self.root)
    
    def plot_confusion_matrix(self, cm):
        for widget in self.plot_frame.winfo_children():
            widget.destroy()
            
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='YlGnBu',
                   xticklabels=self.classifier.labels,
                   yticklabels=self.classifier.labels, ax=ax,
                   annot_kws={"size": 12})
        plt.title('Confusion Matrix', fontsize=16, pad=20)
        plt.xlabel('Predicted Label', fontsize=12)
        plt.ylabel('True Label', fontsize=12)
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=45)
        plt.tight_layout()
        
        canvas = FigureCanvasTkAgg(fig, self.plot_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        plt.close()
        
        # Add hover interaction
        canvas.mpl_connect('motion_notify_event', lambda event: self.on_plot_hover(event, cm))
    
    def on_plot_hover(self, event, cm):
        """Fixed tooltip handling"""
        if hasattr(self, 'current_tooltip') and self.current_tooltip:
            self.current_tooltip.destroy()
            
        if event.inaxes:
            try:
                x, y = int(event.xdata + 0.5), int(event.ydata + 0.5)
                if 0 <= x < cm.shape[1] and 0 <= y < cm.shape[0]:
                    tooltip_text = f"True: {self.classifier.labels[y]}\nPredicted: {self.classifier.labels[x]}\nCount: {cm[y, x]}"
                    self.current_tooltip = ToolTip(self.plot_frame, text=tooltip_text)
            except (ValueError, IndexError):
                pass
    
    def save_model(self):
        if not self.classifier.trained:
            messagebox.showwarning("Warning", "Please train a model first!", parent=self.root)
            return
            
        filepath = filedialog.asksaveasfilename(
            defaultextension=".joblib",
            filetypes=[("Joblib files", "*.joblib"), ("All files", "*.*")]
        )
        if filepath:
            try:
                self.classifier.save_model(filepath)
                messagebox.showinfo("Success", "Model saved successfully!", parent=self.root)
            except Exception as e:
                messagebox.showerror("Error", f"Failed to save model: {str(e)}", parent=self.root)
    
    def load_model(self):
        filepath = filedialog.askopenfilename(
            filetypes=[("Joblib files", "*.joblib"), ("All files", "*.*")]
        )
        if filepath:
            try:
                self.classifier.load_model(filepath)
                self.update_history()
                messagebox.showinfo("Success", "Model loaded successfully!", parent=self.root)
            except Exception as e:
                messagebox.showerror("Error", f"Failed to load model: {str(e)}", parent=self.root)
    
    def export_results(self, format_type):
        filepath = filedialog.asksaveasfilename(
            defaultextension=f".{format_type}",
            filetypes=[(f"{format_type.upper()} files", f"*.{format_type}"), ("All files", "*.*")]
        )
        if filepath:
            try:
                if format_type == 'txt':
                    with open(filepath, 'w') as f:
                        f.write(f"Classification Results - {datetime.datetime.now()}\n\n")
                        f.write(self.results_text.get(1.0, tk.END))
                elif format_type == 'csv':
                    results = self.results_text.get(1.0, tk.END).split('\n')
                    df = pd.DataFrame([line.split(': ') for line in results if ': ' in line], 
                                    columns=['Metric', 'Value'])
                    df.to_csv(filepath, index=False)
                messagebox.showinfo("Success", f"Results exported as {format_type.upper()}!", parent=self.root)
            except Exception as e:
                messagebox.showerror("Error", f"Failed to export results: {str(e)}", parent=self.root)
    
    def update_history(self):
        self.history_text.delete(1.0, tk.END)
        self.history_text.insert(tk.END, "Model Training History:\n\n")
        for entry in self.classifier.model_history:
            self.history_text.insert(tk.END, f"Model: {entry['model']}\n")
            self.history_text.insert(tk.END, f"Time: {entry['timestamp']}\n")
            self.history_text.insert(tk.END, f"ID: {entry['id']}\n\n")

def main():
    root = ttk.Window(themename="flatly")
    app = TextClassifierUI(root)
    root.mainloop()

if __name__ == "__main__":
    main()