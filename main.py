import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer
from gensim.models import LdaModel
import os
# --- 1. Synthetic Data Generation ---
np.random.seed(42) # for reproducibility
# Simulate customer purchase data
num_customers = 100
num_products = 20
num_purchases = 500
customers = [f"Customer_{i}" for i in range(1, num_customers + 1)]
products = [f"Product_{i}" for i in range(1, num_products + 1)]
purchases = []
for _ in range(num_purchases):
    customer = np.random.choice(customers)
    product = np.random.choice(products)
    purchases.append((customer, product))
df = pd.DataFrame(purchases, columns=["Customer", "Product"])
# Create a "purchase history" string for each customer
customer_purchases = df.groupby("Customer")["Product"].agg(list).reset_index()
customer_purchases["PurchaseHistory"] = customer_purchases["Product"].apply(lambda x: " ".join(x))
customer_purchases = customer_purchases[["Customer", "PurchaseHistory"]]
# --- 2. Latent Dirichlet Allocation (LDA) ---
# Create a document-term matrix
vectorizer = CountVectorizer()
dtm = vectorizer.fit_transform(customer_purchases["PurchaseHistory"])
# Train LDA model
lda_model = LdaModel(corpus=dtm, num_topics=5, id2word=vectorizer.vocabulary_, passes=10)
# --- 3.  Topic Visualization (Simplified) ---
#  Visualizing LDA topics directly is complex.  We'll print the top words per topic as a proxy.
print("Top words per topic:")
for idx, topic in lda_model.print_topics(-1, num_words=5):
    print(f"Topic: {idx} Words: {topic}")
# --- 4. Customer Segmentation ---
# Assign customers to topics (segments) based on LDA probabilities
customer_topic_probs = []
for i in range(len(customer_purchases)):
    topic_probs = lda_model.get_document_topics(dtm[i])
    customer_topic_probs.append(topic_probs)
customer_segments = pd.DataFrame(customer_topic_probs)
customer_segments["Customer"] = customer_purchases["Customer"]
customer_segments = customer_segments.set_index('Customer')
#  (Simplified segmentation: assign to the most probable topic)
customer_segments["Segment"] = customer_segments.idxmax(axis=1)
# --- 5. Visualization (Simplified - Bar Chart of Segment Sizes) ---
segment_counts = customer_segments["Segment"].value_counts()
plt.figure(figsize=(8, 6))
plt.bar(segment_counts.index, segment_counts.values)
plt.xlabel("Customer Segment")
plt.ylabel("Number of Customers")
plt.title("Customer Segment Distribution")
plt.savefig("customer_segments.png")
print("Plot saved to customer_segments.png")
# --- Error Handling and Robustness ---
if not os.path.exists("customer_segments.png"):
    print("Error: Plot not saved correctly.")