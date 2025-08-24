# Optimizing Retail Promotions via Latent Dirichlet Allocation-Based Customer Segmentation

## Overview

This project aims to optimize retail promotions by identifying distinct customer segments based on their purchase behavior and product affinity.  We utilize Latent Dirichlet Allocation (LDA), a topic modeling technique, to uncover latent customer segments from transactional data.  The analysis provides insights into the characteristics of each segment, enabling the development of targeted promotional strategies designed to maximize conversion rates and increase customer lifetime value.

## Technologies Used

* Python 3.x
* Pandas
* NumPy
* Scikit-learn
* Gensim
* Matplotlib
* Seaborn

## How to Run

1. **Install Dependencies:**  Ensure you have Python 3.x installed.  Then, install the required Python libraries listed above using pip:

   ```bash
   pip install -r requirements.txt
   ```

2. **Run the Main Script:** Execute the main script using:

   ```bash
   python main.py
   ```

   This script will perform the LDA analysis, segment customers, and generate visualizations.  Ensure that your input data (specified within `main.py`) is correctly formatted and accessible.


## Example Output

The script will print key findings to the console, including:

* The number of identified customer segments.
* The top products associated with each segment.
* Descriptive statistics summarizing the purchase behavior within each segment.

Additionally, the script generates several visualization files (e.g., `segment_product_affinity.png`, `segment_purchase_frequency.png`) within the `output` directory, providing a visual representation of the discovered customer segments and their characteristics.  These visualizations help to understand the identified segments and their potential for targeted marketing campaigns.