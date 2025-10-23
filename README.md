# **Information-Bottleneck-Method**

The **Information Bottleneck (IB) method**, introduced by Tishby et al., provides a principled framework for *extracting relevant information* from a signal while *compressing input data*. Instead of specifying an arbitrary distortion measure, IB compresses the input variable **`X`** while preserving as much information as possible about a target variable **`Y`**.

---

## **Files**

### **1. `information_bottleneck.py`**
- Implements the **Information Bottleneck algorithm**.  
- **Key functions:**  
  1. **`kl_divergence(p, q)`**: Computes KL divergence between two distributions.  
  2. **`mutual_information(p_xy, p_x, p_y)`**: Computes mutual information **I(X;Y)** for discrete distributions.  
  3. **`information_bottleneck(p_x, p_y_given_x, beta, num_clusters, num_iterations=100)`**: Runs the IB iterative procedure and returns the learned mapping **P(T|X)** and mutual information values.

### **2. `plot_information_plane.py`**
- **Visualizes the Information Plane** (plot of **I(X;T)** vs **I(T;Y)**) for different values of **beta**.  
- Uses the IB algorithm from **`information_bottleneck.py`**.  
- Requires **`matplotlib`** and **`numpy`**.

---

## **Usage**

1.**Install dependencies:**  
pip install numpy matplotlib scipy

2.**Run the plot script:** 
python plot_information_plane.py

3.**View results:** 
A plot will appear showing I(X;T) on the x-axis and I(T;Y) on the y-axis for different beta values.
