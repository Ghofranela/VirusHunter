# Add this function to the Streamlit app
def show_data_split_analysis():
    """Show dataset split analysis in Streamlit"""
    st.title("📊 Dataset Split Analysis")
    st.markdown("### EMBER Dataset Training/Validation/Test Split")
    
    try:
        # Load data
        data_dir = Path("data/processed")
        X_train = np.load(data_dir / 'X_train.npy')
        y_train = np.load(data_dir / 'y_train.npy')
        X_val = np.load(data_dir / 'X_val.npy')
        y_val = np.load(data_dir / 'y_val.npy')
        X_test = np.load(data_dir / 'X_test.npy')
        y_test = np.load(data_dir / 'y_test.npy')
        
        splits = {
            'Train': (X_train, y_train),
            'Validation': (X_val, y_val),
            'Test': (X_test, y_test)
        }
        
        # Display statistics
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Total Samples", 
                     f"{len(X_train) + len(X_val) + len(X_test):,}")
        
        with col2:
            st.metric("Feature Dimension", f"{X_train.shape[1]:,}")
        
        with col3:
            st.metric("Total Malware", 
                     f"{y_train.sum() + y_val.sum() + y_test.sum():,}")
        
        # Split details
        st.markdown("### Split Details")
        
        split_data = []
        for split_name, (X, y) in splits.items():
            n_samples = len(X)
            n_malware = y.sum()
            n_benign = len(y) - n_malware
            
            split_data.append({
                'Split': split_name,
                'Samples': n_samples,
                'Malware': n_malware,
                'Benign': n_benign,
                'Malware %': f"{(n_malware/n_samples)*100:.1f}%",
                'Split %': f"{(n_samples/(len(X_train)+len(X_val)+len(X_test)))*100:.1f}%"
            })
        
        # Display as table
        st.dataframe(split_data, use_container_width=True)
        
        # Visualizations
        st.markdown("### Visualizations")
        
        # Pie chart for split proportions
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Split sizes
        sizes = [len(X_train), len(X_val), len(X_test)]
        labels = [f'Train\n{sizes[0]:,}', f'Validation\n{sizes[1]:,}', f'Test\n{sizes[2]:,}']
        ax1.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90)
        ax1.set_title('Dataset Split Proportions')
        
        # Class distribution
        split_names = ['Train', 'Validation', 'Test']
        malware_counts = [y_train.sum(), y_val.sum(), y_test.sum()]
        benign_counts = [len(y_train)-y_train.sum(), len(y_val)-y_val.sum(), len(y_test)-y_test.sum()]
        
        x = np.arange(len(split_names))
        width = 0.35
        
        ax2.bar(x - width/2, benign_counts, width, label='Benign', color='green')
        ax2.bar(x + width/2, malware_counts, width, label='Malware', color='red')
        
        ax2.set_xlabel('Dataset Split')
        ax2.set_ylabel('Number of Samples')
        ax2.set_title('Class Distribution Across Splits')
        ax2.set_xticks(x)
        ax2.set_xticklabels(split_names)
        ax2.legend()
        ax2.grid(axis='y', alpha=0.3)
        
        st.pyplot(fig)
        
    except FileNotFoundError:
        st.error("Processed data not found. Please run preprocessing first.")