import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path
import warnings
import logging
import os
from datetime import datetime

# --- Logging Setup ---
# Define path for the log file
log_dir = 'logs'
plots_log_file_path = os.path.join(log_dir, 'plots_generation.log')

# Ensure log directory exists
os.makedirs(log_dir, exist_ok=True)

# Clear existing handlers to prevent duplicate logs
for handler in logging.root.handlers[:]:
    logging.root.removeHandler(handler)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(plots_log_file_path),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

logger.info("Starting Plot Generation for Gold Layer Visualizations...")

# Set up plotting style
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 10

class GoldLayerVisualizer:
    def __init__(self, data_path):
        """
        Initialize the visualizer with the path to the merged parquet files
        and set up the plot output directory.
        """
        self.data_path = Path(data_path)
        self.df = None
        self.load_data()

        # Determine the base directory of the script
        script_dir = Path(__file__).parent
        self.base_output_dir = script_dir.parent / 'plots'
        
        self.obligated_plots_dir = self.base_output_dir / 'obligated_plots'
        self.additional_plots_dir = self.base_output_dir / 'additional_plots'
        
        self.obligated_plots_dir.mkdir(parents=True, exist_ok=True)
        self.additional_plots_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Obligated plots will be saved to: {self.obligated_plots_dir}")
        logger.info(f"Additional plots will be saved to: {self.additional_plots_dir}")

    def load_data(self):
        """Load the merged parquet data"""
        try:
            if self.data_path.is_file():
                self.df = pd.read_parquet(self.data_path)
            else:
                self.df = pd.read_parquet(self.data_path)

            logger.info(f"Data loaded successfully: {len(self.df)} records, {len(self.df.columns)} columns")

            numeric_cols = ['unit_price', 'line_total', 'net_revenue', 'gross_revenue',
                           'order_subtotal', 'tax_amount', 'freight', 'total_due',
                           'list_price', 'standard_cost', 'weight']

            for col in numeric_cols:
                if col in self.df.columns:
                    if self.df[col].dtype == 'object':
                        self.df[col] = self.df[col].astype(str).str.replace(',', '.').astype(float)

            if 'order_date' in self.df.columns:
                self.df['order_date'] = pd.to_datetime(self.df['order_date'])

        except Exception as e:
            logger.error(f"Error loading data: {e}", exc_info=True)
            logger.error("Please check the data path and ensure parquet files exist")
            self.df = None # Ensure df is None if loading fails

    def _save_plot(self, filename, plot_type='obligated'):
        """Helper function to save and close the plot."""
        if plot_type == 'obligated':
            file_path = self.obligated_plots_dir / filename
        elif plot_type == 'additional':
            file_path = self.additional_plots_dir / filename
        else:
            logger.warning(f"Unknown plot type '{plot_type}'. Saving to obligated_plots folder.")
            file_path = self.obligated_plots_dir / filename

        try:
            plt.savefig(file_path)
            logger.info(f"Plot saved successfully: {file_path}")
        except Exception as e:
            logger.error(f"Error saving plot {filename}: {e}", exc_info=True)
        finally:
            plt.close()

    def plot_revenue_by_category(self):
        """Plot revenue by all product categories"""
        if self.df is None:
            logger.warning("DataFrame is None, skipping plot_revenue_by_category.")
            return

        plt.figure(figsize=(14, 8))
        category_revenue = self.df.groupby('category_name')['net_revenue'].sum().sort_values(ascending=False)

        ax = category_revenue.plot(kind='bar', color='skyblue', edgecolor='black')
        plt.title('Revenue by Product Category', fontsize=16, fontweight='bold')
        plt.xlabel('Product Category', fontsize=12)
        plt.ylabel('Revenue ($)', fontsize=12)
        plt.xticks(rotation=45, ha='right')

        for i, v in enumerate(category_revenue.values):
            ax.text(i, v + max(category_revenue) * 0.01, f'${v:,.0f}',
                   ha='center', va='bottom', fontweight='bold')

        plt.tight_layout()
        self._save_plot('revenue_by_category.png', plot_type='obligated')

    def plot_top_subcategories(self, top_n=10):
        """Plot top N subcategories by revenue"""
        if self.df is None:
            logger.warning("DataFrame is None, skipping plot_top_subcategories.")
            return

        plt.figure(figsize=(14, 8))
        subcategory_revenue = self.df.groupby('subcategory_name')['net_revenue'].sum().sort_values(ascending=False).head(top_n)

        ax = subcategory_revenue.plot(kind='barh', color='lightcoral', edgecolor='black')
        plt.title(f'Top {top_n} Subcategories by Revenue', fontsize=16, fontweight='bold')
        plt.xlabel('Revenue ($)', fontsize=12)
        plt.ylabel('Product Subcategory', fontsize=12)

        for i, v in enumerate(subcategory_revenue.values):
            ax.text(v + max(subcategory_revenue) * 0.01, i, f'${v:,.0f}',
                   ha='left', va='center', fontweight='bold')

        plt.tight_layout()
        self._save_plot(f'top_{top_n}_subcategories.png', plot_type='obligated')

    def plot_top_customers(self, top_n=10):
        """Plot top N customers by revenue"""
        if self.df is None:
            logger.warning("DataFrame is None, skipping plot_top_customers.")
            return

        plt.figure(figsize=(14, 8))
        customer_revenue = self.df.groupby('full_name')['net_revenue'].sum().sort_values(ascending=False).head(top_n)

        ax = customer_revenue.plot(kind='barh', color='lightgreen', edgecolor='black')
        plt.title(f'Top {top_n} Customers by Revenue', fontsize=16, fontweight='bold')
        plt.xlabel('Revenue ($)', fontsize=12)
        plt.ylabel('Customer Name', fontsize=12)

        for i, v in enumerate(customer_revenue.values):
            ax.text(v + max(customer_revenue) * 0.01, i, f'${v:,.0f}',
                   ha='left', va='center', fontweight='bold')

        plt.tight_layout()
        self._save_plot(f'top_{top_n}_customers.png', plot_type='obligated')

    def plot_revenue_by_order_status(self):
        """Plot revenue by all order statuses"""
        if self.df is None:
            logger.warning("DataFrame is None, skipping plot_revenue_by_order_status.")
            return

        plt.figure(figsize=(12, 8))
        status_revenue = self.df.groupby('order_status_desc')['net_revenue'].sum().sort_values(ascending=False)

        colors = plt.cm.Set3(np.linspace(0, 1, len(status_revenue)))
        wedges, texts, autotexts = plt.pie(status_revenue.values, labels=status_revenue.index,
                                          autopct='%1.1f%%', colors=colors, startangle=90)

        plt.title('Revenue Distribution by Order Status', fontsize=16, fontweight='bold')

        for autotext in autotexts:
            autotext.set_color('white')
            autotext.set_fontweight('bold')

        plt.axis('equal')
        self._save_plot('revenue_by_order_status.png', plot_type='obligated')

    def plot_top_countries(self, top_n=10):
        """Plot top N countries by revenue"""
        if self.df is None:
            logger.warning("DataFrame is None, skipping plot_top_countries.")
            return

        plt.figure(figsize=(14, 8))
        country_revenue = self.df.groupby('country_name')['net_revenue'].sum().sort_values(ascending=False).head(top_n)

        ax = country_revenue.plot(kind='bar', color='gold', edgecolor='black')
        plt.title(f'Top {top_n} Countries by Revenue', fontsize=16, fontweight='bold')
        plt.xlabel('Country', fontsize=12)
        plt.ylabel('Revenue ($)', fontsize=12)
        plt.xticks(rotation=45, ha='right')

        for i, v in enumerate(country_revenue.values):
            ax.text(i, v + max(country_revenue) * 0.01, f'${v:,.0f}',
                   ha='center', va='bottom', fontweight='bold')

        plt.tight_layout()
        self._save_plot(f'top_{top_n}_countries.png', plot_type='obligated')

    def plot_top_states(self, top_n=10):
        """Plot top N states by revenue"""
        if self.df is None:
            logger.warning("DataFrame is None, skipping plot_top_states.")
            return

        plt.figure(figsize=(14, 8))
        state_revenue = self.df.groupby('state_name')['net_revenue'].sum().sort_values(ascending=False).head(top_n)

        ax = state_revenue.plot(kind='barh', color='mediumpurple', edgecolor='black')
        plt.title(f'Top {top_n} States by Revenue', fontsize=16, fontweight='bold')
        plt.xlabel('Revenue ($)', fontsize=12)
        plt.ylabel('State', fontsize=12)

        for i, v in enumerate(state_revenue.values):
            ax.text(v + max(state_revenue) * 0.01, i, f'${v:,.0f}',
                   ha='left', va='center', fontweight='bold')

        plt.tight_layout()
        self._save_plot(f'top_{top_n}_states.png', plot_type='obligated')

    def plot_top_cities(self, top_n=10):
        """Plot top N cities by revenue"""
        if self.df is None:
            logger.warning("DataFrame is None, skipping plot_top_cities.")
            return

        plt.figure(figsize=(14, 8))
        city_revenue = self.df.groupby('city')['net_revenue'].sum().sort_values(ascending=False).head(top_n)

        ax = city_revenue.plot(kind='barh', color='orange', edgecolor='black')
        plt.title(f'Top {top_n} Cities by Revenue', fontsize=16, fontweight='bold')
        plt.xlabel('Revenue ($)', fontsize=12)
        plt.ylabel('City', fontsize=12)

        for i, v in enumerate(city_revenue.values):
            ax.text(v + max(city_revenue) * 0.01, i, f'${v:,.0f}',
                   ha='left', va='center', fontweight='bold')

        plt.tight_layout()
        self._save_plot(f'top_{top_n}_cities.png', plot_type='obligated')

    # Additional insightful plots
    def plot_monthly_revenue_trend(self):
        """Plot monthly revenue trend over time"""
        if self.df is None:
            logger.warning("DataFrame is None, skipping plot_monthly_revenue_trend.")
            return

        plt.figure(figsize=(16, 8))
        self.df['order_date'] = pd.to_datetime(self.df['order_date'])
        self.df['order_year'] = self.df['order_date'].dt.year
        self.df['order_month'] = self.df['order_date'].dt.month

        monthly_revenue = self.df.groupby(['order_year', 'order_month'])['net_revenue'].sum().reset_index()

        monthly_revenue['date'] = pd.to_datetime(monthly_revenue['order_year'].astype(str) + '-' +
                                                 monthly_revenue['order_month'].astype(str) + '-01')

        plt.plot(monthly_revenue['date'], monthly_revenue['net_revenue'],
                marker='o', linewidth=2, markersize=6, color='darkblue')
        plt.title('Monthly Revenue Trend', fontsize=16, fontweight='bold')
        plt.xlabel('Date', fontsize=12)
        plt.ylabel('Revenue ($)', fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.xticks(rotation=45)

        z = np.polyfit(range(len(monthly_revenue)), monthly_revenue['net_revenue'], 1)
        p = np.poly1d(z)
        plt.plot(monthly_revenue['date'], p(range(len(monthly_revenue))),
                "r--", alpha=0.8, linewidth=2, label=f'Trend Line')
        plt.legend()

        plt.tight_layout()
        self._save_plot('monthly_revenue_trend.png', plot_type='additional')

    def plot_revenue_by_quarter(self):
        """Plot revenue by quarter"""
        if self.df is None:
            logger.warning("DataFrame is None, skipping plot_revenue_by_quarter.")
            return

        plt.figure(figsize=(12, 8))
        # Ensure order_quarter is available, if not, create it
        if 'order_quarter' not in self.df.columns:
            self.df['order_quarter'] = self.df['order_date'].dt.quarter
            logger.info("Created 'order_quarter' column as it was missing.")

        quarterly_revenue = self.df.groupby('order_quarter')['net_revenue'].sum()

        ax = quarterly_revenue.plot(kind='bar', color='teal', edgecolor='black')
        plt.title('Revenue by Quarter', fontsize=16, fontweight='bold')
        plt.xlabel('Quarter', fontsize=12)
        plt.ylabel('Revenue ($)', fontsize=12)
        plt.xticks(rotation=0)

        for i, v in enumerate(quarterly_revenue.values):
            ax.text(i, v + max(quarterly_revenue) * 0.01, f'${v:,.0f}',
                   ha='center', va='bottom', fontweight='bold')

        plt.tight_layout()
        self._save_plot('revenue_by_quarter.png', plot_type='additional')

    def plot_product_performance_matrix(self):
        """Plot product performance matrix (Revenue vs Quantity)"""
        if self.df is None:
            logger.warning("DataFrame is None, skipping plot_product_performance_matrix.")
            return

        plt.figure(figsize=(14, 10))

        product_perf = self.df.groupby('product_name').agg({
            'net_revenue': 'sum',
            'order_quantity': 'sum',
            'category_name': 'first'
        }).reset_index()

        categories = product_perf['category_name'].unique()
        colors = plt.cm.Set1(np.linspace(0, 1, len(categories)))

        for i, category in enumerate(categories):
            cat_data = product_perf[product_perf['category_name'] == category]
            plt.scatter(cat_data['order_quantity'], cat_data['net_revenue'],
                        alpha=0.7, s=60, color=colors[i], label=category, edgecolors='black')

        plt.title('Product Performance Matrix\n(Revenue vs Quantity Sold)', fontsize=16, fontweight='bold')
        plt.xlabel('Total Quantity Sold', fontsize=12)
        plt.ylabel('Total Revenue ($)', fontsize=12)
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True, alpha=0.3)

        plt.tight_layout()
        self._save_plot('product_performance_matrix.png', plot_type='additional')

    def plot_customer_type_analysis(self):
        """Plot customer type analysis"""
        if self.df is None:
            logger.warning("DataFrame is None, skipping plot_customer_type_analysis.")
            return

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

        customer_type_revenue = self.df.groupby('person_type')['net_revenue'].sum()
        ax1.pie(customer_type_revenue.values, labels=customer_type_revenue.index,
                autopct='%1.1f%%', startangle=90, colors=plt.cm.Pastel1.colors)
        ax1.set_title('Revenue by Customer Type', fontsize=14, fontweight='bold')

        avg_order_value = self.df.groupby('person_type')['net_revenue'].mean()
        bars = ax2.bar(avg_order_value.index, avg_order_value.values,
                       color='lightblue', edgecolor='black')
        ax2.set_title('Average Order Value by Customer Type', fontsize=14, fontweight='bold')
        ax2.set_xlabel('Customer Type')
        ax2.set_ylabel('Average Order Value ($)')

        for bar in bars:
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height + max(avg_order_value) * 0.01,
                    f'${height:,.0f}', ha='center', va='bottom', fontweight='bold')

        plt.tight_layout()
        self._save_plot('customer_type_analysis.png', plot_type='additional')

    def plot_territory_performance(self):
        """Plot territory performance comparison"""
        if self.df is None:
            logger.warning("DataFrame is None, skipping plot_territory_performance.")
            return

        plt.figure(figsize=(14, 8))
        territory_revenue = self.df.groupby('territory_name')['net_revenue'].sum().sort_values(ascending=False)

        ax = territory_revenue.plot(kind='bar', color='darkorange', edgecolor='black')
        plt.title('Revenue by Sales Territory', fontsize=16, fontweight='bold')
        plt.xlabel('Territory', fontsize=12)
        plt.ylabel('Revenue ($)', fontsize=12)
        plt.xticks(rotation=45, ha='right')

        for i, v in enumerate(territory_revenue.values):
            ax.text(i, v + max(territory_revenue) * 0.01, f'${v:,.0f}',
                   ha='center', va='bottom', fontweight='bold')

        plt.tight_layout()
        self._save_plot('territory_performance.png', plot_type='additional')

    def generate_summary_stats(self):
        """Generate and display summary statistics"""
        if self.df is None:
            logger.warning("DataFrame is None, skipping summary statistics generation.")
            return

        logger.info("=" * 50)
        logger.info("GOLD LAYER DATA SUMMARY STATISTICS")
        logger.info("=" * 50)

        logger.info(f"Total Records: {len(self.df):,}")
        
        # Check if 'order_date' column exists and is not empty before calling min/max
        if 'order_date' in self.df.columns and not self.df['order_date'].empty:
            logger.info(f"Date Range: {self.df['order_date'].min()} to {self.df['order_date'].max()}")
        else:
            logger.warning("Order date column is missing or empty, date range not available.")

        logger.info(f"Total Revenue: ${self.df['net_revenue'].sum():,.2f}")
        logger.info(f"Average Order Value: ${self.df['net_revenue'].mean():,.2f}")
        logger.info(f"Total Orders: {self.df['sales_order_id'].nunique():,}")
        logger.info(f"Unique Customers: {self.df['customer_id'].nunique():,}")
        logger.info(f"Unique Products: {self.df['product_id'].nunique():,}")
        logger.info(f"Countries: {self.df['country_name'].nunique()}")
        logger.info(f"States: {self.df['state_name'].nunique()}")
        logger.info(f"Cities: {self.df['city'].nunique()}")

        logger.info("\nTop 5 Revenue Generating:")
        # Check if columns exist before grouping
        if 'category_name' in self.df.columns and 'net_revenue' in self.df.columns:
            logger.info(f"Categories: {list(self.df.groupby('category_name')['net_revenue'].sum().sort_values(ascending=False).head().index)}")
        else:
            logger.warning("Category name or net revenue column missing, skipping top categories.")
        
        if 'country_name' in self.df.columns and 'net_revenue' in self.df.columns:
            logger.info(f"Countries: {list(self.df.groupby('country_name')['net_revenue'].sum().sort_values(ascending=False).head().index)}")
        else:
            logger.warning("Country name or net revenue column missing, skipping top countries.")

        if 'state_name' in self.df.columns and 'net_revenue' in self.df.columns:
            logger.info(f"States: {list(self.df.groupby('state_name')['net_revenue'].sum().sort_values(ascending=False).head().index)}")
        else:
            logger.warning("State name or net revenue column missing, skipping top states.")

    def generate_all_plots(self):
        """Generate all the requested plots plus additional ones"""
        logger.info("Generating comprehensive visualizations...")

        self.generate_summary_stats()

        logger.info("\nGenerating Obligated Plots:")
        self.plot_revenue_by_category()
        self.plot_top_subcategories(10)
        self.plot_top_customers(10)
        self.plot_revenue_by_order_status()
        self.plot_top_countries(10)
        self.plot_top_states(10)
        self.plot_top_cities(10)

        logger.info("\nGenerating Additional Insightful Plots:")
        self.plot_monthly_revenue_trend()
        self.plot_revenue_by_quarter()
        self.plot_product_performance_matrix()
        self.plot_customer_type_analysis()
        self.plot_territory_performance()

        logger.info("\nAll visualizations generated successfully!")

# Usage example
if __name__ == "__main__":
    gold_dir = 'data/Gold'
    comprehensive_data_path = Path(gold_dir) / 'revenue_analysis_comprehensive'

    visualizer = GoldLayerVisualizer(data_path=comprehensive_data_path)
    visualizer.generate_all_plots()