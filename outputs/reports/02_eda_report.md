# Exploratory Data Analysis Report (Phase 2)

## 1. Executive Summary
A thorough exploratory data analysis was conducted on the cleaned dataset of 250,000 real estate records. A total of 12 high-resolution visualizations were generated to analyze univariate distributions, bivariate correlations, and multivariate relationships. All plots have been saved to disk, including interactive assets.

## 2. Top 5 Key Insights

### Insight 1: Size in SqFt is the Dominant Price Driver
- **Observation:** The scatter plot between property size and price shows a strong, highly linear positive correlation. As expected, physical area is the single largest determinant of property value.
- **Visual Reference:** [Size vs Price Scatter](../plots/eda/size_vs_price_scatter.png) and [Interactive Scatter Plot](../plots/eda/size_vs_price_plotly_interactive.html)

### Insight 2: Step-Function Layout Premiums (BHK)
- **Observation:** Property price exhibits a clear step-up behavior with each additional BHK. The median price increases consistently from 1 BHK to 5 BHK, with the interquartile ranges shifting upwards, signifying distinct market segments.
- **Visual Reference:** [BHK vs Price Boxplot](../plots/eda/bhk_vs_price_boxplot.png)

### Insight 3: Location Casing & Baseline City Tiers
- **Observation:** Real estate pricing is heavily localized. Baseline prices vary significantly by city. The highest average prices are found in tier-1 metropolises, showing a steep premium compared to tier-2 and tier-3 cities.
- **Visual Reference:** [City vs Price Boxplot](../plots/eda/city_vs_price_boxplot.png)

### Insight 4: Property Age Depreciation Curve
- **Observation:** There is a notable negative trend line in the Property Age vs Price scatter plot. Older properties sell at a discount compared to brand-new structures, highlighting the impact of physical wear-and-tear and lack of modern amenities.
- **Visual Reference:** [Age vs Price Scatter](../plots/eda/age_vs_price_scatter.png)

### Insight 5: Infrastructure & Convenience Premium
- **Observation:** Properties surrounded by higher densities of nearby schools and hospitals exhibit a clear price premium. Buyers are willing to pay extra for social infrastructure and immediate accessibility to schooling and healthcare.
- **Visual Reference:** [Nearby Schools vs Price](../plots/eda/schools_vs_price_boxplot.png) and [Nearby Hospitals vs Price](../plots/eda/hospitals_vs_price_boxplot.png)

## 3. Visual Assets Directory
All generated plots are saved in `outputs/plots/eda/`:
- `price_in_lakhs_distribution.png` - Target price density
- `size_in_sqft_distribution.png` - Property size distribution
- `age_of_property_distribution.png` - Age distribution
- `property_type_distribution.png` - Property categories count
- `bhk_distribution.png` - BHK frequency
- `furnished_status_distribution.png` - Furnishing states
- `city_distribution_top15.png` - Top 15 cities count
- `size_vs_price_scatter.png` - Price vs Size linear trend
- `bhk_vs_price_boxplot.png` - BHK pricing ranges
- `city_vs_price_boxplot.png` - Top 10 cities baseline pricing
- `age_vs_price_scatter.png` - Property age vs Price depreciation trend
- `schools_vs_price_boxplot.png` - School density impact
- `hospitals_vs_price_boxplot.png` - Hospital density impact
- `correlation_heatmap.png` - Full numeric feature correlations
- `pair_plot_top_features.png` - Combined KDE & scatter pair plots
- `avg_price_per_bhk_per_city.png` - Grouped bar comparisons
- `size_vs_price_plotly_interactive.html` - Interactive Plotly web asset
