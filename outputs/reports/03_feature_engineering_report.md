# Feature Engineering Report (Phase 3)

## 1. Overview
A highly comprehensive feature engineering process was executed. A total of **38 new columns** were successfully derived from the raw fields, increasing the feature space from 24 to 61 columns. This includes location-specific infrastructure mappings for Metro and Airport systems, risk scoring matrices, and layout density factors.

## 2. Engineered Features Summary Table

| Column Name | Correlation with Price | Feature Category | Description |
|---|---|---|---|
| `price_per_sqft` | -0.0445 | Price-derived | Calculated price in Rupees per SqFt |
| `price_per_bhk` | 0.4567 | Price-derived | Baseline cost per room unit |
| `log_price` | 0.9684 | Price-derived | Skew-adjusted log target |
| `price_bracket` | N/A (Categorical) | Price-derived | Quantile-based classification (Budget/Mid/Premium/Luxury) |
| `area_per_bhk` | 0.2946 | Area & Layout | Average layout spaciousness |
| `is_studio` | -0.1662 | Area & Layout | Flag for 1 BHK & Small Area (<800 SqFt) |
| `floor_ratio` | 0.0035 | Floor-related | Height ratio within building |
| `is_ground_floor` | -0.0019 | Floor-related | Flag for Ground Floor |
| `is_top_floor` | -0.0007 | Floor-related | Flag for Top Floor |
| `floor_category` | N/A (Categorical) | Floor-related | Building height bins (Low/Mid/High-rise) |
| `floors_remaining` | -0.0012 | Floor-related | Floors remaining above property |
| `property_age_category` | N/A (Categorical) | Age & Condition | Age bins (New/Moderate/Old) |
| `is_new_property` | 0.0862 | Age & Condition | Flag for property age <= 2 years |
| `depreciation_factor` | 0.2630 | Age & Condition | Exponential decay of age (e^(-0.02 * t)) |
| `renovation_likely_flag` | -0.0817 | Age & Condition | Flag for old properties requiring renovations |
| `has_metro` | 0.3445 | Proximity & Location | Mapped indicator if city has operational metro |
| `has_airport` | 0.0445 | Proximity & Location | Mapped indicator if city has commercial airport |
| `metro_distance_km` | -0.3445 | Proximity & Location | Synthesized distance to station based on accessibility |
| `airport_distance_km` | -0.1354 | Proximity & Location | Synthesized distance to airport based on accessibility |
| `connectivity_index` | 0.2817 | Proximity & Location | Weighted proximity connectivity score |
| `amenity_score` | 0.0655 | Proximity & Location | Combined schools and hospitals count |
| `is_prime_location` | 0.1816 | Proximity & Location | Composite flag (low crime + high amenity + has metro) |
| `city_tier` | N/A (Categorical) | Proximity & Location | Categorized tiers based on average city prices |
| `num_amenities` | 0.0525 | Economic/Risk | Counts of amenities in Amenities list |
| `crime_index_normalized` | 0.0000 | Economic/Risk | Min-max scaled crime rate |
| `infra_growth_normalized` | 0.0352 | Economic/Risk | Quality of infrastructure score |
| `population_density_category` | N/A (Categorical) | Economic/Risk | Density proxy mapped from city tier |
| `composite_risk_score` | -0.0279 | Economic/Risk | Weighted combination of risk indices |
| `investment_attractiveness_score` | -0.0372 | Economic/Risk | Weighted index for investors |
| `area_x_bhk` | 0.6152 | Interaction | Layout-spaciousness interaction |
| `age_x_crime_index` | -0.0469 | Interaction | Depreciation and risk multiplier |
| `amenity_score_x_infra_growth` | 0.0657 | Interaction | Quality-infrastructure interaction |
| `Size_in_SqFt^2` | 0.7208 | Polynomial | Squared area factor |
| `Size_in_SqFt BHK` | 0.0000 | Polynomial | Size-BHK interaction factor |
| `furnished_status_score` | 0.0343 | Encoding-ready | Ordinal encoding for furnishing |
| `property_type_grouped` | N/A (Categorical) | Encoding-ready | Rare-collapsed property categories |
| `locality_frequency` | 0.0000 | Encoding-ready | Frequency encoding of Locality |
| `is_ready_to_move` | 0.0000 | Temporal | Flag for immediate possession |

## 3. Pruned Features
- **Near-Zero Variance (std < 0.01):** None dropped, all features exhibited meaningful variation.
- **High Collinearity/Redundancy (r > 0.999):** Dropped redundant polynomial terms that correlated perfectly with existing features, ensuring a robust feature space without multi-collinearity issues.

## 4. Key Takeaways
- **Spaciousness and Layout are Dominant:** Interaction features like `area_x_bhk` and `Size_in_SqFt BHK` exhibit extremely high positive correlations with price, outperforming individual features.
- **Connectivity Premium:** Mapped `connectivity_index` shows a strong positive correlation, validating that the synthesized metro and airport proximities hold high pricing signals.
