import math
import pandas as pd

# Constants
d = 10  # Distance between each query point in meters
d_km = d / 1000  # Convert to kilometers for consistency with R

# List of radii in kilometers
radii = [1, 5, 10, 11, 12]

# Calculate N for each radius (using the correct formula) and round to the nearest integer
N_values = [round((math.pi * r**2) / (d_km**2)) for r in radii]

# Create a DataFrame for displaying the results
df = pd.DataFrame({
    'Radius (km)': radii,
    'Number of Points (N)': N_values
})

# Display the table
print(df)
