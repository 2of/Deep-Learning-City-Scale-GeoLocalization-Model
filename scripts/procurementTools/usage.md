#### Rough and dirty guide to using these files.

1. Define a location (i.e. chicago is   41.8781 -87.6298  lat long

2. use download shape file to get the shapefile for the maps, you'll need to hop into the code ot change it manually as it uses explicit schema not lat long

3. then run points_from_shapefile.py 
        Change these lines:
            gdf = gpd.read_file("chicago_.shp")
            points_df.to_csv("chicago_road_points.csv", index=False)

4. Now we have the entire set of lat long coords, we filter them by radius because at d = 2, (distnce between each point) we STILL end up at about 1m points

    Run filter_points_to_radius 
                "Usage: python filter_points.py <input_csv> <output_csv> <center_lat> <center_lon> <radius>"

5.  Call mapvis to visualise the points and inspect for anomylies 

a.render(41.8781, -87.6298, True, './filtered_chicago_points.csv', radius=8000)

