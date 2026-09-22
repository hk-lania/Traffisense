from traffic_density import TrafficDensityEstimator

density = TrafficDensityEstimator()

vehicles = [

    {
        "width":60,
        "height":40
    },

    {
        "width":75,
        "height":45
    },

    {
        "width":80,
        "height":50
    }

]

road_area = 800 * 600

result = density.compute(

    vehicles,

    road_area

)

print()

print("Vehicle Area :", result["vehicle_area"])

print("Road Area :", result["road_area"])

print("Traffic Density :", result["density"])