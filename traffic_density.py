

class TrafficDensityEstimator:

    def compute(self, vehicles, road_area):

        if road_area <= 0:

            raise ValueError("Road area must be greater than zero.")

        vehicle_area = 0

        for vehicle in vehicles:

            width = vehicle["width"]

            height = vehicle["height"]

            vehicle_area += width * height

        density = vehicle_area / road_area

        density = min(density, 1.0)

        return {

            "vehicle_area": vehicle_area,

            "road_area": road_area,

            "density": density

        }