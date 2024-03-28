from PyRT_Common import *
from random import randint
# from AppWorkbench import *

# -------------------------------------------------
# Integrator Classes
# -------------------------------------------------
# The integrators also act like a scene class in that-
# it stores all the primitives that are to be ray traced.
# -------------------------------------------------
class Integrator(ABC):
    # Initializer - creates object list
    def __init__(self, filename_, experiment_name=''):
        # self.primitives = []
        self.filename = filename_ + experiment_name
        # self.env_map = None  # not initialized
        self.scene = None

    @abstractmethod
    def compute_color(self, ray):
        pass

    # def add_environment_map(self, env_map_path):
    #    self.env_map = EnvironmentMap(env_map_path)
    def add_scene(self, scene):
        self.scene = scene

    def get_filename(self):
        return self.filename

    # Simple render loop: launches 1 ray per pixel
    def render(self):
        # YOU MUST CHANGE THIS METHOD IN ASSIGNMENTS 1.1 and 1.2:
        cam = self.scene.camera  # camera object
        # ray = Ray()
        print('Rendering Image: ' + self.get_filename())
        for x in range(0, cam.width):
            for y in range(0, cam.height):
                # pixel = RGBColor(x/cam.width, y/cam.height, 0)
                pixel = self.compute_color(Ray(Vector3D(0.0, 0.0, 0.0), cam.get_direction(x,y)))
                self.scene.set_pixel(pixel, x, y)  # save pixel to pixel array
            progress = (x / cam.width) * 100
            print('\r\tProgress: ' + str(progress) + '%', end='')
        # save image to file
        print('\r\tProgress: 100% \n\t', end='')
        full_filename = self.get_filename()
        self.scene.save_image(full_filename)


class LazyIntegrator(Integrator):
    def __init__(self, filename_):
        super().__init__(filename_ + '_Lazy')

    def compute_color(self, ray):
        return BLACK


class IntersectionIntegrator(Integrator):

    def __init__(self, filename_):
        super().__init__(filename_ + '_Intersection')

    def compute_color(self, ray):
        # ASSIGNMENT 1.2: PUT YOUR CODE HERE
        if self.scene.any_hit(ray):
            return RED
        return BLACK


class DepthIntegrator(Integrator):

    def __init__(self, filename_, max_depth_=10):
        super().__init__(filename_ + '_Depth')
        self.max_depth = max_depth_

    def compute_color(self, ray):
        # ASSIGNMENT 1.3: PUT YOUR CODE HERE
        hit_data = self.scene.closest_hit(ray)
        c_i = max(1- (hit_data.hit_distance/self.max_depth), 0)
        return RGBColor(c_i,c_i,c_i)


class NormalIntegrator(Integrator):

    def __init__(self, filename_):
        super().__init__(filename_ + '_Normal')

    def compute_color(self, ray):
        # ASSIGNMENT 1.3: PUT YOUR CODE HERE
        hit_data = self.scene.closest_hit(ray)
        if hit_data.has_hit:
            colors = (hit_data.normal+Vector3D(1.0,1.0,1.0))/2
            return RGBColor(colors.x, colors.y, colors.z)
        return BLACK


class PhongIntegrator(Integrator):

    def __init__(self, filename_):
        super().__init__(filename_ + '_Phong')

    def compute_color(self, ray):
        # ASSIGNMENT 1.4: PUT YOUR CODE HERE
        hit_data=self.scene.closest_hit(ray)

        if hit_data.has_hit:
            # We get the labertian BRDF 
            lambertian = self.scene.object_list[hit_data.primitive_index].get_BRDF()

            # Computing the Amibient Term using k_d from the Lambertian and the ambient intensity (i_a)
            L_a= lambertian.kd.multiply(self.scene.i_a)

            L_d = RGBColor(0,0,0)

            w_o = ray.d

            for light in self.scene.pointLights:

                # Create a the ray path from the hit point to the light source position
                w_i = light.pos - hit_data.hit_point
                ray_l = Ray(hit_data.hit_point, w_i)
                
                # Compute the distance between the hit point and the light source position
                distance = Length(w_i)

                # Computing the closest hitting point of the ray
                hit_closest = self.scene.closest_hit(ray_l)
                
                # Filter not visible points (if not occluded)
                if not self.scene.any_hit(ray_l) or hit_closest.hit_distance >= distance:
                    L_i = light.intensity / (distance * distance)
                    # Computing L_d 
                    L_d += L_i.multiply(lambertian.get_value(w_i, w_o, hit_data.normal))

            return L_a+L_d
            
        return BLACK


class CMCIntegrator(Integrator):  # Classic Monte Carlo Integrator

    def __init__(self, n, filename_, experiment_name='', pdf = UniformPDF()):
        filename_mc = filename_ + '_MC_' + str(n) + '_samples' + experiment_name
        super().__init__(filename_mc)
        self.n_samples = n
        self.pdf = pdf

    def compute_color(self, ray):
        hit_data=self.scene.closest_hit(ray)

        if hit_data.has_hit:
            kd = self.scene.object_list[hit_data.primitive_index].get_BRDF().kd
            # Generate a set of samples (S) distributed over the hemisphere according to a given PDF
            sample_set, sample_prob = sample_set_hemisphere(self.n_samples, self.pdf)

            colors = []
            # For each sample w_j in sample set (S)
            for j, w_j in enumerate(sample_set):
                # Center the sample around the surface normal. yielding w_j_a
                w_j_a = center_around_normal(w_j, hit_data.normal)
                # Create a secondary array r with direction w_j_a
                r = Ray(origin = hit_data.hit_point, direction = w_j_a)
                # Computing the closest hitting point of the ray
                hit_closest = self.scene.closest_hit(r)

                # If r hits the scene geometry, then
                if hit_closest.has_hit:
                    Li_wj = self.scene.object_list[hit_closest.primitive_index].emission
                elif self.scene.env_map:
                    Li_wj = self.scene.env_map.getValue(w_j_a)

                color = Li_wj.multiply(kd)
                color = color*Dot(hit_data.normal,w_j_a)
                colors.append(color)
            return compute_estimate_cmc(sample_prob, colors)
        elif self.scene.env_map:
            return self.scene.env_map.getValue(ray.d)
        return BLACK

class BayesianMonteCarloIntegrator(Integrator):
    def __init__(self, n, myGP, filename_, experiment_name=''):
        filename_bmc = filename_ + '_BMC_' + str(n) + '_samples' + experiment_name
        super().__init__(filename_bmc)
        self.n_samples = n
        self.myGP = myGP

    def compute_color(self, ray):
        pass

def compute_estimate_cmc(sample_prob_, sample_values_):
    # TODO: PUT YOUR CODE HERE
    N = len(sample_values_)
    I = BLACK
    for j in range(0,N):
        I += sample_values_[j]/sample_prob_[j]    
    p = 0
    return I/N
