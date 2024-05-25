//First idea to construct the tree is have each thread try to insert a particle into it and 
//have fencing operations within the tree

/**
 *  Basic idea of this implementation:
 *  1. Create particle data on the GPU
 *  2. Transfer to host and save
 *  3. Then, for each time step that we want:
 *      On CPU:
 *          1. Create initial box
 *      On GPU:
 *          1. Build the tree by inserting each particle into octree
 *          2. Compute forces
 *          3. Update velocities and positions
 *      On CPU:
 *          1. Transfer data back and save to JSON
 *  
 * 
*/
#include <cmath>
#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <random>
#include <cuda_runtime.h>

#define CUDA_ERR_CHECK(err, msg) \
        if ((err) != cudaSuccess) { \
            printf("CUDA ERROR: %s Error code %d\n", msg, err); \
            return false;\
        } \

#define CUDA_ERR_CHECK_NO_R(err, msg, r_value) \
        if ((err) != cudaSuccess) { \
            printf("CUDA ERROR: %s Error code %d\n", msg, err); \
            return r_value;\
        } \

#define EARTH_MASS 5.972e24
#define SUN_MASS 1.989e30
#define MARS_MASS 6.39e23

#define KG_TO_SOLAR_MASS 1.989e30 
#define KM_TO_AU 1.496e8
#define HOURS_TO_YEARS 8760

#define MAX_DEPTH 16
// #define INITIAL_VELOCITY_SCALE 0.25

#define AU_TO_PARSECS 206300

const float PI = 3.14159265359;
const float G = 4 * PI * PI / AU_TO_PARSECS;
const float delta = 10.0;

__constant__ float theta[1];

size_t frames = 1000;
std::vector<float> times;

const uint32_t OCTANT_ONE = 0;
const uint32_t OCTANT_TWO = 1;
const uint32_t OCTANT_THREE = 2;
const uint32_t OCTANT_FOUR = 3;
const uint32_t OCTANT_FIVE = 4;
const uint32_t OCTANT_SIX = 5;
const uint32_t OCTANT_SEVEN = 6;
const uint32_t OCTANT_EIGHT = 7;

//Everything needed is in this single file
struct Vector{
    float x;
    float y;
    float z;

    __host__ void print(){
        std::cout<<"x: "<<x<<" | y: " <<y<<" | z: "<<z<<std::endl;
    }
    __device__ void print_device(){
        printf("x: %.9f | y: %.9f | z: %.9f | %d\n",x,y,z,threadIdx.x);
    }

    __host__ __device__ Vector operator+(const Vector& other) const {
        Vector result;
        result.x = this->x + other.x;
        result.y = this->y + other.y;
        result.z = this->z + other.z;
        return result;
    }

    __host__ __device__ Vector operator*(const float& other) const{
        Vector result;
        result.x = this->x * other;
        result.y = this->y * other;
        result.z = this->z * other;
        return result;
    }

    __host__ __device__ Vector& operator+=(const Vector& other){
        this->x += other.x;
        this->y += other.y;
        this->z += other.z;
        return *this;
    }

    __host__ __device__ Vector operator/(const float scale){
        if(scale == 0){
            return *this;
        }
        return *this * (1.0/scale);
    }

    __host__ __device__ bool operator==(const Vector& other){
        return this->x == other.x && this->y == other.y && this->z == other.z;
    }

    __host__ __device__ float operator()(const Vector& other){
        Vector distance_vector = {other.x - this->x,
                            other.y - this->y,
                            other.z - this->z};
    
        float distance_squared = distance_vector.x * distance_vector.x +
                                distance_vector.y * distance_vector.y + 
                                distance_vector.z * distance_vector.z;
        
        float distance = sqrt(distance_squared);

        return distance;
    }

    //Used for sorting
    __device__ bool operator < (const Vector& other){
        return (x * x) + (y * y) < (other.x * other.x) + (other.y * other.y);
    }
};

struct Star{
    
    Vector position;
    Vector velocity;
    Vector acceleration;

    float mass;
    bool ignored;

    int merged;
    __host__ __device__ Star(){this->merged = 0;}

    __host__ __device__  Star(Vector position, Vector velocity, Vector acceleration, float mass){
        this->position = position;
        this->velocity = velocity;
        this->acceleration = acceleration;
        this->mass = mass;
        this->merged = 0;
    }

    __host__ __device__ bool operator==(const Star& other){
        return this->position == other.position;
    }

    __host__ void print(){
        std::cout<<"Position: ";
        position.print();
    }

    __device__ void print_device(){
        printf("Position: ");
        position.print_device();
        printf("Velocity: ");
        velocity.print_device();
        printf("Acceleration: ");
        acceleration.print_device();

        printf("Mass: %f\n Merged: %d\n\n", mass, merged);
    }

    __device__ bool merge_with(Star* other){
            float new_mass = this->mass + other->mass;
            Vector new_position = (other->position * other->mass + this->position * this->mass) / (this->mass + other->mass);
            Vector new_velocity = (other->velocity * other->mass + this->velocity * this->mass) / (this->mass + other->mass);

            this->mass = new_mass;
            this->position = new_position;
            this->velocity = new_velocity;
    }

};

__constant__ float delta_gpu = 25.0f;

__device__ Vector compute_force_device(Star one, Star two, float delta){
    Vector distance_vector = {two.position.x - one.position.x,
                            two.position.y - one.position.y,
                            two.position.z - one.position.z};
    
    float distance_squared = distance_vector.x * distance_vector.x +
                            distance_vector.y * distance_vector.y + 
                            distance_vector.z * distance_vector.z;
    
    float distance = sqrt(distance_squared);

    float force = (G) * one.mass * two.mass / (distance_squared);
    // printf("%.9f and %.9f creates %.9f\n", one.mass, two.mass, force);

    Vector force_vector = {distance_vector.x/distance, distance_vector.y/distance, distance_vector.z/distance};
    force_vector = force_vector * force;

    Vector relative_acceleration = {force_vector.x/one.mass, force_vector.y/one.mass, force_vector.z/one.mass};

    return relative_acceleration;
}

bool save_to_json(std::vector<Star>& bodies, const std::string& filename){
    //Returns true if successfully save and false otherwise
    std::ofstream file(filename, std::ios::out | std::ios::trunc);

    if(!file.is_open()){
        std::cerr << "Error: Unable to open file for writing: " <<filename << std::endl;
        return false;
    }

    file << "{\n";
    file << " \"bodies\": [\n";

    for(size_t i = 0; i < bodies.size(); ++i){
        const auto& body = bodies[i];

        //Now we have to write to the file
        file << "{\n";
        file << "       \"position\": {\"x\": " << body.position.x
                                  <<", \"y\": " << body.position.y
                                  <<", \"z\": " << body.position.z
                                  <<"}\n";
        file << "   }";

        if(i != bodies.size() - 1){
            file << ",";
        }
        file << "\n";
    }

    file << "   ]\n";
    file <<"}\n";

    file.close();
    
    return true;
}

bool save_times_2(std::vector<float>& times, int main, int side, int total_frames, float host_theta, const std::string& filename) {
    // Returns true if successfully save and false otherwise
    std::ofstream file(filename, std::ios::out | std::ios::trunc);

    if (!file.is_open()) {
        std::cerr << "Error: Unable to open file for writing: " << filename << std::endl;
        return false;
    }

    file << "{\n";
    file << " \"settings\": {\n";
    file << "    \"main\": " << main << ",\n";
    file << "    \"side\": " << side << ",\n";
    file << "    \"total_frames\": " << total_frames << ",\n";
    file << "    \"host_theta\": " << host_theta << "\n";
    file << " },\n";
    file << " \"times\": [\n";

    for (size_t i = 0; i < times.size(); ++i) {
        // Write time along with settings
        file << "  {\n";
        file << "   \"time\": " << times[i] << "\n";
        file << "  }";

        if (i != times.size() - 1) {
            file << ",";
        }
        file << "\n";
    }

    file << " ]\n";
    file << "}\n";

    file.close();

    return true;
}

bool save_times(std::vector<float>& times, const std::string& filename){
    //Returns true if successfully save and false otherwise
    std::ofstream file(filename, std::ios::out | std::ios::trunc);

    if(!file.is_open()){
        std::cerr << "Error: Unable to open file for writing: " <<filename << std::endl;
        return false;
    }

    file << "{\n";
    file << " \"times\": [\n";

    for(size_t i = 0; i < times.size(); ++i){

        //Now we have to write to the file
        file << "{\"time\": " << times[i]
                                  <<"}\n";

        if(i != times.size() - 1){
            file << ",";
        }
        file << "\n";
    }

    file << "   ]\n";
    file <<"}\n";

    file.close();
    
    return true;
}

Node* create_root(std::vector<Star> stars){

    //lowest y, highest z and x
    Vector bottom_right = stars[0].position;

    //lowest z and x, highest y
    Vector top_left = stars[0].position;

    for(int i = 0; i < stars.size(); i++){
        Vector position = stars[i].position;

        float x = position.x; float y = position.y; float z = position.z;
        if(x > bottom_right.x){
            bottom_right.x = x;
        } else
        if(x < top_left.x){
            top_left.x = x;
        }

        if(y > top_left.y){
            top_left.y = y;
        } else
        if(y < bottom_right.y){
            bottom_right.y = y;
        }

    }

    bottom_right.z = 0;
    top_left.z = 0;

    Vector center = (bottom_right + top_left) * 0.5f;

    float length = (bottom_right.x - top_left.x > top_left.y - bottom_right.y) ? 
        (bottom_right.x - top_left.x > bottom_right.z - top_left.z) ?
            bottom_right.x - top_left.x : bottom_right.z - top_left.z
    :
        (top_left.y - bottom_right.y > bottom_right.z - top_left.z) ?
            top_left.y - bottom_right.y : bottom_right.z - top_left.z;

    return (new Node(center.x, center.y, center.z, length));
}

struct VelocityCurve{
        
        static float MS(float r, float core_rad){
            float d = core_rad/2.0f;
            float rho_so = 1.0f;
            float rH = core_rad/2.0f;
            return rho_so*exp(-r/rH) * (r * r) * PI * d;
        }

        static double MH(float r, float core_rad){
            float rho = .15f;
            float rC = 25.0f;
            return rho * (1/(1 + pow(r/rC, 2) * (4 * PI * pow(r,3)/3)));
        }

        static float v(float r, float core_rad){
            float MZ = 100.0f;
            float G = 6.672f * pow(10, -11);

            return 20000 * sqrt((float)(G * (MH(r, core_rad) + MS(r, core_rad) + MZ)/r));

        }

        static float vd(float r, float core_rad){
            float MZ = 100.0f;
            float G = 6.672f * pow(10, -11);
            return 20000 * sqrt((G*MS(r, core_rad) + MZ)/r);
        }

    };

//This section is ported from my work with my sequential galaxy sim in unity:
//https://github.com/NocaToca/Unity-Galaxy/blob/main/DistributionFunction.cs
class DistributionFunction{
    float max_intensity;
    float bulge;
    float disc_scale_length;
    float bulge_rad;
    float start_int;
    float end_int;
    float angular_offset;
    int number_of_points;

    std::vector<float> M1;
    std::vector<float>  X1;
    std::vector<float>  Y1;

    std::vector<float>  M2;
    std::vector<float>  X2;
    std::vector<float>  Y2;

    int two_counts;
    int one_counts;

    // Additional variables
    float gal_rad;
    float core_rad;
    float core_edge_ecc;
    float gal_edge_ecc;
    float orbital_vel_core;
    float orbital_vel_gal;
    bool dark_matter;
    int perturbations;
    int amplitude_pert;
    int dust_render_size;
    int HII_size;
    float dust_temp_base;
    int steps;
    float velocity_scaling;
    int color_offset;
    float star_base_temp;
    int num_HII;
    int number_of_stars;

    float rad_far_feild;

    Vector offset;

    public:
    // Very large constructor!
    DistributionFunction(
        float max_intensity,float bulge, float disc_scale_length, float bulge_rad, float start_int, float end_int, float angular_offset, int number_of_points,
        // Additional parameters
        float gal_rad, float core_rad, float core_edge_ecc, float gal_edge_ecc, float orbital_vel_core, float orbital_vel_gal, bool dark_matter,
        int perturbations, int amplitude_pert, int dust_render_size, int HII_size, float dust_temp_base, int steps, float velocity_scaling, int color_offset, float star_base_temp, int num_HII,
        float rad_far_feild, int number_of_stars,
        Vector offset
    ) {
        // Assign passed-in values to class variables
        this->max_intensity = max_intensity;
        this->bulge = bulge;
        this->disc_scale_length = disc_scale_length;
        this->bulge_rad = bulge_rad;
        this->start_int = start_int;
        this->end_int = end_int;
        this->angular_offset = angular_offset;
        this->number_of_points = number_of_points;

        // Additional variables
        this->gal_rad = gal_rad;
        this->core_rad = core_rad;
        this->core_edge_ecc = core_edge_ecc;
        this->gal_edge_ecc = gal_edge_ecc;
        this->orbital_vel_core = orbital_vel_core;
        this->orbital_vel_gal = orbital_vel_gal;
        this->dark_matter = dark_matter;
        this->perturbations = perturbations;
        this->amplitude_pert = amplitude_pert;
        this->dust_render_size = dust_render_size;
        this->HII_size = HII_size;
        this->dust_temp_base = dust_temp_base;
        this->steps = steps;
        this->velocity_scaling = velocity_scaling;
        this->color_offset = color_offset;
        this->star_base_temp = star_base_temp;
        this->num_HII = num_HII;
        this->max_intensity = max_intensity;
        this->bulge = bulge;
        this->disc_scale_length = disc_scale_length;
        this->bulge_rad = bulge_rad;
        this->start_int = start_int;
        this->end_int = end_int;
        this->angular_offset = angular_offset;
        this->number_of_points = number_of_points;
        this->offset = offset;
        this->rad_far_feild = rad_far_feild;
        this->number_of_stars = number_of_stars;


        two_counts = 0;
        one_counts = 0;

        build();
    }

    ~DistributionFunction(){
    }

    private:

    void build(){
        float h = (end_int - start_int)/steps;
        float x = 0;
        float y = 0;

        //First part
        int index = 1;
        M1.clear();
        X1.clear();
        Y1.clear();

        M2.clear();
        X2.clear();
        Y2.clear();

        X1.push_back(0.0f);
        Y1.push_back(0.0f);
        M1.push_back(0.0f);
        for(int i = 0; i < steps; i += 2){
            x = (i + 2) * h;
            y += (h/3.0f) * (intensity(start_int + i * h) + 4 * 
                (intensity(start_int + (i + 1) * h) + intensity(start_int + (i + 2) * h)));
            
            M1.push_back((y - Y1.back()) / (2 * h));
            X1.push_back(x);
            Y1.push_back(y);    
            index++;
        }

        M1.erase(M1.begin());
        M1.push_back(0.0f);

        if (M1.size() != X1.size() || M1.size() != Y1.size()) {
            std::cout << "Arrays in DistributionFunction have mis-matched sizes 1!" << std::endl;
        }

        for(int i = 0; i < Y1.size(); ++i){
            Y1[i] /= Y1.back();
            M1[i] /= Y1.back();
        }

        one_counts = index;
        index = 1;
        float p;
        h = 1.0f/steps;

        X2.push_back(0.0f);
        Y2.push_back(0.0f);
        M2.push_back(0.0f);

        for(int i = 1, k = 0; i < steps; ++i){
            p = (float)i * h;

            for(; Y1[k+1] <= p; ++k){

            }

            y = X1[k] + (p - Y1[k])/M1[k];

            M2.push_back((y - Y2.back()) / h);
            X2.push_back(p);
            Y2.push_back(y);

            index++;
        }

        M2.erase(M2.begin());
        M2.push_back(0.0f);

        if (M2.size() != X2.size() || M2.size() != Y2.size()) {
            std::cout << "Arrays in DistributionFunction have mis-matched sizes 2!" << std::endl;
        }
        two_counts = index;
    }

    float intensity(float x){
        return (x < bulge_rad) ? intensity_bulge(x, max_intensity, bulge) : 
        intensity_disc(x - bulge_rad, intensity_bulge(bulge_rad, max_intensity, bulge), disc_scale_length);
    }

    float intensity_bulge(float r, float i, float b){
        return i * exp(-b * pow(r, .25));
    }
    float intensity_disc(float r, float i, float s){
        return i * exp(-r/s);
    }

    float val_from_prob(float rad){
        float h = 1.0f / (Y2.size()-1);

        int i = (int)(rad/h);
        float r = rad - i * h;

        return (Y2[i] + M2[i] * r);
    }

    float get_angular_offset(float r){
        return exp(r * angular_offset);
    }

    float get_excentricity(float r){
        if(r < core_rad){
            return 1 + (r/core_rad) * (core_edge_ecc - 1);
        }
        else if(r > core_rad && r <= gal_rad){
            return core_rad + (r - core_rad) / (gal_rad - core_rad) * (gal_edge_ecc - core_edge_ecc);
        }
        else if(r > gal_rad && r < rad_far_feild){
            return gal_edge_ecc + (r - gal_rad)/(rad_far_feild - gal_rad) * (1-gal_edge_ecc);
        }
        else{
            return 1;
        }
    }

    float get_orbital_velocity(float rad){
        float kms = 0;

        kms = VelocityCurve::v(rad, core_rad);

        float u = 2.0f * PI * rad * (3.086f * pow(10, 13)); //Parsecs to Kilometers
        float time = u / (kms * (3.154f * pow(10, 7))); //Seconds in a year

        float val = 360.0f/time;
        val *= velocity_scaling;

        return val;
    }

    Vector compute(float angle, float a, float b, float theta, int pert_N, float pert_amp){
        float alpha = angle;
        float beta = theta * 180.0f/PI;

        float cos_alpha = cos(alpha);
        float sin_alpha = sin(alpha);
        float cos_beta = cos(beta);
        float sin_beta = sin(beta);

        float x = (a * cos_alpha * cos_beta - b * sin_alpha * sin_beta);
        float y = (a * cos_alpha * sin_beta + b * sin_alpha * cos_beta);

        float z = 0;

        if(pert_amp > 0 && pert_N > 0){
            x += (a/pert_amp) * sin(alpha * 2 * pert_N);
            y += (a/pert_amp) * cos(alpha * 2 * pert_N); 
        }

        Vector pos = Vector{x,y,z};

        return pos;

    }

    public:
    Star create_random_star(){
        Vector center = offset;

        float random = ((float)rand())/RAND_MAX;

        float rad = val_from_prob(random);

        float angle = get_angular_offset(rad);
        float a = rad;
        float b = rad * get_excentricity(rad);
        float theta = 360.0f * ((float)rand())/RAND_MAX;
        float angular_velocity = get_orbital_velocity(rad);

        // float temp; unused for now
        // float mag; unuesed for now
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<float> dis(0.1f, 10.0f);

        // Sample a random mass for the star
        float mass = dis(gen);

        Vector position = compute(angle, a, b, theta, perturbations, amplitude_pert);
        position += offset;

        Vector radial = position + (center * -1.0f);

        Vector tangent = Vector{radial.y, -radial.x, 0};

        Vector velocity = tangent * (angular_velocity);

        return Star{position, velocity, {0,0,0}, mass};
    }
};

/*********Brute force**********/
__global__ void brute_force_simulation(Star* in_bodies, Star* out_bodies, int amount){
    int id = threadIdx.x + blockIdx.x * blockDim.x;
    int pvale = 0;

    out_bodies[id].mass = (float)pvale;

    Vector total_acceleration = {0.0f, 0.0f, 0.0f};

    for(int j = id; j < amount; j += blockDim.x){
        Star local_body = in_bodies[j];
        for(int k = 0; k < amount; k++){
            if(j != k){
                Star two = in_bodies[k];
                Vector distance_vector = {two.position.x - local_body.position.x,
                                two.position.y - local_body.position.y,
                                two.position.z - local_body.position.z};
        
                float distance_squared = distance_vector.x * distance_vector.x +
                                        distance_vector.y * distance_vector.y + 
                                        distance_vector.z * distance_vector.z;
                
                float distance = sqrt(distance_squared);

                float force = (G) * local_body.mass * two.mass / (distance_squared);

                Vector force_vector = {distance_vector.x/distance, distance_vector.y/distance, distance_vector.z/distance};
                force_vector = Vector{force_vector.x * force, force_vector.y * force, force_vector.z * force};

                Vector acc = {force_vector.x/local_body.mass, force_vector.y/local_body.mass, force_vector.z/local_body.mass};

                total_acceleration = Vector{acc.x + total_acceleration.x, acc.y + total_acceleration.y, acc.z + total_acceleration.z};
            }
        }
        

        Vector velocity = Vector{local_body.velocity.x + total_acceleration.x * delta,
                            local_body.velocity.y + total_acceleration.y * delta,
                            local_body.velocity.z + total_acceleration.z * delta};
        // velocity.print_device();
        Vector position = Vector{local_body.position.x + velocity.x * delta,
                                local_body.position.y + velocity.y * delta,
                                local_body.position.z + velocity.z * delta};

        out_bodies[id].position = position;
        out_bodies[id].velocity = velocity;
        out_bodies[id].mass = local_body.mass;
        out_bodies[id].acceleration = total_acceleration;
    }

    
}

bool call_kernel(Star* in_bodies, Star* out_bodies, int amount){
    
    //First allocate the device memory
    Star* device_bodies_in;
    Star* device_bodies_out;

    if(cudaMalloc(&device_bodies_in, amount * sizeof(Star)) != cudaSuccess){
        printf("CUDA ERROR: Could not allocate input device bodies.\n");
        return false;
    }
    if(cudaMalloc(&device_bodies_out, amount * sizeof(Star)) != cudaSuccess){
        printf("CUDA ERROR: Could not allocate output device bodies\n");
        return false;
    }

    //Next, copy over the host memory to device memory
    if(cudaMemcpy(device_bodies_in, in_bodies, amount * sizeof(Star), cudaMemcpyHostToDevice) != cudaSuccess){
        printf("CUDA ERROR: Could not copy over input bodies to device\n");
        return false;
    }

    //Now, we can create the kernel properties
    dim3 block_size = dim3(512, 1, 1);
    dim3 grid_size = dim3(1, 1, 1);
    brute_force_simulation<<<grid_size, block_size>>>(device_bodies_in, device_bodies_out, amount);
    cudaDeviceSynchronize();

    if(cudaMemcpy(out_bodies, device_bodies_out, amount * sizeof(Star), cudaMemcpyDeviceToHost) != cudaSuccess){
        printf("CUDA ERROR: Could not copy over output bodies to host.\n");
        return false;
    }

    cudaFree(device_bodies_in);
    cudaFree(device_bodies_out);

    return true;
}

void test_brute_force_gpu(int frames, std::vector<Star>& bodies){
    for(size_t i = 0; i < frames; i++){
        

        Star* array_bodies = (Star*)malloc(sizeof(Star) * bodies.size());
        Star* output_bodies = (Star*)malloc(sizeof(Star) * bodies.size());
        for(int i = 0; i < bodies.size(); i++){
            array_bodies[i] = bodies[i];
        }
        if(!call_kernel(array_bodies, output_bodies, bodies.size())){
            printf("Error in gpu kernel.\n");
            break;
        }

        for(int i = 0; i < bodies.size(); i++){
            bodies[i] = output_bodies[i];
        }
        free(array_bodies);
        free(output_bodies);

        std::string file_name = "Data/Err/gpu_brute_frame_" + std::to_string(i) + ".json";
        save_to_json(bodies, file_name);
    }

    
}

/******New version implementation test********/

struct DeviceNode{
    
    float octant_length;
    Vector center;
    Vector total_center;
    float total_mass;
    bool leaf;
    bool ignored;
    int number_of_bodies;

    int lock;

    __device__ DeviceNode(){
        lock = 0;
        leaf = false;
        ignored = true;

        number_of_bodies = 0;
    }

    __device__ void set_length(float start_distance, int index, Vector center, int max_depth){
        Vector base_center = center;

        int start_index = index;

        int depth = 1;

        while(start_index > 0){
            depth++;
            start_index = (start_index-1)/4;
        }

        float octant_length = start_distance/pow(2, depth-1);
        int number_of_nodes = 0;

        while(index > 0){

            octant_length *= 2.0f;

            switch((index - 1) % 4){
                case 0: base_center.x += octant_length/2.0f; base_center.y += octant_length/2.0f; break;
                case 1: base_center.x -= octant_length/2.0f; base_center.y += octant_length/2.0f; break;
                case 2: base_center.x -= octant_length/2.0f; base_center.y -= octant_length/2.0f; break;
                case 3: base_center.x += octant_length/2.0f; base_center.y -= octant_length/2.0f; break;
            }

            index = (index-1)/4;
        }
        
        leaf = (depth) == max_depth;
        this->octant_length = octant_length;
        this->center = base_center;
    }

    __device__ Star to_body(){
        return Star{total_center/total_mass, {0,0,0}, {0,0,0}, total_mass};
    }

    __device__ int find_quad(Vector position){
        if (position.x < center.x) {
            if (position.y < center.y) {
                return 2; // Bottom-left quadrant
            } else {
                return 1; // Top-left quadrant
            }
        } else {
            if (position.y < center.y) {
                return 3; // Bottom-right quadrant
            } else {
                return 0; // Top-right quadrant
            }
        }
    }

    __device__ bool add_device(Star* star){
        if(atomicCAS(&lock, 0, 1) == 0){
            total_center += star->position * star->mass;
            total_mass += star->mass;

            number_of_bodies++;
            __threadfence();

            atomicExch(&lock, 0);
            ignored = false;
            return true;
        } else {
            return false;
        }
    }
};

__global__ void find_max_depth(Star* bodies, int* found_max_depth, int num_stars, float start_distance){
    __shared__ int depths[512];

    //We're going to be doing a reduction
    //First, let's fill out the depths
    int id = threadIdx.x;
    depths[id] = -1;
    if(id == 0){
        printf("Started max finding\n");
    }
    __syncthreads();

    for(int i = id; i < num_stars; i += blockDim.x){
        Star relative_body = bodies[i];
        float min_distance = (i == 0) ? relative_body.position(bodies[1].position) : relative_body.position(bodies[0].position);
        for(int k = 0; k < num_stars; k++){
            if(k != i){
                float val = relative_body.position(bodies[k].position);
                if(val < min_distance){
                    min_distance = val;
                }
            }
        }

        int depth = ceil(log2(start_distance/min_distance));

        if(i < 512){
            depths[id] = depth;
        } else
        if(depths[id] > depth){
            depths[id] = depth;
        }
    }
    __syncthreads();


    //Now we reduce
    for(int stride = blockDim.x/2; stride > 0; stride /= 2){
        if(id + stride < 512){
            if(depths[id] < depths[id + stride]){
                depths[id] = depths[id + stride];
            }
        }
        __syncthreads();
    }
    
    if(id == 0){
        found_max_depth[0] = depths[0];
        printf("Finished max finding (%d)\n", found_max_depth[0]);
    }
}

__global__ void fill_tree(Star* stars, int number_of_stars, DeviceNode* tree, int max_depth, float start_distance, Vector center , int number_of_nodes){

    //First set octant length
    int id = threadIdx.x;
    int index = id;

    while(index < number_of_nodes){
        tree[index].set_length(start_distance, index, center, max_depth);
        index += blockDim.x;
    }

    __syncthreads();

    for(int i = id; i < number_of_stars; i += blockDim.x){
        //Add each star to the tree
        Star* star = &stars[i];
        index = 0;

        int number_of_nodes = 0;
        int depth = 0;

        while(!tree[index].leaf){
            number_of_nodes += pow(4, depth);

            while(!tree[index].add_device(star));

            int quad = tree[index].find_quad(star->position);

            depth++;
            index = 4 * index + quad + 1;
        }

        tree[index].add_device(star);

    }
}

//0 - nothing placed
// > 0 - something placed
// < 0 - removed, never place
__global__ void filter_kernel(Star* in_stars, Star* out_stars, int* placed_array, int number_of_in_stars, int* number_of_out_stars, float cut_near, float cut_far, int* locks){
    int id = threadIdx.x;

    for(int i = id; i < number_of_in_stars; i += blockDim.x){
        placed_array[i] = 0;
        locks[i] = 0;
    }

    if(id == 0){
        printf("Started filtering in kernel.\n");
        number_of_out_stars[0] = number_of_in_stars;

    }

    for(int i = id; i < number_of_in_stars; i += blockDim.x){
        Star* local_star = in_stars + i;

        Vector pos = local_star->position;
        if(pos.x > cut_far || pos.x < -cut_far || pos.y > cut_far || pos.y < -cut_far){
            placed_array[i] = -1;
        }

        if(placed_array[i] == -1){
            continue;
        }

       for(int k = id+1; k < number_of_in_stars; k++){
            Star* other_star = in_stars + k;
            if(placed_array[i] == -1){
                break;
            }

            if(other_star != local_star && placed_array[k] != -1){

                float dist = local_star->position(other_star->position);
                
                //We want to collide, but we need to make sure of a lot of things beforehand
                if(dist < cut_near){

                    // Set it up so that we want to merge. Although, we also want to make sure another star isn't trying to merge with us
                    if (atomicCAS(&(local_star->merged), 0, 1) == 0){
                        // We have to check first if the other star is waiting for a merge
                        while(true){
                            if(placed_array[k] == -1 || placed_array[i] == -1){
                                atomicExch(&(local_star->merged), 0);
                                break;
                            }
                            if (atomicCAS(&(other_star->merged), 0, 1) == 0){
                                // Great, the other star is not waiting for a merge
                                placed_array[i] = 1;
                                placed_array[k] = -1;
                                local_star->merge_with(other_star);

                                atomicExch(&(local_star->merged), 0);
                                atomicExch(&(other_star->merged), 0);

                                break;
                            } else {
                                atomicMin(number_of_out_stars, i);
                                if(*number_of_out_stars == i){
                                    placed_array[i] = 1;
                                    placed_array[k] = -1;

                                    local_star->merge_with(other_star);

                                    atomicExch(&(local_star->merged), 0);
                                    atomicExch(&(other_star->merged), 0);

                                    atomicExch(number_of_out_stars, number_of_in_stars);
                                }
                            }
                        }
                        
                    }

                }
            }
       }

       if(placed_array[i] != -1){
            placed_array[i] = 1;
       }
    }

    __syncthreads();

    if(id == 0){
        number_of_out_stars[0] = number_of_in_stars;
    }

    for(int i = id; i < number_of_in_stars; i += blockDim.x){
        locks[i] = 0;
    }

    for(int i = id; i < number_of_in_stars; i+= blockDim.x){
        out_stars[i] = in_stars[i];
    }

    __syncthreads();


    for(int i = id; i < number_of_in_stars; i += 512){
        
        int offset = 1;

        bool swap = false;
        if(placed_array[i] <= 0){
            while(true){
                while((placed_array[number_of_in_stars - offset] != 1 && number_of_in_stars - offset > i) || locks[number_of_in_stars - offset] == 1){offset++;}

                if(number_of_in_stars - offset <= i){
                    break;
                }

                if(atomicCAS(&locks[number_of_in_stars - offset], 0 , 1) == 0){
                    swap = true;
                    break;
                } else {
                    offset++;
                }
            }
            
        }

        if(swap){
            atomicMin(number_of_out_stars, number_of_in_stars - offset);
            out_stars[i] = out_stars[number_of_in_stars - offset];
        }


    }

    

    __syncthreads();
    if(id == 0){
        printf("Finished filtering in kernel.\n");
    }
    __syncthreads();

    //Don't forget to process for next time!
    for(int i = id; i < number_of_out_stars[0]; i += blockDim.x){
        out_stars[i].merged = 0;
    }

}

std::vector<float> filter_kernel_times;
Star* start_filter_kernel(Star* in_stars, int number_of_stars, float cut_near, float cut_far, int* size_pass_value, Star** host_filtered){
    Star* stars_gpu;
    Star* out_stars;
    int* placed_gpu;
    int* out_size_value;
    int* locks;

    CUDA_ERR_CHECK_NO_R(cudaMalloc(&stars_gpu, sizeof(Star) * number_of_stars), "Failed to allocated memory for in stars.", nullptr);
    CUDA_ERR_CHECK_NO_R(cudaMalloc(&out_stars, sizeof(Star) * number_of_stars), "Failed to allocated memory for out stars.", nullptr);
    CUDA_ERR_CHECK_NO_R(cudaMalloc(&placed_gpu, sizeof(int) * number_of_stars), "Failed to allocated memory for placed stars.", nullptr);
    CUDA_ERR_CHECK_NO_R(cudaMalloc(&out_size_value, sizeof(int) * 1), "Failed to allocated memory for size pass.", nullptr);
    CUDA_ERR_CHECK_NO_R(cudaMalloc(&locks, sizeof(int) * number_of_stars), "Failed to allocated memory for size pass.", nullptr);
    CUDA_ERR_CHECK_NO_R(cudaMemcpy(stars_gpu, in_stars, sizeof(Star) * number_of_stars, cudaMemcpyHostToDevice), "Failed to transfer star data to device.", nullptr);

    dim3 block_size = dim3(512, 1, 1);
    dim3 grid_size = dim3(1, 1, 1);
    cudaError_t error_check;

    cudaEvent_t start, stop;
    float time = 0.0f;

    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);

    *size_pass_value = number_of_stars;
    int prev_value = number_of_stars;
    cudaDeviceSynchronize();

    filter_kernel<<<grid_size, block_size>>>(stars_gpu, out_stars, placed_gpu, *size_pass_value, out_size_value, cut_near, cut_far, locks);
    cudaDeviceSynchronize();
    CUDA_ERR_CHECK_NO_R(cudaMemcpy(size_pass_value, out_size_value, sizeof(int) * 1, cudaMemcpyDeviceToHost), "Failed to transfer star size to host.", nullptr);

    error_check = cudaGetLastError();
    CUDA_ERR_CHECK_NO_R(error_check, "Filter failed.", nullptr);

    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time, start, stop);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    filter_kernel_times.push_back(time);

    Star* filtered_stars;

    int filtered_size = 0;
    cudaDeviceSynchronize();

    CUDA_ERR_CHECK_NO_R(cudaMalloc(&filtered_stars, sizeof(Star) * size_pass_value[0]), "Failed to allocated memory for filtered stars.", nullptr);
    CUDA_ERR_CHECK_NO_R(cudaMemcpy(filtered_stars, out_stars, sizeof(Star) * size_pass_value[0], cudaMemcpyDeviceToDevice), "Failed to copy star data.", nullptr);

    *host_filtered = (Star*)malloc(size_pass_value[0] * sizeof(Star));
    CUDA_ERR_CHECK_NO_R(cudaMemcpy(*host_filtered, filtered_stars, sizeof(Star) * size_pass_value[0], cudaMemcpyDeviceToHost), "Failed to copy star data.", nullptr);

    cudaFree(stars_gpu);    
    cudaFree(out_stars);
    cudaFree(placed_gpu);
    cudaFree(out_size_value);
    cudaFree(locks);

    return filtered_stars;

}

__global__ void compute_positions(Star* stars, int number_of_stars, DeviceNode* tree, int max_depth){
    //Now we just have to find the forces
    int id = threadIdx.x;

    for(int i = id; i < number_of_stars; i += blockDim.x){
        Vector acceleration{0,0,0};
        Star star = stars[i];

        int index = 0;
        int depth = 0;

        int curr_child[16] = {0};

        bool done = false;
        while(!done){
            // So we will recurse through the tree essentially
            bool stop = false;

            Star body = tree[index].to_body();

            if (tree[index].leaf || tree[index].number_of_bodies <= 1){
                if (tree[index].ignored || body == star){
                    stop = true;
                } else {
                    acceleration += compute_force_device(star, body, delta_gpu);
                    stop = true;
                }
            }

            if (!stop && tree[index].octant_length / (star).position(body.position) < theta[0]){
                acceleration += compute_force_device(star, body, delta_gpu);
                stop = true;
            }
            
            //Adjust index
            if (!stop){
                if(curr_child[depth] < 3){
                    curr_child[depth] += 1;
                    index = 4 * index + curr_child[depth] + 1;
                    depth++;
                    curr_child[depth] = 0;
                } else {
                    // Go back up
                    while (depth > 0 && curr_child[depth] == 3) {
                        index = (index - 1) / 4; // Move to parent index
                        depth--;
                    }
                    if (depth > 0) {
                        curr_child[depth]++;
                        index = 4 * index + curr_child[depth] + 1;
                    } else {
                        done = true;
                        break;
                    }
                }
                
            } else {
                // No children to process or we're done at this node
                if(depth == 0){
                    done = true;
                } else {
                    index = (index - 1) / 4;
                    depth--;
                }
            }
        }

        stars[i].acceleration = acceleration;

        stars[i].velocity = stars[i].velocity + acceleration * delta_gpu;
        stars[i].position = stars[i].position + stars[i].velocity * delta_gpu;
    }

    __syncthreads();

    if(threadIdx.x == 0){
        printf("Finished forces.\n");
    }

}

std::vector<float> compute_positions_time;
bool call_compute_positions(Star* stars, Star* out_stars, int number_of_stars, int max_depth, DeviceNode* tree){

    dim3 block_size = dim3(512, 1, 1);
    dim3 grid_size = dim3(1, 1, 1);
    cudaError_t error_check;

    cudaEvent_t start, stop;
    float time = 0.0f;

    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);

    cudaDeviceSynchronize();
    compute_positions<<<grid_size, block_size>>>(stars, number_of_stars, tree, max_depth);
    error_check = cudaGetLastError();
    cudaDeviceSynchronize();
    CUDA_ERR_CHECK(error_check, "Forces failed.");

    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time, start, stop);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    compute_positions_time.push_back(time);

    CUDA_ERR_CHECK(cudaMemcpy(out_stars, stars, sizeof(Star) * number_of_stars, cudaMemcpyDeviceToHost), "Failed to transfer star data to host.");

    return true;

}

std::vector<float> fill_tree_time;
DeviceNode* call_fill_tree(Star* stars, int number_of_stars, int max_depth, float start_distance, Vector center){
    DeviceNode* tree_gpu;
    // Star* stars_gpu;

    int number_of_nodes = 0;
    for(int i = 0; i < max_depth; i++){
        number_of_nodes += pow(4, i);
    }

    CUDA_ERR_CHECK_NO_R(cudaMalloc(&tree_gpu, sizeof(Node) * number_of_nodes), "Failed to allocate memory for tree nodes.", nullptr);

    dim3 block_size = dim3(512, 1, 1);
    dim3 grid_size = dim3(1, 1, 1);

    cudaEvent_t start, stop;
    float time = 0.0f;

    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);

    cudaDeviceSynchronize();
    fill_tree<<<grid_size, block_size>>>(stars, number_of_stars, tree_gpu, max_depth, start_distance, center, number_of_nodes);
    cudaDeviceSynchronize();

    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time, start, stop);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    fill_tree_time.push_back(time);

    return tree_gpu;
}

std::vector<float> max_depth_time;
int call_max_depth(Star* stars, int number_of_stars, float start_distance){

    int* out_depth;

    CUDA_ERR_CHECK(cudaMalloc(&out_depth, sizeof(int) * 1), "Failed to allocated memory for in out depth.");

    dim3 block_size = dim3(512, 1, 1);
    dim3 grid_size = dim3(1, 1, 1);
    cudaError_t error_check;

    cudaEvent_t start, stop;
    float time = 0.0f;

    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);

    find_max_depth<<<grid_size, block_size>>>(stars, out_depth, number_of_stars, start_distance);
    error_check = cudaGetLastError();
    cudaDeviceSynchronize();

    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time, start, stop);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    max_depth_time.push_back(time);


    int out_depth_host = 0;
    CUDA_ERR_CHECK(cudaMemcpy(&out_depth_host, out_depth, 1 * sizeof(int), cudaMemcpyDeviceToHost), "Failed to copy device depth result to host.");

    cudaFree(out_depth);

    return out_depth_host;
}

//CPU Filter, not used
void filter_close_stars(std::vector<Star>& stars, float filter){

    for(int i = 0; i < stars.size(); i++){
        for(int j = 0; j < stars.size(); j++){
            if(i != j){
                if(stars[i].position(stars[j].position) < filter){
                    if(stars[j].mass > stars[i].mass){
                        stars[j].mass += stars[i].mass;
                        stars[j].position = ((stars[j].position * 1/stars[j].mass )+(stars[i].position * 1/stars[i].mass)) * (stars[i].mass + stars[j].mass) * 0.5;
                        stars[j].velocity = ((stars[j].velocity * 1/stars[j].mass )+(stars[i].velocity * 1/stars[i].mass)) * (stars[i].mass + stars[j].mass) * 0.5;

                        stars.erase(stars.begin() + i);
                    } else {
                        stars[i].mass += stars[j].mass;
                        stars[i].position = ((stars[j].position * 1/stars[j].mass )+(stars[i].position * 1/stars[i].mass)) * (stars[i].mass + stars[j].mass) * 0.5;
                        stars[i].velocity = ((stars[j].velocity * 1/stars[j].mass )+(stars[i].velocity * 1/stars[i].mass)) * (stars[i].mass + stars[j].mass) * 0.5;

                        stars.erase(stars.begin() + j);
                    }
                }
            }
        }

        if(stars[i].position.x > 4000.0f || stars[i].position.y > 4000.0f || stars[i].position.x < -4000.0f || stars[i].position.y < -4000.0f){
            stars.erase(stars.begin() + i);
        }
    }

}


std::vector<float> filter_time;
bool process_sim(std::vector<Star>& stars, int frame, bool save_frames){
    printf("Started frame %d\n", frame);

    Star* input_stars = (Star*)malloc(sizeof(Star) * stars.size());
    for(int i = 0; i < stars.size(); i++){
        input_stars[i] = stars[i];
    }

    printf("Starting filtering.\n");
    int star_size = 0;
    Star* host_filtered;
    Star* filtered_stars = start_filter_kernel(input_stars, stars.size(), 0.75f, 2000.0f, &star_size, &host_filtered);
    printf("Finished filtering (new star length %d)\n", star_size);
    if(filtered_stars == nullptr){
        return false;
    }

    stars.clear();
    stars.resize(star_size);
    for(int i = 0; i < star_size; i++){
        stars[i] = host_filtered[i];
    }

    Node* root = create_root(stars);

    int depth = call_max_depth(filtered_stars, star_size, root->octant_length);
    if(depth == 0){
        return false;
    }

    printf("Found max depth: %d\n", depth);

    DeviceNode* tree = call_fill_tree(filtered_stars, star_size, depth, root->octant_length, root->center);
    if(tree == nullptr){
        return false;
    }

    printf("Filled tree.\n");

    Star* output_stars = (Star*)malloc(sizeof(Star) * star_size);
    if(!call_compute_positions(filtered_stars, output_stars, star_size, depth, tree)){return false;}

    stars.clear();
    stars.resize(star_size);
    for(int i = 0; i < star_size; i++){
        stars[i] = output_stars[i];
        if(stars[i].merged == 1){
        }
        stars[i].merged = 0;
    }

    printf("Finished forces kernel\n");


    if(save_frames){
        std::string file_name = "Frames/GPUBHT/frame_" + std::to_string(frame) + ".json";
        save_to_json(stars, file_name);
    }

    printf("Finished saving data\n");
    
    free(output_stars);
    cudaFree(tree);
    cudaFree(filtered_stars);
    free(root);
    free(input_stars);
    printf("Finished frame %d\n", frame);
    return true;
}

/*********************************************/

std::vector<Star> test_distribution(int number_main, int number_side){
    int m_gal_rad = 90;
    int m_core_rad = 7.5;
    float m_angular_offset = .0004f;
    float m_core_edge_ecc = 0.85f;
    float m_gal_edge_ecc = 0.95f;
    int m_orbital_vel_core = 15;
    int m_orbital_vel_gal = 30;
    int m_num_stars = number_main;
    bool m_dark_matter = true;
    int m_perturbations = 200;
    int m_amplitude_pert = 40;
    int m_dust_render_size = 24;
    int m_HII_render_size = 10;
    float m_dust_temp_base = 9000.0f;
    int m_steps = 15000;
    float m_velocity_scaling = 100.0f;
    int m_colour_offset = 0;
    float m_star_base_temp = 3000.0f;
    int m_num_HII = 100;
    
    float m_rad_far_field = m_gal_rad * 2.0f;

    float m_max_intensity = 1.0f;
    float m_bulge = 0.2f;
    float m_disc_scale_length = m_gal_rad / 3.0f;
    float m_bulge_rad = m_core_rad;
    float m_start_int = 0.0f;
    float m_end_int = m_rad_far_field;

    Vector m_offset = {0,0,0};

    DistributionFunction main_galaxy = DistributionFunction(
        m_max_intensity, m_bulge, m_disc_scale_length, m_bulge_rad, m_start_int, m_end_int,
        m_angular_offset, m_steps, m_gal_rad, m_core_rad, m_core_edge_ecc, m_gal_edge_ecc, 
        m_orbital_vel_core, m_orbital_vel_gal, m_dark_matter, m_perturbations, m_amplitude_pert,
        m_dust_render_size, m_HII_render_size, m_dust_temp_base, m_steps, m_velocity_scaling, 
        m_colour_offset, m_star_base_temp, m_num_HII, m_rad_far_field, m_num_stars, m_offset
    );

    int n_gal_rad = 75;
    int n_core_rad = 15;
    float n_angular_offset = .0004f;
    float n_core_edge_ecc = 0.85f;
    float n_gal_edge_ecc = 0.95f;
    int n_orbital_vel_core = 20;
    int n_orbital_vel_gal = 30;
    int n_num_stars = number_side;
    bool n_dark_matter = true;
    int n_perturbations = 250;
    int n_amplitude_pert = 40;
    int n_dust_render_size = 24;
    int n_HII_render_size = 10;
    float n_dust_temp_base = 9000.0f;
    int n_steps = 10000;
    float n_velocity_scaling = 100.0f;
    int n_colour_offset = 0;
    float n_star_base_temp = 3000.0f;
    int n_num_HII = 100;
    
    float n_rad_far_field = m_gal_rad * 2.0f;

    float n_max_intensity = 1.0f;
    float n_bulge = 0.3f;
    float n_disc_scale_length = m_gal_rad / 3.0f;
    float n_bulge_rad = m_core_rad;
    float n_start_int = 0.0f;
    float n_end_int = m_rad_far_field;

    Vector n_offset = {1000,1000,0};

    
    DistributionFunction side_galaxy = DistributionFunction(
        n_max_intensity, n_bulge, n_disc_scale_length, n_bulge_rad, n_start_int, n_end_int,
        n_angular_offset, n_steps, n_gal_rad, n_core_rad, n_core_edge_ecc, n_gal_edge_ecc, 
        n_orbital_vel_core, n_orbital_vel_gal, n_dark_matter, n_perturbations, n_amplitude_pert,
        n_dust_render_size, n_HII_render_size, n_dust_temp_base, n_steps, n_velocity_scaling, 
        n_colour_offset, n_star_base_temp, n_num_HII, n_rad_far_field, n_num_stars, n_offset
    );
    std::vector<Star> stars(m_num_stars + n_num_stars + 2);

    for(size_t i = 0; i < m_num_stars; i++){
        stars[i] = main_galaxy.create_random_star();
        // stars[i].print();
    }
    for(size_t i = 0; i < n_num_stars; i++){
        stars[i + m_num_stars] = side_galaxy.create_random_star();
        // stars[i].print();
    }
    //6300000   
    stars[m_num_stars + n_num_stars] = Star{m_offset + Vector{0.01, 0.023, 0.2345}, {0,0,0}, {0,0,0}, 63000};
    // stars[m_num_stars + n_num_stars+1] = Star{n_offset, {-.0005f,0,0}, {0,0,0}, 4900000};

    // save_to_json(stars, "Data/init_galaxy.json");

    return stars;

}

std::vector<Star> create_test_stars() {
    std::vector<Star> stars;

    // Define 10 predefined starting variables
    std::vector<Vector> positions = {
        {0.1f, 0.1f, 0.1f},
        {0.3f, 0.2f, 0.2f},
        {0.4f, 0.1f, 0.3f},
        {-0.3f, -0.2f, -0.1f},
        {-0.2f, 0.3f, -0.4f},
        {-0.1f, -0.1f, -0.5f},
        {0.5f, -0.2f, 0.2f},
        {0.2f, -0.3f, 0.3f},
        {-0.4f, 0.2f, -0.3f},
        {0.1f, 0.4f, -0.1f},
    };

    std::vector<Vector> velocities = {
        {0.5f, 0.5f, 0.5f},
        {0.3f, 0.4f, 0.2f},
        {0.1f, 0.2f, 0.6f},
        {-0.3f, -0.1f, 0.2f},
        {0.3f, -0.2f, 0.3f},
        {0.4f, 0.3f, -0.1f},
        {0.2f, 0.1f, 0.1f},
        {-0.4f, -0.3f, -0.2f},
        {0.1f, -0.2f, -0.1f},
        {0.3f, 0.5f, -0.4f},
    };

    std::vector<Vector> accelerations = {
        {0.05f, 0.05f, 0.05f},
        {0.03f, 0.04f, 0.02f},
        {0.01f, 0.02f, 0.06f},
        {-0.03f, -0.01f, 0.02f},
        {0.03f, -0.02f, 0.03f},
        {0.04f, 0.03f, -0.01f},
        {0.02f, 0.01f, 0.01f},
        {-0.04f, -0.03f, -0.02f},
        {0.01f, -0.02f, -0.01f},
        {0.03f, 0.05f, -0.04f},
    };

    std::vector<float> masses = {
        1.0f,
        2.0f,
        3.0f,
        0.5f,
        0.75f,
        1.5f,
        1.2f,
        1.8f,
        0.9f,
        2.5f,
    };

    // Create the 10 stars with the pre-defined values
    for (int i = 0; i < 10; ++i) {
        stars.emplace_back(positions[i], velocities[i], accelerations[i], masses[i]);
    }

    stars.push_back(Star({10,0,0}, {0,0,0}, {0,0,0}, 1));

    return stars;
}

int main(){
    
    srand(100);
    

    std::vector<int> main_nums{15, 30, 75, 100, 150, 500, 1000, 1500, 2500, 4000, 7500, 10000, 12500, 15000};
    std::vector<int> side_nums{8, 16, 32, 50, 80, 160, 300, 500, 800, 1200, 1500, 2000, 5000, 8000};

    std::vector<int> frames{25};
    std::vector<float> thetas{0.1f, 0.25f, 0.5f, 0.75f, 1.0f};

    std::vector<float> total_times;
    std::vector<float> frame_times;

    int iter = 0;

    std::vector<Star> stars = test_distribution(5000, 0);
    // filter_close_stars(stars, 0.5f);

    // std::vector<Star> stars = create_test_stars();
    

    float host_theta = 0.5f;

    // for(int k = 0; k < thetas.size(); k++){
    CUDA_ERR_CHECK_NO_R(cudaMemcpyToSymbol(theta, &host_theta, sizeof(float), 0, cudaMemcpyHostToDevice), "Could not copy theta.", 1);
        // std::vector<Star> copy = stars;

    //Uncomment this to run the simulation

    // for(int k = 0; k < 5000; k++){
    //     process_sim(stars, k, true);
    //     filter_kernel_times.clear();
    //     max_depth_time.clear();
    //     fill_tree_time.clear();
    //     compute_positions_time.clear();
    // // std::string filename = "Data/Err/gpu_bht_"+std::to_string(k) +".json";
    //     // save_to_json(copy, filename);
    //     // for(int i = 0; i < stars.size(); i++){
    //     //     stars[i].print();
    //     // }
    // }
    //     std::string filename = "Data/Error Cmp/gpu_bht_"+std::to_string(k) +".json";
    //     save_to_json(copy, filename);
    // }
    //                 test_brute_force_gpu(25, stars);
    // save_to_json(stars, "Data/Error Cmp/gpu_brute.json");

        // test_brute_force_gpu(1, stars);

        // filter_time.clear();
        // max_depth_time.clear();
        // fill_tree_time.clear();
        // compute_positions_time.clear();

    //This is to test and evaluate the simulation
    for(int i = 0; i < main_nums.size(); i++){
        for(int j = 0; j < frames.size(); j++){
            for(int k = 0; k < thetas.size(); k++){
                filter_time.clear();
                max_depth_time.clear();
                fill_tree_time.clear();
                compute_positions_time.clear();
                frame_times.clear();

                int main = main_nums[i];
                int side = side_nums[i];
                int total_frames = frames[j];
                float host_theta = thetas[k];

                cudaEvent_t total_start, total_stop;
                float total_time = 0.0f;

                cudaEventCreate(&total_start);
                cudaEventCreate(&total_stop);
                cudaEventRecord(total_start, 0);

                printf("Starting settings: main %d, side %d, total frames %d, theta %f.\n", main, side, total_frames, host_theta);

                std::vector<Star> stars = test_distribution(main, side);
                filter_close_stars(stars, 0.5f);

                CUDA_ERR_CHECK_NO_R(cudaMemcpyToSymbol(theta, &host_theta, sizeof(float), 0, cudaMemcpyHostToDevice), "Could not copy theta.", 1);

                for(int f = 0; f < total_frames; f++){
                    cudaEvent_t frame_start, frame_stop;
                    float frame_time = 0.0f;

                    cudaEventCreate(&frame_start);
                    cudaEventCreate(&frame_stop);
                    cudaEventRecord(frame_start, 0);
                    
                    process_sim(stars, f, false);
                    cudaEventRecord(frame_stop, 0);
                    cudaEventSynchronize(frame_stop);
                    cudaEventElapsedTime(&frame_time, frame_start, frame_stop);
                    cudaEventDestroy(frame_start);
                    cudaEventDestroy(frame_stop);
                    frame_times.push_back(frame_time);
                }

                cudaEventRecord(total_stop, 0);
                cudaEventSynchronize(total_stop);
                cudaEventElapsedTime(&total_time, total_start, total_stop);
                cudaEventDestroy(total_start);
                cudaEventDestroy(total_stop);
                total_times.push_back(total_time);

                std::string file_name_filter = "Data/Times/setting_" + std::to_string(iter) + "_filter_times" + ".json";
                std::string file_name_depth = "Data/Times/setting_" + std::to_string(iter) + "_max_depth_times" + ".json";
                std::string file_name_tree = "Data/Times/setting_" + std::to_string(iter)  + "_fill_tree_times" + ".json";
                std::string file_name_positions = "Data/Times/setting_" + std::to_string(iter) + "_compute_positions_times" + ".json";
                std::string file_name_frames = "Data/Times/setting_" + std::to_string(iter)  + "_frame_times_brute" + ".json";
                save_times_2(filter_kernel_times, main, side, total_frames, host_theta, file_name_filter);
                save_times_2(max_depth_time, main, side, total_frames, host_theta, file_name_depth);
                save_times_2(fill_tree_time, main, side, total_frames, host_theta, file_name_tree);
                save_times_2(compute_positions_time, main, side, total_frames, host_theta, file_name_positions);
                save_times_2(frame_times, main, side, total_frames, host_theta, file_name_frames);

                iter++;

            }
        }
    }
}
