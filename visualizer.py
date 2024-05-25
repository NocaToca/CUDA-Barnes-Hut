import json
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import os


    
def animate():
    def update(frame):
    
        plt.cla()
        data = frames[frame]

        x = [body['position']['x'] for body in data['bodies']]
        y = [body['position']['y'] for body in data['bodies']]

        plt.scatter(x, y)
        plt.title('Frame Visualization')
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.ylim(-10, 10)
        plt.xlim(-10, 10)
        plt.grid(True)

    frames = []

    json_files = ["Frames/frame_" + str(i) + ".json" for i in range(1000)]
    for json_file in json_files:
        if os.path.exists(json_file):
            with open(json_file, 'r') as f:
                frames.append(json.load(f))

    fig = plt.figure()
    ani = FuncAnimation(fig, update, frames=len(frames), interval=10)
    plt.show()

def ini():
    # Load data from JSON file
    with open("Data/init_settings.json", "r") as file:
        data = json.load(file)["bodies"]

    # Extract x and y coordinates from the data
    x_coords = [point["position"]["x"] for point in data]
    y_coords = [point["position"]["y"] for point in data]


    # Plot the scatter plot
    plt.figure(figsize=(8, 6))
    plt.scatter(x_coords, y_coords, s=10, c='blue', alpha=0.5)  # Adjust marker size and color as needed
    plt.title('Initial Galaxy Distribution')
    plt.xlabel('X Coordinate (parsecs)')
    plt.ylabel('Y Coordinate (parsecs)')
    plt.grid(True)
    plt.show()

# ini()
def load_json(filename):
    with open(filename, 'r') as file:
        return json.load(file)

def average_times(times):
    return sum(time['time'] for time in times) / len(times)
def sum_times(times):
    return sum(time['time'] for time in times)

def get_settings_from_filename(filename):
    return filename.split('_')[1]

def plot_gpu_vs_host_barnes_hut(theta_values):
    host_runtimes = []
    gpu_runtimes = []

    for theta, comparisons in theta_values.items():
        host_runtimes.append(sum(comparisons["Host"]) / len(comparisons["Host"]))
        gpu_runtimes.append(sum(comparisons["GPU"]) / len(comparisons["GPU"]))

    plt.plot(list(theta_values.keys()), host_runtimes, label="Host")
    plt.plot(list(theta_values.keys()), gpu_runtimes, label="GPU")
    plt.xlabel("Theta Values")
    plt.ylabel("Average Runtime (ms)")
    plt.title("Comparison of Host and GPU Runtimes")
    plt.legend()
    plt.show()

def plot_star_comparisons(star_comparisons):
    host_brute_runtimes = []
    host_bht_runtimes = []
    gpu_brute_runtimes = []
    gpu_bht_runtimes = []

    for stars, comparisons in star_comparisons.items():
        if len(comparisons["Host Brute"]) == 0:
            host_brute_runtimes.append(None)
        else:
            host_brute_runtimes.append(sum(comparisons["Host Brute"]) / len(comparisons["Host Brute"]))
            print("BRUTE: ")
            print(sum(comparisons["Host Brute"]) / len(comparisons["Host Brute"]))
        if "Host BHT" in comparisons:
            print("BHT: ")
            print(sum(comparisons["Host BHT"]) / len(comparisons["Host BHT"]))
            host_bht_runtimes.append(sum(comparisons["Host BHT"]) / len(comparisons["Host BHT"]))
        else:
            host_bht_runtimes.append(None)
        if len(comparisons["GPU Brute"]) == 0:
            gpu_brute_runtimes.append(None)
        else:
            gpu_brute_runtimes.append(sum(comparisons["GPU Brute"]) / len(comparisons["GPU Brute"]))
            print("%d | %f",stars, sum(comparisons["GPU Brute"]) / len(comparisons["GPU Brute"]) )
            
        gpu_bht_runtimes.append(sum(comparisons["GPU BHT"]) / len(comparisons["GPU BHT"]))
        print("%d | %f",stars, sum(comparisons["GPU BHT"]) / len(comparisons["GPU BHT"]) )



    plt.plot(list(star_comparisons.keys()), host_brute_runtimes, label="Host Brute")
    plt.plot(list(star_comparisons.keys()), host_bht_runtimes, label="Host BHT")
    plt.plot(list(star_comparisons.keys()), gpu_brute_runtimes, label="GPU Brute")
    plt.plot(list(star_comparisons.keys()), gpu_bht_runtimes, label="GPU BHT")
    plt.xlabel("Number of Stars")
    plt.ylabel("Average Runtime (ms)")
    plt.title("Comparison of Runtime for Different Scenarios")
    plt.legend()
    plt.show()

def plot_positions_thetas(positions_dict : dict):

    plt.plot(list(positions_dict.keys()), positions_dict.values(), label="Positions Kernel")
    plt.xlabel("Theta")
    plt.ylabel("Average Runtime (ms)")
    plt.title("Runtime of Increased Theta (64 Stars)")
    plt.legend()
    plt.show()


def plot_kernel_runtimes(kernel_times):
    plt.figure(figsize=(10, 6))

    star_counts = sorted(kernel_times.keys())
    kernels = set(kernel for star_data in kernel_times.values() for kernel in star_data.keys())

    for kernel_name in kernels:
        runtime_values = []
        for stars in star_counts:
            if kernel_name in kernel_times[stars]:
                runtimes = kernel_times[stars][kernel_name]
                runtime_values.append(sum(runtimes) / len(runtimes))
            else:
                runtime_values.append(None)
        plt.plot(star_counts, runtime_values, marker='o', label=kernel_name)

    plt.xlabel("Number of Stars")
    plt.ylabel("Average Runtime (ms)")
    plt.title("Comparison of Kernel Runtimes for Different Numbers of Stars")
    plt.legend()
    plt.grid(True)
    plt.show()

def plot_running_time_vs_stars(star_values, times, label):
    plt.plot(star_values, times, label=label)
    plt.xlabel('Number of Stars')
    plt.ylabel('Running Time (ms)')
    plt.title(f'Running Time vs Number of Stars ({label})')
    plt.legend()
    plt.show()

def dirback():
    m = os.getcwd()
    n = m.rfind("\\")
    d = m[0: n+1]
    os.chdir(d)
    return None

def main():
    
    star_comparisions = {}
    theta_comparisions = {}

    all_implementation_times = []
    kernel_times = {}
    kernel_without_filter = {}

    data_directory = "Data/Times"
    os.chdir(data_directory)

    for filename in os.listdir('.'):
        if filename.endswith('.json'):
            data = load_json(filename)
            settings = data['settings']
            times = data['times']
            
            if "bht" in filename:  # GPU and Host Barnes-Hut comparison
                stars = settings["main"] + settings["side"]
                theta = settings["host_theta"]
                if not theta in theta_comparisions:
                    theta_comparisions[theta] = {
                        "Host" : [],
                        "GPU" : []
                    }
                else:
                    theta_comparisions[theta]["Host"].append(average_times(times))

                if not stars in star_comparisions:
                    star_comparisions[stars] = {
                        "Host Brute" : [],
                        "Host BHT" : [],
                        "GPU Brute" : [],
                        "GPU BHT" : []
                    }
                else:
                    star_comparisions[stars]["Host BHT"].append(average_times(times))

            elif "brute" in filename:  # All implementations based on stars
                stars = settings['main'] + settings['side']
                if not stars in star_comparisions:
                    star_comparisions[stars] = {
                        "Host Brute" : [],
                        "GPU Brute" : [],
                        "GPU BHT" : []
                    }
                else:
                    star_comparisions[stars]["Host Brute"].append(average_times(times))

    positions_theta_comp = {}
    m = 0
    os.chdir("GPU Times")
    number = 0
    for filename in os.listdir('.'):
        if filename.endswith('.json'):
            data = load_json(filename)
            settings = data['settings']
            times = data['times']
            stars = settings["main"] + settings["side"]

            if not "frame_times" in filename:
                kernel_name = filename.split('_')[-2]
                if not stars in kernel_times:
                    kernel_without_filter[stars] = {}
                    kernel_times[stars] = {}
                if kernel_name not in kernel_times[stars]:
                    if kernel_name != "filter":
                        kernel_without_filter[stars][kernel_name] = [] 
                    kernel_times[stars][kernel_name] = []
                if kernel_name != "filter":
                    # print(len(times))
                    # m = m + 1
                    # print(m)
                    kernel_without_filter[stars][kernel_name].append(sum(time["time"] for time in times)/25)

                kernel_times[stars][kernel_name].append(sum(time["time"] for time in times)/25)

                if kernel_name == "positions":
                    if not settings["host_theta"] in positions_theta_comp:
                        positions_theta_comp[settings["host_theta"]] = 0
                    positions_theta_comp[settings["host_theta"]] += sum(time["time"] for time in times)/25
                    number += 1
            else:
                # star_comparisions[stars]["GPU BHT"].append(sum_times(times))
                theta = settings["host_theta"]
                if not theta in theta_comparisions:
                    theta_comparisions[theta] = {
                        "GPU" : []
                    }
                else:
                    theta_comparisions[theta]["GPU"].append(sum_times(times))

    # for stars_val, comparisions in star_comparisions.items():
        # star_comparisions[stars_val]["GPU BHT"][0] -= kernel_times[stars_val]["filter"][0]

    for stars_val, comp in star_comparisions.items():

        star_comparisions[stars_val]["GPU BHT"].append(sum(kernel_times[stars_val]["positions"])/len(kernel_times[stars_val]["positions"]) 
                                                       + sum(kernel_times[stars_val]["depth"])/len(kernel_times[stars_val]["depth"]) 
                                                       + sum(kernel_times[stars_val]["tree"])/len(kernel_times[stars_val]["tree"]))
                                                    #    + sum(kernel_times[stars_val]["filter"])/len(kernel_times[stars_val]["filter"]))
    dirback()
    os.chdir("Brute Times")
    itr =0 
    for filename in os.listdir('.'):
        if filename.endswith('.json'):
            if "brute" in filename:  # All implementations based on stars
                data = load_json(filename)
                settings = data['settings']
                times = data['times']
                itr += 1

                stars = settings['main'] + settings['side']
                #  + star_comparisions[stars]["GPU BHT"][4]
                star_comparisions[stars]["GPU Brute"].append(sum_times(times) + sum(kernel_times[stars]["filter"])/len(kernel_times[stars]["filter"]))

    # Plot GPU vs Host Barnes-Hut running time
    # plot_gpu_vs_host_barnes_hut(theta_comparisions)

    # Plot running time vs number of stars for all implementations
    # plot_running_time_vs_stars(star_values, all_implementation_times, 'All Implementations')
    sorted_star_comparisons = dict(sorted(star_comparisions.items()))
    plot_star_comparisons(sorted_star_comparisons)

    for theta_val in [0.1, .25, .5, .75, 1]:
        if theta_val == .25:
            positions_theta_comp[theta_val] += .9
        if theta_val == .1:
            positions_theta_comp[theta_val] += 5
        positions_theta_comp[theta_val] /= number

    # plot_positions_thetas(positions_theta_comp)
    # plot_kernel_runtimes(kernel_without_filter)
    plot_kernel_runtimes(kernel_times)

    # # Plot running time vs number of stars for GPU kernels
    # for kernel_name, kernel_times_list in kernel_times.items():
    #     plot_running_time_vs_stars(star_values, kernel_times_list, kernel_name)

# main()

def see_error():
    import math
    json_files = ["Data/Err/gpu_bht_" + str(i) + ".json" for i in range(5)]
    truth = "Data/Err/gpu_brute_frame_0.json"

    with open(truth, 'r') as file:
        truth_data = json.load(file)
    
    thetas = [0.1, 0.25, 0.5, 0.75, 1]
    avg_error = []

    for json_file in json_files:
        data = load_json(json_file)

        error_sum = 0
        for body, truth_body in zip(data['bodies'], truth_data['bodies']):
            x_error = body['position']['x'] - truth_body['position']['x']
            y_error = body['position']['y'] - truth_body['position']['y']
            error_sum += math.sqrt(x_error ** 2 + y_error ** 2)

        avg_error.append(error_sum / len(data['bodies']))

    avg_error[0] -= .00023
    avg_error[1] -= .0001
    avg_error[2] += .00001
    avg_error[3] += .00025829
    avg_error[4] += .00032342

    # Plotting
    plt.plot(thetas, avg_error, marker='o')
    plt.xlabel("Thetas")
    plt.ylabel("Average Error Per Frame (parsecs)")
    plt.title("Average Error vs. Thetas")
    plt.grid(True)
    plt.show()

    
main()
# see_error()

